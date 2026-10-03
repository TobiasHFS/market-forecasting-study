"""Memory-bounded feature extraction for the equally spaced market bars.

The competition Feather files contain one very large record batch.  Reading the
whole market table therefore expands to more than ten gigabytes.  This module
projects the data in three independent passes instead:

1. ``sample_id`` and ``seconds_before_predict`` build group boundaries and a
   compact uint8 physical-time bucket for every source row;
2. level-1 and level-2 book columns are processed separately; and
3. aggregate-trade columns are processed last.

At no point is ``sample_id`` returned as a feature.  Four columns prefixed with
``ref_`` expose the latest valid price/depth state for later cross-source
normalisation: ``ref_mid``, ``ref_spread``, ``ref_l1_depth`` and cumulative
``ref_l2_depth``.

The implementation assumes rows for each sample are contiguous and sample IDs
are exactly 0, ..., n_samples - 1.  It validates that invariant explicitly.
Rows within a sample need not be perfectly ordered for window membership, but a
timestamp-inversion diagnostic is returned because the return/OFI calculations
use file order as chronological order.
"""

from __future__ import annotations

import gc
import math
import tempfile
from pathlib import Path
from typing import Sequence

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather


DEFAULT_WINDOWS = (10.0, 30.0, 60.0, 180.0, 600.0)
_OUTSIDE_WINDOW = np.uint8(255)


TIME_FEATURES = [
    "market_row_count",
    "market_time_span_s",
    "market_newest_age_s",
    "market_oldest_age_s",
    "market_invalid_time_frac",
    "market_time_inversion_frac",
    "market_duplicate_time_frac",
    "market_row_cap_999",
]

REFERENCE_FEATURES = [
    "ref_mid",
    "ref_spread",
    "ref_l1_depth",
    "ref_l2_depth",
    "ref_level2_depth",
    "ref_bid_price_1",
    "ref_ask_price_1",
    "ref_bid_price_2",
    "ref_ask_price_2",
]

L1_FEATURES = [
    "book_valid_frac",
    "book_invalid_nonpositive_price_frac",
    "book_crossed_frac",
    "book_nonpositive_depth_frac",
    "book_rel_spread_mean",
    "book_rel_spread_last",
    "book_l1_imbalance_mean",
    "book_l1_imbalance_last",
    "book_microprice_rel_mid_mean",
    "book_microprice_spread_units_mean",
    "book_log1p_l1_depth_mean",
    "book_l1_ask_depth_mean",
    "book_l1_bid_depth_mean",
    "book_ask_price_1_mean",
    "book_bid_price_1_mean",
    "book_mid_return",
    "book_mid_realized_vol",
    "book_mid_abs_variation",
    "book_mid_range_rel",
    "book_ofi_l1_depth_scaled",
    "book_valid_count_log1p",
]

L2_FEATURES = [
    "book_level2_valid_frac",
    "book_level2_rel_spread_mean",
    "book_level2_imbalance_mean",
    "book_log1p_level2_depth_mean",
    "book_level2_ask_depth_mean",
    "book_level2_bid_depth_mean",
    "book_ask_price_2_mean",
    "book_bid_price_2_mean",
    "book_ofi_l2_depth_scaled",
]

DERIVED_L2_FEATURES = [
    "book_l2_cumulative_imbalance",
    "book_l2_to_l1_depth_ratio",
    "book_ask_l2_gap_rel_ref_mid",
    "book_bid_l2_gap_rel_ref_mid",
]

TRADE_FEATURES = [
    "market_no_trade_frac",
    "market_trade_aggregate_invalid_frac",
    "market_transaction_volume_log1p",
    "market_transaction_count_log1p",
    "market_transaction_volume_per_trade_log1p",
    "market_transaction_vwap_rel_ref_mid",
    "market_transaction_vwap_spread_units",
    "market_transaction_avgprice_std_rel_ref_mid",
    "market_transaction_last_avgprice_rel_ref_mid",
    "market_transaction_avgprice_range_rel_ref_mid",
    "market_transaction_valid_count_log1p",
]

# Numba cannot treat global reflected Python string lists as compile-time
# constants, so kernels use explicit widths.  The assertions protect these
# constants from drifting when feature definitions are edited.
_N_TIME_FEATURES = 8
_N_L1_FEATURES = 21
_N_L2_FEATURES = 9
_N_DERIVED_L2_FEATURES = 4
_N_TRADE_FEATURES = 11
assert len(TIME_FEATURES) == _N_TIME_FEATURES
assert len(L1_FEATURES) == _N_L1_FEATURES
assert len(L2_FEATURES) == _N_L2_FEATURES
assert len(DERIVED_L2_FEATURES) == _N_DERIVED_L2_FEATURES
assert len(TRADE_FEATURES) == _N_TRADE_FEATURES


def _window_suffix(window: float) -> str:
    if float(window).is_integer():
        return f"{int(window)}s"
    return f"{window:g}s"


def _validate_windows(windows: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(windows), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("windows must be a non-empty one-dimensional sequence")
    if len(values) >= int(_OUTSIDE_WINDOW):
        raise ValueError("at most 254 windows are supported by the uint8 row bucket")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("windows must contain finite, positive seconds")
    if np.any(values[1:] <= values[:-1]):
        raise ValueError("windows must be strictly increasing")
    return values


def _read_projected(
    path: Path, columns: Sequence[str]
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    """Read only requested columns and expose their Arrow buffers as NumPy.

    The competition files have a single record batch.  Refusing multi-chunk
    inputs avoids a silent ``combine_chunks`` copy of a multi-gigabyte pass.
    Nullable columns (notably transaction_avgprice) necessarily copy once so
    that Arrow nulls become NaN.
    """

    table = feather.read_table(
        str(path), columns=list(columns), memory_map=True, use_threads=True
    )
    arrays: dict[str, np.ndarray] = {}
    for name in columns:
        column = table[name]
        if column.num_chunks != 1:
            raise RuntimeError(
                f"{path} column {name!r} has {column.num_chunks} chunks; "
                "this extractor intentionally avoids a memory-heavy combine_chunks copy"
            )
        chunk = column.chunk(0)
        arrays[name] = chunk.to_numpy(zero_copy_only=chunk.null_count == 0)
    return table, arrays


def _release_arrow() -> None:
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except (AttributeError, NotImplementedError):
        pass


@nb.njit(cache=True)
def _group_starts(sample_id: np.ndarray, n_samples: int) -> tuple[np.ndarray, int]:
    """Return group offsets and an error code for contiguous integer IDs."""

    starts = np.full(n_samples + 1, -1, dtype=np.int64)
    if n_samples <= 0 or len(sample_id) == 0:
        return starts, 1
    first = int(sample_id[0])
    if first != 0:
        return starts, 2
    starts[0] = 0
    previous = first
    groups = 1
    for i in range(1, len(sample_id)):
        current = int(sample_id[i])
        if current != previous:
            if current != previous + 1 or current >= n_samples:
                return starts, 3
            starts[current] = i
            previous = current
            groups += 1
    if previous != n_samples - 1 or groups != n_samples:
        return starts, 4
    starts[n_samples] = len(sample_id)
    return starts, 0


@nb.njit(parallel=True, cache=True)
def _time_features_and_bucket(
    starts: np.ndarray, seconds: np.ndarray, windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(starts) - 1
    output = np.full((n_samples, _N_TIME_FEATURES), np.nan, dtype=np.float32)
    bucket = np.full(len(seconds), _OUTSIDE_WINDOW, dtype=np.uint8)

    for sample in nb.prange(n_samples):
        start = starts[sample]
        end = starts[sample + 1]
        count = end - start
        invalid = 0
        inversions = 0
        duplicates = 0
        adjacent_valid = 0
        minimum = math.inf
        maximum = -math.inf
        previous = 0.0
        have_previous = False

        for row in range(start, end):
            age = float(seconds[row])
            if not math.isfinite(age) or age < 0.0:
                invalid += 1
                continue
            if age < minimum:
                minimum = age
            if age > maximum:
                maximum = age
            if have_previous:
                adjacent_valid += 1
                # Source rows are expected to move from older to newer, so the
                # seconds-before-predict value should be non-increasing.
                if age > previous:
                    inversions += 1
                elif age == previous:
                    duplicates += 1
            previous = age
            have_previous = True

            for window_idx in range(len(windows)):
                if age <= windows[window_idx]:
                    bucket[row] = np.uint8(window_idx)
                    break

        output[sample, 0] = np.float32(count)
        if maximum > -math.inf:
            output[sample, 1] = np.float32(maximum - minimum)
            output[sample, 2] = np.float32(minimum)
            output[sample, 3] = np.float32(maximum)
        if count > 0:
            output[sample, 4] = np.float32(invalid / count)
        if adjacent_valid > 0:
            output[sample, 5] = np.float32(inversions / adjacent_valid)
            output[sample, 6] = np.float32(duplicates / adjacent_valid)
        else:
            output[sample, 5] = np.float32(0.0)
            output[sample, 6] = np.float32(0.0)
        output[sample, 7] = np.float32(1.0 if count >= 999 else 0.0)
    return output, bucket


@nb.njit(inline="always")
def _valid_l1(ask: float, bid: float, ask_volume: float, bid_volume: float) -> bool:
    return (
        math.isfinite(ask)
        and math.isfinite(bid)
        and ask > 0.0
        and bid > 0.0
        and ask >= bid
        and ask_volume >= 0.0
        and bid_volume >= 0.0
        and ask_volume + bid_volume > 0.0
    )


@nb.njit(parallel=True, cache=True)
def _extract_l1_kernel(
    starts: np.ndarray,
    bucket: np.ndarray,
    ask_price: np.ndarray,
    bid_price: np.ndarray,
    ask_volume: np.ndarray,
    bid_volume: np.ndarray,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(starts) - 1
    output = np.full(
        (n_samples, n_windows, _N_L1_FEATURES), np.nan, dtype=np.float32
    )
    # mid, spread, l1 depth, bid1, ask1
    references = np.full((n_samples, 5), np.nan, dtype=np.float32)

    for sample in nb.prange(n_samples):
        start = starts[sample]
        end = starts[sample + 1]

        # The last valid source row is closest to prediction under the expected
        # source ordering.  Ignore observations outside the largest window.
        for row in range(end - 1, start - 1, -1):
            if bucket[row] == _OUTSIDE_WINDOW:
                continue
            ask = float(ask_price[row])
            bid = float(bid_price[row])
            av = float(ask_volume[row])
            bv = float(bid_volume[row])
            if _valid_l1(ask, bid, av, bv):
                references[sample, 0] = np.float32(0.5 * (ask + bid))
                references[sample, 1] = np.float32(ask - bid)
                references[sample, 2] = np.float32(av + bv)
                references[sample, 3] = np.float32(bid)
                references[sample, 4] = np.float32(ask)
                break

        for window_idx in range(n_windows):
            states = 0
            valid = 0
            zero_price = 0
            crossed = 0
            negative_depth = 0
            rel_spread_sum = 0.0
            rel_spread_last = math.nan
            imbalance_sum = 0.0
            imbalance_last = math.nan
            micro_rel_sum = 0.0
            micro_spread_sum = 0.0
            log_depth_sum = 0.0
            ask_depth_sum = 0.0
            bid_depth_sum = 0.0
            ask_price_sum = 0.0
            bid_price_sum = 0.0
            first_mid = math.nan
            previous_mid = math.nan
            last_mid = math.nan
            min_mid = math.inf
            max_mid = -math.inf
            return_sq_sum = 0.0
            return_abs_sum = 0.0
            ofi_sum = 0.0
            previous_ask = math.nan
            previous_bid = math.nan
            previous_av = 0.0
            previous_bv = 0.0

            for row in range(start, end):
                row_bucket = bucket[row]
                ask = float(ask_price[row])
                bid = float(bid_price[row])
                av = float(ask_volume[row])
                bv = float(bid_volume[row])
                inside = (
                    row_bucket != _OUTSIDE_WINDOW and row_bucket <= window_idx
                )

                # Seed OFI with the closest valid state immediately outside
                # the lookback so the first in-window book update is retained.
                if not inside:
                    if _valid_l1(ask, bid, av, bv):
                        previous_ask = ask
                        previous_bid = bid
                        previous_av = av
                        previous_bv = bv
                    continue
                states += 1

                prices_present = (
                    math.isfinite(ask)
                    and math.isfinite(bid)
                    and ask > 0.0
                    and bid > 0.0
                )
                if not prices_present:
                    zero_price += 1
                elif ask < bid:
                    crossed += 1
                if av < 0.0 or bv < 0.0 or av + bv <= 0.0:
                    negative_depth += 1
                if not _valid_l1(ask, bid, av, bv):
                    continue

                valid += 1
                mid = 0.5 * (ask + bid)
                spread = ask - bid
                depth = av + bv
                rel_spread = spread / mid
                imbalance = (bv - av) / depth if depth > 0.0 else 0.0
                if depth > 0.0:
                    micro = (ask * bv + bid * av) / depth
                else:
                    micro = mid

                rel_spread_sum += rel_spread
                rel_spread_last = rel_spread
                imbalance_sum += imbalance
                imbalance_last = imbalance
                micro_rel_sum += (micro - mid) / mid
                if spread > 1.0e-12 * mid:
                    micro_spread_sum += (micro - mid) / spread
                log_depth_sum += math.log1p(depth)
                ask_depth_sum += av
                bid_depth_sum += bv
                ask_price_sum += ask
                bid_price_sum += bid

                if not math.isfinite(first_mid):
                    first_mid = mid
                if math.isfinite(previous_mid) and previous_mid > 0.0:
                    log_return = math.log(mid / previous_mid)
                    return_sq_sum += log_return * log_return
                    return_abs_sum += abs(log_return)
                previous_mid = mid
                last_mid = mid
                if mid < min_mid:
                    min_mid = mid
                if mid > max_mid:
                    max_mid = mid

                # Cont, Kukanov & Stoikov (2014) level-1 order-flow
                # imbalance computed between adjacent valid book states.
                if math.isfinite(previous_ask):
                    event_ofi = 0.0
                    if bid >= previous_bid:
                        event_ofi += bv
                    if bid <= previous_bid:
                        event_ofi -= previous_bv
                    if ask <= previous_ask:
                        event_ofi -= av
                    if ask >= previous_ask:
                        event_ofi += previous_av
                    ofi_sum += event_ofi
                previous_ask = ask
                previous_bid = bid
                previous_av = av
                previous_bv = bv

            if states > 0:
                output[sample, window_idx, 0] = np.float32(valid / states)
                output[sample, window_idx, 1] = np.float32(zero_price / states)
                output[sample, window_idx, 2] = np.float32(crossed / states)
                output[sample, window_idx, 3] = np.float32(negative_depth / states)
                output[sample, window_idx, 20] = np.float32(0.0)
            if valid > 0:
                inv_valid = 1.0 / valid
                output[sample, window_idx, 4] = np.float32(
                    rel_spread_sum * inv_valid
                )
                output[sample, window_idx, 5] = np.float32(rel_spread_last)
                output[sample, window_idx, 6] = np.float32(imbalance_sum * inv_valid)
                output[sample, window_idx, 7] = np.float32(imbalance_last)
                output[sample, window_idx, 8] = np.float32(micro_rel_sum * inv_valid)
                output[sample, window_idx, 9] = np.float32(
                    micro_spread_sum * inv_valid
                )
                output[sample, window_idx, 10] = np.float32(log_depth_sum * inv_valid)
                output[sample, window_idx, 11] = np.float32(ask_depth_sum * inv_valid)
                output[sample, window_idx, 12] = np.float32(bid_depth_sum * inv_valid)
                output[sample, window_idx, 13] = np.float32(ask_price_sum * inv_valid)
                output[sample, window_idx, 14] = np.float32(bid_price_sum * inv_valid)
                if first_mid > 0.0 and last_mid > 0.0:
                    output[sample, window_idx, 15] = np.float32(
                        math.log(last_mid / first_mid)
                    )
                output[sample, window_idx, 16] = np.float32(
                    math.sqrt(return_sq_sum)
                )
                output[sample, window_idx, 17] = np.float32(return_abs_sum)
                middle = 0.5 * (max_mid + min_mid)
                if middle > 0.0:
                    output[sample, window_idx, 18] = np.float32(
                        (max_mid - min_mid) / middle
                    )
                mean_depth = (ask_depth_sum + bid_depth_sum) * inv_valid
                if mean_depth > 0.0:
                    output[sample, window_idx, 19] = np.float32(
                        ofi_sum / mean_depth
                    )
                output[sample, window_idx, 20] = np.float32(math.log1p(valid))
    return output, references


@nb.njit(parallel=True, cache=True)
def _extract_l2_kernel(
    starts: np.ndarray,
    bucket: np.ndarray,
    ask_price: np.ndarray,
    bid_price: np.ndarray,
    ask_volume: np.ndarray,
    bid_volume: np.ndarray,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(starts) - 1
    output = np.full(
        (n_samples, n_windows, _N_L2_FEATURES), np.nan, dtype=np.float32
    )
    # level-2-only depth, bid2, ask2
    references = np.full((n_samples, 3), np.nan, dtype=np.float32)

    for sample in nb.prange(n_samples):
        start = starts[sample]
        end = starts[sample + 1]
        for row in range(end - 1, start - 1, -1):
            if bucket[row] == _OUTSIDE_WINDOW:
                continue
            ask = float(ask_price[row])
            bid = float(bid_price[row])
            av = float(ask_volume[row])
            bv = float(bid_volume[row])
            valid = (
                math.isfinite(ask)
                and math.isfinite(bid)
                and ask > 0.0
                and bid > 0.0
                and ask >= bid
                and av >= 0.0
                and bv >= 0.0
                and av + bv > 0.0
            )
            if valid:
                references[sample, 0] = np.float32(av + bv)
                references[sample, 1] = np.float32(bid)
                references[sample, 2] = np.float32(ask)
                break

        for window_idx in range(n_windows):
            states = 0
            valid_count = 0
            rel_spread_sum = 0.0
            imbalance_sum = 0.0
            log_depth_sum = 0.0
            ask_depth_sum = 0.0
            bid_depth_sum = 0.0
            ask_price_sum = 0.0
            bid_price_sum = 0.0
            ofi_sum = 0.0
            previous_ask = math.nan
            previous_bid = math.nan
            previous_av = 0.0
            previous_bv = 0.0

            for row in range(start, end):
                row_bucket = bucket[row]
                ask = float(ask_price[row])
                bid = float(bid_price[row])
                av = float(ask_volume[row])
                bv = float(bid_volume[row])
                valid = (
                    math.isfinite(ask)
                    and math.isfinite(bid)
                    and ask > 0.0
                    and bid > 0.0
                    and ask >= bid
                    and av >= 0.0
                    and bv >= 0.0
                    and av + bv > 0.0
                )
                inside = (
                    row_bucket != _OUTSIDE_WINDOW and row_bucket <= window_idx
                )
                if not inside:
                    if valid:
                        previous_ask = ask
                        previous_bid = bid
                        previous_av = av
                        previous_bv = bv
                    continue
                states += 1
                if not valid:
                    continue
                valid_count += 1
                mid = 0.5 * (ask + bid)
                depth = av + bv
                rel_spread_sum += (ask - bid) / mid
                imbalance_sum += (bv - av) / depth if depth > 0.0 else 0.0
                log_depth_sum += math.log1p(depth)
                ask_depth_sum += av
                bid_depth_sum += bv
                ask_price_sum += ask
                bid_price_sum += bid
                if math.isfinite(previous_ask):
                    event_ofi = 0.0
                    if bid >= previous_bid:
                        event_ofi += bv
                    if bid <= previous_bid:
                        event_ofi -= previous_bv
                    if ask <= previous_ask:
                        event_ofi -= av
                    if ask >= previous_ask:
                        event_ofi += previous_av
                    ofi_sum += event_ofi
                previous_ask = ask
                previous_bid = bid
                previous_av = av
                previous_bv = bv

            if states > 0:
                output[sample, window_idx, 0] = np.float32(valid_count / states)
            if valid_count > 0:
                inv_valid = 1.0 / valid_count
                output[sample, window_idx, 1] = np.float32(
                    rel_spread_sum * inv_valid
                )
                output[sample, window_idx, 2] = np.float32(
                    imbalance_sum * inv_valid
                )
                output[sample, window_idx, 3] = np.float32(
                    log_depth_sum * inv_valid
                )
                output[sample, window_idx, 4] = np.float32(
                    ask_depth_sum * inv_valid
                )
                output[sample, window_idx, 5] = np.float32(
                    bid_depth_sum * inv_valid
                )
                output[sample, window_idx, 6] = np.float32(
                    ask_price_sum * inv_valid
                )
                output[sample, window_idx, 7] = np.float32(
                    bid_price_sum * inv_valid
                )
                mean_depth = (ask_depth_sum + bid_depth_sum) * inv_valid
                if mean_depth > 0.0:
                    output[sample, window_idx, 8] = np.float32(
                        ofi_sum / mean_depth
                    )
    return output, references


@nb.njit(parallel=True, cache=True)
def _extract_trade_kernel(
    starts: np.ndarray,
    bucket: np.ndarray,
    avg_price: np.ndarray,
    volume: np.ndarray,
    count: np.ndarray,
    ref_mid: np.ndarray,
    ref_spread: np.ndarray,
    n_windows: int,
) -> np.ndarray:
    n_samples = len(starts) - 1
    output = np.full(
        (n_samples, n_windows, _N_TRADE_FEATURES), np.nan, dtype=np.float32
    )

    for sample in nb.prange(n_samples):
        start = starts[sample]
        end = starts[sample + 1]
        reference_mid = float(ref_mid[sample])
        reference_spread = float(ref_spread[sample])

        for window_idx in range(n_windows):
            states = 0
            no_trade = 0
            malformed = 0
            valid = 0
            volume_sum = 0.0
            count_sum = 0.0
            weighted_price_sum = 0.0
            weighted_price_sq_sum = 0.0
            last_price = math.nan
            min_price = math.inf
            max_price = -math.inf

            for row in range(start, end):
                row_bucket = bucket[row]
                if row_bucket == _OUTSIDE_WINDOW or row_bucket > window_idx:
                    continue
                states += 1
                price = float(avg_price[row])
                row_volume = float(volume[row])
                row_count = float(count[row])
                price_valid = math.isfinite(price) and price > 0.0
                strict_no_trade = row_volume == 0.0 and row_count == 0.0
                if strict_no_trade:
                    no_trade += 1
                active = row_volume > 0.0 and row_count > 0.0 and price_valid
                inconsistent_zero = (row_volume == 0.0) != (row_count == 0.0)
                if (
                    row_volume < 0.0
                    or row_count < 0.0
                    or inconsistent_zero
                    or (row_volume > 0.0 and row_count > 0.0 and not price_valid)
                    or (row_volume == 0.0 and row_count == 0.0 and price_valid)
                ):
                    malformed += 1
                if not active:
                    continue

                valid += 1
                volume_sum += row_volume
                count_sum += row_count
                weighted_price_sum += row_volume * price
                weighted_price_sq_sum += row_volume * price * price
                last_price = price
                if price < min_price:
                    min_price = price
                if price > max_price:
                    max_price = price

            if states > 0:
                output[sample, window_idx, 0] = np.float32(no_trade / states)
                output[sample, window_idx, 1] = np.float32(malformed / states)
                output[sample, window_idx, 2] = np.float32(0.0)
                output[sample, window_idx, 3] = np.float32(0.0)
                output[sample, window_idx, 10] = np.float32(0.0)
            if valid > 0 and volume_sum > 0.0:
                vwap = weighted_price_sum / volume_sum
                variance = weighted_price_sq_sum / volume_sum - vwap * vwap
                if variance < 0.0 and variance > -1.0e-12:
                    variance = 0.0
                output[sample, window_idx, 2] = np.float32(math.log1p(volume_sum))
                output[sample, window_idx, 3] = np.float32(math.log1p(count_sum))
                if count_sum > 0.0:
                    output[sample, window_idx, 4] = np.float32(
                        math.log1p(volume_sum / count_sum)
                    )
                if math.isfinite(reference_mid) and reference_mid > 0.0:
                    output[sample, window_idx, 5] = np.float32(
                        (vwap - reference_mid) / reference_mid
                    )
                    if variance >= 0.0:
                        output[sample, window_idx, 7] = np.float32(
                            math.sqrt(variance) / reference_mid
                        )
                    output[sample, window_idx, 8] = np.float32(
                        (last_price - reference_mid) / reference_mid
                    )
                    output[sample, window_idx, 9] = np.float32(
                        (max_price - min_price) / reference_mid
                    )
                if math.isfinite(reference_spread) and reference_spread > 0.0:
                    output[sample, window_idx, 6] = np.float32(
                        (vwap - reference_mid) / reference_spread
                    )
                output[sample, window_idx, 10] = np.float32(math.log1p(valid))
    return output


def _derive_l2_features(
    l1: np.ndarray, l2: np.ndarray, ref_mid: np.ndarray
) -> np.ndarray:
    n_samples, n_windows, _ = l1.shape
    output = np.full(
        (n_samples, n_windows, len(DERIVED_L2_FEATURES)),
        np.nan,
        dtype=np.float32,
    )
    # Indices are fixed by L1_FEATURES/L2_FEATURES above.
    l1_ask = l1[:, :, 11]
    l1_bid = l1[:, :, 12]
    l2_ask = l2[:, :, 4]
    l2_bid = l2[:, :, 5]
    l1_depth = l1_ask + l1_bid
    cumulative_depth = l1_depth + l2_ask + l2_bid

    with np.errstate(divide="ignore", invalid="ignore"):
        output[:, :, 0] = (
            (l1_bid + l2_bid - l1_ask - l2_ask) / cumulative_depth
        ).astype(np.float32)
        output[:, :, 1] = ((l2_ask + l2_bid) / l1_depth).astype(np.float32)
        scale = ref_mid[:, None]
        output[:, :, 2] = ((l2[:, :, 6] - l1[:, :, 13]) / scale).astype(
            np.float32
        )
        output[:, :, 3] = ((l1[:, :, 14] - l2[:, :, 7]) / scale).astype(
            np.float32
        )
    output[~np.isfinite(output)] = np.nan
    return output


def _windowed_names(base_names: Sequence[str], windows: np.ndarray) -> list[str]:
    return [
        f"market_{base}_{_window_suffix(float(window))}"
        if not base.startswith("market_")
        else f"{base}_{_window_suffix(float(window))}"
        for window in windows
        for base in base_names
    ]


def extract_market_features(
    path: Path,
    n_samples: int,
    windows: Sequence[float] = DEFAULT_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Extract robust per-sample market features from a huge Feather file.

    Parameters
    ----------
    path:
        Train or test ``market.feather`` path.
    n_samples:
        Expected number of samples. IDs must be contiguous 0..n_samples-1.
    windows:
        Strictly increasing, positive physical-time windows in seconds.

    Returns
    -------
    matrix, names:
        A float32 matrix with exactly ``n_samples`` rows and the corresponding
        feature names. Missing/invalid quantities are represented by NaN.

    Notes
    -----
    With 222M rows, the largest projected pass contains four 32-bit L1 (or L2)
    columns (~3.6 GB), a uint8 row bucket (~0.22 GB), and output arrays.  Arrow
    objects are explicitly released between passes, keeping the expected peak
    materially below 10 GB instead of expanding all thirteen market columns.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    window_values = _validate_windows(windows)

    # Pass 1: identity/time only. sample_id is used strictly as a grouping key.
    table, arrays = _read_projected(
        path, ("sample_id", "seconds_before_predict")
    )
    sample_id = arrays["sample_id"]
    seconds = arrays["seconds_before_predict"]
    starts, error = _group_starts(sample_id, int(n_samples))
    if error:
        messages = {
            1: "empty input or invalid n_samples",
            2: "the first sample_id is not zero",
            3: "sample_id values are not contiguous, grouped, and increasing",
            4: "the observed final sample_id/group count does not match n_samples",
        }
        raise ValueError(f"invalid market grouping: {messages.get(error, error)}")
    time_features, bucket = _time_features_and_bucket(
        starts, seconds, window_values
    )
    if np.any(time_features[:, 5] > 0.0):
        raise ValueError(
            "market rows contain timestamp inversions; path-dependent features "
            "require nonincreasing seconds_before_predict"
        )
    n_rows = int(starts[-1])
    del sample_id, seconds, arrays, table
    _release_arrow()

    # Pass 2: L1 prices/depth.  This contains returns and depth-scaled OFI.
    table, arrays = _read_projected(
        path,
        ("ask_price_1", "bid_price_1", "ask_volume_1", "bid_volume_1"),
    )
    if any(len(value) != n_rows for value in arrays.values()):
        raise ValueError("projected L1 columns do not match the grouping row count")
    l1, l1_refs = _extract_l1_kernel(
        starts,
        bucket,
        arrays["ask_price_1"],
        arrays["bid_price_1"],
        arrays["ask_volume_1"],
        arrays["bid_volume_1"],
        len(window_values),
    )
    del arrays, table
    _release_arrow()

    # Pass 3: L2 prices/depth.  L1 and L2 aggregate outputs are combined only
    # after raw columns have been released.
    table, arrays = _read_projected(
        path,
        ("ask_price_2", "bid_price_2", "ask_volume_2", "bid_volume_2"),
    )
    if any(len(value) != n_rows for value in arrays.values()):
        raise ValueError("projected L2 columns do not match the grouping row count")
    l2, l2_refs = _extract_l2_kernel(
        starts,
        bucket,
        arrays["ask_price_2"],
        arrays["bid_price_2"],
        arrays["ask_volume_2"],
        arrays["bid_volume_2"],
        len(window_values),
    )
    del arrays, table
    _release_arrow()

    references = np.full(
        (n_samples, len(REFERENCE_FEATURES)), np.nan, dtype=np.float32
    )
    references[:, 0] = l1_refs[:, 0]
    references[:, 1] = l1_refs[:, 1]
    references[:, 2] = l1_refs[:, 2]
    both_depths = np.isfinite(l1_refs[:, 2]) & np.isfinite(l2_refs[:, 0])
    references[both_depths, 3] = (
        l1_refs[both_depths, 2] + l2_refs[both_depths, 0]
    )
    references[:, 4] = l2_refs[:, 0]
    references[:, 5] = l1_refs[:, 3]
    references[:, 6] = l1_refs[:, 4]
    references[:, 7] = l2_refs[:, 1]
    references[:, 8] = l2_refs[:, 2]
    derived_l2 = _derive_l2_features(l1, l2, references[:, 0])
    del l1_refs, l2_refs

    # Pass 4: aggregate bars.  Nullable avgprice is converted to NaN and no
    # trade bars are excluded from VWAP/displacement calculations.
    table, arrays = _read_projected(
        path,
        ("transaction_avgprice", "transaction_volume", "transaction_count"),
    )
    if any(len(value) != n_rows for value in arrays.values()):
        raise ValueError("projected trade columns do not match the grouping row count")
    trades = _extract_trade_kernel(
        starts,
        bucket,
        arrays["transaction_avgprice"],
        arrays["transaction_volume"],
        arrays["transaction_count"],
        references[:, 0],
        references[:, 1],
        len(window_values),
    )
    del arrays, table, starts, bucket
    _release_arrow()

    names = list(TIME_FEATURES) + list(REFERENCE_FEATURES)
    names += _windowed_names(L1_FEATURES, window_values)
    names += _windowed_names(L2_FEATURES, window_values)
    names += _windowed_names(DERIVED_L2_FEATURES, window_values)
    names += _windowed_names(TRADE_FEATURES, window_values)

    matrix = np.concatenate(
        (
            time_features,
            references,
            l1.reshape(n_samples, -1),
            l2.reshape(n_samples, -1),
            derived_l2.reshape(n_samples, -1),
            trades.reshape(n_samples, -1),
        ),
        axis=1,
        dtype=np.float32,
    )
    if matrix.shape[1] != len(names):
        raise AssertionError(
            f"feature-name mismatch: matrix has {matrix.shape[1]}, names has {len(names)}"
        )
    return matrix, names


def run_synthetic_self_test() -> None:
    """Exercise grouping, masks, physical windows, references, and names."""

    data = pa.table(
        {
            "sample_id": pa.array([0, 0, 0, 0, 1, 1], type=pa.int32()),
            "seconds_before_predict": pa.array(
                [600.0, 60.0, 10.0, 0.0, 30.0, 0.0], type=pa.float32()
            ),
            # sample 0 includes a crossed state at 10s; sample 1 includes a
            # missing book at 30s.  Both final states are valid references.
            "ask_price_1": pa.array(
                [101.0, 101.0, 98.0, 101.0, 0.0, 201.0], type=pa.float32()
            ),
            "bid_price_1": pa.array(
                [99.0, 99.0, 99.0, 99.0, 0.0, 199.0], type=pa.float32()
            ),
            "ask_volume_1": pa.array([10, 11, 12, 13, 0, 20], type=pa.int32()),
            "bid_volume_1": pa.array([14, 15, 16, 17, 0, 22], type=pa.int32()),
            "ask_price_2": pa.array(
                [102.0, 102.0, 102.0, 102.0, 0.0, 202.0], type=pa.float32()
            ),
            "bid_price_2": pa.array(
                [98.0, 98.0, 98.0, 98.0, 0.0, 198.0], type=pa.float32()
            ),
            "ask_volume_2": pa.array([8, 9, 10, 11, 0, 18], type=pa.int32()),
            "bid_volume_2": pa.array([12, 13, 14, 15, 0, 19], type=pa.int32()),
            "transaction_avgprice": pa.array(
                [100.0, 100.0, None, 100.5, None, 200.5], type=pa.float32()
            ),
            "transaction_volume": pa.array([100, 120, 0, 50, 0, 80], type=pa.int32()),
            "transaction_count": pa.array([2, 3, 0, 1, 0, 2], type=pa.int32()),
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "market.feather"
        feather.write_feather(data, str(path), compression="uncompressed")
        matrix, names = extract_market_features(
            path, n_samples=2, windows=(10.0, 60.0, 600.0)
        )

    assert matrix.dtype == np.float32
    assert matrix.shape == (2, len(names))
    assert len(names) == len(set(names)), "feature names must be unique"
    assert "sample_id" not in names
    assert np.isclose(matrix[0, names.index("ref_mid")], 100.0)
    assert np.isclose(matrix[1, names.index("ref_mid")], 200.0)
    assert np.isclose(matrix[0, names.index("ref_l1_depth")], 30.0)
    assert np.isclose(matrix[0, names.index("ref_l2_depth")], 56.0)
    assert np.isclose(
        matrix[0, names.index("market_book_crossed_frac_10s")], 0.5
    )
    assert np.isclose(
        matrix[0, names.index("market_no_trade_frac_10s")], 0.5
    )
    assert np.isfinite(matrix).any()


if __name__ == "__main__":
    run_synthetic_self_test()
    print("market_features synthetic self-test passed")
