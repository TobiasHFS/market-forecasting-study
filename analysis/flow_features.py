"""Memory-bounded feature extraction for raw order and transaction flow.

The public API intentionally returns only per-sample features and their names;
``sample_id`` is used solely to establish group boundaries and is never emitted
as a model feature.  Each Feather source is projected to the columns required by
its extractor, memory-mapped where PyArrow supports it, and released when the
function returns.

Window features are cumulative physical-time windows ending at the prediction
timestamp.  Undefined quantities (for example, VWAP with no positive volume)
are represented by NaN so that downstream fold-fitted preprocessing can impute
them without confusing missingness with a genuine zero.
"""

from __future__ import annotations

import argparse
import gc
import math
import tempfile
from pathlib import Path
from typing import Sequence

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather


DEFAULT_WINDOWS = (2.0, 5.0, 15.0, 30.0, 60.0)


ORDER_BASE_FEATURES = (
    "order_event_count_all",
    "order_observed_span_s_all",
    "order_newest_age_s_all",
    "order_oldest_age_s_all",
    "order_cap_999_flag",
    "order_invalid_time_fraction_all",
    "order_invalid_price_fraction_all",
    "order_invalid_volume_fraction_all",
    "order_invalid_code_fraction_all",
    "order_adjacent_timestamp_duplicate_fraction_all",
    "order_time_order_violation_fraction_all",
)

ORDER_WINDOW_FEATURES = (
    "order_event_count_{window}",
    "ref_order_abs_volume_total_{window}",
    "ref_order_signed_pressure_total_{window}",
    "order_pressure_volume_imbalance_{window}",
    "order_pressure_count_imbalance_{window}",
    "order_side_volume_imbalance_{window}",
    "order_side_count_imbalance_{window}",
    "order_cancel_volume_fraction_{window}",
    "order_cancel_event_fraction_{window}",
    "order_buy_new_volume_fraction_{window}",
    "order_sell_new_volume_fraction_{window}",
    "order_buy_cancel_volume_fraction_{window}",
    "order_sell_cancel_volume_fraction_{window}",
    "ref_order_vwap_price_{window}",
    "order_price_range_relative_{window}",
    "order_price_log_return_{window}",
    "order_price_abs_log_variation_{window}",
    "order_price_path_efficiency_{window}",
    "order_max_volume_share_{window}",
    "order_event_rate_hz_{window}",
)


TRANSACTION_BASE_FEATURES = (
    "trade_event_count_all",
    "trade_observed_span_s_all",
    "trade_newest_age_s_all",
    "trade_oldest_age_s_all",
    "trade_cap_999_flag",
    "trade_invalid_time_fraction_all",
    "trade_invalid_price_fraction_all",
    "trade_invalid_volume_fraction_all",
    "trade_invalid_side_fraction_all",
    "trade_adjacent_timestamp_duplicate_fraction_all",
    "trade_time_order_violation_fraction_all",
)

TRANSACTION_WINDOW_FEATURES = (
    "trade_event_count_{window}",
    "ref_trade_abs_volume_total_{window}",
    "ref_trade_signed_volume_total_{window}",
    "trade_signed_volume_imbalance_{window}",
    "trade_signed_count_imbalance_{window}",
    "ref_trade_vwap_price_{window}",
    "trade_price_range_relative_{window}",
    "trade_price_log_return_{window}",
    "trade_price_abs_log_variation_{window}",
    "trade_price_rms_log_variation_{window}",
    "trade_price_path_efficiency_{window}",
    "trade_max_volume_share_{window}",
    "trade_event_rate_hz_{window}",
    "trade_volume_rate_per_s_{window}",
)


def _validate_windows(windows: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(windows), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("windows must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("windows must contain only finite positive durations")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("windows must be strictly increasing")
    return np.ascontiguousarray(values)


def _window_label(value: float) -> str:
    if float(value).is_integer():
        return f"w{int(value)}s"
    return "w" + f"{value:g}".replace(".", "p") + "s"


def _feature_names(
    base_names: Sequence[str], window_names: Sequence[str], windows: np.ndarray
) -> list[str]:
    names = list(base_names)
    for window in windows:
        label = _window_label(float(window))
        names.extend(template.format(window=label) for template in window_names)
    if len(names) != len(set(names)):
        raise RuntimeError("feature-name construction produced duplicates")
    return names


def _numpy_column(
    table: pa.Table, name: str, dtype: np.dtype, null_fill: float | int
) -> np.ndarray:
    """Return a contiguous NumPy column, filling Arrow nulls explicitly."""

    column = table.column(name)
    if column.num_chunks == 1:
        array = column.chunk(0)
    else:
        # Feather files in this competition normally have one chunk.  This copy
        # is the bounded fallback for other legal IPC layouts.
        array = column.combine_chunks()
    if array.null_count:
        array = pc.fill_null(array, pa.scalar(null_fill, type=array.type))
    values = array.to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(values, dtype=dtype)


@nb.njit(cache=True)
def _group_offsets(sample_id: np.ndarray, n_samples: int) -> np.ndarray:
    """Build CSR-style offsets while validating grouped, in-range IDs."""

    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    previous = -1
    for row in range(sample_id.size):
        current = int(sample_id[row])
        if current < 0 or current >= n_samples:
            raise ValueError("sample_id outside [0, n_samples)")
        if current < previous:
            raise ValueError("sample_id must be grouped in nondecreasing order")
        offsets[current + 1] += 1
        previous = current
    for sample in range(n_samples):
        offsets[sample + 1] += offsets[sample]
    return offsets


@nb.njit(cache=True, inline="always")
def _first_at_or_inside_window(
    seconds: np.ndarray, start: int, end: int, window: float
) -> int:
    """Binary-search a nonincreasing, finite, nonnegative time group."""

    lo = start
    hi = end
    while lo < hi:
        middle = (lo + hi) // 2
        if float(seconds[middle]) > window:
            lo = middle + 1
        else:
            hi = middle
    return lo


@nb.njit(cache=True, parallel=True)
def _order_kernel(
    offsets: np.ndarray,
    seconds: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    side: np.ndarray,
    action: np.ndarray,
    windows: np.ndarray,
    output: np.ndarray,
) -> None:
    n_base = len(ORDER_BASE_FEATURES)
    n_per_window = len(ORDER_WINDOW_FEATURES)

    for sample in nb.prange(offsets.size - 1):
        start = int(offsets[sample])
        end = int(offsets[sample + 1])
        row_count = end - start
        output[sample, 0] = float(row_count)
        output[sample, 4] = 1.0 if row_count >= 999 else 0.0

        valid_time = 0
        invalid_time = 0
        invalid_price = 0
        invalid_volume = 0
        invalid_code = 0
        duplicates = 0
        order_violations = 0
        newest = math.inf
        oldest = -math.inf
        have_previous_time = False
        previous_time = 0.0

        for row in range(start, end):
            time = float(seconds[row])
            if math.isfinite(time) and time >= 0.0:
                valid_time += 1
                if time < newest:
                    newest = time
                if time > oldest:
                    oldest = time
                if have_previous_time:
                    if time == previous_time:
                        duplicates += 1
                    elif time > previous_time:
                        order_violations += 1
                previous_time = time
                have_previous_time = True
            else:
                invalid_time += 1
                # Only raw-adjacent valid timestamps should contribute to the
                # duplicate/order diagnostics.
                have_previous_time = False

            row_price = float(price[row])
            if not math.isfinite(row_price) or row_price <= 0.0:
                invalid_price += 1
            if int(volume[row]) < 0:
                invalid_volume += 1
            row_side = int(side[row])
            row_action = int(action[row])
            if (row_side != 0 and row_side != 1) or (
                row_action != 0 and row_action != 1
            ):
                invalid_code += 1

        if valid_time:
            output[sample, 1] = oldest - newest
            output[sample, 2] = newest
            output[sample, 3] = oldest
        if row_count:
            denominator = float(row_count)
            output[sample, 5] = invalid_time / denominator
            output[sample, 6] = invalid_price / denominator
            output[sample, 7] = invalid_volume / denominator
            output[sample, 8] = invalid_code / denominator
        if valid_time > 1:
            adjacency_denominator = float(valid_time - 1)
            output[sample, 9] = duplicates / adjacency_denominator
            output[sample, 10] = order_violations / adjacency_denominator

        monotone_for_search = invalid_time == 0 and order_violations == 0

        for window_index in range(windows.size):
            window = float(windows[window_index])
            column = n_base + window_index * n_per_window
            scan_start = start
            if monotone_for_search:
                scan_start = _first_at_or_inside_window(
                    seconds, start, end, window
                )

            event_count = 0
            valid_code_count = 0
            pressure_count = 0.0
            side_count = 0.0
            cancel_count = 0
            total_volume = 0.0
            pressure_volume = 0.0
            side_volume = 0.0
            buy_new_volume = 0.0
            sell_new_volume = 0.0
            buy_cancel_volume = 0.0
            sell_cancel_volume = 0.0
            max_volume = 0.0
            price_volume = 0.0
            vwap_volume = 0.0
            minimum_price = math.inf
            maximum_price = -math.inf
            first_price = math.nan
            last_price = math.nan
            previous_price = math.nan
            absolute_log_variation = 0.0

            for row in range(scan_start, end):
                time = float(seconds[row])
                if not math.isfinite(time) or time < 0.0 or time > window:
                    continue
                event_count += 1
                row_price = float(price[row])
                row_volume = float(volume[row])
                row_side = int(side[row])
                row_action = int(action[row])
                valid_price = math.isfinite(row_price) and row_price > 0.0
                valid_volume = row_volume >= 0.0
                valid_code = (row_side == 0 or row_side == 1) and (
                    row_action == 0 or row_action == 1
                )

                # Keep all price/size summaries on the same valid-code
                # population as the signed-flow denominator.  Otherwise a
                # malformed large row can create a max-volume "share" > 1.
                if valid_code and valid_volume and row_volume > max_volume:
                    max_volume = row_volume
                if valid_price and valid_code:
                    if row_price < minimum_price:
                        minimum_price = row_price
                    if row_price > maximum_price:
                        maximum_price = row_price
                    if math.isnan(first_price):
                        first_price = row_price
                    if not math.isnan(previous_price):
                        absolute_log_variation += abs(
                            math.log(row_price / previous_price)
                        )
                    previous_price = row_price
                    last_price = row_price
                    if valid_volume and row_volume > 0.0:
                        price_volume += row_price * row_volume
                        vwap_volume += row_volume

                if valid_code:
                    valid_code_count += 1
                    side_sign = 1.0 - 2.0 * row_side
                    action_sign = 1.0 - 2.0 * row_action
                    pressure_sign = side_sign * action_sign
                    pressure_count += pressure_sign
                    side_count += side_sign
                    if row_action == 1:
                        cancel_count += 1
                    if valid_volume:
                        total_volume += row_volume
                        pressure_volume += pressure_sign * row_volume
                        side_volume += side_sign * row_volume
                        if row_side == 0 and row_action == 0:
                            buy_new_volume += row_volume
                        elif row_side == 1 and row_action == 0:
                            sell_new_volume += row_volume
                        elif row_side == 0 and row_action == 1:
                            buy_cancel_volume += row_volume
                        else:
                            sell_cancel_volume += row_volume

            output[sample, column] = float(event_count)
            output[sample, column + 1] = total_volume
            output[sample, column + 2] = pressure_volume
            if total_volume > 0.0:
                output[sample, column + 3] = pressure_volume / total_volume
                output[sample, column + 5] = side_volume / total_volume
                output[sample, column + 7] = (
                    buy_cancel_volume + sell_cancel_volume
                ) / total_volume
                output[sample, column + 9] = buy_new_volume / total_volume
                output[sample, column + 10] = sell_new_volume / total_volume
                output[sample, column + 11] = buy_cancel_volume / total_volume
                output[sample, column + 12] = sell_cancel_volume / total_volume
                output[sample, column + 18] = max_volume / total_volume
            if valid_code_count:
                output[sample, column + 4] = pressure_count / valid_code_count
                output[sample, column + 6] = side_count / valid_code_count
                output[sample, column + 8] = cancel_count / valid_code_count
            if vwap_volume > 0.0:
                vwap = price_volume / vwap_volume
                output[sample, column + 13] = vwap
                if maximum_price >= minimum_price:
                    output[sample, column + 14] = (
                        maximum_price - minimum_price
                    ) / vwap
            if not math.isnan(first_price):
                net_log_return = math.log(last_price / first_price)
                output[sample, column + 15] = net_log_return
                output[sample, column + 16] = absolute_log_variation
                output[sample, column + 17] = (
                    abs(net_log_return) / absolute_log_variation
                    if absolute_log_variation > 0.0
                    else 0.0
                )
            output[sample, column + 19] = event_count / window


@nb.njit(cache=True, parallel=True)
def _transaction_kernel(
    offsets: np.ndarray,
    seconds: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    side: np.ndarray,
    windows: np.ndarray,
    output: np.ndarray,
) -> None:
    n_base = len(TRANSACTION_BASE_FEATURES)
    n_per_window = len(TRANSACTION_WINDOW_FEATURES)

    for sample in nb.prange(offsets.size - 1):
        start = int(offsets[sample])
        end = int(offsets[sample + 1])
        row_count = end - start
        output[sample, 0] = float(row_count)
        output[sample, 4] = 1.0 if row_count >= 999 else 0.0

        valid_time = 0
        invalid_time = 0
        invalid_price = 0
        invalid_volume = 0
        invalid_side = 0
        duplicates = 0
        order_violations = 0
        newest = math.inf
        oldest = -math.inf
        have_previous_time = False
        previous_time = 0.0

        for row in range(start, end):
            time = float(seconds[row])
            if math.isfinite(time) and time >= 0.0:
                valid_time += 1
                if time < newest:
                    newest = time
                if time > oldest:
                    oldest = time
                if have_previous_time:
                    if time == previous_time:
                        duplicates += 1
                    elif time > previous_time:
                        order_violations += 1
                previous_time = time
                have_previous_time = True
            else:
                invalid_time += 1
                have_previous_time = False

            row_price = float(price[row])
            if not math.isfinite(row_price) or row_price <= 0.0:
                invalid_price += 1
            if int(volume[row]) < 0:
                invalid_volume += 1
            row_side = int(side[row])
            if row_side != 0 and row_side != 1:
                invalid_side += 1

        if valid_time:
            output[sample, 1] = oldest - newest
            output[sample, 2] = newest
            output[sample, 3] = oldest
        if row_count:
            denominator = float(row_count)
            output[sample, 5] = invalid_time / denominator
            output[sample, 6] = invalid_price / denominator
            output[sample, 7] = invalid_volume / denominator
            output[sample, 8] = invalid_side / denominator
        if valid_time > 1:
            adjacency_denominator = float(valid_time - 1)
            output[sample, 9] = duplicates / adjacency_denominator
            output[sample, 10] = order_violations / adjacency_denominator

        monotone_for_search = invalid_time == 0 and order_violations == 0

        for window_index in range(windows.size):
            window = float(windows[window_index])
            column = n_base + window_index * n_per_window
            scan_start = start
            if monotone_for_search:
                scan_start = _first_at_or_inside_window(
                    seconds, start, end, window
                )

            event_count = 0
            valid_side_count = 0
            signed_count = 0.0
            total_volume = 0.0
            directed_volume = 0.0
            signed_volume = 0.0
            max_volume = 0.0
            price_volume = 0.0
            vwap_volume = 0.0
            minimum_price = math.inf
            maximum_price = -math.inf
            first_price = math.nan
            last_price = math.nan
            previous_price = math.nan
            absolute_log_variation = 0.0
            squared_log_variation = 0.0
            price_steps = 0

            for row in range(scan_start, end):
                time = float(seconds[row])
                if not math.isfinite(time) or time < 0.0 or time > window:
                    continue
                event_count += 1
                row_price = float(price[row])
                row_volume = float(volume[row])
                row_side = int(side[row])
                valid_price = math.isfinite(row_price) and row_price > 0.0
                valid_volume = row_volume >= 0.0
                valid_side = row_side == 0 or row_side == 1

                if valid_volume:
                    total_volume += row_volume
                    if row_volume > max_volume:
                        max_volume = row_volume
                    if valid_side:
                        directed_volume += row_volume
                        signed_volume += (1.0 - 2.0 * row_side) * row_volume
                if valid_side:
                    valid_side_count += 1
                    signed_count += 1.0 - 2.0 * row_side
                if valid_price:
                    if row_price < minimum_price:
                        minimum_price = row_price
                    if row_price > maximum_price:
                        maximum_price = row_price
                    if math.isnan(first_price):
                        first_price = row_price
                    if not math.isnan(previous_price):
                        log_change = math.log(row_price / previous_price)
                        absolute_log_variation += abs(log_change)
                        squared_log_variation += log_change * log_change
                        price_steps += 1
                    previous_price = row_price
                    last_price = row_price
                    if valid_volume and row_volume > 0.0:
                        price_volume += row_price * row_volume
                        vwap_volume += row_volume

            output[sample, column] = float(event_count)
            output[sample, column + 1] = total_volume
            output[sample, column + 2] = signed_volume
            if directed_volume > 0.0:
                output[sample, column + 3] = signed_volume / directed_volume
            if total_volume > 0.0:
                output[sample, column + 11] = max_volume / total_volume
            if valid_side_count:
                output[sample, column + 4] = signed_count / valid_side_count
            if vwap_volume > 0.0:
                vwap = price_volume / vwap_volume
                output[sample, column + 5] = vwap
                if maximum_price >= minimum_price:
                    output[sample, column + 6] = (
                        maximum_price - minimum_price
                    ) / vwap
            if not math.isnan(first_price):
                net_log_return = math.log(last_price / first_price)
                output[sample, column + 7] = net_log_return
                output[sample, column + 8] = absolute_log_variation
                output[sample, column + 9] = (
                    math.sqrt(squared_log_variation / price_steps)
                    if price_steps
                    else 0.0
                )
                output[sample, column + 10] = (
                    abs(net_log_return) / absolute_log_variation
                    if absolute_log_variation > 0.0
                    else 0.0
                )
            output[sample, column + 12] = event_count / window
            output[sample, column + 13] = total_volume / window


def extract_order_features(
    path: Path,
    n_samples: int,
    windows: Sequence[float] = DEFAULT_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Extract grouped order-flow features from one Feather source.

    Signed order pressure follows the requested convention exactly:
    ``(1 - 2*side) * (1 - 2*order_action) * volume``.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    window_values = _validate_windows(windows)
    names = _feature_names(
        ORDER_BASE_FEATURES, ORDER_WINDOW_FEATURES, window_values
    )
    table = feather.read_table(
        str(Path(path)),
        columns=(
            "sample_id",
            "seconds_before_predict",
            "price",
            "volume",
            "side",
            "order_action",
        ),
        memory_map=True,
        use_threads=True,
    )
    sample_id = _numpy_column(table, "sample_id", np.int32, -1)
    seconds = _numpy_column(
        table, "seconds_before_predict", np.float32, np.nan
    )
    price = _numpy_column(table, "price", np.float32, np.nan)
    volume = _numpy_column(table, "volume", np.int32, -1)
    side = _numpy_column(table, "side", np.int8, -1)
    action = _numpy_column(table, "order_action", np.int8, -1)
    offsets = _group_offsets(sample_id, int(n_samples))
    del sample_id, table
    gc.collect()
    output = np.full((n_samples, len(names)), np.nan, dtype=np.float32)
    _order_kernel(
        offsets,
        seconds,
        price,
        volume,
        side,
        action,
        window_values,
        output,
    )
    return output, names


def extract_transaction_features(
    path: Path,
    n_samples: int,
    windows: Sequence[float] = DEFAULT_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Extract grouped trade-flow features from one Feather source.

    Signed transaction volume follows the requested convention exactly:
    ``(1 - 2*side) * volume``.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    window_values = _validate_windows(windows)
    names = _feature_names(
        TRANSACTION_BASE_FEATURES,
        TRANSACTION_WINDOW_FEATURES,
        window_values,
    )
    table = feather.read_table(
        str(Path(path)),
        columns=(
            "sample_id",
            "seconds_before_predict",
            "price",
            "volume",
            "side",
        ),
        memory_map=True,
        use_threads=True,
    )
    sample_id = _numpy_column(table, "sample_id", np.int32, -1)
    seconds = _numpy_column(
        table, "seconds_before_predict", np.float32, np.nan
    )
    price = _numpy_column(table, "price", np.float32, np.nan)
    volume = _numpy_column(table, "volume", np.int32, -1)
    side = _numpy_column(table, "side", np.int8, -1)
    offsets = _group_offsets(sample_id, int(n_samples))
    del sample_id, table
    gc.collect()
    output = np.full((n_samples, len(names)), np.nan, dtype=np.float32)
    _transaction_kernel(
        offsets,
        seconds,
        price,
        volume,
        side,
        window_values,
        output,
    )
    return output, names


def _self_test() -> None:
    """Exercise exact sign conventions, windows, caps, and path features."""

    with tempfile.TemporaryDirectory(prefix="flow_features_") as directory:
        root = Path(directory)
        order_path = root / "order.feather"
        transaction_path = root / "transaction.feather"

        order_sample_0 = 3
        order_sample_1 = 999
        order_ids = np.concatenate(
            (
                np.zeros(order_sample_0, dtype=np.int32),
                np.ones(order_sample_1, dtype=np.int32),
            )
        )
        order_seconds = np.concatenate(
            (
                np.array([60.0, 10.0, 1.0], dtype=np.float32),
                np.linspace(60.0, 0.0, order_sample_1, dtype=np.float32),
            )
        )
        order_price = np.concatenate(
            (
                np.array([100.0, 101.0, 102.0], dtype=np.float32),
                np.full(order_sample_1, 100.0, dtype=np.float32),
            )
        )
        order_volume = np.concatenate(
            (
                np.array([10, 20, 30], dtype=np.int32),
                np.ones(order_sample_1, dtype=np.int32),
            )
        )
        order_side = np.concatenate(
            (
                np.array([0, 1, 0], dtype=np.int8),
                np.zeros(order_sample_1, dtype=np.int8),
            )
        )
        order_action = np.concatenate(
            (
                np.array([0, 0, 1], dtype=np.int8),
                np.zeros(order_sample_1, dtype=np.int8),
            )
        )
        feather.write_feather(
            pa.table(
                {
                    "sample_id": order_ids,
                    "seconds_before_predict": order_seconds,
                    "price": order_price,
                    "volume": order_volume,
                    "side": order_side,
                    "order_action": order_action,
                }
            ),
            order_path,
        )

        trade_ids = np.array([0, 0, 1], dtype=np.int32)
        feather.write_feather(
            pa.table(
                {
                    "sample_id": trade_ids,
                    "seconds_before_predict": np.array(
                        [10.0, 1.0, 0.5], dtype=np.float32
                    ),
                    "price": np.array([100.0, 101.0, 50.0], dtype=np.float32),
                    "volume": np.array([10, 5, 7], dtype=np.int32),
                    "side": np.array([0, 1, 0], dtype=np.int8),
                }
            ),
            transaction_path,
        )

        order_matrix, order_names = extract_order_features(order_path, 2)
        trade_matrix, trade_names = extract_transaction_features(
            transaction_path, 2
        )

        assert order_matrix.shape == (2, len(ORDER_BASE_FEATURES) + 5 * 20)
        assert trade_matrix.shape == (
            2,
            len(TRANSACTION_BASE_FEATURES) + 5 * 14,
        )
        assert not any("sample_id" in name for name in order_names + trade_names)
        assert order_matrix[1, order_names.index("order_cap_999_flag")] == 1.0
        assert order_matrix[
            0, order_names.index("order_event_count_w2s")
        ] == 1.0
        np.testing.assert_allclose(
            order_matrix[
                0,
                order_names.index("ref_order_signed_pressure_total_w2s"),
            ],
            -30.0,
        )
        np.testing.assert_allclose(
            order_matrix[
                0,
                order_names.index("order_pressure_volume_imbalance_w2s"),
            ],
            -1.0,
        )
        np.testing.assert_allclose(
            trade_matrix[
                0, trade_names.index("ref_trade_signed_volume_total_w2s")
            ],
            -5.0,
        )
        np.testing.assert_allclose(
            trade_matrix[
                0, trade_names.index("trade_signed_volume_imbalance_w2s")
            ],
            -1.0,
        )
        order_share_columns = [
            index
            for index, name in enumerate(order_names)
            if "order_max_volume_share_" in name
        ]
        defined_order_shares = order_matrix[:, order_share_columns]
        defined_order_shares = defined_order_shares[
            np.isfinite(defined_order_shares)
        ]
        assert np.all((defined_order_shares >= 0.0) & (defined_order_shares <= 1.0))
        print(
            "flow_features self-test passed: "
            f"{len(order_names)} order features, "
            f"{len(trade_names)} transaction features"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a small synthetic feature-contract test",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported by this module")
    _self_test()


if __name__ == "__main__":
    main()
