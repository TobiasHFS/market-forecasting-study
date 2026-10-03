"""Audited NumPy primitives for a future V3 extraction, independent of V2.

These are correctness/reference kernels, not a full-data extractor. All ages
are ``seconds_before_predict``: larger ages occurred earlier. Quote rows must
be ordered from oldest to newest within each sample. Invalid times/books never
seed state. No target, sample identifier feature, cache, or model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _offsets(value: np.ndarray, length: int) -> np.ndarray:
    result = np.asarray(value)
    if (result.ndim != 1 or result.size < 1
            or not np.issubdtype(result.dtype, np.integer)
            or result[0] != 0 or result[-1] != length
            or np.any(result[1:] < result[:-1])):
        raise ValueError("offsets must be nondecreasing integers from 0 to row count")
    return result


def valid_l1_quotes(bid: np.ndarray, ask: np.ndarray,
                    bid_volume: np.ndarray, ask_volume: np.ndarray) -> np.ndarray:
    """A locked positive book is valid; its spread normalization is undefined."""
    bid, ask, bv, av = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64) for x in (bid, ask, bid_volume, ask_volume)]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        depth = bv + av
    return (np.isfinite(bid) & np.isfinite(ask) & np.isfinite(bv) & np.isfinite(av)
            & (bid > 0) & (ask >= bid) & (bv >= 0) & (av >= 0)
            & np.isfinite(depth) & (depth > 0))


def spread_units(value: np.ndarray, reference_mid: np.ndarray,
                 reference_spread: np.ndarray) -> np.ndarray:
    """Return (value - mid) / spread, with explicit NaN for missing references.

    The minimum spread matches V2's guard. Locked, crossed, nonfinite, or
    excessively small reference spreads are missing, never zero displacement.
    """
    value, mid, spread = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64)
          for x in (value, reference_mid, reference_spread)]
    )
    valid = (np.isfinite(value) & (value > 0) & np.isfinite(mid) & (mid > 0)
             & np.isfinite(spread) & (spread > np.maximum(1e-12, np.abs(mid) * 1e-8)))
    result = np.full(value.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(value - mid, spread, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def depth_units(value: np.ndarray, reference_depth: np.ndarray) -> np.ndarray:
    """Scale a signed quantity only when its finite positive depth is known."""
    value, depth = np.broadcast_arrays(
        np.asarray(value, dtype=np.float64), np.asarray(reference_depth, dtype=np.float64)
    )
    result = np.full(value.shape, np.nan)
    valid = np.isfinite(value) & np.isfinite(depth) & (depth > 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(value, depth, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return result


def _quote_inputs(offsets: np.ndarray, seconds: np.ndarray, bid: np.ndarray,
                  ask: np.ndarray, bid_volume: np.ndarray, ask_volume: np.ndarray):
    arrays = [_vector(x, name) for x, name in zip(
        (seconds, bid, ask, bid_volume, ask_volume),
        ("seconds", "bid", "ask", "bid_volume", "ask_volume"), strict=True)]
    ages, bid, ask, bv, av = arrays
    if any(x.size != ages.size for x in arrays):
        raise ValueError("quote columns must have equal lengths")
    offsets = _offsets(offsets, ages.size)
    time_valid = np.isfinite(ages) & (ages >= 0)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        sample_ages = ages[start:end][time_valid[start:end]]
        if np.any(sample_ages[1:] > sample_ages[:-1]):
            raise ValueError("quote ages must be nonincreasing within each sample")
    valid = time_valid & valid_l1_quotes(bid, ask, bv, av)
    return offsets, ages, bid, ask, bv, av, valid


def ofi_l1_events(offsets: np.ndarray, seconds: np.ndarray, bid: np.ndarray,
                  ask: np.ndarray, bid_volume: np.ndarray,
                  ask_volume: np.ndarray) -> np.ndarray:
    """Cont L1 OFI between consecutive valid states, returned at current rows.

    Invalid rows and each sample's first valid observation return NaN: there
    is no observed transition there. Equal-price states use the difference in
    depth. Invalid quotes are skipped without replacing the previous valid
    state. Equal-time quote rows follow their source order, which must already
    be meaningful if used to interpret intra-timestamp transitions.
    """
    offsets, _, bid, ask, bv, av, valid = _quote_inputs(
        offsets, seconds, bid, ask, bid_volume, ask_volume)
    result = np.full(bid.size, np.nan)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        rows = np.flatnonzero(valid[start:end]) + start
        previous, current = rows[:-1], rows[1:]
        result[current] = (
            (bid[current] >= bid[previous]) * bv[current]
            - (bid[current] <= bid[previous]) * bv[previous]
            - (ask[current] <= ask[previous]) * av[current]
            + (ask[current] >= ask[previous]) * av[previous]
        )
    result[~np.isfinite(result)] = np.nan
    return result


@dataclass(frozen=True)
class AsOfQuotes:
    """One result per query; row_index=-1 and NaNs mean no eligible reference."""

    row_index: np.ndarray
    mid: np.ndarray
    spread: np.ndarray
    depth: np.ndarray
    lag_seconds: np.ndarray


def asof_l1_quotes(query_offsets: np.ndarray, query_seconds: np.ndarray,
                   quote_offsets: np.ndarray, quote_seconds: np.ndarray,
                   bid: np.ndarray, ask: np.ndarray, bid_volume: np.ndarray,
                   ask_volume: np.ndarray, *, allow_exact_matches: bool = False,
                   max_lag_seconds: float | None = None) -> AsOfQuotes:
    """Join each event to its most recent valid *earlier* same-sample quote.

    A backward join requires quote_age >= event_age, not <=. Exact timestamp
    matches are excluded by default because cross-stream order at a shared
    timestamp is unknown. Opt in only if timestamp availability is documented.
    Tied quote ages select the last eligible source row. Query order may be
    arbitrary. No eligible quote returns NaNs; there is no terminal-mid or
    future-bin fallback. Optional max_lag_seconds rejects stale references;
    otherwise every selected quote's lag is emitted for downstream masking.
    """
    qo, ages, bid, ask, bv, av, valid = _quote_inputs(
        quote_offsets, quote_seconds, bid, ask, bid_volume, ask_volume)
    query = _vector(query_seconds, "query_seconds")
    eo = _offsets(query_offsets, query.size)
    if eo.size != qo.size:
        raise ValueError("quote and query offsets must describe the same samples")
    if max_lag_seconds is not None and (
            not np.isfinite(max_lag_seconds) or max_lag_seconds < 0):
        raise ValueError("max_lag_seconds must be finite and nonnegative")
    index = np.full(query.size, -1, dtype=np.int64)
    lag = np.full(query.size, np.nan)
    side = "right" if allow_exact_matches else "left"
    for sample in range(qo.size - 1):
        rows = np.flatnonzero(valid[qo[sample]:qo[sample + 1]]) + qo[sample]
        events = np.arange(eo[sample], eo[sample + 1], dtype=np.int64)
        events = events[np.isfinite(query[events]) & (query[events] >= 0)]
        if not rows.size or not events.size:
            continue
        # Negated ages are conventional ascending timestamps.
        positions = np.searchsorted(-ages[rows], -query[events], side=side) - 1
        events, positions = events[positions >= 0], positions[positions >= 0]
        chosen = rows[positions]
        elapsed = ages[chosen] - query[events]
        keep = np.ones(events.size, dtype=bool)
        if max_lag_seconds is not None:
            keep &= elapsed <= max_lag_seconds
        index[events[keep]] = chosen[keep]
        lag[events[keep]] = elapsed[keep]
    mid, spread, depth = [np.full(query.size, np.nan) for _ in range(3)]
    found = index >= 0
    selected = index[found]
    mid[found] = 0.5 * bid[selected] + 0.5 * ask[selected]
    spread[found] = ask[selected] - bid[selected]
    depth[found] = bv[selected] + av[selected]
    return AsOfQuotes(index, mid, spread, depth, lag)


def weighted_bin_mean(offsets: np.ndarray, seconds: np.ndarray, value: np.ndarray,
                      weight: np.ndarray, *, n_bins: int = 10,
                      bin_width_seconds: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and observed weight for each sample/bin, using identical masks.

    Bin 0 is [0, width), with the horizon endpoint included in the last bin
    for compatibility with V2. Missing values never enter the denominator.
    A bin with no defined, positive-weight observations has mean NaN, weight
    zero. Pass unit weights for a per-channel mean; retain returned weight as
    an explicit coverage feature alongside total event volume/count.
    """
    ages, values, weights = [_vector(x, name) for x, name in zip(
        (seconds, value, weight), ("seconds", "value", "weight"), strict=True)]
    if not (ages.size == values.size == weights.size):
        raise ValueError("binning columns must have equal lengths")
    offsets = _offsets(offsets, ages.size)
    if (isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer))
            or n_bins <= 0 or not np.isfinite(bin_width_seconds)
            or bin_width_seconds <= 0):
        raise ValueError("n_bins and bin_width_seconds must be positive")
    horizon = n_bins * bin_width_seconds
    if not np.isfinite(horizon):
        raise ValueError("bin horizon must be finite")
    observed = np.zeros((offsets.size - 1, n_bins), dtype=np.float64)
    result = np.full(observed.shape, np.nan)
    valid = (np.isfinite(ages) & (ages >= 0) & (ages <= horizon)
             & np.isfinite(values) & np.isfinite(weights) & (weights > 0))
    for sample, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        rows = np.flatnonzero(valid[start:end]) + start
        bins = np.minimum((ages[rows] / bin_width_seconds).astype(np.int64), n_bins - 1)
        denominator = np.bincount(bins, weights=weights[rows], minlength=n_bins)
        with np.errstate(over="ignore", invalid="ignore"):
            numerator = np.bincount(bins, weights=values[rows] * weights[rows], minlength=n_bins)
            np.divide(numerator, denominator, out=result[sample], where=denominator > 0)
        observed[sample] = denominator
    result[~np.isfinite(result)] = np.nan
    observed[~np.isfinite(observed)] = np.nan
    return result, observed
