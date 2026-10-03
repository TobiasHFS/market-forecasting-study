"""Extract fixed-clock microstructure paths from the final 60 seconds.

The v1 feature set summarizes nested trailing windows.  That is robust, but it
throws away the path within a window.  This module adds a complementary,
book-aligned representation inspired by the useful part of modern LOB models:
fixed clock-time bins, signed event marks, contemporaneous price displacement,
and depth-normalized order-flow imbalance.  It deliberately avoids a giant raw
event Transformer; the resulting tensors can be used by LightGBM, TabM, or a
small temporal network on an 8 GB GPU.

Bin zero is closest to the prediction timestamp.  All inputs precede that
timestamp.  ``sample_id`` is used only for row alignment and is never emitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "ms-capital-real-financial-market-forecasting"
BASE_FEATURE_ROOT = PROJECT_ROOT / "artifacts" / "features"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "features"

SPLIT_ROWS = {"train": 1_257_637, "test": 647_896}
N_BINS = 10
BIN_WIDTH_SECONDS = 6.0
HORIZON_SECONDS = N_BINS * BIN_WIDTH_SECONDS

MARKET_CHANNELS: tuple[tuple[str, str], ...] = (
    ("snapshot_count_log1p", "identity"),
    ("mid_position_terminal_spread_units_mean", "asinh_signed"),
    ("microprice_terminal_spread_units_mean", "asinh_signed"),
    ("l1_imbalance_mean", "bounded"),
    ("l2_cumulative_imbalance_mean", "bounded"),
    ("relative_spread_mean", "identity"),
    ("log_l1_depth_ratio_mean", "asinh_signed"),
    ("ofi_l1_terminal_depth_scaled_sum", "asinh_signed"),
    ("trade_vwap_terminal_spread_units", "asinh_signed"),
    ("transaction_volume_log1p", "identity"),
    ("transaction_count_log1p", "identity"),
)

ORDER_CHANNELS: tuple[tuple[str, str], ...] = (
    ("event_count_log1p", "identity"),
    ("volume_log1p", "identity"),
    ("pressure_count_imbalance", "bounded"),
    ("pressure_volume_imbalance", "bounded"),
    ("side_count_imbalance", "bounded"),
    ("side_volume_imbalance", "bounded"),
    ("cancel_event_fraction", "bounded"),
    ("cancel_volume_fraction", "bounded"),
    ("vwap_contemporaneous_mid_terminal_spread_units", "asinh_signed"),
    ("new_directional_offset_terminal_spread_units", "asinh_signed"),
    ("cancel_directional_offset_terminal_spread_units", "asinh_signed"),
)

TRADE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("event_count_log1p", "identity"),
    ("volume_log1p", "identity"),
    ("signed_count_imbalance", "bounded"),
    ("signed_volume_imbalance", "bounded"),
    ("vwap_contemporaneous_mid_terminal_spread_units", "asinh_signed"),
    ("directional_offset_terminal_spread_units", "asinh_signed"),
)

N_MARKET_CHANNELS = 11
N_ORDER_CHANNELS = 11
N_TRADE_CHANNELS = 6

assert len(MARKET_CHANNELS) == N_MARKET_CHANNELS
assert len(ORDER_CHANNELS) == N_ORDER_CHANNELS
assert len(TRADE_CHANNELS) == N_TRADE_CHANNELS


def _names(prefix: str, channels: Sequence[tuple[str, str]]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    kinds: list[str] = []
    for bin_index in range(N_BINS):
        left = bin_index * BIN_WIDTH_SECONDS
        right = (bin_index + 1) * BIN_WIDTH_SECONDS
        label = f"b{bin_index}_{left:g}to{right:g}s"
        for channel, kind in channels:
            names.append(f"{prefix}_{channel}_{label}")
            kinds.append(kind)
    return names, kinds


def market_schema() -> tuple[list[str], list[str]]:
    return _names("market_path", MARKET_CHANNELS)


def flow_schema() -> tuple[list[str], list[str]]:
    order_names, order_kinds = _names("order_path", ORDER_CHANNELS)
    trade_names, trade_kinds = _names("trade_path", TRADE_CHANNELS)
    return order_names + trade_names, order_kinds + trade_kinds


def _numpy_column(
    table: pa.Table, name: str, dtype: np.dtype, null_fill: float | int
) -> np.ndarray:
    column = table.column(name)
    array = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    if array.null_count:
        array = pc.fill_null(array, pa.scalar(null_fill, type=array.type))
    return np.ascontiguousarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


@nb.njit(cache=True)
def _group_offsets(sample_id: np.ndarray, n_samples: int) -> np.ndarray:
    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    previous = -1
    for row in range(sample_id.size):
        current = int(sample_id[row])
        if current < 0 or current >= n_samples:
            raise ValueError("sample_id outside expected range")
        if current < previous:
            raise ValueError("sample_id is not grouped in nondecreasing order")
        offsets[current + 1] += 1
        previous = current
    for sample in range(n_samples):
        offsets[sample + 1] += offsets[sample]
    return offsets


@nb.njit(cache=True, inline="always")
def _time_bin(value: float) -> int:
    if not math.isfinite(value) or value < 0.0 or value > HORIZON_SECONDS:
        return -1
    result = int(value / BIN_WIDTH_SECONDS)
    return N_BINS - 1 if result >= N_BINS else result


@nb.njit(cache=True, inline="always")
def _valid_reference(mid: float, spread: float) -> bool:
    return (
        math.isfinite(mid)
        and mid > 0.0
        and math.isfinite(spread)
        and spread > max(1e-12, abs(mid) * 1e-8)
    )


@nb.njit(cache=True, parallel=True)
def _market_kernel(
    offsets: np.ndarray,
    seconds: np.ndarray,
    transaction_avgprice: np.ndarray,
    transaction_volume: np.ndarray,
    transaction_count: np.ndarray,
    ask_price_1: np.ndarray,
    ask_volume_1: np.ndarray,
    bid_price_1: np.ndarray,
    bid_volume_1: np.ndarray,
    ask_volume_2: np.ndarray,
    bid_volume_2: np.ndarray,
    reference_mid: np.ndarray,
    reference_spread: np.ndarray,
    reference_depth: np.ndarray,
    output: np.ndarray,
) -> None:
    channels = N_MARKET_CHANNELS
    for sample in nb.prange(offsets.size - 1):
        ref_mid = float(reference_mid[sample])
        ref_spread = float(reference_spread[sample])
        ref_depth = float(reference_depth[sample])
        good_reference = _valid_reference(ref_mid, ref_spread)
        good_depth = math.isfinite(ref_depth) and ref_depth > 0.0
        start = int(offsets[sample])
        end = int(offsets[sample + 1])

        previous_bid_price = math.nan
        previous_ask_price = math.nan
        previous_bid_volume = math.nan
        previous_ask_volume = math.nan

        for row in range(start, end):
            current_bid_price = float(bid_price_1[row])
            current_ask_price = float(ask_price_1[row])
            current_bid_volume = float(bid_volume_1[row])
            current_ask_volume = float(ask_volume_1[row])
            bin_index = _time_bin(float(seconds[row]))

            if bin_index >= 0:
                base = bin_index * channels
                valid_book = (
                    math.isfinite(current_bid_price)
                    and math.isfinite(current_ask_price)
                    and current_bid_price > 0.0
                    and current_ask_price >= current_bid_price
                    and current_bid_volume >= 0.0
                    and current_ask_volume >= 0.0
                )
                if valid_book:
                    output[sample, base] += 1.0
                    mid = 0.5 * (current_bid_price + current_ask_price)
                    spread = current_ask_price - current_bid_price
                    depth = current_bid_volume + current_ask_volume
                    if good_reference:
                        output[sample, base + 1] += (mid - ref_mid) / ref_spread
                    if depth > 0.0:
                        output[sample, base + 3] += (
                            current_bid_volume - current_ask_volume
                        ) / depth
                        microprice = (
                            current_ask_price * current_bid_volume
                            + current_bid_price * current_ask_volume
                        ) / depth
                        if good_reference:
                            output[sample, base + 2] += (
                                microprice - ref_mid
                            ) / ref_spread
                    level2_bid = float(bid_volume_2[row])
                    level2_ask = float(ask_volume_2[row])
                    cumulative_depth = depth + level2_bid + level2_ask
                    if level2_bid >= 0.0 and level2_ask >= 0.0 and cumulative_depth > 0.0:
                        output[sample, base + 4] += (
                            current_bid_volume
                            + level2_bid
                            - current_ask_volume
                            - level2_ask
                        ) / cumulative_depth
                    if mid > 0.0:
                        output[sample, base + 5] += spread / mid
                    if good_depth and depth > 0.0:
                        output[sample, base + 6] += math.log(depth / ref_depth)

                    if (
                        math.isfinite(previous_bid_price)
                        and math.isfinite(previous_ask_price)
                        and good_depth
                    ):
                        bid_event = 0.0
                        if current_bid_price >= previous_bid_price:
                            bid_event += current_bid_volume
                        if current_bid_price <= previous_bid_price:
                            bid_event -= previous_bid_volume
                        ask_event = 0.0
                        if current_ask_price <= previous_ask_price:
                            ask_event -= current_ask_volume
                        if current_ask_price >= previous_ask_price:
                            ask_event += previous_ask_volume
                        output[sample, base + 7] += (bid_event + ask_event) / ref_depth

                    bar_volume = float(transaction_volume[row])
                    bar_price = float(transaction_avgprice[row])
                    if bar_volume > 0.0:
                        output[sample, base + 9] += bar_volume
                        if good_reference and math.isfinite(bar_price) and bar_price > 0.0:
                            output[sample, base + 8] += (
                                (bar_price - ref_mid) / ref_spread
                            ) * bar_volume
                    bar_count = float(transaction_count[row])
                    if bar_count > 0.0:
                        output[sample, base + 10] += bar_count

            if (
                math.isfinite(current_bid_price)
                and math.isfinite(current_ask_price)
                and current_bid_volume >= 0.0
                and current_ask_volume >= 0.0
            ):
                previous_bid_price = current_bid_price
                previous_ask_price = current_ask_price
                previous_bid_volume = current_bid_volume
                previous_ask_volume = current_ask_volume

        for bin_index in range(N_BINS):
            base = bin_index * channels
            count = float(output[sample, base])
            volume = float(output[sample, base + 9])
            if count > 0.0:
                for channel in range(1, 7):
                    output[sample, base + channel] /= count
                output[sample, base] = math.log1p(count)
            else:
                for channel in range(1, 8):
                    output[sample, base + channel] = math.nan
            if volume > 0.0:
                output[sample, base + 8] /= volume
            else:
                output[sample, base + 8] = math.nan
            output[sample, base + 9] = math.log1p(max(volume, 0.0))
            output[sample, base + 10] = math.log1p(
                max(float(output[sample, base + 10]), 0.0)
            )


@nb.njit(cache=True, inline="always")
def _contemporaneous_mid(
    sample: int,
    bin_index: int,
    reference_mid: np.ndarray,
    reference_spread: np.ndarray,
    market_path: np.ndarray,
) -> float:
    ref_mid = float(reference_mid[sample])
    ref_spread = float(reference_spread[sample])
    units = float(
        market_path[
            sample,
            bin_index * N_MARKET_CHANNELS + 1,
        ]
    )
    if _valid_reference(ref_mid, ref_spread) and math.isfinite(units):
        return ref_mid + units * ref_spread
    return ref_mid


@nb.njit(cache=True, parallel=True)
def _order_kernel(
    offsets: np.ndarray,
    seconds: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    side: np.ndarray,
    action: np.ndarray,
    reference_mid: np.ndarray,
    reference_spread: np.ndarray,
    market_path: np.ndarray,
    output: np.ndarray,
) -> None:
    channels = N_ORDER_CHANNELS
    for sample in nb.prange(offsets.size - 1):
        ref_mid = float(reference_mid[sample])
        ref_spread = float(reference_spread[sample])
        good_reference = _valid_reference(ref_mid, ref_spread)
        start = int(offsets[sample])
        end = int(offsets[sample + 1])
        for row in range(start, end):
            bin_index = _time_bin(float(seconds[row]))
            if bin_index < 0:
                continue
            row_side = int(side[row])
            row_action = int(action[row])
            row_volume = float(volume[row])
            row_price = float(price[row])
            if (
                (row_side != 0 and row_side != 1)
                or (row_action != 0 and row_action != 1)
                or row_volume < 0.0
            ):
                continue
            base = bin_index * channels
            side_sign = 1.0 - 2.0 * row_side
            action_sign = 1.0 - 2.0 * row_action
            pressure_sign = side_sign * action_sign
            output[sample, base] += 1.0
            output[sample, base + 1] += row_volume
            output[sample, base + 2] += pressure_sign
            output[sample, base + 3] += pressure_sign * row_volume
            output[sample, base + 4] += side_sign
            output[sample, base + 5] += side_sign * row_volume
            if row_action == 1:
                output[sample, base + 6] += 1.0
                output[sample, base + 7] += row_volume
            if good_reference and math.isfinite(row_price) and row_price > 0.0 and row_volume > 0.0:
                aligned_mid = _contemporaneous_mid(
                    sample, bin_index, reference_mid, reference_spread, market_path
                )
                displacement = (row_price - aligned_mid) / ref_spread
                output[sample, base + 8] += displacement * row_volume
                directional = side_sign * displacement
                if row_action == 0:
                    output[sample, base + 9] += directional * row_volume
                else:
                    output[sample, base + 10] += directional * row_volume

        for bin_index in range(N_BINS):
            base = bin_index * channels
            count = float(output[sample, base])
            total_volume = float(output[sample, base + 1])
            cancel_count = float(output[sample, base + 6])
            cancel_volume = float(output[sample, base + 7])
            new_volume = total_volume - cancel_volume
            if count > 0.0:
                output[sample, base + 2] /= count
                output[sample, base + 4] /= count
                output[sample, base + 6] /= count
            else:
                output[sample, base + 2] = math.nan
                output[sample, base + 4] = math.nan
                output[sample, base + 6] = math.nan
            if total_volume > 0.0:
                output[sample, base + 3] /= total_volume
                output[sample, base + 5] /= total_volume
                output[sample, base + 7] /= total_volume
                output[sample, base + 8] /= total_volume
            else:
                for channel in (3, 5, 7, 8):
                    output[sample, base + channel] = math.nan
            if new_volume > 0.0:
                output[sample, base + 9] /= new_volume
            else:
                output[sample, base + 9] = math.nan
            if cancel_volume > 0.0:
                output[sample, base + 10] /= cancel_volume
            else:
                output[sample, base + 10] = math.nan
            output[sample, base] = math.log1p(max(count, 0.0))
            output[sample, base + 1] = math.log1p(max(total_volume, 0.0))


@nb.njit(cache=True, parallel=True)
def _trade_kernel(
    offsets: np.ndarray,
    seconds: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    side: np.ndarray,
    reference_mid: np.ndarray,
    reference_spread: np.ndarray,
    market_path: np.ndarray,
    output: np.ndarray,
) -> None:
    channels = N_TRADE_CHANNELS
    destination_offset = N_BINS * N_ORDER_CHANNELS
    for sample in nb.prange(offsets.size - 1):
        ref_mid = float(reference_mid[sample])
        ref_spread = float(reference_spread[sample])
        good_reference = _valid_reference(ref_mid, ref_spread)
        start = int(offsets[sample])
        end = int(offsets[sample + 1])
        for row in range(start, end):
            bin_index = _time_bin(float(seconds[row]))
            if bin_index < 0:
                continue
            row_side = int(side[row])
            row_volume = float(volume[row])
            row_price = float(price[row])
            if (row_side != 0 and row_side != 1) or row_volume < 0.0:
                continue
            base = destination_offset + bin_index * channels
            side_sign = 1.0 - 2.0 * row_side
            output[sample, base] += 1.0
            output[sample, base + 1] += row_volume
            output[sample, base + 2] += side_sign
            output[sample, base + 3] += side_sign * row_volume
            if good_reference and math.isfinite(row_price) and row_price > 0.0 and row_volume > 0.0:
                aligned_mid = _contemporaneous_mid(
                    sample, bin_index, reference_mid, reference_spread, market_path
                )
                displacement = (row_price - aligned_mid) / ref_spread
                output[sample, base + 4] += displacement * row_volume
                output[sample, base + 5] += side_sign * displacement * row_volume

        for bin_index in range(N_BINS):
            base = destination_offset + bin_index * channels
            count = float(output[sample, base])
            total_volume = float(output[sample, base + 1])
            if count > 0.0:
                output[sample, base + 2] /= count
            else:
                output[sample, base + 2] = math.nan
            if total_volume > 0.0:
                for channel in range(3, 6):
                    output[sample, base + channel] /= total_volume
            else:
                for channel in range(3, 6):
                    output[sample, base + channel] = math.nan
            output[sample, base] = math.log1p(max(count, 0.0))
            output[sample, base + 1] = math.log1p(max(total_volume, 0.0))


def _reference_columns(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = json.loads(
        (BASE_FEATURE_ROOT / f"{split}_market.names.json").read_text(encoding="utf-8")
    )
    matrix = np.load(
        BASE_FEATURE_ROOT / f"{split}_market.npy", mmap_mode="r", allow_pickle=False
    )
    index = {name: position for position, name in enumerate(names)}
    return (
        matrix[:, index["ref_mid"]],
        matrix[:, index["ref_spread"]],
        matrix[:, index["ref_l1_depth"]],
    )


def _write_schema(stem: Path, names: list[str], kinds: list[str], metadata: dict) -> None:
    stem.with_suffix(".names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    stem.with_suffix(".kinds.json").write_text(json.dumps(kinds, indent=2), encoding="utf-8")
    stem.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_market(split: str, *, overwrite: bool = False) -> Path:
    n_samples = SPLIT_ROWS[split]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{split}_market_path.npy"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    names, kinds = market_schema()
    started = time.perf_counter()
    table = feather.read_table(
        DATA_ROOT / split / "market.feather",
        columns=[
            "sample_id",
            "seconds_before_predict",
            "transaction_avgprice",
            "transaction_volume",
            "transaction_count",
            "ask_price_1",
            "ask_volume_1",
            "bid_price_1",
            "bid_volume_1",
            "ask_volume_2",
            "bid_volume_2",
        ],
        memory_map=True,
        use_threads=True,
    )
    sample_id = _numpy_column(table, "sample_id", np.int32, -1)
    offsets = _group_offsets(sample_id, n_samples)
    del sample_id
    columns = {
        "seconds": _numpy_column(table, "seconds_before_predict", np.float32, np.nan),
        "transaction_avgprice": _numpy_column(table, "transaction_avgprice", np.float32, np.nan),
        "transaction_volume": _numpy_column(table, "transaction_volume", np.int32, -1),
        "transaction_count": _numpy_column(table, "transaction_count", np.int32, -1),
        "ask_price_1": _numpy_column(table, "ask_price_1", np.float32, np.nan),
        "ask_volume_1": _numpy_column(table, "ask_volume_1", np.int32, -1),
        "bid_price_1": _numpy_column(table, "bid_price_1", np.float32, np.nan),
        "bid_volume_1": _numpy_column(table, "bid_volume_1", np.int32, -1),
        "ask_volume_2": _numpy_column(table, "ask_volume_2", np.int32, -1),
        "bid_volume_2": _numpy_column(table, "bid_volume_2", np.int32, -1),
    }
    del table
    reference_mid, reference_spread, reference_depth = _reference_columns(split)
    temporary = path.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(n_samples, len(names))
    )
    output[:] = 0.0
    _market_kernel(
        offsets,
        columns["seconds"],
        columns["transaction_avgprice"],
        columns["transaction_volume"],
        columns["transaction_count"],
        columns["ask_price_1"],
        columns["ask_volume_1"],
        columns["bid_price_1"],
        columns["bid_volume_1"],
        columns["ask_volume_2"],
        columns["bid_volume_2"],
        reference_mid,
        reference_spread,
        reference_depth,
        output,
    )
    output.flush()
    del output, columns, offsets
    gc.collect()
    os.replace(temporary, path)
    metadata = {
        "split": split,
        "source": "market_path",
        "rows": n_samples,
        "features": len(names),
        "n_bins": N_BINS,
        "bin_width_seconds": BIN_WIDTH_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "sha256": _sha256(path),
    }
    _write_schema(path.with_suffix(""), names, kinds, metadata)
    return path


def _load_flow_table(path: Path, order: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    requested = ["sample_id", "seconds_before_predict", "price", "volume", "side"]
    if order:
        requested.append("order_action")
    table = feather.read_table(path, columns=requested, memory_map=True, use_threads=True)
    values = {
        "sample_id": _numpy_column(table, "sample_id", np.int32, -1),
        "seconds": _numpy_column(table, "seconds_before_predict", np.float32, np.nan),
        "price": _numpy_column(table, "price", np.float32, np.nan),
        "volume": _numpy_column(table, "volume", np.int32, -1),
        "side": _numpy_column(table, "side", np.int8, -1),
    }
    if order:
        values["action"] = _numpy_column(table, "order_action", np.int8, -1)
    del table
    return values, values.pop("sample_id")


def extract_flow(split: str, *, overwrite: bool = False) -> Path:
    n_samples = SPLIT_ROWS[split]
    market_path_file = OUTPUT_ROOT / f"{split}_market_path.npy"
    if not market_path_file.exists():
        raise FileNotFoundError(
            f"extract market path before flow path: {market_path_file}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{split}_flow_path.npy"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    names, kinds = flow_schema()
    started = time.perf_counter()
    market_path = np.load(market_path_file, mmap_mode="r", allow_pickle=False)
    reference_mid, reference_spread, _ = _reference_columns(split)
    temporary = path.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(n_samples, len(names))
    )
    output[:] = 0.0

    values, sample_id = _load_flow_table(DATA_ROOT / split / "order.feather", True)
    offsets = _group_offsets(sample_id, n_samples)
    del sample_id
    _order_kernel(
        offsets,
        values["seconds"],
        values["price"],
        values["volume"],
        values["side"],
        values["action"],
        reference_mid,
        reference_spread,
        market_path,
        output,
    )
    del values, offsets
    gc.collect()

    values, sample_id = _load_flow_table(
        DATA_ROOT / split / "transaction.feather", False
    )
    offsets = _group_offsets(sample_id, n_samples)
    del sample_id
    _trade_kernel(
        offsets,
        values["seconds"],
        values["price"],
        values["volume"],
        values["side"],
        reference_mid,
        reference_spread,
        market_path,
        output,
    )
    output.flush()
    del output, values, offsets, market_path
    gc.collect()
    os.replace(temporary, path)
    metadata = {
        "split": split,
        "source": "flow_path",
        "rows": n_samples,
        "features": len(names),
        "n_bins": N_BINS,
        "bin_width_seconds": BIN_WIDTH_SECONDS,
        "elapsed_seconds": time.perf_counter() - started,
        "sha256": _sha256(path),
    }
    _write_schema(path.with_suffix(""), names, kinds, metadata)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument(
        "--source", choices=("market", "flow", "all"), default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        if args.source in {"market", "all"}:
            print(f"extracting {split} market path", flush=True)
            print(extract_market(split, overwrite=args.overwrite), flush=True)
        if args.source in {"flow", "all"}:
            print(f"extracting {split} flow path", flush=True)
            print(extract_flow(split, overwrite=args.overwrite), flush=True)


if __name__ == "__main__":
    main()
