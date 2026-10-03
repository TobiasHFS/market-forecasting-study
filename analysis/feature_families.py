"""Feature-family manifests, mechanics, and memory-mapped matrix assembly.

The hierarchy is deliberately empirical rather than inherited from a generic
"Tier 1/Tier 2" template:

``observability -> invariant_core -> multiscale_dynamics -> liquidity_mechanics``

Absolute-scale, tied-event path, and independently aggregated cross-level L2
features are quarantined challengers.  The latter are available for diagnosis
but are excluded from every default candidate because the memory-bounded raw
passes cannot guarantee common row support across book levels.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pipeline_config import (
    CORE_FLOW_WINDOWS,
    CORE_MARKET_WINDOWS,
    FEATURE_ROOT,
    FLOW_WINDOWS,
    MARKET_WINDOWS,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    ensure_artifact_directories,
)


RAW_SOURCES = ("market", "order", "transaction")
FAMILY_ORDER = (
    "observability",
    "invariant_core",
    "multiscale_dynamics",
    "liquidity_mechanics",
    "scale_satellite",
    "path_satellite",
    "unaligned_l2_satellite",
)

FEATURE_SET_FAMILIES: dict[str, tuple[str, ...]] = {
    "market_core": ("observability", "invariant_core"),
    "all_core": ("observability", "invariant_core"),
    "multiscale": (
        "observability",
        "invariant_core",
        "multiscale_dynamics",
    ),
    "multiscale_mechanics": (
        "observability",
        "invariant_core",
        "multiscale_dynamics",
        "liquidity_mechanics",
    ),
    "multiscale_scale": (
        "observability",
        "invariant_core",
        "multiscale_dynamics",
        "scale_satellite",
    ),
    "multiscale_path": (
        "observability",
        "invariant_core",
        "multiscale_dynamics",
        "path_satellite",
    ),
    "multiscale_mechanics_scale": (
        "observability",
        "invariant_core",
        "multiscale_dynamics",
        "liquidity_mechanics",
        "scale_satellite",
    ),
}


_WINDOW_PATTERN = re.compile(r"_(?:w)?(?P<seconds>\d+(?:p\d+)?)s$")
_PATH_TOKENS = (
    "_price_log_return_",
    "_price_abs_log_variation_",
    "_price_rms_log_variation_",
    "_price_path_efficiency_",
    "_max_volume_share_",
)
_UNALIGNED_L2_TOKENS = (
    "book_l2_cumulative_imbalance",
    "book_l2_to_l1_depth_ratio",
    "book_ask_l2_gap_rel_ref_mid",
    "book_bid_l2_gap_rel_ref_mid",
)
_SCALE_TOKENS = (
    "_ask_depth_mean_",
    "_bid_depth_mean_",
    "_ask_price_",
    "_bid_price_",
    "_log1p_l1_depth_mean_",
    "_log1p_level2_depth_mean_",
    "_event_count_",
    "_event_rate_hz_",
    "_abs_volume_total_",
    "_volume_rate_per_s_",
    "_vwap_price_",
    "market_transaction_volume_log1p",
    "market_transaction_count_log1p",
    "market_transaction_volume_per_trade_log1p",
    "market_transaction_valid_count_log1p",
    "book_valid_count_log1p",
)


def _window_seconds(name: str) -> float | None:
    match = _WINDOW_PATTERN.search(name)
    if match is None:
        return None
    return float(match.group("seconds").replace("p", "."))


def classify_raw_feature(source: str, name: str) -> str:
    """Assign every extracted feature to exactly one modeling family."""

    if source not in RAW_SOURCES:
        raise ValueError(f"unknown raw source: {source}")
    if "sample_id" in name.lower() or name in {"month", "target"}:
        raise ValueError(f"forbidden identifier/label feature: {name}")
    if name.startswith("ref_"):
        return "scale_satellite"
    if any(token in name for token in _UNALIGNED_L2_TOKENS):
        return "unaligned_l2_satellite"
    if source in {"order", "transaction"} and any(
        token in name for token in _PATH_TOKENS
    ):
        return "path_satellite"

    window = _window_seconds(name)
    if window is None:
        # The only non-reference, non-window columns are source integrity and
        # coverage controls (counts, spans, ages, validity, caps).
        return "observability"
    if any(token in name for token in _SCALE_TOKENS):
        return "scale_satellite"

    core_windows = CORE_MARKET_WINDOWS if source == "market" else CORE_FLOW_WINDOWS
    return "invariant_core" if int(window) in core_windows else "multiscale_dynamics"


def infer_transform_kind(name: str) -> str:
    """Return the predeclared train-fitted transform for one feature."""

    lower = name.lower()
    if any(
        token in lower
        for token in (
            "_frac",
            "_fraction",
            "_imbalance",
            "_share",
            "_efficiency",
            "_flag",
            "_violation",
            "_agreement",
        )
    ):
        return "bounded"
    if any(
        token in lower
        for token in (
            "signed_",
            "pressure_total",
            "_return",
            "_ofi_",
            "spread_units",
            "minus_",
            "_gap_",
        )
    ):
        return "asinh_signed"
    if "log1p" in lower:
        return "identity"
    if any(
        token in lower
        for token in (
            "event_count",
            "row_count",
            "_volume_total",
            "_volume_rate",
            "_depth",
            "_span_s",
            "_age_s",
            "_rate_hz",
        )
    ):
        return "log1p_nonnegative"
    return "identity"


def _load_names(split: str, source: str) -> list[str]:
    path = FEATURE_ROOT / f"{split}_{source}.names.json"
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise TypeError(f"invalid feature-name manifest: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate feature names in {path}")
    return names


def _load_array(split: str, source: str) -> np.ndarray:
    path = FEATURE_ROOT / f"{split}_{source}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != expected:
        raise ValueError(f"invalid feature array contract: {path} -> {array.shape}/{array.dtype}")
    return array


def validate_train_test_manifests() -> None:
    for source in RAW_SOURCES:
        train = _load_names("train", source)
        test = _load_names("test", source)
        if train != test:
            raise ValueError(f"train/test feature schema mismatch for {source}")
        for name in train:
            classify_raw_feature(source, name)


def family_manifest() -> dict[str, list[tuple[str, int, str]]]:
    """Map families to ``(source, column_index, feature_name)`` entries."""

    validate_train_test_manifests()
    result = {family: [] for family in FAMILY_ORDER}
    for source in RAW_SOURCES:
        for index, name in enumerate(_load_names("train", source)):
            result[classify_raw_feature(source, name)].append((source, index, name))

    mechanics_path = FEATURE_ROOT / "train_mechanics.names.json"
    if mechanics_path.exists():
        mechanics_names = _load_names("train", "mechanics")
        if mechanics_names != _load_names("test", "mechanics"):
            raise ValueError("train/test mechanics schema mismatch")
        result["liquidity_mechanics"].extend(
            ("mechanics", index, name)
            for index, name in enumerate(mechanics_names)
        )
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def _safe_spread_ratio(
    numerator: np.ndarray, spread: np.ndarray, mid: np.ndarray
) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(spread)
        & np.isfinite(mid)
        & (spread > np.maximum(1e-12, np.abs(mid) * 1e-8))
    )
    np.divide(numerator, spread, out=out, where=valid)
    return out


def build_liquidity_mechanics(split: str, *, overwrite: bool = False) -> Path:
    """Build row-aligned, book-normalized cross-source mechanics features."""

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    ensure_artifact_directories()
    output_path = FEATURE_ROOT / f"{split}_mechanics.npy"
    names_path = FEATURE_ROOT / f"{split}_mechanics.names.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True")

    arrays = {source: _load_array(split, source) for source in RAW_SOURCES}
    names = {source: _load_names(split, source) for source in RAW_SOURCES}
    index = {
        source: {name: column for column, name in enumerate(source_names)}
        for source, source_names in names.items()
    }

    def column(source: str, name: str) -> np.ndarray:
        try:
            return np.asarray(arrays[source][:, index[source][name]], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f"required mechanics input missing: {source}/{name}") from exc

    mid = column("market", "ref_mid")
    spread = column("market", "ref_spread")
    l1_depth = column("market", "ref_l1_depth")
    mechanics_names: list[str] = []
    mechanics_values: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        if name in mechanics_names:
            raise ValueError(f"duplicate mechanics feature: {name}")
        mechanics_names.append(name)
        mechanics_values.append(np.asarray(values, dtype=np.float32))

    market_match = {2: 10, 5: 10, 15: 30, 30: 30, 60: 60}
    for flow_window in (int(value) for value in FLOW_WINDOWS):
        market_window = market_match[flow_window]
        suffix = f"w{flow_window}s"
        market_suffix = f"{market_window}s"
        order_signed = column(
            "order", f"ref_order_signed_pressure_total_{suffix}"
        )
        order_abs = column("order", f"ref_order_abs_volume_total_{suffix}")
        trade_signed = column(
            "transaction", f"ref_trade_signed_volume_total_{suffix}"
        )
        trade_abs = column(
            "transaction", f"ref_trade_abs_volume_total_{suffix}"
        )
        order_vwap = column("order", f"ref_order_vwap_price_{suffix}")
        trade_vwap = column("transaction", f"ref_trade_vwap_price_{suffix}")
        order_imbalance = column(
            "order", f"order_pressure_volume_imbalance_{suffix}"
        )
        trade_imbalance = column(
            "transaction", f"trade_signed_volume_imbalance_{suffix}"
        )
        book_imbalance = column(
            "market", f"market_book_l1_imbalance_last_{market_suffix}"
        )
        book_ofi = column(
            "market", f"market_book_ofi_l1_depth_scaled_{market_suffix}"
        )

        add(f"mechanics_order_signed_to_l1_depth_{suffix}", _safe_ratio(order_signed, l1_depth))
        add(f"mechanics_order_abs_to_l1_depth_{suffix}", _safe_ratio(order_abs, l1_depth))
        add(f"mechanics_trade_signed_to_l1_depth_{suffix}", _safe_ratio(trade_signed, l1_depth))
        add(f"mechanics_trade_abs_to_l1_depth_{suffix}", _safe_ratio(trade_abs, l1_depth))
        order_displacement = order_vwap - mid
        trade_displacement = trade_vwap - mid
        add(f"mechanics_order_vwap_mid_relative_{suffix}", _safe_ratio(order_displacement, mid))
        add(f"mechanics_order_vwap_spread_units_{suffix}", _safe_spread_ratio(order_displacement, spread, mid))
        add(f"mechanics_trade_vwap_mid_relative_{suffix}", _safe_ratio(trade_displacement, mid))
        add(f"mechanics_trade_vwap_spread_units_{suffix}", _safe_spread_ratio(trade_displacement, spread, mid))
        add(f"mechanics_order_book_agreement_{suffix}", order_imbalance * book_imbalance)
        add(f"mechanics_trade_book_agreement_{suffix}", trade_imbalance * book_imbalance)
        add(f"mechanics_order_trade_agreement_{suffix}", order_imbalance * trade_imbalance)
        add(f"mechanics_order_minus_trade_imbalance_{suffix}", order_imbalance - trade_imbalance)
        bounded_ofi = np.tanh(book_ofi).astype(np.float32)
        add(f"mechanics_order_ofi_agreement_{suffix}", order_imbalance * bounded_ofi)
        add(f"mechanics_trade_ofi_agreement_{suffix}", trade_imbalance * bounded_ofi)
        add(f"mechanics_order_minus_ofi_{suffix}", order_imbalance - bounded_ofi)
        add(f"mechanics_trade_minus_ofi_{suffix}", trade_imbalance - bounded_ofi)
        add(
            f"mechanics_order_minus_trade_vwap_spread_units_{suffix}",
            _safe_spread_ratio(order_vwap - trade_vwap, spread, mid),
        )

    for market_window in (int(value) for value in MARKET_WINDOWS):
        suffix = f"{market_window}s"
        l1_imbalance = column(
            "market", f"market_book_l1_imbalance_mean_{suffix}"
        )
        l2_imbalance = column(
            "market", f"market_book_level2_imbalance_mean_{suffix}"
        )
        l1_ofi = column(
            "market", f"market_book_ofi_l1_depth_scaled_{suffix}"
        )
        l2_ofi = column(
            "market", f"market_book_ofi_l2_depth_scaled_{suffix}"
        )
        add(f"mechanics_l1_minus_l2_imbalance_{suffix}", l1_imbalance - l2_imbalance)
        add(f"mechanics_l1_l2_imbalance_agreement_{suffix}", l1_imbalance * l2_imbalance)
        add(f"mechanics_l1_minus_l2_ofi_{suffix}", l1_ofi - l2_ofi)
        add(f"mechanics_l1_l2_ofi_agreement_{suffix}", np.tanh(l1_ofi) * np.tanh(l2_ofi))

    n_rows = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    temporary = output_path.with_suffix(".npy.tmp.npy")
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, len(mechanics_names)),
    )
    for destination, values in enumerate(mechanics_values):
        if values.shape != (n_rows,):
            raise ValueError(f"mechanics column has wrong shape: {mechanics_names[destination]}")
        matrix[:, destination] = values
    matrix.flush()
    del matrix
    os.replace(temporary, output_path)
    names_path.write_text(json.dumps(mechanics_names, indent=2), encoding="utf-8")
    return output_path


@dataclass(frozen=True)
class MaterializedFeatures:
    matrix: np.ndarray
    names: list[str]
    kinds: list[str]
    families: list[str]


def materialize_feature_set(split: str, feature_set: str) -> MaterializedFeatures:
    """Copy one declared candidate into a contiguous float32 matrix."""

    if feature_set not in FEATURE_SET_FAMILIES:
        raise ValueError(f"unknown feature set: {feature_set}")
    manifest = family_manifest()
    selected_families = set(FEATURE_SET_FAMILIES[feature_set])
    entries: list[tuple[str, int, str, str]] = []
    for family in FAMILY_ORDER:
        if family not in selected_families:
            continue
        for source, index, name in manifest[family]:
            if feature_set == "market_core" and source != "market":
                continue
            entries.append((source, index, name, family))
    if not entries:
        raise ValueError(f"feature set {feature_set} is empty")
    feature_names = [entry[2] for entry in entries]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError(f"feature set {feature_set} contains duplicate names")
    if any(name in {"sample_id", "month", "target"} for name in feature_names):
        raise ValueError("identifier/label leaked into materialized feature set")

    source_arrays = {
        source: _load_array(split, source) for source in sorted({e[0] for e in entries})
    }
    n_rows = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    matrix = np.empty((n_rows, len(entries)), dtype=np.float32)
    for destination, (source, source_column, _name, _family) in enumerate(entries):
        matrix[:, destination] = source_arrays[source][:, source_column]
    return MaterializedFeatures(
        matrix=matrix,
        names=feature_names,
        kinds=[infer_transform_kind(name) for name in feature_names],
        families=[entry[3] for entry in entries],
    )


def write_family_manifest(path: Path) -> Path:
    manifest = family_manifest()
    rows = [
        {
            "family": family,
            "source": source,
            "column_index": index,
            "feature": name,
            "transform": infer_transform_kind(name),
        }
        for family in FAMILY_ORDER
        for source, index, name in manifest[family]
    ]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path
