"""CLI for deterministic, source-at-a-time feature extraction.

The source files are much wider in memory than on disk.  This driver therefore
extracts and persists one source at a time, validates the row/column contract,
and releases the arrays before continuing.  Row position is the alignment key;
``sample_id`` is deliberately not written as a model feature.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from flow_features import extract_order_features, extract_transaction_features
from market_features import extract_market_features
from pipeline_config import (
    DATA_ROOT,
    FEATURE_ROOT,
    FLOW_WINDOWS,
    MARKET_WINDOWS,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    ensure_artifact_directories,
)


Extractor = Callable[[Path, int, tuple[float, ...]], tuple[np.ndarray, list[str]]]


SOURCE_SPECS: dict[str, tuple[str, Extractor, tuple[float, ...]]] = {
    "market": ("market.feather", extract_market_features, MARKET_WINDOWS),
    "order": ("order.feather", extract_order_features, FLOW_WINDOWS),
    "transaction": (
        "transaction.feather",
        extract_transaction_features,
        FLOW_WINDOWS,
    ),
}


def _validate_features(
    features: np.ndarray, names: list[str], expected_rows: int, source: str
) -> dict[str, object]:
    if features.dtype != np.float32:
        raise TypeError(f"{source}: expected float32, received {features.dtype}")
    if features.ndim != 2 or features.shape[0] != expected_rows:
        raise ValueError(
            f"{source}: expected ({expected_rows}, p), received {features.shape}"
        )
    if features.shape[1] != len(names):
        raise ValueError(f"{source}: matrix/name count mismatch")
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"{source}: duplicate feature names: {duplicates[:10]}")
    if any("sample_id" in name.lower() for name in names):
        raise ValueError(f"{source}: sample_id leaked into feature names")

    finite = np.isfinite(features)
    finite_rate = float(finite.mean()) if features.size else 1.0
    all_missing = int(np.sum(~finite.any(axis=0)))
    if all_missing:
        raise ValueError(f"{source}: {all_missing} feature columns are entirely non-finite")
    return {
        "rows": int(features.shape[0]),
        "columns": int(features.shape[1]),
        "dtype": str(features.dtype),
        "finite_rate": finite_rate,
        "all_missing_columns": all_missing,
    }


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_save_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def extract_one(split: str, source: str, overwrite: bool = False) -> Path:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if source not in SOURCE_SPECS:
        raise ValueError(f"Unknown source: {source}")

    expected_rows = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    filename, extractor, windows = SOURCE_SPECS[source]
    input_path = DATA_ROOT / split / filename
    output_path = FEATURE_ROOT / f"{split}_{source}.npy"
    names_path = FEATURE_ROOT / f"{split}_{source}.names.json"
    metadata_path = FEATURE_ROOT / f"{split}_{source}.metadata.json"

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to regenerate it"
        )

    started = time.perf_counter()
    features, names = extractor(input_path, expected_rows, windows)
    summary = _validate_features(features, names, expected_rows, source)
    _atomic_save_array(output_path, features)
    _atomic_save_json(names_path, names)
    stat = input_path.stat()
    metadata = {
        "split": split,
        "source": source,
        "input_path": str(input_path.resolve()),
        "input_bytes": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "windows_seconds": list(windows),
        "elapsed_seconds": time.perf_counter() - started,
        **summary,
    }
    _atomic_save_json(metadata_path, metadata)
    del features
    gc.collect()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("train", "test", "both"), default="both"
    )
    parser.add_argument(
        "--source",
        choices=("market", "order", "transaction", "all"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_artifact_directories()
    splits = ("train", "test") if args.split == "both" else (args.split,)
    sources = tuple(SOURCE_SPECS) if args.source == "all" else (args.source,)
    for source in sources:
        for split in splits:
            output = extract_one(split, source, overwrite=args.overwrite)
            print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()

