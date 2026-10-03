"""Versioned feature-set assembly for research challengers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_families import materialize_feature_set
from pipeline_config import TEST_SAMPLES, TRAIN_SAMPLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "features"
BASE_FEATURE_SET = "multiscale_mechanics_scale"
VALID_FEATURE_SETS = {
    "base_only",
    "sequence_stationary",
    "sequence_all",
    "base_plus_sequence_stationary",
    "base_plus_sequence_all",
}


@dataclass
class V2Features:
    matrix: np.ndarray
    names: list[str]
    kinds: list[str]


def _read_json_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise TypeError(f"invalid string-list manifest: {path}")
    return payload


def _load_path(split: str, source: str) -> tuple[np.ndarray, list[str], list[str]]:
    path = FEATURE_ROOT / f"{split}_{source}.npy"
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    names = _read_json_list(path.with_suffix(".names.json"))
    kinds = _read_json_list(path.with_suffix(".kinds.json"))
    expected_rows = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    if matrix.shape != (expected_rows, len(names)) or len(names) != len(kinds):
        raise ValueError(f"invalid path feature contract: {path}")
    return matrix, names, kinds


def _is_stationary_path_feature(name: str) -> bool:
    """Exclude absolute activity scale while retaining relative state/path."""

    shifted_scale_tokens = (
        "snapshot_count_log1p",
        "event_count_log1p",
        "_volume_log1p",
        "transaction_count_log1p",
    )
    return not any(token in name for token in shifted_scale_tokens)


def materialize_v2_feature_set(
    split: str, feature_set: str, *, overwrite: bool = False
) -> V2Features:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(f"unknown v2 feature set: {feature_set}")
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_ROOT / f"{split}_{feature_set}.npy"
    names_path = cache.with_suffix(".names.json")
    kinds_path = cache.with_suffix(".kinds.json")
    if cache.exists() and not overwrite:
        names = _read_json_list(names_path)
        kinds = _read_json_list(kinds_path)
        matrix = np.load(cache, mmap_mode="r", allow_pickle=False)
        if matrix.shape[1] != len(names) or len(names) != len(kinds):
            raise ValueError(f"cached feature schema mismatch: {cache}")
        return V2Features(matrix=matrix, names=names, kinds=kinds)

    arrays: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[str] = []
    if feature_set == "base_only" or feature_set.startswith("base_plus_"):
        base = materialize_feature_set(split, BASE_FEATURE_SET)
        arrays.append(base.matrix)
        names.extend(base.names)
        kinds.extend(base.kinds)

    if feature_set != "base_only":
        stationary = feature_set.endswith("stationary")
        for source in ("market_path", "flow_path"):
            matrix, source_names, source_kinds = _load_path(split, source)
            selected = np.arange(len(source_names))
            if stationary:
                selected = np.array(
                    [i for i, name in enumerate(source_names) if _is_stationary_path_feature(name)],
                    dtype=np.int64,
                )
            arrays.append(matrix[:, selected])
            names.extend(source_names[i] for i in selected)
            kinds.extend(source_kinds[i] for i in selected)

    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names in v2 materialization")
    n_rows = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    temporary = cache.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(n_rows, len(names))
    )
    destination = 0
    for array in arrays:
        width = array.shape[1]
        output[:, destination : destination + width] = array
        destination += width
    output.flush()
    del output, arrays
    os.replace(temporary, cache)
    names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    kinds_path.write_text(json.dumps(kinds, indent=2), encoding="utf-8")
    return V2Features(
        matrix=np.load(cache, mmap_mode="r", allow_pickle=False),
        names=names,
        kinds=kinds,
    )
