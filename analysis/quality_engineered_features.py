"""Quality, leakage-risk, and train/test drift checks for extracted features."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from feature_families import (
    FAMILY_ORDER,
    FEATURE_ROOT,
    RAW_SOURCES,
    classify_raw_feature,
    family_manifest,
    infer_transform_kind,
)
from pipeline_config import DATA_ROOT, DIAGNOSTIC_ROOT, RANDOM_SEED, ensure_artifact_directories


PROFILE_SAMPLE = 100_000


def _names(split: str, source: str) -> list[str]:
    return json.loads(
        (FEATURE_ROOT / f"{split}_{source}.names.json").read_text(encoding="utf-8")
    )


def _array(split: str, source: str) -> np.ndarray:
    return np.load(
        FEATURE_ROOT / f"{split}_{source}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )


def _sample_indices(n_rows: int, size: int = PROFILE_SAMPLE) -> np.ndarray:
    if n_rows <= size:
        return np.arange(n_rows, dtype=np.int64)
    # Even chronological coverage avoids a random draw accidentally missing a
    # late regime while remaining deterministic and inexpensive.
    return np.linspace(0, n_rows - 1, size, dtype=np.int64)


def _finite_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return math.nan
    xv = x[valid].astype(np.float64, copy=False)
    yv = y[valid].astype(np.float64, copy=False)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denominator = math.sqrt(float(np.dot(xv, xv) * np.dot(yv, yv)))
    return float(np.dot(xv, yv) / denominator) if denominator > 0.0 else math.nan


def _bounds(name: str) -> tuple[float, float] | None:
    lower = name.lower()
    if "spread_units" in lower or "signed_to" in lower:
        return None
    if "_minus_" in lower and "_imbalance" in lower:
        return (-2.000001, 2.000001)
    if any(token in lower for token in ("_frac", "_fraction", "_share", "_efficiency", "_flag")):
        return (0.0, 1.0)
    if "_imbalance" in lower or "_agreement" in lower:
        return (-1.000001, 1.000001)
    return None


def build_quality_profile() -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_artifact_directories()
    label = feather.read_table(
        DATA_ROOT / "train" / "label.feather", columns=["target", "month", "sample_id"]
    ).to_pandas()
    if not np.array_equal(label["sample_id"].to_numpy(), np.arange(len(label))):
        raise ValueError("label sample_id is not exact row alignment")
    target = label["target"].to_numpy(dtype=np.float64)

    rows: list[dict[str, object]] = []
    sources = RAW_SOURCES + ("mechanics",)
    for source in sources:
        train_names = _names("train", source)
        test_names = _names("test", source)
        if train_names != test_names:
            raise ValueError(f"train/test schema mismatch: {source}")
        train = _array("train", source)
        test = _array("test", source)
        train_index = _sample_indices(train.shape[0])
        test_index = _sample_indices(test.shape[0])
        train_sample_matrix = np.asarray(train[train_index, :], dtype=np.float64)
        test_sample_matrix = np.asarray(test[test_index, :], dtype=np.float64)
        sampled_target = target[train_index]
        normalized_index = train_index.astype(np.float64) / max(train.shape[0] - 1, 1)

        n_columns = len(train_names)
        train_finite_count = np.zeros(n_columns, dtype=np.int64)
        test_finite_count = np.zeros(n_columns, dtype=np.int64)
        train_inf_count = np.zeros(n_columns, dtype=np.int64)
        test_inf_count = np.zeros(n_columns, dtype=np.int64)
        lower_violations = np.zeros(n_columns, dtype=np.int64)
        upper_violations = np.zeros(n_columns, dtype=np.int64)
        bounds = [_bounds(name) for name in train_names]

        for array, finite_count, inf_count, count_bounds in (
            (train, train_finite_count, train_inf_count, True),
            (test, test_finite_count, test_inf_count, False),
        ):
            for start in range(0, array.shape[0], 50_000):
                block = np.asarray(array[start : start + 50_000, :])
                finite = np.isfinite(block)
                finite_count += finite.sum(axis=0, dtype=np.int64)
                inf_count += np.isinf(block).sum(axis=0, dtype=np.int64)
                if count_bounds:
                    for bound_column, bound in enumerate(bounds):
                        if bound is None:
                            continue
                        valid_values = block[finite[:, bound_column], bound_column]
                        lower_violations[bound_column] += np.sum(valid_values < bound[0])
                        upper_violations[bound_column] += np.sum(valid_values > bound[1])

        for column, name in enumerate(train_names):
            train_sample = train_sample_matrix[:, column]
            test_sample = test_sample_matrix[:, column]
            train_finite = np.isfinite(train_sample)
            test_finite = np.isfinite(test_sample)
            train_values = train_sample[train_finite]
            test_values = test_sample[test_finite]
            train_q = (
                np.quantile(train_values, [0.01, 0.25, 0.5, 0.75, 0.99])
                if train_values.size
                else np.full(5, np.nan)
            )
            test_q = (
                np.quantile(test_values, [0.01, 0.25, 0.5, 0.75, 0.99])
                if test_values.size
                else np.full(5, np.nan)
            )
            robust_scale = max(float(train_q[3] - train_q[1]), 1e-12)
            family = (
                "liquidity_mechanics"
                if source == "mechanics"
                else classify_raw_feature(source, name)
            )
            rows.append(
                {
                    "source": source,
                    "feature": name,
                    "family": family,
                    "transform": infer_transform_kind(name),
                    "train_missing_rate": float(1.0 - train_finite_count[column] / train.shape[0]),
                    "test_missing_rate": float(1.0 - test_finite_count[column] / test.shape[0]),
                    "missing_rate_shift": float(
                        train_finite_count[column] / train.shape[0]
                        - test_finite_count[column] / test.shape[0]
                    ),
                    "train_inf_count": int(train_inf_count[column]),
                    "test_inf_count": int(test_inf_count[column]),
                    "train_q01": float(train_q[0]),
                    "train_median": float(train_q[2]),
                    "train_q99": float(train_q[4]),
                    "test_q01": float(test_q[0]),
                    "test_median": float(test_q[2]),
                    "test_q99": float(test_q[4]),
                    "robust_median_shift_iqr": float((test_q[2] - train_q[2]) / robust_scale),
                    "sample_target_correlation": _finite_correlation(train_sample, sampled_target),
                    "sample_index_correlation": _finite_correlation(train_sample, normalized_index),
                    "bound_lower_violations": int(lower_violations[column]),
                    "bound_upper_violations": int(upper_violations[column]),
                }
            )

    frame = pd.DataFrame(rows)
    manifest = family_manifest()
    summary: dict[str, object] = {
        "rows_profiled": int(len(frame)),
        "family_feature_counts": {family: len(manifest[family]) for family in FAMILY_ORDER},
        "any_infinite_values": bool(
            (frame["train_inf_count"] + frame["test_inf_count"]).sum() > 0
        ),
        "total_bound_violations": int(
            (frame["bound_lower_violations"] + frame["bound_upper_violations"]).sum()
        ),
        "largest_absolute_missing_shift": float(frame["missing_rate_shift"].abs().max()),
        "features_missing_shift_over_5pct": int((frame["missing_rate_shift"].abs() > 0.05).sum()),
        "features_robust_median_shift_over_1_iqr": int((frame["robust_median_shift_iqr"].abs() > 1.0).sum()),
        "features_abs_index_correlation_over_0_95": int((frame["sample_index_correlation"].abs() > 0.95).sum()),
        "features_abs_target_correlation_over_0_5": int((frame["sample_target_correlation"].abs() > 0.5).sum()),
        "profile_sample_rows": PROFILE_SAMPLE,
        "random_seed_reserved": RANDOM_SEED,
    }
    return frame, summary


def main() -> None:
    frame, summary = build_quality_profile()
    output_csv = DIAGNOSTIC_ROOT / "engineered_feature_quality.csv"
    output_json = DIAGNOSTIC_ROOT / "engineered_feature_quality_summary.json"
    frame.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
