"""Memory-conscious profiling for the MS Capital market-forecasting data.

The Feather files contain one very large Arrow record batch.  This script reads
only the columns needed for each check, releases them between files, and writes
a compact JSON profile that can be audited or regenerated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE / "ms-capital-real-financial-market-forecasting"
OUTPUT_DIR = WORKSPACE / "analysis" / "output"


def json_value(value: Any) -> Any:
    """Convert numpy/pandas values into finite JSON-native values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def quantiles(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {}
    probs = np.array([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    result = np.quantile(values, probs)
    return {f"q{int(p * 100):02d}": json_value(v) for p, v in zip(probs, result)}


def arrow_schema(path: Path) -> dict[str, Any]:
    reader = ipc.open_file(pa.memory_map(str(path), "r"))
    return {
        "record_batches": reader.num_record_batches,
        "columns": [{"name": field.name, "type": str(field.type)} for field in reader.schema],
    }


def numpy_column(table: pa.Table, name: str) -> np.ndarray:
    column = table.column(name)
    if column.num_chunks != 1:
        column = column.combine_chunks()
    else:
        column = column.chunk(0)
    return column.to_numpy(zero_copy_only=False)


def row_structure(path: Path, expected_samples: int) -> dict[str, Any]:
    print(f"Profiling row structure: {path.relative_to(WORKSPACE)}", flush=True)
    table = feather.read_table(
        path,
        columns=["sample_id", "seconds_before_predict"],
        memory_map=True,
        use_threads=True,
    )
    sample_ids = numpy_column(table, "sample_id")
    seconds = numpy_column(table, "seconds_before_predict")
    n_rows = int(sample_ids.size)

    change_positions = np.flatnonzero(sample_ids[1:] != sample_ids[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change_positions.astype(np.int64)))
    ends = np.concatenate((change_positions.astype(np.int64), np.array([n_rows], dtype=np.int64)))
    group_ids = sample_ids[starts]
    counts = ends - starts

    same_group = sample_ids[1:] == sample_ids[:-1]
    within_diffs = seconds[1:][same_group] - seconds[:-1][same_group]
    nondecreasing = float(np.mean(within_diffs >= 0)) if within_diffs.size else None
    nonincreasing = float(np.mean(within_diffs <= 0)) if within_diffs.size else None
    repeated = float(np.mean(within_diffs == 0)) if within_diffs.size else None

    # Inspect a deterministic, bounded set of complete samples for time-grid shape.
    max_groups = 4096
    inspected_group_indexes = np.unique(
        np.linspace(0, max(len(starts) - 1, 0), min(max_groups, len(starts)), dtype=np.int64)
    )
    unique_times = np.empty(inspected_group_indexes.size, dtype=np.int32)
    time_span = np.empty(inspected_group_indexes.size, dtype=np.float32)
    for out_index, group_index in enumerate(inspected_group_indexes):
        group_seconds = seconds[starts[group_index] : ends[group_index]]
        unique_times[out_index] = np.unique(group_seconds).size
        time_span[out_index] = np.nanmax(group_seconds) - np.nanmin(group_seconds)

    observed_unique_ids = int(np.unique(group_ids).size)
    missing_ids = expected_samples - observed_unique_ids
    result = {
        "rows": n_rows,
        "expected_samples": expected_samples,
        "observed_sample_groups": int(group_ids.size),
        "observed_unique_sample_ids": observed_unique_ids,
        "missing_sample_ids": int(missing_ids),
        "duplicate_noncontiguous_groups": int(group_ids.size - observed_unique_ids),
        "sample_id_min": int(sample_ids.min()),
        "sample_id_max": int(sample_ids.max()),
        "sample_ids_nondecreasing": bool(np.all(sample_ids[1:] >= sample_ids[:-1])),
        "mean_rows_per_observed_sample": float(np.mean(counts)),
        "rate_at_999_row_cap": float(np.mean(counts == 999)),
        "rows_per_observed_sample": quantiles(counts),
        "seconds_before_predict": {
            "min": json_value(np.nanmin(seconds)),
            "max": json_value(np.nanmax(seconds)),
            "nan_rate": float(np.mean(~np.isfinite(seconds))),
            "within_sample_nondecreasing_rate": nondecreasing,
            "within_sample_nonincreasing_rate": nonincreasing,
            "within_sample_repeated_rate": repeated,
            "first_value_quantiles": quantiles(seconds[starts]),
            "last_value_quantiles": quantiles(seconds[ends - 1]),
            "unique_times_per_inspected_sample": quantiles(unique_times),
            "span_per_inspected_sample": quantiles(time_span),
            "inspected_samples": int(inspected_group_indexes.size),
        },
    }

    del table, sample_ids, seconds, change_positions, starts, ends, group_ids, counts
    del same_group, within_diffs, unique_times, time_span
    gc.collect()
    return result


def label_profile(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = pd.read_feather(path)
    target = labels["target"].to_numpy(dtype=np.float64)
    by_month = (
        labels.groupby("month", sort=True)["target"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    by_month["zero_rate"] = labels.groupby("month", sort=True)["target"].apply(
        lambda series: float((series == 0).mean())
    ).to_numpy()
    by_month["positive_rate"] = labels.groupby("month", sort=True)["target"].apply(
        lambda series: float((series > 0).mean())
    ).to_numpy()

    lag_correlations: dict[str, float] = {}
    absolute_lag_correlations: dict[str, float] = {}
    for lag in [1, 2, 5, 10, 50, 100, 500, 1000]:
        lag_correlations[str(lag)] = float(np.corrcoef(target[:-lag], target[lag:])[0, 1])
        absolute_lag_correlations[str(lag)] = float(
            np.corrcoef(np.abs(target[:-lag]), np.abs(target[lag:]))[0, 1]
        )

    profile = {
        "rows": int(len(labels)),
        "months": int(labels["month"].nunique()),
        "month_min": int(labels["month"].min()),
        "month_max": int(labels["month"].max()),
        "sample_id_min": int(labels["sample_id"].min()),
        "sample_id_max": int(labels["sample_id"].max()),
        "sample_ids_contiguous": bool(
            labels["sample_id"].is_monotonic_increasing
            and labels["sample_id"].iloc[0] == 0
            and labels["sample_id"].iloc[-1] == len(labels) - 1
        ),
        "duplicate_sample_ids": int(labels["sample_id"].duplicated().sum()),
        "target": {
            "mean": float(np.mean(target)),
            "std": float(np.std(target, ddof=1)),
            "skew": float(pd.Series(target).skew()),
            "excess_kurtosis": float(pd.Series(target).kurt()),
            "zero_rate": float(np.mean(target == 0)),
            "positive_rate": float(np.mean(target > 0)),
            "negative_rate": float(np.mean(target < 0)),
            "quantiles": quantiles(target),
            "lag_correlations": lag_correlations,
            "absolute_lag_correlations": absolute_lag_correlations,
        },
    }
    return profile, by_month


def submission_profile(path: Path) -> dict[str, Any]:
    submission = pd.read_csv(path)
    ids = submission["sample_id"].to_numpy()
    return {
        "rows": int(len(submission)),
        "sample_id_min": int(ids.min()),
        "sample_id_max": int(ids.max()),
        "sample_ids_contiguous": bool(np.all(ids == np.arange(len(ids)))),
        "duplicate_sample_ids": int(submission["sample_id"].duplicated().sum()),
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "competition_profile.json",
        help="Path for the compact JSON profile.",
    )
    args = parser.parse_args()

    label_path = DATA_ROOT / "train" / "label.feather"
    submission_path = DATA_ROOT / "submission.csv"
    label_summary, month_table = label_profile(label_path)
    submission_summary = submission_profile(submission_path)

    result: dict[str, Any] = {
        "data_root": DATA_ROOT.name,
        "label": label_summary,
        "submission": submission_summary,
        "files": {},
    }
    expected = {"train": label_summary["rows"], "test": submission_summary["rows"]}
    for split in ["train", "test"]:
        for source in ["market", "order", "transaction"]:
            path = DATA_ROOT / split / f"{source}.feather"
            result["files"][f"{split}/{source}"] = {
                "bytes": path.stat().st_size,
                "arrow": arrow_schema(path),
                "structure": row_structure(path, int(expected[split])),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=json_value)
    month_path = args.output.with_name("target_by_month.csv")
    month_table.to_csv(month_path, index=False)
    print(f"Wrote {args.output.relative_to(WORKSPACE)}", flush=True)
    print(f"Wrote {month_path.relative_to(WORKSPACE)}", flush=True)


if __name__ == "__main__":
    run()
