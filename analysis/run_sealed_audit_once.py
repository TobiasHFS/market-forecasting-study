"""Run the single, already-frozen audit on months 59--70.

This script is intentionally fail-closed.  It verifies the immutable pipeline
manifest, fits and predicts without reading the sealed validation target, then
atomically claims a permanent audit ledger immediately before loading that
target.  Once the ledger exists, a second audit is refused even if the first
attempt failed after the claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from lightgbm import LGBMRegressor

from feature_families import FEATURE_SET_FAMILIES, materialize_feature_set
from modeling import (
    RobustPreprocessor,
    cosine_score,
    monthly_diagnostics,
    summarize_monthly_diagnostics,
)
from pipeline_config import (
    DATA_ROOT,
    DIAGNOSTIC_ROOT,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRAIN_SAMPLES,
    ensure_artifact_directories,
)
from run_gbdt_experiment import SPECS


FEATURE_SET = "multiscale_mechanics_scale"
MODEL_SPEC_NAME = "slow"
TRAIN_MONTHS = (0, 58)
VALIDATION_MONTHS = (59, 70)

FROZEN_PATH = DIAGNOSTIC_ROOT / "frozen_pipeline.json"
LEDGER_PATH = DIAGNOSTIC_ROOT / "sealed_audit_ledger.json"
PREDICTION_PATH = DIAGNOSTIC_ROOT / "sealed_audit_predictions.npz"
MONTHLY_PATH = DIAGNOSTIC_ROOT / "sealed_audit_monthly.csv"
SUMMARY_PATH = DIAGNOSTIC_ROOT / "sealed_audit_summary.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(
            f"frozen pipeline mismatch for {name}: "
            f"expected {expected!r}, found {actual!r}"
        )


def _load_and_validate_frozen_manifest() -> dict[str, Any]:
    if not FROZEN_PATH.is_file():
        raise FileNotFoundError(
            f"{FROZEN_PATH} is required; freeze the pipeline before any sealed audit"
        )
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict):
        raise TypeError("frozen pipeline manifest must contain a JSON object")

    recorded_pipeline_hash = frozen.get("pipeline_sha256")
    hash_payload = dict(frozen)
    hash_payload.pop("pipeline_sha256", None)
    _require_equal(
        "pipeline_sha256",
        recorded_pipeline_hash,
        _sha256_json(hash_payload),
    )
    _require_equal("status", frozen.get("status"), "frozen_before_sealed_audit")
    _require_equal("feature_set", frozen.get("feature_set"), FEATURE_SET)
    _require_equal(
        "feature_families",
        frozen.get("feature_families"),
        list(FEATURE_SET_FAMILIES[FEATURE_SET]),
    )
    _require_equal("model_spec_name", frozen.get("model_spec_name"), MODEL_SPEC_NAME)
    _require_equal("model_spec", frozen.get("model_spec"), asdict(SPECS[MODEL_SPEC_NAME]))
    _require_equal(
        "preprocessor",
        frozen.get("preprocessor"),
        {
            "clip": 8.0,
            "add_missing_indicators": True,
            "fit_scope": "training rows only",
        },
    )
    _require_equal("objective", frozen.get("objective"), "regression_l2")
    _require_equal("history_weighting", frozen.get("history_weighting"), "equal")
    _require_equal("final_component_rule", frozen.get("final_component_rule"), "pure_gbdt")
    _require_equal("ridge_weight", frozen.get("ridge_weight"), 0.0)
    _require_equal("sealed_audit_months", frozen.get("sealed_audit_months"), [59, 70])
    _require_equal("random_seed", frozen.get("random_seed"), RANDOM_SEED)

    source_hashes = frozen.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("frozen manifest has no source hashes")
    project_root = PROJECT_ROOT.resolve()
    for relative, recorded_hash in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(recorded_hash, str):
            raise TypeError("frozen source hash entries must be string pairs")
        candidate = (project_root / relative).resolve()
        if candidate != project_root and project_root not in candidate.parents:
            raise RuntimeError(f"frozen source path escapes project root: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {relative}")
        _require_equal(f"source_sha256[{relative}]", _sha256_file(candidate), recorded_hash)
    return frozen


def _validate_materialized_schema(
    frozen: dict[str, Any], names: list[str], kinds: list[str]
) -> None:
    _require_equal("feature_count", len(names), frozen.get("feature_count"))
    _require_equal("feature_names", names, frozen.get("feature_names"))
    _require_equal("feature_kinds", kinds, frozen.get("feature_kinds"))
    _require_equal(
        "feature_names_sha256",
        _sha256_json(names),
        frozen.get("feature_names_sha256"),
    )
    _require_equal(
        "feature_kinds_sha256",
        _sha256_json(kinds),
        frozen.get("feature_kinds_sha256"),
    )


def _load_label_metadata() -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather", columns=["sample_id", "month"]
    )
    sample_id = table["sample_id"].to_numpy()
    months = table["month"].to_numpy().astype(np.int16, copy=False)
    expected_ids = np.arange(TRAIN_SAMPLES, dtype=sample_id.dtype)
    if sample_id.shape != (TRAIN_SAMPLES,) or not np.array_equal(sample_id, expected_ids):
        raise ValueError("label sample_id is not exact row alignment")
    if months.shape != (TRAIN_SAMPLES,) or np.any(months[1:] < months[:-1]):
        raise ValueError("label months are missing or nonchronological")
    if int(months[0]) != 0 or int(months[-1]) != 70:
        raise ValueError("unexpected train month coverage")
    return sample_id, months


def _month_slice(months: np.ndarray, start: int, end: int) -> slice:
    left = int(np.searchsorted(months, start, side="left"))
    right = int(np.searchsorted(months, end, side="right"))
    if left >= right or int(months[left]) != start or int(months[right - 1]) != end:
        raise ValueError(f"incomplete month slice {start}..{end}")
    return slice(left, right)


def _load_target_slice(rows: slice) -> np.ndarray:
    """Convert only the requested memory-mapped Arrow rows to NumPy.

    The label Feather file contains one record batch, so a high-level Feather
    read would eagerly materialize the entire target column.  Memory mapping
    and slicing the Arrow array before conversion keeps the validation values
    out of the train-time NumPy process state.
    """

    start = 0 if rows.start is None else int(rows.start)
    stop = TRAIN_SAMPLES if rows.stop is None else int(rows.stop)
    if start < 0 or stop > TRAIN_SAMPLES or start >= stop:
        raise ValueError("invalid target row slice")
    label_path = DATA_ROOT / "train" / "label.feather"
    pieces: list[np.ndarray] = []
    cursor = 0
    with pa.memory_map(str(label_path), "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        target_index = reader.schema.get_field_index("target")
        if target_index < 0:
            raise ValueError("label file has no target column")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            batch_start = cursor
            batch_stop = cursor + batch.num_rows
            overlap_start = max(start, batch_start)
            overlap_stop = min(stop, batch_stop)
            if overlap_start < overlap_stop:
                arrow_slice = batch.column(target_index).slice(
                    overlap_start - batch_start, overlap_stop - overlap_start
                )
                # copy=True detaches the selected values before the mmap closes.
                pieces.append(
                    arrow_slice.to_numpy().astype(np.float64, copy=True)
                )
            cursor = batch_stop
            if cursor >= stop:
                break
    if not pieces:
        raise ValueError("requested target slice is outside the label record batches")
    target = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    if target.shape != (stop - start,) or not np.all(np.isfinite(target)):
        raise ValueError("target slice is incomplete or non-finite")
    return target


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=path.parent,
            prefix=path.stem + ".",
            suffix=".npz",
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _claim_ledger(pipeline_hash: str, prediction_rows: int) -> dict[str, Any]:
    """Atomically spend the audit before any sealed target value is loaded."""

    claim = {
        "status": "claimed_before_sealed_target_access",
        "pipeline_sha256": pipeline_hash,
        "claimed_at_utc": _utc_now(),
        "process_id": os.getpid(),
        "feature_set": FEATURE_SET,
        "model_spec_name": MODEL_SPEC_NAME,
        "validation_months": list(VALIDATION_MONTHS),
        "prediction_rows_ready": int(prediction_rows),
    }
    try:
        with LEDGER_PATH.open("x", encoding="utf-8", newline="") as handle:
            json.dump(claim, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(
            f"{LEDGER_PATH} already exists; the sealed audit has already been spent"
        ) from exc
    return claim


def _refuse_existing_audit_artifacts() -> None:
    if LEDGER_PATH.exists():
        raise RuntimeError(
            f"{LEDGER_PATH} already exists; the sealed audit cannot be rerun"
        )
    orphaned = [
        str(path)
        for path in (PREDICTION_PATH, MONTHLY_PATH, SUMMARY_PATH)
        if path.exists()
    ]
    if orphaned:
        raise RuntimeError(
            "sealed output exists without a ledger; refusing overwrite: "
            + ", ".join(orphaned)
        )


def run_once(*, confirm_frozen: bool = False) -> dict[str, Any]:
    if not confirm_frozen:
        raise RuntimeError(
            "sealed audit remains closed; pass --confirm-frozen for the one final audit"
        )
    ensure_artifact_directories()
    _refuse_existing_audit_artifacts()
    frozen = _load_and_validate_frozen_manifest()
    pipeline_hash = str(frozen["pipeline_sha256"])
    started = time.perf_counter()

    sample_id, months = _load_label_metadata()
    train_rows = _month_slice(months, *TRAIN_MONTHS)
    validation_rows = _month_slice(months, *VALIDATION_MONTHS)
    if train_rows.start != 0 or train_rows.stop != validation_rows.start:
        raise ValueError("train and sealed validation rows are not contiguous")

    materialized = materialize_feature_set("train", FEATURE_SET)
    _validate_materialized_schema(frozen, materialized.names, materialized.kinds)
    X = materialized.matrix
    if X.shape != (TRAIN_SAMPLES, int(frozen["feature_count"])):
        raise ValueError(f"unexpected materialized feature shape: {X.shape}")

    # This function slices before NumPy conversion.  No validation target is
    # selected here; only months 0--58 are available to model.fit.
    y_train = _load_target_slice(train_rows)
    preprocessor = RobustPreprocessor(
        feature_names=materialized.names,
        feature_kinds=materialized.kinds,
        clip=8.0,
        add_missing_indicators=True,
        output_dtype=np.float32,
    )
    transform_started = time.perf_counter()
    preprocessor.fit(X[train_rows])
    X_train = preprocessor.transform(X[train_rows])
    X_validation = preprocessor.transform(X[validation_rows])
    transform_seconds = time.perf_counter() - transform_started
    transformed_feature_count = int(X_train.shape[1])
    del X, materialized
    gc.collect()

    params = dict(frozen["model_spec"])
    params.update(
        {
            "force_col_wise": True,
            "deterministic": True,
            "bagging_seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
        }
    )
    model = LGBMRegressor(**params)
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    prediction = np.asarray(model.predict(X_validation), dtype=np.float64).reshape(-1)
    fit_predict_seconds = time.perf_counter() - fit_started
    expected_validation_rows = int(validation_rows.stop - validation_rows.start)
    if prediction.shape != (expected_validation_rows,):
        raise ValueError("sealed prediction has the wrong number of rows")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("sealed prediction contains NaN or infinity")
    prediction_rms = float(np.sqrt(np.mean(np.square(prediction, dtype=np.float64))))
    if prediction_rms <= 1e-15:
        raise ValueError("sealed prediction has zero norm")
    del X_train, X_validation, y_train, model, preprocessor
    gc.collect()

    # Nothing may be inserted between this exclusive claim and the first
    # selection of the sealed target.  The claim permanently prevents a
    # second candidate from being evaluated on months 59--70.
    claim = _claim_ledger(pipeline_hash, prediction.size)
    try:
        y_validation = _load_target_slice(validation_rows)
        # Exactly one pooled, competition-equivalent composite score is made.
        pooled_cosine = cosine_score(y_validation, prediction)
        validation_month_vector = months[validation_rows]
        diagnostics = monthly_diagnostics(
            y_validation, prediction, validation_month_vector
        )
        robustness_series = summarize_monthly_diagnostics(
            diagnostics, pooled_cosine
        )
        robustness = {
            str(key): float(value) for key, value in robustness_series.items()
        }
        validation_sample_id = sample_id[validation_rows].astype(np.int32, copy=False)
        validation_row_index = np.arange(
            validation_rows.start, validation_rows.stop, dtype=np.int64
        )

        _atomic_write_npz(
            PREDICTION_PATH,
            row_index=validation_row_index,
            sample_id=validation_sample_id,
            month=validation_month_vector.astype(np.int16, copy=False),
            prediction=prediction,
            pipeline_sha256=np.asarray(pipeline_hash),
        )
        _atomic_write_frame(MONTHLY_PATH, diagnostics)

        summary: dict[str, Any] = {
            "status": "complete_single_frozen_pipeline_audit",
            "pipeline_sha256": pipeline_hash,
            "feature_set": FEATURE_SET,
            "raw_feature_count": int(frozen["feature_count"]),
            "transformed_feature_count": transformed_feature_count,
            "preprocessor": frozen["preprocessor"],
            "model_spec_name": MODEL_SPEC_NAME,
            "model_spec": frozen["model_spec"],
            "history_weighting": "equal",
            "component_rule": "pure_gbdt",
            "train_months": list(TRAIN_MONTHS),
            "validation_months": list(VALIDATION_MONTHS),
            "train_rows": int(train_rows.stop - train_rows.start),
            "validation_rows": expected_validation_rows,
            "pooled_cosine": float(pooled_cosine),
            "monthly_robustness": robustness,
            "prediction_mean": float(np.mean(prediction)),
            "prediction_rms": prediction_rms,
            "prediction_min": float(np.min(prediction)),
            "prediction_max": float(np.max(prediction)),
            "transform_seconds": float(transform_seconds),
            "fit_predict_seconds": float(fit_predict_seconds),
            "total_seconds": float(time.perf_counter() - started),
            "completed_at_utc": _utc_now(),
            "outputs": {
                "predictions": str(PREDICTION_PATH),
                "monthly_diagnostics": str(MONTHLY_PATH),
                "summary": str(SUMMARY_PATH),
                "ledger": str(LEDGER_PATH),
            },
        }
        _atomic_write_json(SUMMARY_PATH, summary)

        complete_ledger = {
            **claim,
            "status": "complete",
            "completed_at_utc": summary["completed_at_utc"],
            "pooled_cosine": float(pooled_cosine),
            "outputs_sha256": {
                "predictions": _sha256_file(PREDICTION_PATH),
                "monthly_diagnostics": _sha256_file(MONTHLY_PATH),
                "summary": _sha256_file(SUMMARY_PATH),
            },
        }
        _atomic_write_json(LEDGER_PATH, complete_ledger)
        return summary
    except Exception as exc:
        failed_ledger = {
            **claim,
            "status": "failed_after_claim_no_retry_permitted",
            "failed_at_utc": _utc_now(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        _atomic_write_json(LEDGER_PATH, failed_ledger)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="confirm this is the sole pre-frozen audit on months 59--70",
    )
    args = parser.parse_args()
    summary = run_once(confirm_frozen=args.confirm_frozen)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
