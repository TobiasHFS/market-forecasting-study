"""Independently audit the final competition CSV and prediction artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "ms-capital-real-financial-market-forecasting"
SUBMISSION_PATH = PROJECT_ROOT / "artifacts" / "submissions" / "submission_final.csv"
PREDICTION_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_test_predictions.npz"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_training_manifest.json"
FROZEN_PATH = PROJECT_ROOT / "artifacts" / "diagnostics" / "frozen_pipeline.json"
LEDGER_PATH = PROJECT_ROOT / "artifacts" / "diagnostics" / "sealed_audit_ledger.json"
AUDIT_PATH = PROJECT_ROOT / "artifacts" / "diagnostics" / "submission_audit.json"
EXPECTED_ROWS = 647_896


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _rms(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("RMS requires a nonempty finite vector")
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("cosine inputs have incompatible shapes")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("cosine inputs must be finite")
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator <= 0.0:
        raise ValueError("cosine inputs must have nonzero norm")
    return float(np.dot(a, b) / denominator)


def _sample_submission_path() -> Path:
    for candidate in (
        DATA_ROOT / "sample_submission.csv",
        DATA_ROOT / "submission.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("organizer sample-submission file is absent")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def audit_submission(*, overwrite: bool = False) -> dict[str, Any]:
    if AUDIT_PATH.exists() and not overwrite:
        raise FileExistsError(f"{AUDIT_PATH} exists; pass --overwrite to replace it")
    required_paths = (
        SUBMISSION_PATH,
        PREDICTION_PATH,
        MANIFEST_PATH,
        FROZEN_PATH,
        LEDGER_PATH,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    sample_path = _sample_submission_path()
    first_line = SUBMISSION_PATH.open("r", encoding="utf-8-sig").readline().rstrip("\r\n")
    # round_trip asks pandas' C parser to recover the original float whenever
    # possible, making this a stringent check of the actual serialized CSV.
    submitted = pd.read_csv(SUBMISSION_PATH, float_precision="round_trip")
    template = pd.read_csv(sample_path)
    frozen = _read_json(FROZEN_PATH)
    ledger = _read_json(LEDGER_PATH)
    manifest = _read_json(MANIFEST_PATH)
    with np.load(PREDICTION_PATH, allow_pickle=False) as saved:
        saved_files = set(saved.files)
        if not {"sample_id", "prediction", "raw_prediction"}.issubset(saved_files):
            raise ValueError(
                "saved prediction NPZ must contain sample_id, prediction, and raw_prediction"
            )
        saved_id = np.asarray(saved["sample_id"], dtype=np.int64)
        saved_prediction = np.asarray(saved["prediction"], dtype=np.float64)
        raw_prediction = np.asarray(saved["raw_prediction"], dtype=np.float64)

    checks: dict[str, bool] = {}
    checks["header_exact"] = first_line == "sample_id,prediction"
    checks["columns_exact"] = submitted.columns.tolist() == ["sample_id", "prediction"]
    checks["template_columns_exact"] = template.columns.tolist() == [
        "sample_id",
        "prediction",
    ]
    checks["row_count_exact"] = len(submitted) == EXPECTED_ROWS
    checks["template_row_count_exact"] = len(template) == EXPECTED_ROWS
    checks["sample_id_integer"] = pd.api.types.is_integer_dtype(submitted["sample_id"])

    submitted_id = submitted["sample_id"].to_numpy(dtype=np.int64, copy=False)
    template_id = template["sample_id"].to_numpy(dtype=np.int64, copy=False)
    prediction = submitted["prediction"].to_numpy(dtype=np.float64, copy=False)
    checks["sample_id_matches_template_order"] = np.array_equal(submitted_id, template_id)
    checks["sample_id_unique"] = len(np.unique(submitted_id)) == EXPECTED_ROWS
    checks["saved_sample_id_matches_csv"] = np.array_equal(saved_id, submitted_id)
    checks["prediction_shape_exact"] = prediction.shape == (EXPECTED_ROWS,)
    checks["saved_prediction_shape_exact"] = saved_prediction.shape == (EXPECTED_ROWS,)
    checks["raw_prediction_shape_exact"] = raw_prediction.shape == (EXPECTED_ROWS,)
    checks["prediction_all_finite"] = bool(np.all(np.isfinite(prediction)))
    checks["saved_prediction_all_finite"] = bool(np.all(np.isfinite(saved_prediction)))
    checks["raw_prediction_all_finite"] = bool(np.all(np.isfinite(raw_prediction)))

    prediction_rms = _rms(prediction) if checks["prediction_all_finite"] else float("nan")
    checks["prediction_nonzero_norm"] = bool(np.isfinite(prediction_rms) and prediction_rms > 0.0)
    checks["prediction_rms_is_one"] = bool(abs(prediction_rms - 1.0) <= 1e-9)
    saved_cosine = (
        _cosine(prediction, saved_prediction)
        if checks["prediction_all_finite"] and checks["saved_prediction_all_finite"]
        else float("nan")
    )
    max_abs_roundtrip_error = (
        float(np.max(np.abs(prediction - saved_prediction)))
        if prediction.shape == saved_prediction.shape
        else float("inf")
    )
    roundtrip_tolerance = 1e-12 * max(
        1.0,
        float(np.max(np.abs(saved_prediction)))
        if saved_prediction.size and np.all(np.isfinite(saved_prediction))
        else 1.0,
    )
    checks["csv_roundtrip_cosine"] = bool(
        np.isfinite(saved_cosine) and saved_cosine >= 1.0 - 1e-12
    )
    checks["csv_roundtrip_absolute_error"] = bool(
        max_abs_roundtrip_error <= roundtrip_tolerance
    )

    pipeline_hash = frozen.get("pipeline_sha256")
    checks["ledger_complete"] = ledger.get("status") == "complete"
    checks["ledger_pipeline_matches_frozen"] = ledger.get("pipeline_sha256") == pipeline_hash
    checks["manifest_complete"] = manifest.get("status") == "complete"
    checks["manifest_pipeline_matches_frozen"] = manifest.get("pipeline_sha256") == pipeline_hash
    checks["manifest_submission_hash_matches"] = (
        manifest.get("outputs", {}).get("submission_sha256")
        == _sha256_file(SUBMISSION_PATH)
    )
    checks["manifest_prediction_hash_matches"] = (
        manifest.get("outputs", {}).get("predictions_sha256")
        == _sha256_file(PREDICTION_PATH)
    )

    quantile_levels = np.asarray(
        [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
    )
    quantiles = (
        np.quantile(prediction, quantile_levels)
        if checks["prediction_all_finite"]
        else np.full(quantile_levels.shape, np.nan)
    )
    ready = all(checks.values())
    payload: dict[str, Any] = {
        "status": "ready" if ready else "failed",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "submission_path": str(SUBMISSION_PATH.resolve()),
        "submission_sha256": _sha256_file(SUBMISSION_PATH),
        "submission_bytes": SUBMISSION_PATH.stat().st_size,
        "prediction_artifact_path": str(PREDICTION_PATH.resolve()),
        "prediction_artifact_sha256": _sha256_file(PREDICTION_PATH),
        "sample_submission_path": str(sample_path.resolve()),
        "sample_submission_sha256": _sha256_file(sample_path),
        "pipeline_sha256": pipeline_hash,
        "rows": int(len(submitted)),
        "unique_sample_ids": int(len(np.unique(submitted_id))),
        "zero_prediction_count": int(np.count_nonzero(prediction == 0.0)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction, dtype=np.float64)),
        "prediction_rms": prediction_rms,
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "prediction_quantiles": {
            f"{level:g}": float(value)
            for level, value in zip(quantile_levels, quantiles, strict=True)
        },
        "saved_prediction_cosine_after_csv_roundtrip": saved_cosine,
        "saved_prediction_max_abs_csv_roundtrip_error": max_abs_roundtrip_error,
        "saved_prediction_roundtrip_tolerance": roundtrip_tolerance,
    }
    _atomic_json(AUDIT_PATH, payload)
    if not ready:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"submission audit failed checks: {failed}")
    print(
        f"READY: {len(submitted):,} rows, RMS={prediction_rms:.12f}, "
        f"sha256={payload['submission_sha256']}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_submission(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
