"""Train the frozen v2 model and publish byte-identical submission mirrors.

This is the only final-training entry point for the v2 candidate.  The model,
feature family, preprocessing, and signed-power exponent are constants selected
on Dev1--Dev2 and confirmed on Dev3.  The completed SealedAudit artifacts are
verified and recorded for provenance, but no value from that audit is used to
change the frozen pipeline.

Publication intentionally requires ``--overwrite`` because two destinations
replace the archived v1 submission.  Every destination is written to a sibling
temporary file and replaced with ``os.replace`` only after all temporary CSVs
have been serialized successfully.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import joblib  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import pyarrow.feather as feather  # noqa: E402
import sklearn  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

from modeling import RobustPreprocessor, cosine_score, rms, rms_normalize  # noqa: E402
from pipeline_config import (  # noqa: E402
    DATA_ROOT,
    RANDOM_SEED,
    TEST_SAMPLES,
    TRAIN_MONTH_MAX,
    TRAIN_MONTH_MIN,
    TRAIN_SAMPLES,
)
from v2.run_sequence_experiment import SPECS  # noqa: E402
from v2.v2_features import FEATURE_ROOT, materialize_v2_feature_set  # noqa: E402


FEATURE_SET = "base_plus_sequence_all"
MODEL_SPEC_NAME = "capacity"
PREDICTION_POWER = 1.2
PREPROCESSOR_CLIP = 8.0
ADD_MISSING_INDICATORS = True
EXPECTED_RAW_FEATURES = 754
EXPECTED_TRANSFORMED_FEATURES = 1_326

V2_ROOT = PROJECT_ROOT / "artifacts" / "v2"
V2_MODEL_ROOT = V2_ROOT / "models"
V2_SUBMISSION_ROOT = V2_ROOT / "submissions"
V2_DIAGNOSTIC_ROOT = V2_ROOT / "diagnostics"
EXPERIMENT_ROOT = V2_ROOT / "experiments"

CANONICAL_SUBMISSION_PATH = V2_SUBMISSION_ROOT / "submission_final.csv"
ROOT_ARTIFACT_SUBMISSION_PATH = (
    PROJECT_ROOT / "artifacts" / "submissions" / "submission_final.csv"
)
WORKSPACE_SUBMISSION_PATH = PROJECT_ROOT / "submission_final.csv"
PREDICTION_PATH = V2_MODEL_ROOT / "final_test_predictions.npz"
MODEL_PATH = V2_MODEL_ROOT / "final_model.joblib"
PREPROCESSOR_PATH = V2_MODEL_ROOT / "final_preprocessor.joblib"
MANIFEST_PATH = V2_MODEL_ROOT / "final_training_manifest.json"

SCREEN_SUMMARY_PATH = (
    EXPERIMENT_ROOT
    / "sequence_base_plus_sequence_all_capacity_Dev1-Dev2_summary.json"
)
CONFIRMATION_SUMMARY_PATH = (
    EXPERIMENT_ROOT
    / "sequence_base_plus_sequence_all_capacity_Dev3_summary.json"
)
SEALED_STEM = "sequence_base_plus_sequence_all_capacity_SealedAudit"
SEALED_ARTIFACT_PATHS = {
    "summary": EXPERIMENT_ROOT / f"{SEALED_STEM}_summary.json",
    "predictions": EXPERIMENT_ROOT / f"{SEALED_STEM}_oof.npz",
    "monthly": EXPERIMENT_ROOT / f"{SEALED_STEM}_monthly.csv",
    "feature_importance": EXPERIMENT_ROOT / f"{SEALED_STEM}_feature_importance.csv",
}

SOURCE_PATHS = (
    ANALYSIS_ROOT / "pipeline_config.py",
    ANALYSIS_ROOT / "modeling.py",
    ANALYSIS_ROOT / "market_features.py",
    ANALYSIS_ROOT / "flow_features.py",
    ANALYSIS_ROOT / "feature_families.py",
    ANALYSIS_ROOT / "v2" / "sequence_features.py",
    ANALYSIS_ROOT / "v2" / "v2_features.py",
    ANALYSIS_ROOT / "v2" / "run_sequence_experiment.py",
    Path(__file__).resolve(),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _signed_power(prediction: np.ndarray, exponent: float) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("prediction exponent must be finite and positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("signed-power input contains NaN or infinity")
    return np.sign(values) * np.power(np.abs(values), exponent)


def _power_record(summary: dict[str, Any], exponent: float) -> dict[str, Any]:
    rows = summary.get("power_calibration")
    if not isinstance(rows, list):
        raise TypeError("experiment summary has no power_calibration list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and abs(float(row.get("exponent", float("nan"))) - exponent) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one q={exponent:g} calibration record")
    return matches[0]


def _verify_summary_contract(
    summary: dict[str, Any], expected_folds: Iterable[str]
) -> None:
    expected_spec = asdict(SPECS[MODEL_SPEC_NAME])
    if summary.get("feature_set") != FEATURE_SET:
        raise ValueError("validation artifact has the wrong feature set")
    if summary.get("spec_name") != MODEL_SPEC_NAME:
        raise ValueError("validation artifact has the wrong model specification name")
    if summary.get("spec") != expected_spec:
        raise ValueError("validation artifact model specification differs from frozen code")
    if summary.get("random_seed") != RANDOM_SEED:
        raise ValueError("validation artifact has the wrong random seed")
    if summary.get("raw_feature_count") != EXPECTED_RAW_FEATURES:
        raise ValueError("validation artifact has the wrong raw feature count")
    folds = summary.get("folds")
    if not isinstance(folds, list):
        raise TypeError("validation artifact folds must be a list")
    observed = [row.get("fold") for row in folds if isinstance(row, dict)]
    if observed != list(expected_folds):
        raise ValueError(f"unexpected validation folds: {observed}")
    for row in folds:
        if row.get("raw_features") != EXPECTED_RAW_FEATURES:
            raise ValueError("validation fold raw feature count is inconsistent")
        if row.get("transformed_features") != EXPECTED_TRANSFORMED_FEATURES:
            raise ValueError("validation fold transformed feature count is inconsistent")
        if not np.isfinite(float(row.get("fold_cosine", float("nan")))):
            raise ValueError("validation fold cosine is not finite")


def _load_training_labels() -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "train" / "label.feather"
    table = feather.read_table(path, columns=["sample_id", "month", "target"])
    sample_id = table["sample_id"].to_numpy(zero_copy_only=False)
    months = table["month"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    target = table["target"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    del table
    expected_id = np.arange(TRAIN_SAMPLES, dtype=sample_id.dtype)
    if not np.array_equal(sample_id, expected_id):
        raise ValueError("training label sample_id does not equal feature-row order")
    if months.shape != (TRAIN_SAMPLES,) or target.shape != (TRAIN_SAMPLES,):
        raise ValueError("training labels have the wrong shape")
    if np.any(months[1:] < months[:-1]):
        raise ValueError("training months are not chronological")
    if not np.array_equal(
        np.unique(months), np.arange(TRAIN_MONTH_MIN, TRAIN_MONTH_MAX + 1)
    ):
        raise ValueError("training month coverage is not exactly 0..70")
    if not np.all(np.isfinite(target)) or rms(target) <= 0.0:
        raise ValueError("training target is non-finite or zero norm")
    return months, target


def verify_validation_provenance(
    months: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    """Verify development evidence and the non-selective sealed audit."""

    screen = _read_json_object(SCREEN_SUMMARY_PATH)
    confirmation = _read_json_object(CONFIRMATION_SUMMARY_PATH)
    sealed = _read_json_object(SEALED_ARTIFACT_PATHS["summary"])
    _verify_summary_contract(screen, ("Dev1", "Dev2"))
    _verify_summary_contract(confirmation, ("Dev3",))
    _verify_summary_contract(sealed, ("SealedAudit",))

    screen_power = _power_record(screen, PREDICTION_POWER)
    confirmation_power = _power_record(confirmation, PREDICTION_POWER)
    sealed_power = _power_record(sealed, PREDICTION_POWER)
    screen_rows = screen["power_calibration"]
    best_screen = max(float(row["pooled_cosine"]) for row in screen_rows)
    if abs(float(screen_power["pooled_cosine"]) - best_screen) > 1e-15:
        raise ValueError("q=1.2 is no longer the Dev1--Dev2-selected exponent")

    for path in SEALED_ARTIFACT_PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(SEALED_ARTIFACT_PATHS["predictions"], allow_pickle=False) as saved:
        required = {"row_indices", "months", "target", "prediction"}
        if not required.issubset(saved.files):
            raise ValueError("sealed OOF artifact is missing required arrays")
        row_indices = np.asarray(saved["row_indices"], dtype=np.int64)
        saved_months = np.asarray(saved["months"], dtype=np.int16)
        saved_target = np.asarray(saved["target"], dtype=np.float64)
        saved_prediction = np.asarray(saved["prediction"], dtype=np.float64)
    expected_indices = np.flatnonzero((months >= 59) & (months <= 70)).astype(np.int64)
    expected_shape = expected_indices.shape
    if not all(
        array.shape == expected_shape
        for array in (row_indices, saved_months, saved_target, saved_prediction)
    ):
        raise ValueError("sealed OOF arrays have inconsistent shapes")
    if not np.array_equal(row_indices, expected_indices):
        raise ValueError("sealed OOF row indices are not exactly months 59..70")
    if not np.array_equal(saved_months, months[row_indices]):
        raise ValueError("sealed OOF months do not match labels")
    if not np.array_equal(saved_target, target[row_indices]):
        raise ValueError("sealed OOF targets do not match label.feather")
    if not np.all(np.isfinite(saved_prediction)) or np.ptp(saved_prediction) <= 0.0:
        raise ValueError("sealed OOF predictions are non-finite or constant")
    raw_score = cosine_score(saved_target, saved_prediction)
    powered_score = cosine_score(
        saved_target, _signed_power(saved_prediction, PREDICTION_POWER)
    )
    reported_raw = float(sealed["summary"]["pooled_cosine"])
    reported_powered = float(sealed_power["pooled_cosine"])
    if abs(raw_score - reported_raw) > 1e-12:
        raise ValueError("sealed raw cosine does not reproduce its summary")
    if abs(powered_score - reported_powered) > 1e-12:
        raise ValueError("sealed q=1.2 cosine does not reproduce its summary")

    monthly = pd.read_csv(SEALED_ARTIFACT_PATHS["monthly"])
    required_monthly = {"month", "n", "cosine"}
    if not required_monthly.issubset(monthly.columns):
        raise ValueError("sealed monthly diagnostic has the wrong schema")
    if not np.array_equal(monthly["month"].to_numpy(), np.arange(59, 71)):
        raise ValueError("sealed monthly diagnostic does not cover exactly 59..70")
    if int(monthly["n"].sum()) != row_indices.size:
        raise ValueError("sealed monthly diagnostic row counts do not reconcile")
    importance = pd.read_csv(SEALED_ARTIFACT_PATHS["feature_importance"])
    if (
        importance.empty
        or not {"fold", "feature", "gain"}.issubset(importance.columns)
        or set(importance["fold"]) != {"SealedAudit"}
        or not np.all(np.isfinite(importance["gain"]))
        or not np.all(importance["gain"] > 0.0)
    ):
        raise ValueError("sealed feature-importance artifact is invalid")

    artifact_records = {
        "development_screen_summary": _file_record(SCREEN_SUMMARY_PATH),
        "development_confirmation_summary": _file_record(CONFIRMATION_SUMMARY_PATH),
        **{
            f"sealed_{name}": _file_record(path)
            for name, path in SEALED_ARTIFACT_PATHS.items()
        },
    }
    return {
        "selection_source": "Dev1-Dev2 only",
        "confirmation_source": "Dev3",
        "prediction_power": PREDICTION_POWER,
        "development_screen_powered_cosine": float(screen_power["pooled_cosine"]),
        "development_confirmation_powered_cosine": float(
            confirmation_power["pooled_cosine"]
        ),
        "sealed_used_for_selection": False,
        "sealed_raw_cosine": raw_score,
        "sealed_powered_cosine_recorded_only": powered_score,
        "sealed_months": [59, 70],
        "sealed_rows": int(row_indices.size),
        "artifacts": artifact_records,
    }


def _sample_submission_path() -> Path:
    for path in (DATA_ROOT / "sample_submission.csv", DATA_ROOT / "submission.csv"):
        if path.is_file():
            return path
    raise FileNotFoundError("organizer sample-submission file is absent")


def _load_template_ids(path: Path) -> np.ndarray:
    template = pd.read_csv(path)
    if template.columns.tolist() != ["sample_id", "prediction"]:
        raise ValueError(f"unexpected template columns: {template.columns.tolist()}")
    if len(template) != TEST_SAMPLES:
        raise ValueError(f"template has {len(template):,} rows, expected {TEST_SAMPLES:,}")
    if not pd.api.types.is_integer_dtype(template["sample_id"]):
        raise TypeError("template sample_id must be integer")
    sample_id = template["sample_id"].to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(np.sort(sample_id), np.arange(TEST_SAMPLES, dtype=np.int64)):
        raise ValueError("template IDs are not exactly the complete 0..N-1 set")
    return sample_id


def _validate_feature_contract(
    matrix: np.ndarray, names: list[str], kinds: list[str], split: str
) -> None:
    expected_rows = TRAIN_SAMPLES if split == "train" else TEST_SAMPLES
    if matrix.shape != (expected_rows, EXPECTED_RAW_FEATURES):
        raise ValueError(f"{split} feature matrix has shape {matrix.shape}")
    if matrix.dtype != np.float32:
        raise TypeError(f"{split} feature matrix must be float32, got {matrix.dtype}")
    if len(names) != EXPECTED_RAW_FEATURES or len(kinds) != EXPECTED_RAW_FEATURES:
        raise ValueError(f"{split} feature schema has the wrong width")
    if len(set(names)) != len(names):
        raise ValueError(f"{split} feature names are not unique")
    forbidden = {"sample_id", "month", "target"}.intersection(names)
    if forbidden:
        raise ValueError(f"forbidden fields entered {split} features: {sorted(forbidden)}")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv_mirrors(frame: pd.DataFrame, paths: Iterable[Path]) -> None:
    """Serialize every mirror first, then atomically replace each destination."""

    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.csv")
            frame.to_csv(temporary, index=False, float_format="%.17g")
            temporary_paths.append((temporary, path))
        temporary_hashes = {_sha256_file(temporary) for temporary, _ in temporary_paths}
        if len(temporary_hashes) != 1:
            raise RuntimeError("submission mirrors did not serialize to identical bytes")
        for temporary, path in temporary_paths:
            os.replace(temporary, path)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def _all_output_paths() -> tuple[Path, ...]:
    return (
        CANONICAL_SUBMISSION_PATH,
        ROOT_ARTIFACT_SUBMISSION_PATH,
        WORKSPACE_SUBMISSION_PATH,
        PREDICTION_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        MANIFEST_PATH,
    )


def _require_explicit_overwrite(overwrite: bool) -> None:
    if not overwrite:
        existing = [str(path) for path in _all_output_paths() if path.exists()]
        suffix = "\nExisting outputs:\n  " + "\n  ".join(existing) if existing else ""
        raise PermissionError(
            "final v2 publication replaces canonical submission paths; "
            "pass --overwrite explicitly" + suffix
        )


def _feature_cache_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        matrix_path = FEATURE_ROOT / f"{split}_{FEATURE_SET}.npy"
        for suffix, path in (
            ("matrix", matrix_path),
            ("names", matrix_path.with_suffix(".names.json")),
            ("kinds", matrix_path.with_suffix(".kinds.json")),
        ):
            records[f"{split}_{suffix}"] = _file_record(path)
    return records


def train_final(*, overwrite: bool = False) -> dict[str, Any]:
    _require_explicit_overwrite(overwrite)
    for directory in (V2_MODEL_ROOT, V2_SUBMISSION_ROOT, V2_DIAGNOSTIC_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    months, target = _load_training_labels()
    validation = verify_validation_provenance(months, target)
    template_path = _sample_submission_path()
    submission_ids = _load_template_ids(template_path)

    print("materializing frozen v2 training features", flush=True)
    train_features = materialize_v2_feature_set("train", FEATURE_SET)
    _validate_feature_contract(
        train_features.matrix, train_features.names, train_features.kinds, "train"
    )
    raw_feature_names = list(train_features.names)
    raw_feature_kinds = list(train_features.kinds)
    preprocessor = RobustPreprocessor(
        feature_names=raw_feature_names,
        feature_kinds=raw_feature_kinds,
        clip=PREPROCESSOR_CLIP,
        add_missing_indicators=ADD_MISSING_INDICATORS,
        output_dtype=np.float32,
    )
    print("fitting train-only robust preprocessing on all months 0..70", flush=True)
    preprocessor.fit(train_features.matrix)
    X_train = preprocessor.transform(train_features.matrix)
    if X_train.shape != (TRAIN_SAMPLES, EXPECTED_TRANSFORMED_FEATURES):
        raise ValueError(f"transformed training matrix has shape {X_train.shape}")
    if not np.all(np.isfinite(X_train)):
        raise ValueError("transformed training matrix contains NaN or infinity")
    transformed_names = preprocessor.get_feature_names_out().tolist()
    del train_features
    gc.collect()

    effective_params = asdict(SPECS[MODEL_SPEC_NAME])
    effective_params.update(
        {
            "force_col_wise": True,
            "deterministic": True,
            "bagging_seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
        }
    )
    print(
        f"fitting deterministic capacity LightGBM on {TRAIN_SAMPLES:,} rows",
        flush=True,
    )
    model = LGBMRegressor(**effective_params)
    model.fit(X_train, target)
    del X_train
    gc.collect()

    print("materializing and transforming frozen v2 test features", flush=True)
    test_features = materialize_v2_feature_set("test", FEATURE_SET)
    _validate_feature_contract(
        test_features.matrix, test_features.names, test_features.kinds, "test"
    )
    if test_features.names != raw_feature_names or test_features.kinds != raw_feature_kinds:
        raise ValueError("test feature schema/order differs from training")
    X_test = preprocessor.transform(test_features.matrix)
    if X_test.shape != (TEST_SAMPLES, EXPECTED_TRANSFORMED_FEATURES):
        raise ValueError(f"transformed test matrix has shape {X_test.shape}")
    if not np.all(np.isfinite(X_test)):
        raise ValueError("transformed test matrix contains NaN or infinity")
    del test_features
    gc.collect()

    raw_prediction_by_row = np.asarray(model.predict(X_test), dtype=np.float64)
    del X_test
    gc.collect()
    if raw_prediction_by_row.shape != (TEST_SAMPLES,):
        raise ValueError("model returned the wrong number of predictions")
    if not np.all(np.isfinite(raw_prediction_by_row)) or np.ptp(raw_prediction_by_row) <= 0.0:
        raise ValueError("model prediction is non-finite or constant")
    powered_by_row = _signed_power(raw_prediction_by_row, PREDICTION_POWER)
    prediction_by_row = rms_normalize(powered_by_row)

    # Feature row i corresponds to sample_id i.  Explicit indexing preserves
    # the organizer's exact template order even if that order is not sorted.
    raw_prediction = raw_prediction_by_row[submission_ids]
    powered_prediction = powered_by_row[submission_ids]
    prediction = prediction_by_row[submission_ids]
    if abs(rms(prediction) - 1.0) > 1e-12 or np.ptp(prediction) <= 0.0:
        raise ValueError("final prediction normalization/nonconstancy check failed")

    source_hashes = {_relative(path): _sha256_file(path) for path in SOURCE_PATHS}
    feature_cache_records = _feature_cache_records()
    pipeline_contract: dict[str, Any] = {
        "pipeline_version": "v2",
        "feature_set": FEATURE_SET,
        "raw_feature_count": EXPECTED_RAW_FEATURES,
        "raw_feature_names_sha256": _sha256_json(raw_feature_names),
        "raw_feature_kinds_sha256": _sha256_json(raw_feature_kinds),
        "preprocessor": {
            "class": "RobustPreprocessor",
            "clip": PREPROCESSOR_CLIP,
            "add_missing_indicators": ADD_MISSING_INDICATORS,
            "fit_scope": "all train rows, months 0..70 only",
        },
        "transformed_feature_count": EXPECTED_TRANSFORMED_FEATURES,
        "model_family": "LightGBM L2 regression",
        "model_spec_name": MODEL_SPEC_NAME,
        "model_spec": asdict(SPECS[MODEL_SPEC_NAME]),
        "effective_model_params": effective_params,
        "training_months": [TRAIN_MONTH_MIN, TRAIN_MONTH_MAX],
        "history_weighting": "equal",
        "blend": {"enabled": False, "components": ["capacity_lightgbm"]},
        "prediction_transform": {
            "name": "signed_power_then_global_rms",
            "exponent": PREDICTION_POWER,
            "centering": "none",
        },
        "selection": {
            "model_and_power": "Dev1-Dev2",
            "confirmation": "Dev3",
            "sealed_audit_used_for_selection": False,
        },
        "random_seed": RANDOM_SEED,
        "source_sha256": source_hashes,
        "development_evidence_sha256": {
            "screen": validation["artifacts"]["development_screen_summary"]["sha256"],
            "confirmation": validation["artifacts"]
            ["development_confirmation_summary"]["sha256"],
        },
    }
    pipeline_hash = _sha256_json(pipeline_contract)

    print("writing model, prediction artifact, and three CSV mirrors atomically", flush=True)
    _atomic_joblib(MODEL_PATH, model)
    _atomic_joblib(PREPROCESSOR_PATH, preprocessor)
    _atomic_npz(
        PREDICTION_PATH,
        sample_id=submission_ids,
        raw_prediction=raw_prediction,
        powered_prediction=powered_prediction,
        prediction=prediction,
    )
    submission = pd.DataFrame({"sample_id": submission_ids, "prediction": prediction})
    _atomic_csv_mirrors(
        submission,
        (
            CANONICAL_SUBMISSION_PATH,
            ROOT_ARTIFACT_SUBMISSION_PATH,
            WORKSPACE_SUBMISSION_PATH,
        ),
    )

    outputs = {
        "canonical_submission": _file_record(CANONICAL_SUBMISSION_PATH),
        "root_artifact_submission": _file_record(ROOT_ARTIFACT_SUBMISSION_PATH),
        "workspace_submission": _file_record(WORKSPACE_SUBMISSION_PATH),
        "prediction_artifact": _file_record(PREDICTION_PATH),
        "model": _file_record(MODEL_PATH),
        "preprocessor": _file_record(PREPROCESSOR_PATH),
    }
    submission_hashes = {
        outputs[name]["sha256"]
        for name in (
            "canonical_submission",
            "root_artifact_submission",
            "workspace_submission",
        )
    }
    if len(submission_hashes) != 1:
        raise RuntimeError("published submission mirrors are not byte-identical")

    manifest: dict[str, Any] = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": pipeline_hash,
        "pipeline_contract": pipeline_contract,
        "validation_provenance": validation,
        "input_artifacts": {
            "training_labels": _file_record(DATA_ROOT / "train" / "label.feather"),
            "sample_submission": _file_record(template_path),
            "feature_caches": feature_cache_records,
        },
        "training_rows": TRAIN_SAMPLES,
        "test_rows": TEST_SAMPLES,
        "raw_feature_names": raw_feature_names,
        "raw_feature_kinds": raw_feature_kinds,
        "transformed_feature_names": transformed_names,
        "prediction_statistics": {
            "raw_mean": float(np.mean(raw_prediction)),
            "raw_rms": rms(raw_prediction),
            "powered_mean": float(np.mean(powered_prediction)),
            "powered_rms": rms(powered_prediction),
            "final_mean": float(np.mean(prediction)),
            "final_std": float(np.std(prediction, dtype=np.float64)),
            "final_rms": rms(prediction),
            "final_min": float(np.min(prediction)),
            "final_max": float(np.max(prediction)),
        },
        "outputs": outputs,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(MANIFEST_PATH, manifest)
    print(f"saved canonical v2 submission: {CANONICAL_SUBMISSION_PATH}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace v2 outputs and the two canonical v1-era paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_final(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
