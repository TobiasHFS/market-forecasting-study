"""Independently audit the frozen v2 model artifacts and submission mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline_config import DATA_ROOT, RANDOM_SEED, TEST_SAMPLES, TRAIN_SAMPLES  # noqa: E402
from v2.run_sequence_experiment import SPECS  # noqa: E402


FEATURE_SET = "base_plus_sequence_all"
MODEL_SPEC_NAME = "capacity"
PREDICTION_POWER = 1.2
EXPECTED_RAW_FEATURES = 754
EXPECTED_TRANSFORMED_FEATURES = 1_326

V2_ROOT = PROJECT_ROOT / "artifacts" / "v2"
CANONICAL_SUBMISSION_PATH = V2_ROOT / "submissions" / "submission_final.csv"
ROOT_ARTIFACT_SUBMISSION_PATH = (
    PROJECT_ROOT / "artifacts" / "submissions" / "submission_final.csv"
)
WORKSPACE_SUBMISSION_PATH = PROJECT_ROOT / "submission_final.csv"
PREDICTION_PATH = V2_ROOT / "models" / "final_test_predictions.npz"
MODEL_PATH = V2_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = V2_ROOT / "models" / "final_preprocessor.joblib"
MANIFEST_PATH = V2_ROOT / "models" / "final_training_manifest.json"
AUDIT_PATH = V2_ROOT / "diagnostics" / "submission_audit.json"

EXPECTED_OUTPUT_PATHS = {
    "canonical_submission": CANONICAL_SUBMISSION_PATH,
    "root_artifact_submission": ROOT_ARTIFACT_SUBMISSION_PATH,
    "workspace_submission": WORKSPACE_SUBMISSION_PATH,
    "prediction_artifact": PREDICTION_PATH,
    "model": MODEL_PATH,
    "preprocessor": PREPROCESSOR_PATH,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve_project_relative(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError("manifest path must be a nonempty string")
    candidate = (PROJECT_ROOT / value).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {value}") from exc
    return candidate


def _rms(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("RMS requires a nonempty finite vector")
    return float(np.sqrt(np.mean(np.square(vector, dtype=np.float64))))


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("signed-power input contains NaN or infinity")
    return np.sign(vector) * np.power(np.abs(vector), exponent)


def _sample_submission_path() -> Path:
    for candidate in (DATA_ROOT / "sample_submission.csv", DATA_ROOT / "submission.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("organizer sample-submission file is absent")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_check(
    checks: dict[str, bool], name: str, condition: bool
) -> None:
    checks[name] = bool(condition)


def _verify_file_record(
    checks: dict[str, bool], prefix: str, record: Any, expected_path: Path | None = None
) -> Path | None:
    if not isinstance(record, dict):
        checks[f"{prefix}_record_valid"] = False
        return None
    try:
        path = _resolve_project_relative(record.get("path"))
    except (TypeError, ValueError):
        checks[f"{prefix}_record_valid"] = False
        return None
    _record_check(checks, f"{prefix}_record_valid", True)
    if expected_path is not None:
        _record_check(checks, f"{prefix}_path_exact", path == expected_path.resolve())
    exists = path.is_file()
    _record_check(checks, f"{prefix}_exists", exists)
    if not exists:
        return path
    _record_check(checks, f"{prefix}_bytes_match", path.stat().st_size == record.get("bytes"))
    _record_check(checks, f"{prefix}_sha256_match", _sha256_file(path) == record.get("sha256"))
    return path


def audit_submission(*, overwrite: bool = False) -> dict[str, Any]:
    if AUDIT_PATH.exists() and not overwrite:
        raise FileExistsError(f"{AUDIT_PATH} exists; pass --overwrite to replace it")
    for path in (
        CANONICAL_SUBMISSION_PATH,
        ROOT_ARTIFACT_SUBMISSION_PATH,
        WORKSPACE_SUBMISSION_PATH,
        PREDICTION_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        MANIFEST_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = _read_json_object(MANIFEST_PATH)
    template_path = _sample_submission_path()
    template = pd.read_csv(template_path)
    first_line = (
        CANONICAL_SUBMISSION_PATH.open("r", encoding="utf-8-sig")
        .readline()
        .rstrip("\r\n")
    )
    submission = pd.read_csv(CANONICAL_SUBMISSION_PATH, float_precision="round_trip")
    with np.load(PREDICTION_PATH, allow_pickle=False) as saved:
        required_arrays = {
            "sample_id",
            "raw_prediction",
            "powered_prediction",
            "prediction",
        }
        saved_arrays_present = required_arrays.issubset(saved.files)
        if not saved_arrays_present:
            raise ValueError("prediction artifact is missing required arrays")
        saved_id = np.asarray(saved["sample_id"], dtype=np.int64)
        raw_prediction = np.asarray(saved["raw_prediction"], dtype=np.float64)
        powered_prediction = np.asarray(saved["powered_prediction"], dtype=np.float64)
        saved_prediction = np.asarray(saved["prediction"], dtype=np.float64)

    checks: dict[str, bool] = {}
    _record_check(checks, "manifest_complete", manifest.get("status") == "complete")
    contract = manifest.get("pipeline_contract")
    _record_check(checks, "pipeline_contract_is_object", isinstance(contract, dict))
    if not isinstance(contract, dict):
        contract = {}
    _record_check(
        checks,
        "pipeline_hash_reproduces",
        manifest.get("pipeline_sha256") == _sha256_json(contract),
    )
    _record_check(checks, "pipeline_version_v2", contract.get("pipeline_version") == "v2")
    _record_check(checks, "feature_set_frozen", contract.get("feature_set") == FEATURE_SET)
    _record_check(
        checks,
        "raw_feature_count_frozen",
        contract.get("raw_feature_count") == EXPECTED_RAW_FEATURES,
    )
    _record_check(
        checks,
        "transformed_feature_count_frozen",
        contract.get("transformed_feature_count") == EXPECTED_TRANSFORMED_FEATURES,
    )
    preprocessor_contract = contract.get("preprocessor", {})
    _record_check(
        checks,
        "preprocessor_clip_frozen",
        preprocessor_contract.get("clip") == 8.0,
    )
    _record_check(
        checks,
        "missing_indicators_frozen",
        preprocessor_contract.get("add_missing_indicators") is True,
    )
    _record_check(checks, "model_spec_name_frozen", contract.get("model_spec_name") == MODEL_SPEC_NAME)
    _record_check(checks, "model_spec_frozen", contract.get("model_spec") == asdict(SPECS[MODEL_SPEC_NAME]))
    _record_check(checks, "random_seed_frozen", contract.get("random_seed") == RANDOM_SEED)
    _record_check(checks, "training_months_exact", contract.get("training_months") == [0, 70])
    _record_check(checks, "equal_history_weighting", contract.get("history_weighting") == "equal")
    blend = contract.get("blend", {})
    _record_check(checks, "no_blend", blend.get("enabled") is False)
    transform = contract.get("prediction_transform", {})
    _record_check(checks, "signed_power_transform", transform.get("name") == "signed_power_then_global_rms")
    _record_check(checks, "prediction_power_frozen", transform.get("exponent") == PREDICTION_POWER)
    _record_check(checks, "prediction_not_centered", transform.get("centering") == "none")
    selection = contract.get("selection", {})
    _record_check(checks, "selection_on_dev1_dev2", selection.get("model_and_power") == "Dev1-Dev2")
    _record_check(checks, "confirmation_on_dev3", selection.get("confirmation") == "Dev3")
    _record_check(
        checks,
        "sealed_not_used_for_selection",
        selection.get("sealed_audit_used_for_selection") is False,
    )
    _record_check(checks, "training_rows_exact", manifest.get("training_rows") == TRAIN_SAMPLES)
    _record_check(checks, "test_rows_exact", manifest.get("test_rows") == TEST_SAMPLES)

    raw_names = manifest.get("raw_feature_names")
    raw_kinds = manifest.get("raw_feature_kinds")
    transformed_names = manifest.get("transformed_feature_names")
    _record_check(checks, "raw_names_count_exact", isinstance(raw_names, list) and len(raw_names) == EXPECTED_RAW_FEATURES)
    _record_check(checks, "raw_kinds_count_exact", isinstance(raw_kinds, list) and len(raw_kinds) == EXPECTED_RAW_FEATURES)
    _record_check(
        checks,
        "transformed_names_count_exact",
        isinstance(transformed_names, list)
        and len(transformed_names) == EXPECTED_TRANSFORMED_FEATURES,
    )
    if isinstance(raw_names, list):
        _record_check(
            checks,
            "raw_names_hash_matches_contract",
            _sha256_json(raw_names) == contract.get("raw_feature_names_sha256"),
        )
        _record_check(checks, "raw_names_unique", len(raw_names) == len(set(raw_names)))
        _record_check(
            checks,
            "identifier_and_labels_excluded",
            not {"sample_id", "month", "target"}.intersection(raw_names),
        )
    if isinstance(raw_kinds, list):
        _record_check(
            checks,
            "raw_kinds_hash_matches_contract",
            _sha256_json(raw_kinds) == contract.get("raw_feature_kinds_sha256"),
        )

    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
    for name, expected_path in EXPECTED_OUTPUT_PATHS.items():
        _verify_file_record(checks, f"output_{name}", outputs.get(name), expected_path)

    input_artifacts = manifest.get("input_artifacts", {})
    if not isinstance(input_artifacts, dict):
        input_artifacts = {}
    _verify_file_record(
        checks,
        "input_training_labels",
        input_artifacts.get("training_labels"),
        DATA_ROOT / "train" / "label.feather",
    )
    _verify_file_record(
        checks,
        "input_sample_submission",
        input_artifacts.get("sample_submission"),
        template_path,
    )
    feature_caches = input_artifacts.get("feature_caches", {})
    _record_check(checks, "feature_cache_records_present", isinstance(feature_caches, dict) and len(feature_caches) == 6)
    if isinstance(feature_caches, dict):
        for name, record in feature_caches.items():
            _verify_file_record(checks, f"feature_cache_{name}", record)

    source_hashes = contract.get("source_sha256", {})
    _record_check(checks, "source_hashes_present", isinstance(source_hashes, dict) and bool(source_hashes))
    if isinstance(source_hashes, dict):
        for relative, expected_hash in source_hashes.items():
            try:
                source_path = _resolve_project_relative(relative)
                matches = source_path.is_file() and _sha256_file(source_path) == expected_hash
            except (TypeError, ValueError):
                matches = False
            _record_check(checks, f"source_hash_{str(relative).replace('/', '_')}", matches)

    validation = manifest.get("validation_provenance", {})
    _record_check(checks, "validation_provenance_present", isinstance(validation, dict))
    if not isinstance(validation, dict):
        validation = {}
    _record_check(checks, "validation_selection_source", validation.get("selection_source") == "Dev1-Dev2 only")
    _record_check(checks, "validation_confirmation_source", validation.get("confirmation_source") == "Dev3")
    _record_check(checks, "validation_power_frozen", validation.get("prediction_power") == PREDICTION_POWER)
    _record_check(checks, "validation_sealed_nonselective", validation.get("sealed_used_for_selection") is False)
    _record_check(checks, "sealed_months_exact", validation.get("sealed_months") == [59, 70])
    _record_check(checks, "sealed_metrics_finite", all(np.isfinite(float(validation.get(key, float("nan")))) for key in ("sealed_raw_cosine", "sealed_powered_cosine_recorded_only")))
    validation_artifacts = validation.get("artifacts", {})
    _record_check(checks, "validation_artifact_records_present", isinstance(validation_artifacts, dict) and len(validation_artifacts) == 6)
    if isinstance(validation_artifacts, dict):
        for name, record in validation_artifacts.items():
            _verify_file_record(checks, f"validation_artifact_{name}", record)

    _record_check(checks, "header_exact", first_line == "sample_id,prediction")
    _record_check(checks, "columns_exact", submission.columns.tolist() == ["sample_id", "prediction"])
    _record_check(checks, "template_columns_exact", template.columns.tolist() == ["sample_id", "prediction"])
    _record_check(checks, "row_count_exact", len(submission) == TEST_SAMPLES)
    _record_check(checks, "template_row_count_exact", len(template) == TEST_SAMPLES)
    _record_check(checks, "sample_id_integer", pd.api.types.is_integer_dtype(submission["sample_id"]))
    submitted_id = submission["sample_id"].to_numpy(dtype=np.int64, copy=False)
    template_id = template["sample_id"].to_numpy(dtype=np.int64, copy=False)
    prediction = submission["prediction"].to_numpy(dtype=np.float64, copy=False)
    expected_ids = np.arange(TEST_SAMPLES, dtype=np.int64)
    _record_check(checks, "sample_id_matches_template_order", np.array_equal(submitted_id, template_id))
    _record_check(checks, "sample_id_complete_set", np.array_equal(np.sort(submitted_id), expected_ids))
    _record_check(checks, "sample_id_unique", np.unique(submitted_id).size == TEST_SAMPLES)
    _record_check(checks, "saved_sample_id_matches_csv", np.array_equal(saved_id, submitted_id))

    expected_shape = (TEST_SAMPLES,)
    _record_check(checks, "prediction_shape_exact", prediction.shape == expected_shape)
    _record_check(checks, "saved_prediction_shape_exact", saved_prediction.shape == expected_shape)
    _record_check(checks, "raw_prediction_shape_exact", raw_prediction.shape == expected_shape)
    _record_check(checks, "powered_prediction_shape_exact", powered_prediction.shape == expected_shape)
    _record_check(checks, "prediction_all_finite", np.all(np.isfinite(prediction)))
    _record_check(checks, "saved_prediction_all_finite", np.all(np.isfinite(saved_prediction)))
    _record_check(checks, "raw_prediction_all_finite", np.all(np.isfinite(raw_prediction)))
    _record_check(checks, "powered_prediction_all_finite", np.all(np.isfinite(powered_prediction)))
    _record_check(checks, "prediction_nonconstant", np.ptp(prediction) > 0.0 and np.unique(prediction).size > 1)
    _record_check(checks, "raw_prediction_nonconstant", np.ptp(raw_prediction) > 0.0 and np.unique(raw_prediction).size > 1)

    prediction_rms = _rms(prediction)
    _record_check(checks, "prediction_rms_is_one", abs(prediction_rms - 1.0) <= 1e-9)
    expected_powered = _signed_power(raw_prediction, PREDICTION_POWER)
    power_scale = max(1.0, float(np.max(np.abs(expected_powered))))
    powered_error = float(np.max(np.abs(expected_powered - powered_prediction)))
    _record_check(checks, "signed_power_relation_exact", powered_error <= 2e-15 * power_scale)
    expected_final = expected_powered / _rms(expected_powered)
    final_scale = max(1.0, float(np.max(np.abs(expected_final))))
    normalized_error = float(np.max(np.abs(expected_final - saved_prediction)))
    _record_check(checks, "global_rms_relation_exact", normalized_error <= 2e-15 * final_scale)
    roundtrip_scale = max(1.0, float(np.max(np.abs(saved_prediction))))
    roundtrip_error = float(np.max(np.abs(prediction - saved_prediction)))
    _record_check(checks, "csv_roundtrip_exact", roundtrip_error <= 1e-15 * roundtrip_scale)

    canonical_hash = _sha256_file(CANONICAL_SUBMISSION_PATH)
    _record_check(checks, "root_artifact_mirror_identical", _sha256_file(ROOT_ARTIFACT_SUBMISSION_PATH) == canonical_hash)
    _record_check(checks, "workspace_mirror_identical", _sha256_file(WORKSPACE_SUBMISSION_PATH) == canonical_hash)

    model_loads = preprocessor_loads = False
    model_feature_count_ok = preprocessor_feature_count_ok = False
    transformed_names_ok = False
    model_param_contract_ok = False
    try:
        model = joblib.load(MODEL_PATH)
        model_loads = True
        model_feature_count_ok = int(model.n_features_in_) == EXPECTED_TRANSFORMED_FEATURES
        params = model.get_params()
        expected_params = contract.get("effective_model_params", {})
        model_param_contract_ok = all(params.get(key) == value for key, value in expected_params.items())
    except Exception:
        model = None
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        preprocessor_loads = True
        preprocessor_feature_count_ok = int(preprocessor.n_features_in_) == EXPECTED_RAW_FEATURES
        transformed_names_ok = (
            preprocessor.get_feature_names_out().tolist() == transformed_names
        )
    except Exception:
        preprocessor = None
    _record_check(checks, "model_joblib_loads", model_loads)
    _record_check(checks, "preprocessor_joblib_loads", preprocessor_loads)
    _record_check(checks, "model_feature_count_exact", model_feature_count_ok)
    _record_check(checks, "preprocessor_feature_count_exact", preprocessor_feature_count_ok)
    _record_check(checks, "preprocessor_names_exact", transformed_names_ok)
    _record_check(checks, "model_params_match_contract", model_param_contract_ok)

    quantile_levels = np.asarray([0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0])
    quantiles = np.quantile(prediction, quantile_levels)
    ready = all(checks.values())
    payload: dict[str, Any] = {
        "status": "ready" if ready else "failed",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "pipeline_sha256": manifest.get("pipeline_sha256"),
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": _sha256_file(MANIFEST_PATH),
        "canonical_submission_path": str(CANONICAL_SUBMISSION_PATH.resolve()),
        "canonical_submission_sha256": canonical_hash,
        "canonical_submission_bytes": CANONICAL_SUBMISSION_PATH.stat().st_size,
        "rows": int(len(submission)),
        "unique_sample_ids": int(np.unique(submitted_id).size),
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
        "signed_power_max_abs_error": powered_error,
        "global_rms_max_abs_error": normalized_error,
        "csv_roundtrip_max_abs_error": roundtrip_error,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
    _atomic_json(AUDIT_PATH, payload)
    if not ready:
        raise ValueError(f"v2 submission audit failed: {payload['failed_checks']}")
    print(
        f"READY: {len(submission):,} rows, RMS={prediction_rms:.12f}, "
        f"sha256={canonical_hash}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing v2 audit record",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_submission(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
