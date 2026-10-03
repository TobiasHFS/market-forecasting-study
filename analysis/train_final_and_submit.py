"""Fit the frozen final model and write the competition submission.

This command is intentionally downstream of the one-time sealed audit.  It
refuses to train unless the frozen pipeline is internally intact, its recorded
source hashes still match the working tree, and the sealed-audit ledger says
that the same pipeline hash completed successfully.

The feature matrix is materialized and transformed one split at a time.  The
training arrays are released after LightGBM has fitted before test features are
materialized, which keeps peak memory substantially below a train+test join.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.feather as feather
import sklearn
from lightgbm import LGBMRegressor

from feature_families import FEATURE_SET_FAMILIES, materialize_feature_set
from modeling import RobustPreprocessor, rms, rms_normalize
from pipeline_config import (
    DATA_ROOT,
    DIAGNOSTIC_ROOT,
    MODEL_ROOT,
    PROJECT_ROOT,
    RANDOM_SEED,
    SUBMISSION_ROOT,
    TEST_SAMPLES,
    TRAIN_MONTH_MAX,
    TRAIN_MONTH_MIN,
    TRAIN_SAMPLES,
    ensure_artifact_directories,
)
from run_gbdt_experiment import SPECS


FEATURE_SET = "multiscale_mechanics_scale"
MODEL_SPEC_NAME = "slow"
FROZEN_PATH = DIAGNOSTIC_ROOT / "frozen_pipeline.json"
SEALED_LEDGER_PATH = DIAGNOSTIC_ROOT / "sealed_audit_ledger.json"

SUBMISSION_PATH = SUBMISSION_ROOT / "submission_final.csv"
PREDICTION_PATH = MODEL_ROOT / "final_test_predictions.npz"
MODEL_PATH = MODEL_ROOT / "final_model.joblib"
PREPROCESSOR_PATH = MODEL_ROOT / "final_preprocessor.joblib"
MANIFEST_PATH = MODEL_ROOT / "final_training_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json_list(values: list[str]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _canonical_pipeline_hash(frozen: dict[str, Any]) -> str:
    body = dict(frozen)
    recorded = body.pop("pipeline_sha256", None)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("frozen pipeline has no valid pipeline_sha256")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if computed != recorded:
        raise ValueError(
            "frozen pipeline JSON was modified after hashing: "
            f"recorded={recorded}, computed={computed}"
        )
    return computed


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_frozen_and_sealed() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable pipeline and its one-time sealed-audit claim."""

    frozen = _read_json_object(FROZEN_PATH)
    pipeline_hash = _canonical_pipeline_hash(frozen)
    required = {
        "status": "frozen_before_sealed_audit",
        "feature_set": FEATURE_SET,
        "model_spec_name": MODEL_SPEC_NAME,
        "history_weighting": "equal",
        "final_component_rule": "pure_gbdt",
        "ridge_weight": 0.0,
        "prediction_centering": "none",
        "random_seed": RANDOM_SEED,
    }
    for key, expected in required.items():
        if frozen.get(key) != expected:
            raise ValueError(
                f"frozen pipeline contract mismatch for {key!r}: "
                f"expected {expected!r}, received {frozen.get(key)!r}"
            )
    expected_families = list(FEATURE_SET_FAMILIES[FEATURE_SET])
    if frozen.get("feature_families") != expected_families:
        raise ValueError("frozen feature-family list no longer matches code")
    if frozen.get("model_spec") != asdict(SPECS[MODEL_SPEC_NAME]):
        raise ValueError("frozen slow LightGBM specification no longer matches code")

    source_hashes = frozen.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("frozen pipeline has no source-hash manifest")
    for relative, expected_hash in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TypeError("invalid frozen source-hash entry")
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        observed_hash = _sha256_file(source_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"frozen source changed after selection: {relative} "
                f"({expected_hash} -> {observed_hash})"
            )

    ledger = _read_json_object(SEALED_LEDGER_PATH)
    if ledger.get("status") != "complete":
        raise ValueError("sealed audit ledger is not complete")
    if ledger.get("pipeline_sha256") != pipeline_hash:
        raise ValueError("sealed audit and frozen pipeline hashes do not match")
    for key in ("summary_path", "prediction_path"):
        value = ledger.get(key)
        if value is not None:
            if not isinstance(value, str) or not _resolve_recorded_path(value).is_file():
                raise FileNotFoundError(
                    f"sealed ledger {key} does not resolve to an existing file: {value!r}"
                )
    return frozen, ledger


def _validate_materialized_schema(
    names: list[str], kinds: list[str], frozen: dict[str, Any], split: str
) -> None:
    frozen_names = frozen.get("feature_names")
    frozen_kinds = frozen.get("feature_kinds")
    if names != frozen_names or kinds != frozen_kinds:
        raise ValueError(f"{split} feature schema differs from frozen pipeline")
    if len(names) != frozen.get("feature_count"):
        raise ValueError(f"{split} feature count differs from frozen pipeline")
    if _sha256_json_list(names) != frozen.get("feature_names_sha256"):
        raise ValueError(f"{split} feature-name digest differs from frozen pipeline")
    if _sha256_json_list(kinds) != frozen.get("feature_kinds_sha256"):
        raise ValueError(f"{split} feature-kind digest differs from frozen pipeline")


def _load_training_labels() -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "train" / "label.feather"
    table = feather.read_table(path, columns=["sample_id", "month", "target"])
    sample_id = table["sample_id"].to_numpy(zero_copy_only=False)
    months = table["month"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    target = table["target"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    del table
    expected_id = np.arange(TRAIN_SAMPLES, dtype=sample_id.dtype)
    if sample_id.shape != expected_id.shape or not np.array_equal(sample_id, expected_id):
        raise ValueError("training label sample_id is not exact feature-row alignment")
    if months.shape != (TRAIN_SAMPLES,) or target.shape != (TRAIN_SAMPLES,):
        raise ValueError("training labels have the wrong row count")
    if np.any(months[1:] < months[:-1]):
        raise ValueError("training months are not chronological")
    if int(months.min()) != TRAIN_MONTH_MIN or int(months.max()) != TRAIN_MONTH_MAX:
        raise ValueError("training month coverage is not exactly 0..70")
    if not np.array_equal(
        np.unique(months), np.arange(TRAIN_MONTH_MIN, TRAIN_MONTH_MAX + 1)
    ):
        raise ValueError("one or more training months are absent")
    if not np.all(np.isfinite(target)) or rms(target) <= 0.0:
        raise ValueError("training target is non-finite or zero norm")
    return months, target


def _sample_submission_path() -> Path:
    candidates = (
        DATA_ROOT / "sample_submission.csv",
        DATA_ROOT / "submission.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "neither sample_submission.csv nor submission.csv exists in the data root"
    )


def _load_submission_ids(path: Path) -> np.ndarray:
    template = pd.read_csv(path)
    if template.columns.tolist() != ["sample_id", "prediction"]:
        raise ValueError(f"unexpected sample-submission header: {template.columns.tolist()}")
    if len(template) != TEST_SAMPLES:
        raise ValueError(f"sample submission has {len(template):,} rows, expected {TEST_SAMPLES:,}")
    if not pd.api.types.is_integer_dtype(template["sample_id"]):
        raise TypeError("sample-submission sample_id must be integer")
    sample_id = template["sample_id"].to_numpy(dtype=np.int64, copy=True)
    if len(np.unique(sample_id)) != TEST_SAMPLES:
        raise ValueError("sample-submission sample_id contains duplicates")
    expected_set = np.arange(TEST_SAMPLES, dtype=np.int64)
    if not np.array_equal(np.sort(sample_id), expected_set):
        raise ValueError(
            "sample-submission IDs are not the complete feature-row ID set 0..N-1"
        )
    return sample_id


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp" + path.suffix)
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _preflight_outputs(overwrite: bool) -> None:
    paths = (
        SUBMISSION_PATH,
        PREDICTION_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        MANIFEST_PATH,
    )
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "final artifacts already exist; refusing to overwrite without --overwrite:\n  "
            + joined
        )


def train_final(*, overwrite: bool = False) -> dict[str, Any]:
    ensure_artifact_directories()
    _preflight_outputs(overwrite)
    frozen, ledger = verify_frozen_and_sealed()
    pipeline_hash = str(frozen["pipeline_sha256"])
    template_path = _sample_submission_path()
    submission_ids = _load_submission_ids(template_path)
    months, target = _load_training_labels()
    started = time.perf_counter()

    print("materializing frozen training features", flush=True)
    train_features = materialize_feature_set("train", FEATURE_SET)
    _validate_materialized_schema(
        train_features.names, train_features.kinds, frozen, "train"
    )
    preprocessor_config = frozen.get("preprocessor")
    if not isinstance(preprocessor_config, dict):
        raise TypeError("frozen preprocessor specification is invalid")
    preprocessor = RobustPreprocessor(
        feature_names=train_features.names,
        feature_kinds=train_features.kinds,
        clip=float(preprocessor_config["clip"]),
        add_missing_indicators=bool(preprocessor_config["add_missing_indicators"]),
        output_dtype=np.float32,
    )
    print("fitting train-only preprocessing on all months 0..70", flush=True)
    preprocessor.fit(train_features.matrix)
    X_train = preprocessor.transform(train_features.matrix)
    transformed_names = preprocessor.get_feature_names_out().tolist()
    raw_feature_count = len(train_features.names)
    del train_features
    gc.collect()

    effective_params = dict(frozen["model_spec"])
    effective_params.update(
        {
            "force_col_wise": True,
            "deterministic": True,
            "bagging_seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
        }
    )
    print(
        f"fitting deterministic {MODEL_SPEC_NAME} LightGBM on {len(target):,} rows",
        flush=True,
    )
    model = LGBMRegressor(**effective_params)
    model.fit(X_train, target)
    del X_train
    gc.collect()

    print("materializing and transforming test features", flush=True)
    test_features = materialize_feature_set("test", FEATURE_SET)
    _validate_materialized_schema(
        test_features.names, test_features.kinds, frozen, "test"
    )
    X_test = preprocessor.transform(test_features.matrix)
    del test_features
    gc.collect()
    raw_prediction_by_row = np.asarray(model.predict(X_test), dtype=np.float64)
    del X_test
    gc.collect()
    if raw_prediction_by_row.shape != (TEST_SAMPLES,):
        raise ValueError("model returned the wrong number of test predictions")
    if not np.all(np.isfinite(raw_prediction_by_row)):
        raise ValueError("model returned non-finite test predictions")
    normalized_by_row = rms_normalize(raw_prediction_by_row)

    # Extracted feature row i corresponds to raw sample_id i.  Reindex the
    # prediction vector to the organizer's template order rather than assuming
    # that the template itself is sorted.
    raw_prediction = raw_prediction_by_row[submission_ids]
    prediction = normalized_by_row[submission_ids]
    if abs(rms(prediction) - 1.0) > 1e-12:
        raise ValueError("whole-vector RMS normalization failed")

    print("writing final model, predictions, and CSV atomically", flush=True)
    _atomic_joblib(MODEL_PATH, model)
    _atomic_joblib(PREPROCESSOR_PATH, preprocessor)
    _atomic_npz(
        PREDICTION_PATH,
        sample_id=submission_ids,
        raw_prediction=raw_prediction,
        prediction=prediction,
    )
    submission = pd.DataFrame(
        {"sample_id": submission_ids, "prediction": prediction}
    )
    temporary_csv = SUBMISSION_PATH.with_suffix(".tmp.csv")
    submission.to_csv(temporary_csv, index=False, float_format="%.17g")
    os.replace(temporary_csv, SUBMISSION_PATH)

    manifest: dict[str, Any] = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": pipeline_hash,
        "frozen_config_path": str(FROZEN_PATH.resolve()),
        "frozen_config_file_sha256": _sha256_file(FROZEN_PATH),
        "sealed_audit_ledger_path": str(SEALED_LEDGER_PATH.resolve()),
        "sealed_audit_ledger_file_sha256": _sha256_file(SEALED_LEDGER_PATH),
        "sealed_audit_status": ledger.get("status"),
        "feature_set": FEATURE_SET,
        "feature_families": frozen["feature_families"],
        "raw_feature_count": raw_feature_count,
        "transformed_feature_count": len(transformed_names),
        "transformed_feature_names": transformed_names,
        "model_spec_name": MODEL_SPEC_NAME,
        "effective_model_params": effective_params,
        "history_weighting": "equal",
        "training_months": [TRAIN_MONTH_MIN, TRAIN_MONTH_MAX],
        "training_rows": TRAIN_SAMPLES,
        "test_rows": TEST_SAMPLES,
        "prediction_normalization": "uncentered RMS once over complete test vector",
        "raw_prediction_mean": float(np.mean(raw_prediction)),
        "raw_prediction_rms": rms(raw_prediction),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_rms": rms(prediction),
        "sample_submission_path": str(template_path.resolve()),
        "sample_submission_sha256": _sha256_file(template_path),
        "outputs": {
            "submission": str(SUBMISSION_PATH.resolve()),
            "submission_sha256": _sha256_file(SUBMISSION_PATH),
            "predictions": str(PREDICTION_PATH.resolve()),
            "predictions_sha256": _sha256_file(PREDICTION_PATH),
            "model": str(MODEL_PATH.resolve()),
            "model_sha256": _sha256_file(MODEL_PATH),
            "preprocessor": str(PREPROCESSOR_PATH.resolve()),
            "preprocessor_sha256": _sha256_file(PREPROCESSOR_PATH),
        },
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
    print(f"saved {SUBMISSION_PATH}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a complete set of existing final artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_final(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
