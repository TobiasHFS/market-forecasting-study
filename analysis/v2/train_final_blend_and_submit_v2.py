"""Train and publish the frozen TabM-mini/capacity-LightGBM blend.

The immutable contract lives in
``artifacts/v2/diagnostics/frozen_blend_before_dev3.json``.  This entry point
verifies that file, every source/cache hash it froze, the screen-only grid, and
the one-shot Dev3 promotion gate before fitting anything.  It also verifies the
completed one-time sealed audit and applies its predeclared safety veto; a gate
failure refuses the blend in favor of the already-frozen capacity q=1.2
fallback.  Missing post-freeze artifacts produce a concise PENDING preflight
result.

Training is sequential to control memory.  A unique generation directory is
created with an ``in_progress`` manifest and is changed to ``complete`` only
after every generation artifact has been hashed.  The three public CSV paths
are then replaced individually and atomically, followed by a stable pointer.
The set of several files is intentionally not described as one atomic
transaction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
LOCAL_DEPS = PROJECT_ROOT / ".analysis_deps"
for dependency in (LOCAL_DEPS, ANALYSIS_ROOT, ANALYSIS_ROOT / "v2"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import joblib  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import pyarrow.feather as feather  # noqa: E402
import sklearn  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

from feature_families import materialize_feature_set  # noqa: E402
from modeling import RobustPreprocessor, cosine_score, rms, rms_normalize  # noqa: E402
from pipeline_config import (  # noqa: E402
    DATA_ROOT,
    FEATURE_ROOT,
    RANDOM_SEED,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
)
from v2.run_sequence_experiment import SPECS  # noqa: E402
from v2.run_tabm_mini_challenger import (  # noqa: E402
    TabMMiniSpec,
    _build_tabm_mini,
    _cache_training_matrix,
    _predict_members,
    _run_epoch,
    _set_determinism,
    _smooth_clip_inplace,
)
from v2.v2_features import FEATURE_ROOT as V2_FEATURE_ROOT  # noqa: E402
from v2.v2_features import materialize_v2_feature_set  # noqa: E402


GBDT_FEATURE_SET = "base_plus_sequence_all"
TABM_FEATURE_SET = "multiscale_mechanics_scale"
GBDT_SPEC_NAME = "capacity"
TABM_WEIGHT = 0.6
GBDT_WEIGHT = 0.4
POST_BLEND_POWER = 1.1
PREPROCESSOR_CLIP = 8.0
GBDT_RAW_FEATURES = 754
GBDT_TRANSFORMED_FEATURES = 1_326
TABM_RAW_FEATURES = 474
TABM_TRANSFORMED_FEATURES = 836
FINITE_CHUNK_ROWS = 65_536

V2_ROOT = PROJECT_ROOT / "artifacts" / "v2"
DIAGNOSTIC_ROOT = V2_ROOT / "diagnostics"
EXPERIMENT_ROOT = V2_ROOT / "experiments"
TABM_VALIDATION_ROOT = V2_ROOT / "tabm_mini"
GENERATION_ROOT = V2_ROOT / "generations"
STABLE_POINTER_PATH = V2_ROOT / "models" / "final_blend_pointer.json"
CANONICAL_SUBMISSION_PATH = V2_ROOT / "submissions" / "submission_final.csv"
ROOT_ARTIFACT_SUBMISSION_PATH = (
    PROJECT_ROOT / "artifacts" / "submissions" / "submission_final.csv"
)
WORKSPACE_SUBMISSION_PATH = PROJECT_ROOT / "submission_final.csv"

FROZEN_PATH = DIAGNOSTIC_ROOT / "frozen_blend_before_dev3.json"
SCREEN_GRID_PATH = DIAGNOSTIC_ROOT / "tabm_capacity_screen_grid.csv"
DEV3_AUDIT_PATH = DIAGNOSTIC_ROOT / "tabm_capacity_frozen_dev3_audit.json"
DEV3_BLEND_PREDICTION_PATH = (
    DIAGNOSTIC_ROOT / "tabm_capacity_frozen_dev3_predictions.npz"
)
POOLED_REPORT_PATH = DIAGNOSTIC_ROOT / "tabm_capacity_pooled_report.json"
MONTHLY_REPORT_PATH = DIAGNOSTIC_ROOT / "tabm_capacity_monthly_scores.csv"
COMPARISON_PATH = DIAGNOSTIC_ROOT / "tabm_capacity_blend_comparison.json"

SEALED_ROOT = V2_ROOT / "sealed_blend"
SEALED_TABM_PREDICTION_PATH = SEALED_ROOT / "tabm_sealed_predictions.npz"
SEALED_BLEND_VIEWS_PATH = SEALED_ROOT / "sealed_blend_views.npz"
SEALED_CHECKPOINT_PATH = SEALED_ROOT / "tabm_final_checkpoint.pt"
SEALED_PREPROCESSOR_PATH = SEALED_ROOT / "tabm_final_preprocessor.joblib"
SEALED_HISTORY_PATH = SEALED_ROOT / "tabm_final_training_history.csv"
SEALED_MONTHLY_PATH = SEALED_ROOT / "sealed_blend_monthly.csv"
SEALED_SUMMARY_PATH = SEALED_ROOT / "sealed_blend_summary.json"
SEALED_LEDGER_PATH = SEALED_ROOT / "audit_ledger.json"
SEALED_CAPACITY_STEM = "sequence_base_plus_sequence_all_capacity_SealedAudit"
SEALED_CAPACITY_PREDICTION_PATH = (
    EXPERIMENT_ROOT / f"{SEALED_CAPACITY_STEM}_oof.npz"
)
SEALED_CAPACITY_SUMMARY_PATH = (
    EXPERIMENT_ROOT / f"{SEALED_CAPACITY_STEM}_summary.json"
)
SEALED_OUTPUT_PATHS = {
    "tabm_predictions": SEALED_TABM_PREDICTION_PATH,
    "blend_views": SEALED_BLEND_VIEWS_PATH,
    "checkpoint": SEALED_CHECKPOINT_PATH,
    "preprocessor": SEALED_PREPROCESSOR_PATH,
    "training_history": SEALED_HISTORY_PATH,
    "monthly": SEALED_MONTHLY_PATH,
    "summary": SEALED_SUMMARY_PATH,
}

POST_FREEZE_PATHS = (
    SCREEN_GRID_PATH,
    DEV3_AUDIT_PATH,
    DEV3_BLEND_PREDICTION_PATH,
    POOLED_REPORT_PATH,
    MONTHLY_REPORT_PATH,
    COMPARISON_PATH,
    TABM_VALIDATION_ROOT / "dev3_summary.json",
    TABM_VALIDATION_ROOT / "dev3_predictions.npz",
    TABM_VALIDATION_ROOT / "dev3_checkpoint.pt",
    SEALED_LEDGER_PATH,
    *SEALED_OUTPUT_PATHS.values(),
    SEALED_CAPACITY_PREDICTION_PATH,
    SEALED_CAPACITY_SUMMARY_PATH,
)

SOURCE_PATHS = (
    ANALYSIS_ROOT / "pipeline_config.py",
    ANALYSIS_ROOT / "modeling.py",
    ANALYSIS_ROOT / "feature_families.py",
    ANALYSIS_ROOT / "v2" / "v2_features.py",
    ANALYSIS_ROOT / "v2" / "sequence_features.py",
    ANALYSIS_ROOT / "v2" / "run_sequence_experiment.py",
    ANALYSIS_ROOT / "v2" / "run_tabm_mini_challenger.py",
    Path(__file__).resolve(),
)


class PendingArtifactsError(RuntimeError):
    """Raised before training when required post-freeze evidence is pending."""

    def __init__(self, missing: Sequence[Path]) -> None:
        self.missing = tuple(missing)
        super().__init__("post-freeze artifacts are pending")


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


def _resolve_relative(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    return path


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _verify_file_record(record: Any, expected_path: Path, context: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{context} file record is missing")
    try:
        path = _resolve_relative(record.get("path"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} file record has an invalid path") from exc
    if path != expected_path.resolve():
        raise ValueError(f"{context} file record points to the wrong path")
    observed = _file_record(path)
    if observed["bytes"] != record.get("bytes"):
        raise ValueError(f"{context} byte count differs from its audit record")
    if observed["sha256"].lower() != str(record.get("sha256", "")).lower():
        raise ValueError(f"{context} SHA-256 differs from its audit record")
    return observed


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(torch: Any, path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
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


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.csv")
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_finite_chunked(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim not in (1, 2) or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty vector/matrix")
    for left in range(0, matrix.shape[0], FINITE_CHUNK_ROWS):
        right = min(left + FINITE_CHUNK_ROWS, matrix.shape[0])
        if not np.all(np.isfinite(matrix[left:right])):
            raise ValueError(f"{name} contains NaN or infinity in rows {left}:{right}")


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    _assert_finite_chunked(vector, "signed-power input")
    return np.sign(vector) * np.power(np.abs(vector), exponent)


def _validate_frozen_contract(frozen: dict[str, Any]) -> None:
    if frozen.get("status") != "frozen_before_dev3_outer_labels":
        raise ValueError("frozen blend status is invalid")
    if frozen.get("selection_months") != [23, 46]:
        raise ValueError("selection months changed")
    if frozen.get("confirmation_months") != [47, 58]:
        raise ValueError("confirmation months changed")
    if frozen.get("sealed_audit_months") != [59, 70]:
        raise ValueError("sealed months changed")
    if frozen.get("dev3_result") is not None or frozen.get("sealed_result") is not None:
        raise ValueError("the pre-Dev3 frozen file was modified with later results")
    components = frozen.get("components", {})
    gbdt = components.get("capacity_lightgbm", {})
    if gbdt != {
        "feature_set": GBDT_FEATURE_SET,
        "raw_feature_count": GBDT_RAW_FEATURES,
        "preprocessor": {
            "class": "RobustPreprocessor",
            "clip": PREPROCESSOR_CLIP,
            "add_missing_indicators": True,
        },
        "spec": asdict(SPECS[GBDT_SPEC_NAME]),
    }:
        raise ValueError("frozen capacity-LightGBM component contract changed")
    tabm = components.get("tabm_mini", {})
    expected_tabm = {
        "feature_set": TABM_FEATURE_SET,
        "raw_feature_count": TABM_RAW_FEATURES,
        "architecture": "16-member parameter-efficient MiniEnsemble with shared 2x256 ReLU backbone and member-specific input scales/heads",
        "k": 16,
        "hidden_size": 256,
        "hidden_layers": 2,
        "dropout": 0.1,
        "batch_size": 1024,
        "learning_rate": 0.002,
        "weight_decay": 0.0003,
        "inner_validation_months": 3,
        "smooth_clip_scale": 3.0,
        "loss": "mean of 16 standardized-target member MSE losses",
        "inference": "mean of 16 de-standardized member predictions",
        "final_epoch_rule": "median of the three inner-selected development-fold epochs; no outer-validation or sealed label chooses the epoch",
        "random_seed": RANDOM_SEED,
    }
    if tabm != expected_tabm:
        raise ValueError("frozen TabM-mini component contract changed")
    blend = frozen.get("blend_contract", {})
    if blend != {
        "tabm_weight": TABM_WEIGHT,
        "capacity_lightgbm_weight": GBDT_WEIGHT,
        "component_normalization": "uncentered unit RMS once over the complete evaluation vector for each component",
        "post_blend_transform": "sign(p) * abs(p)^1.1",
        "final_normalization": "uncentered unit RMS once over the complete transformed blend",
        "centering": "none",
    }:
        raise ValueError("frozen blend arithmetic changed")
    grid = frozen.get("selection_grid", {})
    if grid != {
        "tabm_weights": [value / 10.0 for value in range(11)],
        "post_blend_signed_power_exponents": [1.0, 1.1, 1.2, 1.3],
    }:
        raise ValueError("frozen selection grid changed")
    if abs(float(frozen.get("selected_dev1_dev2_cosine", np.nan)) - 0.14445559865492444) > 1e-15:
        raise ValueError("frozen screen score changed")
    gates = frozen.get("predeclared_promotion_gates", {})
    if gates != {
        "dev3": {
            "candidate_cosine_must_exceed": 0.1401980574532448,
            "comparison": "frozen capacity-LightGBM q=1.2 Dev3 score",
            "minimum_positive_month_share": 1.0,
        },
        "sealed_audit_safety_veto_only": {
            "maximum_allowed_deficit_vs_capacity_q1p2": 0.001,
            "apply_to": [
                "pooled months 59-70 cosine",
                "pooled months 59-70 cosine excluding high-energy month 66",
            ],
            "minimum_positive_month_share": 0.9166666666666666,
            "failure_action": (
                "fall back to the already-frozen capacity-LightGBM q=1.2; "
                "do not tune the blend"
            ),
        },
    }:
        raise ValueError("frozen promotion gates changed")


def verify_frozen_file_and_hashes() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = _read_json(FROZEN_PATH)
    _validate_frozen_contract(frozen)
    verified: dict[str, Any] = {
        "frozen_file": _file_record(FROZEN_PATH),
        "validation_inputs": {},
        "selection_artifacts": {},
    }
    for section, destination in (
        ("validation_input_sha256", "validation_inputs"),
        ("selection_artifact_sha256", "selection_artifacts"),
    ):
        records = frozen.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"frozen file has no {section}")
        for relative, expected in records.items():
            path = _resolve_relative(relative)
            observed = _sha256_file(path)
            if observed.lower() != str(expected).lower():
                raise ValueError(f"frozen hash mismatch: {relative}")
            verified[destination][relative] = {
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
    return frozen, verified


def _labels() -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather", columns=["sample_id", "month", "target"]
    )
    sample_id = table["sample_id"].to_numpy(zero_copy_only=False)
    months = table["month"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    target = table["target"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    if not np.array_equal(sample_id, np.arange(TRAIN_SAMPLES, dtype=sample_id.dtype)):
        raise ValueError("training label IDs are not feature-row aligned")
    if months.shape != (TRAIN_SAMPLES,) or target.shape != (TRAIN_SAMPLES,):
        raise ValueError("training labels have the wrong shape")
    if not np.array_equal(np.unique(months), np.arange(71)):
        raise ValueError("training labels do not cover exactly months 0..70")
    _assert_finite_chunked(target, "target")
    return months, target


def _load_oof(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"row_indices", "months", "target", "prediction"}
        if not required.issubset(saved.files):
            raise ValueError(f"OOF artifact missing arrays: {path}")
        return {name: np.asarray(saved[name]) for name in required}


def _concat_oof(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    pieces = [_load_oof(path) for path in paths]
    result = {
        name: np.concatenate([piece[name] for piece in pieces])
        for name in ("row_indices", "months", "target", "prediction")
    }
    order = np.argsort(result["row_indices"])
    return {name: values[order] for name, values in result.items()}


def _aligned(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return all(
        np.array_equal(left[name], right[name])
        for name in ("row_indices", "months", "target")
    )


def _blend_view(tabm: np.ndarray, gbdt: np.ndarray, weight: float, power: float) -> np.ndarray:
    tabm_unit = rms_normalize(tabm)
    gbdt_unit = rms_normalize(gbdt)
    blend = weight * tabm_unit + (1.0 - weight) * gbdt_unit
    return rms_normalize(_signed_power(blend, power))


def _monthly_cosines(
    months: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    expected_months: Sequence[int],
) -> dict[int, float]:
    month_vector = np.asarray(months).reshape(-1)
    target_vector = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction_vector = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not (
        month_vector.shape == target_vector.shape == prediction_vector.shape
        and np.array_equal(np.unique(month_vector), np.asarray(expected_months))
    ):
        raise ValueError("monthly cosine inputs have the wrong shape/month coverage")
    result: dict[int, float] = {}
    for month in expected_months:
        selected = month_vector == month
        result[int(month)] = cosine_score(
            target_vector[selected], prediction_vector[selected]
        )
    return result


def _verify_tabm_fold(
    torch: Any,
    fold: str,
    month_range: tuple[int, int],
    months: np.ndarray,
    target: np.ndarray,
    frozen_tabm: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    lower = fold.lower()
    summary_path = TABM_VALIDATION_ROOT / f"{lower}_summary.json"
    prediction_path = TABM_VALIDATION_ROOT / f"{lower}_predictions.npz"
    checkpoint_path = TABM_VALIDATION_ROOT / f"{lower}_checkpoint.pt"
    summary = _read_json(summary_path)
    if summary.get("fold") != fold or summary.get("outer_validation_months") != list(month_range):
        raise ValueError(f"{fold} summary has the wrong outer block")
    if summary.get("raw_features") != TABM_RAW_FEATURES:
        raise ValueError(f"{fold} summary has the wrong raw feature count")
    if summary.get("refit_transformed_features") != TABM_TRANSFORMED_FEATURES:
        raise ValueError(f"{fold} summary has the wrong transformed feature count")
    if summary.get("refit_from_scratch_on_all_outer_train_months") is not True:
        raise ValueError(f"{fold} was not honestly refit")
    if summary.get("loss_contract") != "mean of k individual-member MSE values":
        raise ValueError(f"{fold} has the wrong loss contract")
    if summary.get("inference_contract") != "arithmetic mean of k member predictions":
        raise ValueError(f"{fold} has the wrong inference contract")
    outer_train_end = month_range[0] - 1
    inner_months = int(frozen_tabm["inner_validation_months"])
    inner_start = outer_train_end - inner_months + 1
    expected_ranges = {
        "outer_train_months": [0, outer_train_end],
        "inner_fit_months": [0, inner_start - 1],
        "inner_validation_months": [inner_start, outer_train_end],
        "outer_validation_months": list(month_range),
    }
    for key, expected in expected_ranges.items():
        if summary.get(key) != expected:
            raise ValueError(f"{fold} summary has the wrong {key}")

    # Match the bounded development schedule used by the sealed audit.  The
    # dataclass defaults are intentionally broader and are not the frozen run.
    validation_spec = TabMMiniSpec(max_epochs=8, min_epochs=4, patience=4)
    best_epoch_value = summary.get("best_epoch")
    if (
        isinstance(best_epoch_value, bool)
        or not isinstance(best_epoch_value, int)
        or not 1 <= best_epoch_value <= validation_spec.max_epochs
    ):
        raise ValueError(f"{fold} has no valid inner-selected epoch")
    best_epoch = best_epoch_value
    selection_rows = summary.get("selection_epochs")
    if not isinstance(selection_rows, list) or not selection_rows:
        raise ValueError(f"{fold} selection epoch trace is missing")
    reproduced_epoch = 0
    reproduced_score = -np.inf
    observed_epochs: list[int] = []
    for row in selection_rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{fold} selection epoch row is invalid")
        epoch = row.get("epoch")
        score = float(row.get("inner_ensemble_cosine", np.nan))
        if (
            row.get("fold") != fold
            or row.get("stage") != "selection"
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not np.isfinite(score)
        ):
            raise ValueError(f"{fold} selection epoch evidence is invalid")
        observed_epochs.append(epoch)
        improved = score > reproduced_score + 1e-7
        if row.get("improved") is not improved:
            raise ValueError(f"{fold} selection improvement flags do not reproduce")
        if improved:
            reproduced_epoch = epoch
            reproduced_score = score
    if observed_epochs != list(range(1, len(observed_epochs) + 1)):
        raise ValueError(f"{fold} selection epochs are not contiguous")
    if len(observed_epochs) > validation_spec.max_epochs:
        raise ValueError(f"{fold} selection trace exceeds the frozen epoch budget")
    if best_epoch != reproduced_epoch:
        raise ValueError(f"{fold} best epoch does not reproduce inner selection")
    if not np.isclose(
        float(summary.get("best_inner_ensemble_cosine", np.nan)),
        reproduced_score,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError(f"{fold} best inner score does not reproduce")
    refit_rows = summary.get("refit_epochs")
    if not isinstance(refit_rows, list) or any(
        not isinstance(row, Mapping) for row in refit_rows
    ):
        raise ValueError(f"{fold} refit epoch trace differs from inner selection")
    if [row.get("epoch") for row in refit_rows] != list(range(1, best_epoch + 1)):
        raise ValueError(f"{fold} refit epoch trace differs from inner selection")
    if any(
        row.get("fold") != fold or row.get("stage") != "refit"
        for row in refit_rows
    ):
        raise ValueError(f"{fold} refit epoch identities are invalid")
    oof = _load_oof(prediction_path)
    expected_indices = np.flatnonzero(
        (months >= month_range[0]) & (months <= month_range[1])
    ).astype(np.int64)
    if not np.array_equal(oof["row_indices"].astype(np.int64), expected_indices):
        raise ValueError(f"{fold} prediction rows are wrong")
    if not np.array_equal(oof["months"].astype(np.int16), months[expected_indices]):
        raise ValueError(f"{fold} prediction months are wrong")
    if not np.array_equal(oof["target"].astype(np.float64), target[expected_indices]):
        raise ValueError(f"{fold} targets differ from labels")
    prediction = oof["prediction"].astype(np.float64)
    _assert_finite_chunked(prediction, f"{fold} prediction")
    reproduced = cosine_score(target[expected_indices], prediction)
    if abs(reproduced - float(summary["outer_ensemble_cosine_raw"])) > 1e-12:
        raise ValueError(f"{fold} raw score does not reproduce")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("fold") != fold or int(checkpoint.get("selected_epoch", 0)) != best_epoch:
        raise ValueError(f"{fold} checkpoint metadata differs from summary")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError(f"{fold} checkpoint state dictionary is missing")
    if int(checkpoint.get("input_dim", 0)) != TABM_TRANSFORMED_FEATURES:
        raise ValueError(f"{fold} checkpoint input dimension is wrong")
    spec = checkpoint.get("spec", {})
    if spec != asdict(validation_spec):
        raise ValueError(f"{fold} checkpoint was not trained with the frozen schedule")
    mapping = {
        "feature_set": "feature_set",
        "k": "k",
        "hidden_size": "hidden_size",
        "hidden_layers": "hidden_layers",
        "dropout": "dropout",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "inner_val_months": "inner_validation_months",
        "smooth_clip_scale": "smooth_clip_scale",
        "seed": "random_seed",
    }
    for checkpoint_key, frozen_key in mapping.items():
        if spec.get(checkpoint_key) != frozen_tabm.get(frozen_key):
            raise ValueError(f"{fold} checkpoint spec differs for {checkpoint_key}")
    return best_epoch, {
        "summary": _file_record(summary_path),
        "predictions": _file_record(prediction_path),
        "checkpoint": _file_record(checkpoint_path),
        "raw_cosine": reproduced,
    }


def _verify_sealed_safety_veto(
    frozen: Mapping[str, Any],
    months: np.ndarray,
    target: np.ndarray,
    selected_epochs: Mapping[str, int],
    final_epoch: int,
) -> dict[str, Any]:
    summary = _read_json(SEALED_SUMMARY_PATH)
    ledger = _read_json(SEALED_LEDGER_PATH)
    if summary.get("status") != "complete_one_time_sealed_audit_diagnostic_only":
        raise ValueError("sealed blend summary is not complete")
    if ledger.get("status") != "complete_one_time_non_selective_sealed_audit":
        raise ValueError("sealed blend audit ledger is not complete")
    for key, expected in {
        "configuration_locked_before_labels_loaded": True,
        "outer_validation_scores_used_for_configuration": False,
        "sealed_labels_used_for_configuration": False,
        "sealed_results_are_selection_eligible": False,
        "sealed_results_used_only_for_predeclared_safety_veto": True,
    }.items():
        if ledger.get(key) is not expected:
            raise ValueError(f"sealed audit ledger has invalid {key}")

    _verify_file_record(ledger.get("frozen_contract"), FROZEN_PATH, "sealed frozen contract")
    output_records = ledger.get("outputs")
    if not isinstance(output_records, Mapping) or set(output_records) != set(
        SEALED_OUTPUT_PATHS
    ):
        raise ValueError("sealed audit output record set is incomplete")
    verified_outputs = {
        name: _verify_file_record(
            output_records.get(name), path, f"sealed output {name}"
        )
        for name, path in SEALED_OUTPUT_PATHS.items()
    }
    capacity_records = ledger.get("capacity_sealed_inputs")
    if not isinstance(capacity_records, Mapping) or set(capacity_records) != {
        "summary",
        "predictions",
    }:
        raise ValueError("sealed capacity input records are incomplete")
    verified_capacity = {
        "summary": _verify_file_record(
            capacity_records.get("summary"),
            SEALED_CAPACITY_SUMMARY_PATH,
            "sealed capacity summary",
        ),
        "predictions": _verify_file_record(
            capacity_records.get("predictions"),
            SEALED_CAPACITY_PREDICTION_PATH,
            "sealed capacity predictions",
        ),
    }
    if summary.get("capacity_inputs") != capacity_records:
        raise ValueError("sealed summary and ledger capacity inputs differ")
    expected_output_paths = {
        "tabm_predictions": _relative(SEALED_TABM_PREDICTION_PATH),
        "blend_views": _relative(SEALED_BLEND_VIEWS_PATH),
        "checkpoint": _relative(SEALED_CHECKPOINT_PATH),
        "preprocessor": _relative(SEALED_PREPROCESSOR_PATH),
        "training_history": _relative(SEALED_HISTORY_PATH),
        "monthly": _relative(SEALED_MONTHLY_PATH),
        "audit_ledger": _relative(SEALED_LEDGER_PATH),
    }
    if summary.get("output_paths") != expected_output_paths:
        raise ValueError("sealed summary output paths are not exact")

    epoch_evidence = ledger.get("development_epoch_evidence")
    expected_epoch_list = [int(selected_epochs[name]) for name in ("Dev1", "Dev2", "Dev3")]
    if not isinstance(epoch_evidence, Mapping):
        raise ValueError("sealed audit has no development epoch evidence")
    if epoch_evidence.get("inner_selected_epochs") != expected_epoch_list:
        raise ValueError("sealed audit selected epochs differ from strict inner traces")
    if epoch_evidence.get("final_epoch") != final_epoch:
        raise ValueError("sealed audit final epoch differs from the frozen median rule")
    if epoch_evidence.get("rule") != (
        "median of Dev1/Dev2/Dev3 inner-selected epochs; outer scores ignored"
    ):
        raise ValueError("sealed audit epoch rule changed")
    if summary.get("epoch_evidence") != epoch_evidence:
        raise ValueError("sealed summary and ledger epoch evidence differ")
    evidence_rows = epoch_evidence.get("folds")
    if not isinstance(evidence_rows, list) or any(
        not isinstance(row, Mapping) for row in evidence_rows
    ):
        raise ValueError("sealed audit epoch fold evidence is incomplete")
    if [row.get("fold") for row in evidence_rows] != ["Dev1", "Dev2", "Dev3"]:
        raise ValueError("sealed audit epoch fold evidence is incomplete")
    for row, fold in zip(evidence_rows, ("Dev1", "Dev2", "Dev3"), strict=True):
        if row.get("inner_selected_epoch") != selected_epochs[fold]:
            raise ValueError(f"sealed audit {fold} epoch differs")
        if row.get("outer_scores_used_for_configuration") is not False:
            raise ValueError(f"sealed audit {fold} used an outer score for configuration")
        stem = fold.lower()
        for name, path in {
            "summary": TABM_VALIDATION_ROOT / f"{stem}_summary.json",
            "checkpoint": TABM_VALIDATION_ROOT / f"{stem}_checkpoint.pt",
            "predictions": TABM_VALIDATION_ROOT / f"{stem}_predictions.npz",
        }.items():
            _verify_file_record(row.get(name), path, f"sealed epoch evidence {fold} {name}")

    locked = summary.get("locked_configuration")
    if not isinstance(locked, Mapping):
        raise ValueError("sealed summary has no locked configuration")
    locked_sha256 = _sha256_json(locked)
    if (
        summary.get("locked_configuration_sha256") != locked_sha256
        or ledger.get("locked_configuration_sha256") != locked_sha256
    ):
        raise ValueError("sealed locked-configuration hash does not reproduce")
    expected_locked = {
        "locked_before_training_or_sealed_labels_loaded": True,
        "training_months": [0, 58],
        "sealed_prediction_months": [59, 70],
        "final_epoch": final_epoch,
        "final_epoch_rule": epoch_evidence["rule"],
        "inner_selected_epochs": expected_epoch_list,
        "tabm_weight": TABM_WEIGHT,
        "capacity_lightgbm_weight": GBDT_WEIGHT,
        "component_normalization": "complete-vector uncentered unit RMS",
        "post_blend_signed_power": POST_BLEND_POWER,
        "final_normalization": "complete-vector uncentered unit RMS",
        "centering": "none",
        "random_seed": RANDOM_SEED,
    }
    for key, expected in expected_locked.items():
        if locked.get(key) != expected:
            raise ValueError(f"sealed locked configuration differs for {key}")
    locked_spec = locked.get("tabm_spec")
    frozen_tabm = frozen["components"]["tabm_mini"]
    if not isinstance(locked_spec, Mapping):
        raise ValueError("sealed locked TabM specification is missing")
    for locked_key, frozen_key in {
        "feature_set": "feature_set",
        "k": "k",
        "hidden_size": "hidden_size",
        "hidden_layers": "hidden_layers",
        "dropout": "dropout",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "inner_val_months": "inner_validation_months",
        "smooth_clip_scale": "smooth_clip_scale",
        "seed": "random_seed",
    }.items():
        if locked_spec.get(locked_key) != frozen_tabm.get(frozen_key):
            raise ValueError(f"sealed locked TabM spec differs for {locked_key}")

    with np.load(SEALED_BLEND_VIEWS_PATH, allow_pickle=False) as saved:
        required = {
            "row_indices",
            "months",
            "target",
            "tabm_raw",
            "capacity_lightgbm_raw",
            "tabm_unit_rms",
            "capacity_lightgbm_unit_rms",
            "linear_blend_tabm_0p6_capacity_0p4",
            "post_blend_signed_power_q1p1",
            "final_unit_rms",
        }
        if not required.issubset(saved.files):
            raise ValueError("sealed blend view artifact is incomplete")
        arrays = {name: np.asarray(saved[name]) for name in required}
    expected_indices = np.flatnonzero((months >= 59) & (months <= 70)).astype(np.int64)
    if not np.array_equal(arrays["row_indices"].astype(np.int64), expected_indices):
        raise ValueError("sealed blend row indices are not exact")
    sealed_months = arrays["months"].astype(np.int16)
    sealed_target = arrays["target"].astype(np.float64)
    if not np.array_equal(sealed_months, months[expected_indices]):
        raise ValueError("sealed blend months are not label-aligned")
    if not np.array_equal(sealed_target, target[expected_indices]):
        raise ValueError("sealed blend targets are not label-aligned")
    for name in required.difference({"row_indices", "months"}):
        vector = np.asarray(arrays[name], dtype=np.float64)
        if vector.shape != expected_indices.shape:
            raise ValueError(f"sealed blend array {name} has the wrong shape")
        _assert_finite_chunked(vector, f"sealed blend {name}")

    tabm_raw = arrays["tabm_raw"].astype(np.float64)
    capacity_raw = arrays["capacity_lightgbm_raw"].astype(np.float64)

    # Tie the saved blend views back to both prediction artifacts that the
    # sealed ledger hashes.  Hash verification alone would not establish that
    # these were the vectors actually used to build the views.
    sealed_tabm = _load_oof(SEALED_TABM_PREDICTION_PATH)
    sealed_capacity = _load_oof(SEALED_CAPACITY_PREDICTION_PATH)
    for name, artifact in {
        "TabM": sealed_tabm,
        "capacity-LightGBM": sealed_capacity,
    }.items():
        if not np.array_equal(
            artifact["row_indices"].astype(np.int64), expected_indices
        ):
            raise ValueError(f"sealed {name} prediction rows are not exact")
        if not np.array_equal(artifact["months"].astype(np.int16), sealed_months):
            raise ValueError(f"sealed {name} prediction months are not aligned")
        if not np.array_equal(
            artifact["target"].astype(np.float64), sealed_target
        ):
            raise ValueError(f"sealed {name} prediction targets are not aligned")
        artifact_prediction = artifact["prediction"].astype(np.float64)
        if artifact_prediction.shape != expected_indices.shape:
            raise ValueError(f"sealed {name} prediction has the wrong shape")
        _assert_finite_chunked(artifact_prediction, f"sealed {name} prediction")
    if not np.array_equal(
        sealed_tabm["prediction"].astype(np.float64), tabm_raw
    ):
        raise ValueError("sealed TabM prediction differs from the blend view")
    if not np.array_equal(
        sealed_capacity["prediction"].astype(np.float64), capacity_raw
    ):
        raise ValueError("sealed capacity prediction differs from the blend view")

    capacity_summary = _read_json(SEALED_CAPACITY_SUMMARY_PATH)
    frozen_capacity = frozen["components"]["capacity_lightgbm"]
    if (
        capacity_summary.get("feature_set") != frozen_capacity["feature_set"]
        or capacity_summary.get("spec_name") != GBDT_SPEC_NAME
        or capacity_summary.get("spec") != frozen_capacity["spec"]
        or capacity_summary.get("raw_feature_count")
        != frozen_capacity["raw_feature_count"]
    ):
        raise ValueError("sealed capacity summary differs from the frozen component")
    capacity_folds = capacity_summary.get("folds")
    if (
        not isinstance(capacity_folds, list)
        or len(capacity_folds) != 1
        or not isinstance(capacity_folds[0], Mapping)
        or capacity_folds[0].get("fold") != "SealedAudit"
    ):
        raise ValueError("sealed capacity summary is not the one sealed-audit fold")

    expected_tabm_unit = rms_normalize(tabm_raw)
    expected_capacity_unit = rms_normalize(capacity_raw)
    expected_linear = TABM_WEIGHT * expected_tabm_unit + GBDT_WEIGHT * expected_capacity_unit
    expected_powered = _signed_power(expected_linear, POST_BLEND_POWER)
    expected_blend = rms_normalize(expected_powered)
    for name, expected in {
        "tabm_unit_rms": expected_tabm_unit,
        "capacity_lightgbm_unit_rms": expected_capacity_unit,
        "linear_blend_tabm_0p6_capacity_0p4": expected_linear,
        "post_blend_signed_power_q1p1": expected_powered,
        "final_unit_rms": expected_blend,
    }.items():
        if not np.allclose(
            arrays[name].astype(np.float64), expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"sealed blend stored view {name} does not reproduce")

    capacity_q1p2 = rms_normalize(_signed_power(capacity_raw, 1.2))
    full_blend_cosine = cosine_score(sealed_target, expected_blend)
    full_capacity_cosine = cosine_score(sealed_target, capacity_q1p2)
    monthly_scores = _monthly_cosines(
        sealed_months, sealed_target, expected_blend, range(59, 71)
    )
    positive_month_share = float(
        np.mean(np.asarray(list(monthly_scores.values()), dtype=np.float64) > 0.0)
    )
    exclude_66 = sealed_months != 66
    if np.count_nonzero(~exclude_66) == 0:
        raise ValueError("sealed audit does not contain month 66")
    excluded_target = sealed_target[exclude_66]
    # The predeclared producer filters the already complete-vector-normalized
    # views.  Re-normalizing each component on this subset would change their
    # relative 60/40 scale and would no longer reproduce the frozen decision.
    excluded_blend_cosine = cosine_score(
        excluded_target, expected_blend[exclude_66]
    )
    excluded_capacity_cosine = cosine_score(
        excluded_target, capacity_q1p2[exclude_66]
    )

    reported = summary.get("diagnostic_scores", {}).get(
        "frozen_blend_final_q1p1", {}
    )
    if (
        not isinstance(reported, Mapping)
        or not np.isclose(
            float(reported.get("pooled_cosine", np.nan)),
            full_blend_cosine,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(reported.get("positive_month_share", np.nan)),
            positive_month_share,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("sealed summary diagnostics do not reproduce")

    gate = frozen["predeclared_promotion_gates"]["sealed_audit_safety_veto_only"]
    maximum_deficit = float(gate["maximum_allowed_deficit_vs_capacity_q1p2"])
    minimum_positive_share = float(gate["minimum_positive_month_share"])
    full_deficit = full_capacity_cosine - full_blend_cosine
    excluded_deficit = excluded_capacity_cosine - excluded_blend_cosine
    full_passed = full_deficit <= maximum_deficit
    excluded_passed = excluded_deficit <= maximum_deficit
    months_passed = positive_month_share >= minimum_positive_share
    passed = bool(full_passed and excluded_passed and months_passed)

    reported_gate = summary.get("predeclared_safety_gate")
    ledger_gate = ledger.get("predeclared_safety_gate")
    if not isinstance(reported_gate, Mapping) or dict(reported_gate) != ledger_gate:
        raise ValueError("sealed summary and ledger safety-gate decisions differ")
    expected_gate_strings = {
        "status": "passed" if passed else "vetoed",
        "passed": passed,
        "candidate": "frozen_blend_final_q1p1",
        "benchmark": "capacity_lightgbm_q1p2",
        "failure_action": gate["failure_action"],
        "selection_scope": "predeclared safety veto only; no tuning",
    }
    for key, expected in expected_gate_strings.items():
        if reported_gate.get(key) != expected:
            raise ValueError(f"sealed safety-gate decision differs for {key}")
    expected_gate_metrics = {
        "maximum_allowed_deficit": maximum_deficit,
        "minimum_positive_month_share": minimum_positive_share,
        "candidate_pooled_cosine": full_blend_cosine,
        "benchmark_pooled_cosine": full_capacity_cosine,
        "pooled_deficit": full_deficit,
        "candidate_excluding_month_66_cosine": excluded_blend_cosine,
        "benchmark_excluding_month_66_cosine": excluded_capacity_cosine,
        "excluding_month_66_deficit": excluded_deficit,
        "candidate_positive_month_share": positive_month_share,
    }
    for key, expected in expected_gate_metrics.items():
        if not np.isclose(
            float(reported_gate.get(key, np.nan)), expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"sealed safety-gate metric differs for {key}")
    if not passed:
        raise ValueError(
            "frozen blend failed the predeclared sealed safety veto; use the "
            "already-frozen capacity-LightGBM q=1.2 fallback "
            f"(full deficit={full_deficit:.12g}, excluding-66 deficit="
            f"{excluded_deficit:.12g}, positive-month share={positive_month_share:.12g})"
        )
    return {
        "status": "verified_passed",
        "decision": "publish_frozen_tabm_capacity_blend",
        "fallback_if_failed": "capacity_lightgbm_q1p2",
        "maximum_allowed_deficit": maximum_deficit,
        "minimum_positive_month_share": minimum_positive_share,
        "full_months_59_70": {
            "blend_q1p1_cosine": full_blend_cosine,
            "capacity_q1p2_cosine": full_capacity_cosine,
            "deficit": full_deficit,
            "passed": bool(full_passed),
        },
        "excluding_month_66": {
            "blend_q1p1_cosine": excluded_blend_cosine,
            "capacity_q1p2_cosine": excluded_capacity_cosine,
            "deficit": excluded_deficit,
            "passed": bool(excluded_passed),
        },
        "positive_month_share": positive_month_share,
        "monthly_cosines": {str(key): value for key, value in monthly_scores.items()},
        "positive_month_share_passed": bool(months_passed),
        "artifacts": {
            "ledger": _file_record(SEALED_LEDGER_PATH),
            "outputs": verified_outputs,
            "capacity_inputs": verified_capacity,
        },
    }


def verify_post_freeze_artifacts(
    frozen: dict[str, Any], months: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    missing = [path for path in POST_FREEZE_PATHS if not path.is_file()]
    if missing:
        raise PendingArtifactsError(missing)
    import torch

    fold_records: dict[str, Any] = {}
    epochs: list[int] = []
    for fold, bounds in (("Dev1", (23, 34)), ("Dev2", (35, 46)), ("Dev3", (47, 58))):
        epoch, record = _verify_tabm_fold(
            torch, fold, bounds, months, target, frozen["components"]["tabm_mini"]
        )
        epochs.append(epoch)
        fold_records[fold] = record

    tabm_screen = _concat_oof(
        [TABM_VALIDATION_ROOT / "dev1_predictions.npz", TABM_VALIDATION_ROOT / "dev2_predictions.npz"]
    )
    gbdt_screen = _load_oof(
        EXPERIMENT_ROOT
        / "sequence_base_plus_sequence_all_capacity_Dev1-Dev2_oof.npz"
    )
    if not _aligned(tabm_screen, gbdt_screen):
        raise ValueError("screen component OOF artifacts are not aligned")
    y_screen = tabm_screen["target"].astype(np.float64)
    tabm_screen_prediction = tabm_screen["prediction"].astype(np.float64)
    gbdt_screen_prediction = gbdt_screen["prediction"].astype(np.float64)
    expected_grid: dict[tuple[float, float], float] = {}
    for weight in frozen["selection_grid"]["tabm_weights"]:
        for power in frozen["selection_grid"]["post_blend_signed_power_exponents"]:
            prediction = _blend_view(
                tabm_screen_prediction, gbdt_screen_prediction, float(weight), float(power)
            )
            expected_grid[(float(weight), float(power))] = cosine_score(y_screen, prediction)
    selected_key = max(expected_grid, key=expected_grid.get)
    if selected_key != (TABM_WEIGHT, POST_BLEND_POWER):
        raise ValueError(f"independent screen selected {selected_key}, not frozen blend")
    if abs(expected_grid[selected_key] - float(frozen["selected_dev1_dev2_cosine"])) > 1e-12:
        raise ValueError("independent screen score differs from frozen score")

    grid = pd.read_csv(SCREEN_GRID_PATH)
    required_columns = {"tabm_weight", "power_exponent", "screen_cosine", "rank", "selected"}
    if not required_columns.issubset(grid.columns) or len(grid) != len(expected_grid):
        raise ValueError("strict screen grid schema/size is invalid")
    forbidden = [
        column
        for column in grid.columns
        if any(token in column.lower() for token in ("dev3", "confirmation", "sealed", "audit"))
    ]
    if forbidden:
        raise ValueError(f"screen grid contains post-screen columns: {forbidden}")
    observed_keys: set[tuple[float, float]] = set()
    for row in grid.itertuples(index=False):
        key = (float(row.tabm_weight), float(row.power_exponent))
        observed_keys.add(key)
        if key not in expected_grid or abs(float(row.screen_cosine) - expected_grid[key]) > 1e-12:
            raise ValueError(f"screen grid row does not reproduce: {key}")
    if observed_keys != set(expected_grid):
        raise ValueError("screen grid candidates differ from frozen grid")
    selected_rows = grid[grid["selected"].astype(bool)]
    if len(selected_rows) != 1:
        raise ValueError("screen grid does not mark exactly one winner")
    selected_row = selected_rows.iloc[0]
    if (float(selected_row["tabm_weight"]), float(selected_row["power_exponent"])) != selected_key:
        raise ValueError("screen grid winner differs from frozen winner")

    tabm_dev3 = _load_oof(TABM_VALIDATION_ROOT / "dev3_predictions.npz")
    gbdt_dev3 = _load_oof(
        EXPERIMENT_ROOT / "sequence_base_plus_sequence_all_capacity_Dev3_oof.npz"
    )
    if not _aligned(tabm_dev3, gbdt_dev3):
        raise ValueError("Dev3 component OOF artifacts are not aligned")
    dev3_prediction = _blend_view(
        tabm_dev3["prediction"].astype(np.float64),
        gbdt_dev3["prediction"].astype(np.float64),
        TABM_WEIGHT,
        POST_BLEND_POWER,
    )
    dev3_target = tabm_dev3["target"].astype(np.float64)
    dev3_months = tabm_dev3["months"].astype(np.int16)
    dev3_cosine = cosine_score(dev3_target, dev3_prediction)
    dev3_monthly_scores = _monthly_cosines(
        dev3_months, dev3_target, dev3_prediction, range(47, 59)
    )
    dev3_positive_share = float(
        np.mean(
            np.asarray(list(dev3_monthly_scores.values()), dtype=np.float64) > 0.0
        )
    )
    dev3_gate = frozen["predeclared_promotion_gates"]["dev3"]
    dev3_cosine_passed = dev3_cosine > float(
        dev3_gate["candidate_cosine_must_exceed"]
    )
    dev3_months_passed = dev3_positive_share >= float(
        dev3_gate["minimum_positive_month_share"]
    )
    dev3_gate_passed = bool(dev3_cosine_passed and dev3_months_passed)
    dev3_audit = _read_json(DEV3_AUDIT_PATH)
    if dev3_audit.get("status") not in {"complete", "complete_one_shot_dev3_audit"}:
        raise ValueError("one-shot Dev3 audit is not complete")
    if str(dev3_audit.get("frozen_config_sha256", "")).lower() != _sha256_file(FROZEN_PATH).lower():
        raise ValueError("Dev3 audit references a different frozen file")
    if float(dev3_audit.get("tabm_weight", np.nan)) != TABM_WEIGHT:
        raise ValueError("Dev3 audit used a different TabM weight")
    if float(dev3_audit.get("capacity_lightgbm_weight", np.nan)) != GBDT_WEIGHT:
        raise ValueError("Dev3 audit used a different capacity-LightGBM weight")
    if float(dev3_audit.get("power_exponent", np.nan)) != POST_BLEND_POWER:
        raise ValueError("Dev3 audit used a different exponent")
    if abs(float(dev3_audit.get("dev3_cosine", np.nan)) - dev3_cosine) > 1e-12:
        raise ValueError("Dev3 audit cosine does not independently reproduce")
    if not np.isclose(
        float(dev3_audit.get("dev3_positive_month_share", np.nan)),
        dev3_positive_share,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Dev3 audit positive-month share does not reproduce")
    reported_gate = dev3_audit.get("promotion_gate")
    if not isinstance(reported_gate, Mapping) or reported_gate != {
        "candidate_cosine_must_exceed": float(
            dev3_gate["candidate_cosine_must_exceed"]
        ),
        "minimum_positive_month_share": float(
            dev3_gate["minimum_positive_month_share"]
        ),
        "cosine_passed": bool(dev3_cosine_passed),
        "positive_month_share_passed": bool(dev3_months_passed),
        "passed": dev3_gate_passed,
    }:
        raise ValueError("Dev3 audit promotion gate does not reproduce")
    if not dev3_gate_passed:
        raise ValueError(
            "frozen blend failed the predeclared Dev3 promotion gate; use the "
            "already-frozen capacity-LightGBM q=1.2 fallback"
        )
    expected_prediction_hash = _sha256_file(DEV3_BLEND_PREDICTION_PATH)
    if str(dev3_audit.get("prediction_artifact_sha256", "")).lower() != expected_prediction_hash.lower():
        raise ValueError("Dev3 audit prediction hash differs")

    reported_dev3 = _load_oof(DEV3_BLEND_PREDICTION_PATH)
    if not np.array_equal(reported_dev3["row_indices"], tabm_dev3["row_indices"]):
        raise ValueError("Dev3 blend prediction rows differ")
    if not np.array_equal(reported_dev3["months"].astype(np.int16), dev3_months):
        raise ValueError("Dev3 blend prediction months differ")
    if not np.array_equal(reported_dev3["target"].astype(np.float64), dev3_target):
        raise ValueError("Dev3 blend prediction targets differ")
    if not np.allclose(
        reported_dev3["prediction"].astype(np.float64), dev3_prediction, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Dev3 blend prediction vector does not reproduce")

    monthly = pd.read_csv(MONTHLY_REPORT_PATH)
    if monthly.empty or "month" not in monthly.columns:
        raise ValueError("pooled monthly context is invalid")
    pooled = _read_json(POOLED_REPORT_PATH)
    comparison = _read_json(COMPARISON_PATH)
    selected_epochs = {
        name: epoch
        for name, epoch in zip(("Dev1", "Dev2", "Dev3"), epochs, strict=True)
    }
    final_epoch = int(median(epochs))
    if pooled.get("status") != "reporting_only_no_selection":
        raise ValueError("pooled post-freeze report has the wrong status")
    if pooled.get("selected_epochs") != selected_epochs or pooled.get(
        "final_tabm_refit_epoch"
    ) != final_epoch:
        raise ValueError("pooled report epoch evidence differs")
    expected_summary_hashes = {
        _relative(TABM_VALIDATION_ROOT / f"{fold.lower()}_summary.json"): _sha256_file(
            TABM_VALIDATION_ROOT / f"{fold.lower()}_summary.json"
        )
        for fold in ("Dev1", "Dev2", "Dev3")
    }
    if {
        str(key): str(value).lower()
        for key, value in pooled.get("tabm_summary_sha256", {}).items()
    } != {key: value.lower() for key, value in expected_summary_hashes.items()}:
        raise ValueError("pooled report summary hashes differ")
    if comparison.get("status") != "frozen_dev3_audit_complete":
        raise ValueError("post-freeze comparison has the wrong status")
    comparison_dev3 = comparison.get("dev3_audit", {})
    if (
        not isinstance(comparison_dev3, Mapping)
        or comparison_dev3.get("promotion_gate_passed") is not True
        or not np.isclose(
            float(comparison_dev3.get("cosine", np.nan)),
            dev3_cosine,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(comparison_dev3.get("positive_month_share", np.nan)),
            dev3_positive_share,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("post-freeze comparison Dev3 gate differs")
    if comparison.get("tabm_epoch_rule") != {
        "selected_epochs": selected_epochs,
        "median_final_refit_epoch": final_epoch,
    }:
        raise ValueError("post-freeze comparison epoch evidence differs")

    sealed_safety = _verify_sealed_safety_veto(
        frozen, months, target, selected_epochs, final_epoch
    )

    paths = {
        "screen_grid": SCREEN_GRID_PATH,
        "dev3_audit": DEV3_AUDIT_PATH,
        "dev3_blend_predictions": DEV3_BLEND_PREDICTION_PATH,
        "pooled_report": POOLED_REPORT_PATH,
        "monthly_report": MONTHLY_REPORT_PATH,
        "comparison": COMPARISON_PATH,
        "gbdt_screen_oof": EXPERIMENT_ROOT
        / "sequence_base_plus_sequence_all_capacity_Dev1-Dev2_oof.npz",
        "gbdt_dev3_oof": EXPERIMENT_ROOT
        / "sequence_base_plus_sequence_all_capacity_Dev3_oof.npz",
    }
    return {
        "status": "verified",
        "selected_epochs": selected_epochs,
        "final_epoch": final_epoch,
        "screen_cosine": expected_grid[selected_key],
        "dev3_promotion_gate": {
            "passed": dev3_gate_passed,
            "cosine": dev3_cosine,
            "positive_month_share": dev3_positive_share,
            "monthly_cosines": {
                str(key): value for key, value in dev3_monthly_scores.items()
            },
        },
        "dev3_used_for_epoch_rule_only": True,
        "dev3_used_for_weight_or_power_selection": False,
        "dev3_used_for_predeclared_promotion_gate": True,
        "sealed_safety_veto": sealed_safety,
        "fold_artifacts": fold_records,
        "post_freeze_artifacts": {name: _file_record(path) for name, path in paths.items()},
    }


def preflight() -> dict[str, Any]:
    frozen, frozen_verification = verify_frozen_file_and_hashes()
    months, target = _labels()
    try:
        post_freeze = verify_post_freeze_artifacts(frozen, months, target)
    except PendingArtifactsError as exc:
        return {
            "status": "pending",
            "frozen_contract_verified": True,
            "frozen_file_sha256": frozen_verification["frozen_file"]["sha256"],
            "missing": [_relative(path) for path in exc.missing],
        }
    return {
        "status": "ready",
        "frozen_contract_verified": True,
        "frozen_verification": frozen_verification,
        "post_freeze": post_freeze,
    }


def _template() -> tuple[Path, np.ndarray]:
    candidates = (DATA_ROOT / "sample_submission.csv", DATA_ROOT / "submission.csv")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("organizer submission template is missing")
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["sample_id", "prediction"] or len(frame) != TEST_SAMPLES:
        raise ValueError("organizer submission template has the wrong schema/rows")
    if not pd.api.types.is_integer_dtype(frame["sample_id"]):
        raise TypeError("template sample_id must be integer")
    sample_id = frame["sample_id"].to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(np.sort(sample_id), np.arange(TEST_SAMPLES, dtype=np.int64)):
        raise ValueError("template IDs are not the complete test feature-row set")
    return path, sample_id


def _make_preprocessor(names: list[str], kinds: list[str]) -> RobustPreprocessor:
    return RobustPreprocessor(
        feature_names=names,
        feature_kinds=kinds,
        clip=PREPROCESSOR_CLIP,
        add_missing_indicators=True,
        output_dtype=np.float32,
    )


def _predict_gbdt_chunks(
    model: LGBMRegressor, preprocessor: RobustPreprocessor, raw: np.ndarray
) -> np.ndarray:
    result = np.empty(raw.shape[0], dtype=np.float64)
    for left in range(0, raw.shape[0], FINITE_CHUNK_ROWS):
        right = min(left + FINITE_CHUNK_ROWS, raw.shape[0])
        transformed = preprocessor.transform(raw[left:right])
        _assert_finite_chunked(transformed, "GBDT transformed test chunk")
        result[left:right] = np.asarray(model.predict(transformed), dtype=np.float64)
        del transformed
    _assert_finite_chunked(result, "GBDT raw prediction")
    return result


def _predict_tabm_chunks(
    torch: Any,
    model: Any,
    preprocessor: RobustPreprocessor,
    raw: np.ndarray,
    spec: TabMMiniSpec,
    device: Any,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    result = np.empty(raw.shape[0], dtype=np.float64)
    use_amp = device.type == "cuda"
    for left in range(0, raw.shape[0], FINITE_CHUNK_ROWS):
        right = min(left + FINITE_CHUNK_ROWS, raw.shape[0])
        transformed = np.ascontiguousarray(preprocessor.transform(raw[left:right]))
        _assert_finite_chunked(transformed, "TabM transformed test chunk")
        _smooth_clip_inplace(transformed, spec.smooth_clip_scale)
        members = _predict_members(
            torch,
            model,
            transformed,
            device=device,
            batch_size=spec.inference_batch_size,
            k=spec.k,
            use_amp=use_amp,
        )
        result[left:right] = (
            members.astype(np.float64) * target_std + target_mean
        ).mean(axis=1)
        del transformed, members
    _assert_finite_chunked(result, "TabM raw prediction")
    return result


def _input_cache_records() -> dict[str, Any]:
    paths = [
        V2_FEATURE_ROOT / f"test_{GBDT_FEATURE_SET}.npy",
        V2_FEATURE_ROOT / f"test_{GBDT_FEATURE_SET}.names.json",
        V2_FEATURE_ROOT / f"test_{GBDT_FEATURE_SET}.kinds.json",
    ]
    for source in ("market", "order", "transaction", "mechanics"):
        paths.extend(
            [FEATURE_ROOT / f"test_{source}.npy", FEATURE_ROOT / f"test_{source}.names.json"]
        )
    return {_relative(path): _file_record(path) for path in paths}


def train_and_publish(*, overwrite: bool = False) -> dict[str, Any]:
    if not overwrite:
        raise PermissionError("publishing the final blend requires explicit --overwrite")
    preflight_result = preflight()
    if preflight_result["status"] != "ready":
        missing = [_resolve_relative(path) for path in preflight_result["missing"]]
        raise PendingArtifactsError(missing)
    frozen = _read_json(FROZEN_PATH)
    months, target = _labels()
    template_path, template_ids = _template()
    final_epoch = int(preflight_result["post_freeze"]["final_epoch"])

    generation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    generation = GENERATION_ROOT / generation_id
    model_root = generation / "models"
    prediction_root = generation / "predictions"
    submission_root = generation / "submission"
    generation.mkdir(parents=True, exist_ok=False)
    manifest_path = generation / "manifest.json"
    started = time.perf_counter()
    in_progress: dict[str, Any] = {
        "status": "in_progress",
        "generation_id": generation_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "preflight_complete",
        "frozen_file": _file_record(FROZEN_PATH),
        "post_freeze_verification": preflight_result["post_freeze"],
    }
    _atomic_json(manifest_path, in_progress)

    # Component 1: capacity LightGBM.  Release its transformed training matrix
    # and test feature view before materializing the TabM training matrix.
    print("[1/2] fitting frozen capacity LightGBM", flush=True)
    gbdt_train = materialize_v2_feature_set("train", GBDT_FEATURE_SET)
    if gbdt_train.matrix.shape != (TRAIN_SAMPLES, GBDT_RAW_FEATURES):
        raise ValueError("GBDT training feature shape changed")
    gbdt_names, gbdt_kinds = list(gbdt_train.names), list(gbdt_train.kinds)
    gbdt_preprocessor = _make_preprocessor(gbdt_names, gbdt_kinds)
    gbdt_preprocessor.fit(gbdt_train.matrix)
    X_gbdt = gbdt_preprocessor.transform(gbdt_train.matrix)
    if X_gbdt.shape != (TRAIN_SAMPLES, GBDT_TRANSFORMED_FEATURES):
        raise ValueError("GBDT transformed feature shape changed")
    _assert_finite_chunked(X_gbdt, "GBDT transformed training matrix")
    gbdt_params = asdict(SPECS[GBDT_SPEC_NAME])
    gbdt_params.update(
        force_col_wise=True,
        deterministic=True,
        bagging_seed=RANDOM_SEED,
        feature_fraction_seed=RANDOM_SEED,
    )
    gbdt_model = LGBMRegressor(**gbdt_params)
    gbdt_model.fit(X_gbdt, target)
    del X_gbdt, gbdt_train
    gc.collect()
    gbdt_test = materialize_v2_feature_set("test", GBDT_FEATURE_SET)
    if gbdt_test.matrix.shape != (TEST_SAMPLES, GBDT_RAW_FEATURES):
        raise ValueError("GBDT test feature shape changed")
    if gbdt_test.names != gbdt_names or gbdt_test.kinds != gbdt_kinds:
        raise ValueError("GBDT train/test schemas differ")
    gbdt_by_row = _predict_gbdt_chunks(gbdt_model, gbdt_preprocessor, gbdt_test.matrix)
    del gbdt_test
    gc.collect()
    gbdt_model_path = model_root / "capacity_lightgbm.joblib"
    gbdt_preprocessor_path = model_root / "capacity_preprocessor.joblib"
    _atomic_joblib(gbdt_model_path, gbdt_model)
    _atomic_joblib(gbdt_preprocessor_path, gbdt_preprocessor)
    in_progress["stage"] = "capacity_complete"
    _atomic_json(manifest_path, in_progress)

    # Component 2: final TabM-mini, with epoch fixed by the three inner folds.
    print(f"[2/2] fitting frozen TabM-mini for {final_epoch} epochs", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the frozen blend") from exc
    tabm_spec = TabMMiniSpec(
        feature_set=TABM_FEATURE_SET,
        k=16,
        hidden_size=256,
        hidden_layers=2,
        dropout=0.1,
        batch_size=1024,
        inference_batch_size=4096,
        max_epochs=max(final_epoch, 1),
        min_epochs=min(final_epoch, 4),
        patience=4,
        learning_rate=0.002,
        weight_decay=0.0003,
        inner_val_months=3,
        smooth_clip_scale=3.0,
        seed=RANDOM_SEED,
    )
    _set_determinism(torch, RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tabm_train = materialize_feature_set("train", TABM_FEATURE_SET)
    if tabm_train.matrix.shape != (TRAIN_SAMPLES, TABM_RAW_FEATURES):
        raise ValueError("TabM training feature shape changed")
    tabm_names, tabm_kinds = list(tabm_train.names), list(tabm_train.kinds)
    tabm_preprocessor = _make_preprocessor(tabm_names, tabm_kinds)
    tabm_preprocessor.fit(tabm_train.matrix)
    X_tabm = np.ascontiguousarray(tabm_preprocessor.transform(tabm_train.matrix))
    if X_tabm.shape != (TRAIN_SAMPLES, TABM_TRANSFORMED_FEATURES):
        raise ValueError("TabM transformed feature shape changed")
    _assert_finite_chunked(X_tabm, "TabM transformed training matrix")
    _smooth_clip_inplace(X_tabm, tabm_spec.smooth_clip_scale)
    target_mean = float(np.mean(target))
    target_std = float(np.std(target, dtype=np.float64))
    if not np.isfinite(target_std) or target_std <= 0.0:
        raise ValueError("target standard deviation is invalid")
    target_standard = ((target - target_mean) / target_std).astype(np.float32)
    tabm_model = _build_tabm_mini(torch, TABM_TRANSFORMED_FEATURES, tabm_spec).to(device)
    optimizer = torch.optim.AdamW(
        tabm_model.parameters(), lr=tabm_spec.learning_rate, weight_decay=tabm_spec.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    X_tensor, y_tensor, data_device, _cached = _cache_training_matrix(
        torch, X_tabm, target_standard, device
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, final_epoch + 1):
        epoch_started = time.perf_counter()
        loss = _run_epoch(
            torch,
            model=tabm_model,
            optimizer=optimizer,
            scaler=scaler,
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            data_device=data_device,
            device=device,
            spec=tabm_spec,
            use_amp=use_amp,
        )
        row = {
            "epoch": epoch,
            "member_mse_standardized": loss,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(f"TabM final epoch {epoch}/{final_epoch}: mse={loss:.6f}", flush=True)
    del X_tensor, y_tensor, X_tabm, target_standard, optimizer, scaler, tabm_train
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tabm_test = materialize_feature_set("test", TABM_FEATURE_SET)
    if tabm_test.matrix.shape != (TEST_SAMPLES, TABM_RAW_FEATURES):
        raise ValueError("TabM test feature shape changed")
    if tabm_test.names != tabm_names or tabm_test.kinds != tabm_kinds:
        raise ValueError("TabM train/test schemas differ")
    tabm_by_row = _predict_tabm_chunks(
        torch,
        tabm_model,
        tabm_preprocessor,
        tabm_test.matrix,
        tabm_spec,
        device,
        target_mean,
        target_std,
    )
    del tabm_test
    gc.collect()

    tabm_preprocessor_path = model_root / "tabm_preprocessor.joblib"
    tabm_checkpoint_path = model_root / "tabm_checkpoint.pt"
    tabm_spec_path = model_root / "tabm_spec.json"
    history_path = model_root / "tabm_training_history.csv"
    _atomic_joblib(tabm_preprocessor_path, tabm_preprocessor)
    checkpoint = {
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in tabm_model.state_dict().items()
        },
        "input_dim": TABM_TRANSFORMED_FEATURES,
        "target_mean": target_mean,
        "target_std": target_std,
        "final_epoch": final_epoch,
        "selected_fold_epochs": preflight_result["post_freeze"]["selected_epochs"],
        "spec": asdict(tabm_spec),
        "seed": RANDOM_SEED,
    }
    _atomic_torch_save(torch, tabm_checkpoint_path, checkpoint)
    _atomic_json(
        tabm_spec_path,
        {
            "spec": asdict(tabm_spec),
            "final_epoch": final_epoch,
            "selected_fold_epochs": preflight_result["post_freeze"]["selected_epochs"],
            "target_mean": target_mean,
            "target_std": target_std,
            "device_used_for_training": str(device),
            "amp": use_amp,
        },
    )
    _atomic_csv(history_path, pd.DataFrame(history))
    del tabm_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    in_progress["stage"] = "tabm_complete"
    _atomic_json(manifest_path, in_progress)

    # All vector arithmetic is performed once over the complete template-order
    # vector; there is no per-month or per-partition normalization.
    gbdt_raw = gbdt_by_row[template_ids]
    tabm_raw = tabm_by_row[template_ids]
    gbdt_scale = rms(gbdt_raw)
    tabm_scale = rms(tabm_raw)
    gbdt_unit = gbdt_raw / gbdt_scale
    tabm_unit = tabm_raw / tabm_scale
    blend_pre_power = TABM_WEIGHT * tabm_unit + GBDT_WEIGHT * gbdt_unit
    powered_blend = _signed_power(blend_pre_power, POST_BLEND_POWER)
    powered_blend_scale = rms(powered_blend)
    prediction = powered_blend / powered_blend_scale
    _assert_finite_chunked(prediction, "final blend")
    if abs(rms(prediction) - 1.0) > 1e-12 or np.ptp(prediction) <= 0.0:
        raise ValueError("final blend is not finite, nonconstant, unit RMS")

    prediction_path = prediction_root / "test_blend_predictions.npz"
    _atomic_npz(
        prediction_path,
        sample_id=template_ids,
        capacity_raw_prediction=gbdt_raw,
        tabm_raw_prediction=tabm_raw,
        capacity_component_rms=np.asarray([gbdt_scale], dtype=np.float64),
        tabm_component_rms=np.asarray([tabm_scale], dtype=np.float64),
        capacity_unit_rms_view=gbdt_unit,
        tabm_unit_rms_view=tabm_unit,
        blend_pre_power=blend_pre_power,
        powered_blend=powered_blend,
        powered_blend_rms=np.asarray([powered_blend_scale], dtype=np.float64),
        tabm_weight=np.asarray([TABM_WEIGHT], dtype=np.float64),
        capacity_weight=np.asarray([GBDT_WEIGHT], dtype=np.float64),
        power_exponent=np.asarray([POST_BLEND_POWER], dtype=np.float64),
        prediction=prediction,
    )
    generation_submission_path = submission_root / "submission_final.csv"
    submission = pd.DataFrame({"sample_id": template_ids, "prediction": prediction})
    _atomic_csv(generation_submission_path, submission)

    component_artifacts = {
        "capacity_model": _file_record(gbdt_model_path),
        "capacity_preprocessor": _file_record(gbdt_preprocessor_path),
        "tabm_preprocessor": _file_record(tabm_preprocessor_path),
        "tabm_checkpoint": _file_record(tabm_checkpoint_path),
        "tabm_spec": _file_record(tabm_spec_path),
        "tabm_training_history": _file_record(history_path),
        "prediction_artifact": _file_record(prediction_path),
        "generation_submission": _file_record(generation_submission_path),
    }
    source_hashes = {_relative(path): _sha256_file(path) for path in SOURCE_PATHS}
    pipeline_contract = {
        "feature_sets": {
            "capacity_lightgbm": GBDT_FEATURE_SET,
            "tabm_mini": TABM_FEATURE_SET,
        },
        "preprocessor": {"clip": PREPROCESSOR_CLIP, "add_missing_indicators": True},
        "capacity_spec": asdict(SPECS[GBDT_SPEC_NAME]),
        "tabm_spec": asdict(tabm_spec),
        "tabm_final_epoch": final_epoch,
        "tabm_epoch_rule": "median of Dev1, Dev2, Dev3 inner-selected epochs",
        "blend": {
            "tabm_weight": TABM_WEIGHT,
            "capacity_weight": GBDT_WEIGHT,
            "component_normalization": "complete-vector uncentered RMS",
            "signed_power": POST_BLEND_POWER,
            "final_normalization": "complete-vector uncentered RMS",
            "centering": "none",
        },
        "training_months": [0, 70],
        "random_seed": RANDOM_SEED,
        "frozen_config_sha256": _sha256_file(FROZEN_PATH),
        "source_sha256": source_hashes,
    }
    manifest: dict[str, Any] = {
        "status": "complete",
        "generation_id": generation_id,
        "created_utc": in_progress["created_utc"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": _sha256_json(pipeline_contract),
        "pipeline_contract": pipeline_contract,
        "frozen_verification": preflight_result["frozen_verification"],
        "post_freeze_verification": preflight_result["post_freeze"],
        "input_artifacts": {
            "labels": _file_record(DATA_ROOT / "train" / "label.feather"),
            "template": _file_record(template_path),
            "test_feature_caches": _input_cache_records(),
        },
        "feature_schemas": {
            "capacity_raw_names": gbdt_names,
            "capacity_raw_kinds": gbdt_kinds,
            "capacity_transformed_names": gbdt_preprocessor.get_feature_names_out().tolist(),
            "tabm_raw_names": tabm_names,
            "tabm_raw_kinds": tabm_kinds,
            "tabm_transformed_names": tabm_preprocessor.get_feature_names_out().tolist(),
        },
        "prediction_statistics": {
            "capacity_raw_rms": gbdt_scale,
            "tabm_raw_rms": tabm_scale,
            "powered_blend_rms": powered_blend_scale,
            "final_rms": rms(prediction),
            "final_mean": float(np.mean(prediction)),
            "final_std": float(np.std(prediction, dtype=np.float64)),
            "final_min": float(np.min(prediction)),
            "final_max": float(np.max(prediction)),
        },
        "artifacts": component_artifacts,
        "replay_tolerances": {
            "component_rtol": 1e-6,
            "component_atol": 1e-10,
            "derived_max_abs": 2e-15,
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
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "publication": {
            "status": "not_yet_published",
            "multi_file_transactional": False,
            "individual_replacements_atomic": True,
        },
    }
    _atomic_json(manifest_path, manifest)

    # Publish only a complete generation.  Pointer replacement is last, so an
    # interrupted mirror update cannot make the incomplete generation current.
    for destination in (
        CANONICAL_SUBMISSION_PATH,
        ROOT_ARTIFACT_SUBMISSION_PATH,
        WORKSPACE_SUBMISSION_PATH,
    ):
        _atomic_copy(generation_submission_path, destination)
    published_hashes = {
        _sha256_file(path)
        for path in (
            CANONICAL_SUBMISSION_PATH,
            ROOT_ARTIFACT_SUBMISSION_PATH,
            WORKSPACE_SUBMISSION_PATH,
        )
    }
    if published_hashes != {_sha256_file(generation_submission_path)}:
        raise RuntimeError("published CSV mirrors are not byte-identical")
    pointer = {
        "status": "complete",
        "generation_id": generation_id,
        "generation_manifest": _relative(manifest_path),
        "generation_manifest_sha256": _sha256_file(manifest_path),
        "submission_sha256": _sha256_file(generation_submission_path),
        "published_paths": [
            _relative(CANONICAL_SUBMISSION_PATH),
            _relative(ROOT_ARTIFACT_SUBMISSION_PATH),
            _relative(WORKSPACE_SUBMISSION_PATH),
        ],
        "multi_file_transactional": False,
        "individual_replacements_atomic": True,
        "published_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(STABLE_POINTER_PATH, pointer)
    print(f"published frozen blend generation {generation_id}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="verify prerequisites only")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly publish over the current canonical CSV paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight:
        result = preflight()
        print(json.dumps(result, indent=2), flush=True)
        return
    try:
        train_and_publish(overwrite=args.overwrite)
    except PendingArtifactsError as exc:
        print(
            json.dumps(
                {"status": "pending", "missing": [_relative(path) for path in exc.missing]},
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
