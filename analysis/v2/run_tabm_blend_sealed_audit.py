"""Run the frozen TabM--capacity-LightGBM blend on the sealed audit once.

This entry point is intentionally narrower than a model-search script:

1. Verify ``frozen_blend_before_dev3.json`` and every SHA-256 recorded in it.
2. Read only the *inner-selected epoch* from each completed TabM development
   fold and lock the final epoch to their median.
3. Refit the unchanged TabM-mini pipeline from scratch on months 0..58.
4. Predict months 59..70 once and combine those predictions with the already
   materialized capacity-LightGBM sealed predictions under the frozen blend.
5. Report the sealed labels diagnostically and evaluate only the predeclared
   safety veto.  They never tune or modify any feature, epoch, component
   weight, transform, or normalization rule; a veto can only fall back to the
   already-frozen capacity-LightGBM q=1.2 candidate.

The actual fit is gated by ``--confirm-one-time-sealed-audit``.  The default
invocation cannot train or inspect sealed outcomes.  Outputs are isolated under
``artifacts/v2/sealed_blend`` and a non-empty output directory is never
overwritten by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
LOCAL_DEPS = PROJECT_ROOT / ".analysis_deps"
V2_SOURCE_ROOT = ANALYSIS_ROOT / "v2"
for dependency_path in (LOCAL_DEPS, ANALYSIS_ROOT, V2_SOURCE_ROOT):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modeling import (  # noqa: E402
    DEVELOPMENT_FOLDS,
    RobustPreprocessor,
    cosine_score,
    monthly_diagnostics,
    rms,
    rms_normalize,
    summarize_monthly_diagnostics,
)
from run_tabm_mini_challenger import (  # noqa: E402
    TabMMiniSpec,
    _build_tabm_mini,
    _cache_training_matrix,
    _load_labels,
    _make_preprocessor,
    _month_slice,
    _predict_members,
    _run_epoch,
    _set_determinism,
    _smooth_clip_inplace,
)
from feature_families import materialize_feature_set  # noqa: E402


FROZEN_PATH = (
    PROJECT_ROOT / "artifacts" / "v2" / "diagnostics" / "frozen_blend_before_dev3.json"
)
TABM_DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "tabm_mini"
EXPERIMENT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "experiments"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "sealed_blend"

CAPACITY_STEM = "sequence_base_plus_sequence_all_capacity_SealedAudit"
CAPACITY_PREDICTIONS_PATH = EXPERIMENT_ROOT / f"{CAPACITY_STEM}_oof.npz"
CAPACITY_SUMMARY_PATH = EXPERIMENT_ROOT / f"{CAPACITY_STEM}_summary.json"

TABM_PREDICTIONS_PATH = OUTPUT_ROOT / "tabm_sealed_predictions.npz"
BLEND_VIEWS_PATH = OUTPUT_ROOT / "sealed_blend_views.npz"
CHECKPOINT_PATH = OUTPUT_ROOT / "tabm_final_checkpoint.pt"
PREPROCESSOR_PATH = OUTPUT_ROOT / "tabm_final_preprocessor.joblib"
TRAINING_HISTORY_PATH = OUTPUT_ROOT / "tabm_final_training_history.csv"
MONTHLY_PATH = OUTPUT_ROOT / "sealed_blend_monthly.csv"
SUMMARY_PATH = OUTPUT_ROOT / "sealed_blend_summary.json"
AUDIT_LEDGER_PATH = OUTPUT_ROOT / "audit_ledger.json"

TRAIN_MONTHS = (0, 58)
SEALED_MONTHS = (59, 70)
TABM_WEIGHT = 0.60
CAPACITY_WEIGHT = 0.40
POST_BLEND_POWER = 1.10
CAPACITY_BENCHMARK_POWER = 1.20
EXPECTED_RAW_TABM_FEATURES = 474
EXPECTED_CAPACITY_FEATURE_SET = "base_plus_sequence_all"
EXPECTED_CAPACITY_SPEC_NAME = "capacity"


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


def _resolve_frozen_relative_path(relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TypeError("frozen hash path must be a non-empty string")
    candidate = (PROJECT_ROOT / Path(relative_path)).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"frozen hash path escapes project root: {relative_path}") from exc
    return candidate


def _verify_hash_map(name: str, mapping: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(mapping, dict) or not mapping:
        raise TypeError(f"{name} must be a non-empty object")
    records: dict[str, dict[str, Any]] = {}
    for relative_path, expected_hash in mapping.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"invalid frozen SHA-256 for {relative_path}")
        path = _resolve_frozen_relative_path(relative_path)
        record = _file_record(path)
        if record["sha256"].casefold() != expected_hash.casefold():
            raise ValueError(
                f"frozen SHA-256 mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {record['sha256']}"
            )
        record["frozen_expected_sha256"] = expected_hash.casefold()
        records[relative_path] = record
    return records


def _validate_frozen_contract(frozen: Mapping[str, Any]) -> None:
    if frozen.get("status") != "frozen_before_dev3_outer_labels":
        raise ValueError("blend contract was not frozen before Dev3 outer labels")
    if frozen.get("selection_months") != [23, 46]:
        raise ValueError("unexpected frozen selection months")
    if frozen.get("confirmation_months") != [47, 58]:
        raise ValueError("unexpected frozen confirmation months")
    if frozen.get("sealed_audit_months") != [59, 70]:
        raise ValueError("unexpected frozen sealed-audit months")
    if frozen.get("primary_metric") != "pooled uncentered cosine over all rows":
        raise ValueError("unexpected frozen primary metric")
    if "No value from Dev3 or months 59-70" not in str(frozen.get("guardrail", "")):
        raise ValueError("frozen guardrail is absent")

    components = frozen.get("components")
    if not isinstance(components, dict):
        raise TypeError("frozen components must be an object")
    tabm = components.get("tabm_mini")
    capacity = components.get("capacity_lightgbm")
    if not isinstance(tabm, dict) or not isinstance(capacity, dict):
        raise TypeError("frozen component definitions are incomplete")
    # Match the immutable development-run budget exactly.  The dataclass has
    # deliberately generous defaults for ad-hoc runs, whereas the frozen
    # challenger used the bounded 8/4/4 selection schedule recorded in every
    # development checkpoint and trace.
    spec = TabMMiniSpec(max_epochs=8, min_epochs=4, patience=4)
    expected_tabm = {
        "feature_set": spec.feature_set,
        "raw_feature_count": EXPECTED_RAW_TABM_FEATURES,
        "k": spec.k,
        "hidden_size": spec.hidden_size,
        "hidden_layers": spec.hidden_layers,
        "dropout": spec.dropout,
        "batch_size": spec.batch_size,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "inner_validation_months": spec.inner_val_months,
        "smooth_clip_scale": spec.smooth_clip_scale,
        "random_seed": spec.seed,
    }
    for key, expected in expected_tabm.items():
        if tabm.get(key) != expected:
            raise ValueError(f"frozen TabM {key} differs from unchanged challenger")
    if tabm.get("final_epoch_rule") != (
        "median of the three inner-selected development-fold epochs; "
        "no outer-validation or sealed label chooses the epoch"
    ):
        raise ValueError("unexpected frozen final-epoch rule")
    if tabm.get("loss") != "mean of 16 standardized-target member MSE losses":
        raise ValueError("unexpected frozen TabM loss")
    if tabm.get("inference") != "mean of 16 de-standardized member predictions":
        raise ValueError("unexpected frozen TabM inference rule")

    if capacity.get("feature_set") != EXPECTED_CAPACITY_FEATURE_SET:
        raise ValueError("unexpected frozen capacity-LightGBM feature set")
    if int(capacity.get("raw_feature_count", -1)) != 754:
        raise ValueError("unexpected frozen capacity-LightGBM feature width")
    if not isinstance(capacity.get("spec"), dict):
        raise TypeError("frozen capacity-LightGBM specification is missing")

    blend = frozen.get("blend_contract")
    if not isinstance(blend, dict):
        raise TypeError("frozen blend contract is missing")
    expected_blend = {
        "tabm_weight": TABM_WEIGHT,
        "capacity_lightgbm_weight": CAPACITY_WEIGHT,
        "component_normalization": (
            "uncentered unit RMS once over the complete evaluation vector for each component"
        ),
        "post_blend_transform": "sign(p) * abs(p)^1.1",
        "final_normalization": (
            "uncentered unit RMS once over the complete transformed blend"
        ),
        "centering": "none",
    }
    for key, expected in expected_blend.items():
        if blend.get(key) != expected:
            raise ValueError(f"frozen blend field {key!r} has changed")


def verify_frozen_preflight() -> dict[str, Any]:
    """Verify the immutable contract and every input/source hash it recorded."""

    frozen = _read_json_object(FROZEN_PATH)
    _validate_frozen_contract(frozen)
    validation_hashes = _verify_hash_map(
        "validation_input_sha256", frozen.get("validation_input_sha256")
    )
    selection_hashes = _verify_hash_map(
        "selection_artifact_sha256", frozen.get("selection_artifact_sha256")
    )
    return {
        "frozen": frozen,
        "frozen_file": _file_record(FROZEN_PATH),
        "verified_validation_inputs": validation_hashes,
        "verified_selection_artifacts": selection_hashes,
    }


def _expected_fold_ranges(fold: Any, inner_val_months: int) -> dict[str, list[int]]:
    inner_start = fold.train_end - inner_val_months + 1
    return {
        "outer_train_months": [fold.train_start, fold.train_end],
        "inner_fit_months": [fold.train_start, inner_start - 1],
        "inner_validation_months": [inner_start, fold.train_end],
        "outer_validation_months": [fold.validation_start, fold.validation_end],
    }


def _validate_inner_epoch_summary(
    summary: Mapping[str, Any], fold: Any, spec: TabMMiniSpec
) -> int:
    """Return only the epoch justified by the fold's inner validation trace."""

    if summary.get("fold") != fold.name:
        raise ValueError(f"{fold.name}: summary fold identity mismatch")
    for key, expected in _expected_fold_ranges(fold, spec.inner_val_months).items():
        if summary.get(key) != expected:
            raise ValueError(f"{fold.name}: unexpected {key}")
    if summary.get("raw_features") != EXPECTED_RAW_TABM_FEATURES:
        raise ValueError(f"{fold.name}: unexpected raw feature count")
    if summary.get("refit_from_scratch_on_all_outer_train_months") is not True:
        raise ValueError(f"{fold.name}: honest refit flag is absent")
    if summary.get("loss_contract") != "mean of k individual-member MSE values":
        raise ValueError(f"{fold.name}: TabM member-wise loss contract changed")
    if summary.get("inference_contract") != "arithmetic mean of k member predictions":
        raise ValueError(f"{fold.name}: TabM ensemble inference contract changed")

    best_epoch = summary.get("best_epoch")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= spec.max_epochs:
        raise ValueError(f"{fold.name}: invalid inner-selected epoch")
    selection_rows = summary.get("selection_epochs")
    if not isinstance(selection_rows, list) or not selection_rows:
        raise TypeError(f"{fold.name}: selection epoch trace is missing")
    epoch_values: list[int] = []
    inner_scores: list[float] = []
    for row in selection_rows:
        if not isinstance(row, dict):
            raise TypeError(f"{fold.name}: invalid selection trace row")
        if row.get("fold") != fold.name or row.get("stage") != "selection":
            raise ValueError(f"{fold.name}: invalid selection trace identity")
        epoch = row.get("epoch")
        score = float(row.get("inner_ensemble_cosine", float("nan")))
        if not isinstance(epoch, int) or not np.isfinite(score):
            raise ValueError(f"{fold.name}: invalid inner epoch evidence")
        epoch_values.append(epoch)
        inner_scores.append(score)
    if epoch_values != list(range(1, len(epoch_values) + 1)):
        raise ValueError(f"{fold.name}: non-contiguous selection epoch trace")
    # Reproduce the challenger's exact 1e-7 improvement threshold instead of
    # using an unconstrained argmax, which could disagree on a numerically tiny
    # (<1e-7) late improvement.
    reproduced_best_epoch = 0
    reproduced_best_score = -np.inf
    for epoch, score in zip(epoch_values, inner_scores, strict=True):
        if score > reproduced_best_score + 1e-7:
            reproduced_best_epoch = epoch
            reproduced_best_score = score
    if best_epoch != reproduced_best_epoch:
        raise ValueError(f"{fold.name}: best_epoch does not reproduce inner selection")
    reported_best = float(summary.get("best_inner_ensemble_cosine", float("nan")))
    if not np.isclose(reported_best, reproduced_best_score, rtol=0.0, atol=1e-14):
        raise ValueError(f"{fold.name}: best inner score does not reproduce")

    refit_rows = summary.get("refit_epochs")
    if not isinstance(refit_rows, list):
        raise TypeError(f"{fold.name}: refit epoch trace is missing")
    if [row.get("epoch") for row in refit_rows] != list(range(1, best_epoch + 1)):
        raise ValueError(f"{fold.name}: refit did not use the selected epoch count")
    return best_epoch


def _load_development_epoch_evidence(torch: Any, spec: TabMMiniSpec) -> dict[str, Any]:
    """Load only inner-epoch evidence; outer predictions/scores are non-selective."""

    rows: list[dict[str, Any]] = []
    for fold in DEVELOPMENT_FOLDS:
        stem = fold.name.lower()
        summary_path = TABM_DEVELOPMENT_ROOT / f"{stem}_summary.json"
        checkpoint_path = TABM_DEVELOPMENT_ROOT / f"{stem}_checkpoint.pt"
        prediction_path = TABM_DEVELOPMENT_ROOT / f"{stem}_predictions.npz"
        for path in (summary_path, checkpoint_path, prediction_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"all three development folds must finish before the sealed audit: {path}"
                )
        summary = _read_json_object(summary_path)
        epoch = _validate_inner_epoch_summary(summary, fold, spec)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{fold.name}: checkpoint is not a dictionary")
        if checkpoint.get("fold") != fold.name:
            raise ValueError(f"{fold.name}: checkpoint fold mismatch")
        if checkpoint.get("selected_epoch") != epoch:
            raise ValueError(f"{fold.name}: checkpoint epoch differs from inner evidence")
        if checkpoint.get("spec") != asdict(spec):
            raise ValueError(f"{fold.name}: checkpoint was not trained with unchanged TabM spec")
        if int(checkpoint.get("input_dim", -1)) != int(
            summary.get("refit_transformed_features", -2)
        ):
            raise ValueError(f"{fold.name}: checkpoint input dimension mismatch")
        if not isinstance(checkpoint.get("state_dict"), dict):
            raise TypeError(f"{fold.name}: checkpoint state_dict is absent")
        del checkpoint

        # Read alignment metadata only.  The stored outer target and outer
        # predictions are deliberately not accessed by epoch selection.
        with np.load(prediction_path, allow_pickle=False) as saved:
            if not {"row_indices", "months"}.issubset(saved.files):
                raise ValueError(f"{fold.name}: prediction artifact lacks alignment arrays")
            row_indices = np.asarray(saved["row_indices"], dtype=np.int64)
            saved_months = np.asarray(saved["months"], dtype=np.int16)
        if row_indices.ndim != 1 or saved_months.shape != row_indices.shape:
            raise ValueError(f"{fold.name}: invalid prediction alignment shapes")
        if not np.all(
            (saved_months >= fold.validation_start)
            & (saved_months <= fold.validation_end)
        ):
            raise ValueError(f"{fold.name}: prediction artifact covers wrong months")

        rows.append(
            {
                "fold": fold.name,
                "inner_selected_epoch": epoch,
                "inner_validation_months": summary["inner_validation_months"],
                "summary": _file_record(summary_path),
                "checkpoint": _file_record(checkpoint_path),
                "predictions": _file_record(prediction_path),
                "outer_scores_used_for_configuration": False,
            }
        )

    epochs = [int(row["inner_selected_epoch"]) for row in rows]
    if len(epochs) != 3:
        raise AssertionError("the final epoch rule requires exactly three folds")
    median_epoch = statistics.median(epochs)
    if not isinstance(median_epoch, int):
        raise AssertionError("median of three integer epochs must be an integer")
    return {
        "folds": rows,
        "inner_selected_epochs": epochs,
        "final_epoch": median_epoch,
        "rule": (
            "median of Dev1/Dev2/Dev3 inner-selected epochs; outer scores ignored"
        ),
    }


def _validate_feature_contract(
    matrix: np.ndarray, names: Sequence[str], kinds: Sequence[str]
) -> None:
    if matrix.ndim != 2 or matrix.dtype != np.float32:
        raise ValueError("TabM raw features must be a two-dimensional float32 matrix")
    if matrix.shape[1] != EXPECTED_RAW_TABM_FEATURES:
        raise ValueError(f"unexpected TabM raw feature width: {matrix.shape[1]}")
    if len(names) != matrix.shape[1] or len(kinds) != matrix.shape[1]:
        raise ValueError("TabM feature schema does not match matrix width")
    if len(set(names)) != len(names):
        raise ValueError("TabM feature names are not unique")
    forbidden = {"sample_id", "month", "target"}.intersection(names)
    if forbidden:
        raise ValueError(f"forbidden columns in TabM feature matrix: {sorted(forbidden)}")


def _validate_capacity_artifact(
    *,
    frozen: Mapping[str, Any],
    months: np.ndarray,
    target: np.ndarray,
    expected_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    summary = _read_json_object(CAPACITY_SUMMARY_PATH)
    frozen_capacity = frozen["components"]["capacity_lightgbm"]
    if summary.get("feature_set") != frozen_capacity["feature_set"]:
        raise ValueError("sealed capacity feature set differs from frozen contract")
    if summary.get("spec_name") != EXPECTED_CAPACITY_SPEC_NAME:
        raise ValueError("sealed capacity model spec name differs from frozen contract")
    if summary.get("spec") != frozen_capacity["spec"]:
        raise ValueError("sealed capacity parameters differ from frozen contract")
    if summary.get("raw_feature_count") != frozen_capacity["raw_feature_count"]:
        raise ValueError("sealed capacity feature width differs from frozen contract")
    folds = summary.get("folds")
    if not isinstance(folds, list) or [row.get("fold") for row in folds] != [
        "SealedAudit"
    ]:
        raise ValueError("capacity summary is not the one sealed-audit fold")

    with np.load(CAPACITY_PREDICTIONS_PATH, allow_pickle=False) as saved:
        required = {"row_indices", "months", "target", "prediction"}
        if not required.issubset(saved.files):
            raise ValueError("capacity sealed artifact lacks required arrays")
        row_indices = np.asarray(saved["row_indices"], dtype=np.int64)
        saved_months = np.asarray(saved["months"], dtype=np.int16)
        saved_target = np.asarray(saved["target"], dtype=np.float64)
        prediction = np.asarray(saved["prediction"], dtype=np.float64)
    if not np.array_equal(row_indices, expected_indices):
        raise ValueError("capacity sealed row indices are not exact")
    if not np.array_equal(saved_months, months[expected_indices]):
        raise ValueError("capacity sealed months are not aligned")
    if not np.array_equal(saved_target, target[expected_indices]):
        raise ValueError("capacity sealed targets are not aligned")
    if prediction.shape != expected_indices.shape:
        raise ValueError("capacity sealed prediction shape is wrong")
    if not np.all(np.isfinite(prediction)) or np.ptp(prediction) <= 0.0:
        raise ValueError("capacity sealed prediction is non-finite or constant")
    return prediction, {
        "summary": _file_record(CAPACITY_SUMMARY_PATH),
        "predictions": _file_record(CAPACITY_PREDICTIONS_PATH),
    }


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("signed-power input contains NaN or infinity")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("signed-power exponent must be finite and positive")
    return np.sign(prediction) * np.power(np.abs(prediction), exponent)


def _build_blend_views(
    tabm_raw: np.ndarray, capacity_raw: np.ndarray
) -> dict[str, np.ndarray]:
    tabm = np.asarray(tabm_raw, dtype=np.float64).reshape(-1)
    capacity = np.asarray(capacity_raw, dtype=np.float64).reshape(-1)
    if tabm.shape != capacity.shape or tabm.size == 0:
        raise ValueError("sealed component predictions are not aligned")
    tabm_unit = rms_normalize(tabm)
    capacity_unit = rms_normalize(capacity)
    linear = TABM_WEIGHT * tabm_unit + CAPACITY_WEIGHT * capacity_unit
    powered = _signed_power(linear, POST_BLEND_POWER)
    final = rms_normalize(powered)
    if not (
        abs(rms(tabm_unit) - 1.0) <= 1e-12
        and abs(rms(capacity_unit) - 1.0) <= 1e-12
        and abs(rms(final) - 1.0) <= 1e-12
    ):
        raise AssertionError("frozen uncentered RMS normalization failed")
    return {
        "tabm_raw": tabm,
        "capacity_lightgbm_raw": capacity,
        "tabm_unit_rms": tabm_unit,
        "capacity_lightgbm_unit_rms": capacity_unit,
        "linear_blend_tabm_0p6_capacity_0p4": linear,
        "post_blend_signed_power_q1p1": powered,
        "final_unit_rms": final,
    }


def _temporary_sibling(path: Path, suffix: str | None = None) -> Path:
    extension = path.suffix if suffix is None else suffix
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{extension}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_sibling(path)
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = _temporary_sibling(path, ".npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(path: Path, value: Any) -> None:
    temporary = _temporary_sibling(path)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_checkpoint(torch: Any, path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary_sibling(path)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _known_outputs() -> tuple[Path, ...]:
    return (
        TABM_PREDICTIONS_PATH,
        BLEND_VIEWS_PATH,
        CHECKPOINT_PATH,
        PREPROCESSOR_PATH,
        TRAINING_HISTORY_PATH,
        MONTHLY_PATH,
        SUMMARY_PATH,
        AUDIT_LEDGER_PATH,
    )


def _require_clean_output_root() -> None:
    existing = list(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.exists() else []
    if existing:
        detail = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "sealed-blend audit directory is non-empty and is never overwritten:\n  "
            + detail
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _diagnostic_rows(
    target: np.ndarray,
    months: np.ndarray,
    views: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    monthly_frames: list[pd.DataFrame] = []
    global_scores: dict[str, Any] = {}
    for name, prediction in views.items():
        diagnostics = monthly_diagnostics(target, prediction, months)
        diagnostics.insert(0, "model", name)
        monthly_frames.append(diagnostics)
        summary = summarize_monthly_diagnostics(
            diagnostics.drop(columns="model"), cosine_score(target, prediction)
        )
        global_scores[name] = summary.to_dict()
    return pd.concat(monthly_frames, ignore_index=True), global_scores


def run(*, confirm_one_time_sealed_audit: bool = False) -> dict[str, Any]:
    if not confirm_one_time_sealed_audit:
        raise PermissionError(
            "sealed audit remains closed; pass --confirm-one-time-sealed-audit only "
            "after the three development folds have completed"
        )
    _require_clean_output_root()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    # Stage 1: verify the pre-Dev3 freeze before reading any development epoch
    # metadata or any target vector.
    preflight = verify_frozen_preflight()
    frozen = preflight["frozen"]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable for frozen TabM audit") from exc

    # Stage 2: derive and hash the entire train-time configuration.  This
    # function exposes only inner-selected epochs; it deliberately ignores all
    # outer-fold targets and scores present in the development artifacts.
    # Match the immutable development-run budget exactly.  The dataclass has
    # deliberately generous defaults for ad-hoc runs, whereas the frozen
    # challenger used the bounded 8/4/4 selection schedule recorded in every
    # development checkpoint and trace.
    spec = TabMMiniSpec(max_epochs=8, min_epochs=4, patience=4)
    epoch_evidence = _load_development_epoch_evidence(torch, spec)
    final_epoch = int(epoch_evidence["final_epoch"])
    locked_configuration: dict[str, Any] = {
        "locked_before_training_or_sealed_labels_loaded": True,
        "tabm_spec": asdict(spec),
        "training_months": list(TRAIN_MONTHS),
        "sealed_prediction_months": list(SEALED_MONTHS),
        "final_epoch": final_epoch,
        "final_epoch_rule": epoch_evidence["rule"],
        "inner_selected_epochs": epoch_evidence["inner_selected_epochs"],
        "tabm_weight": TABM_WEIGHT,
        "capacity_lightgbm_weight": CAPACITY_WEIGHT,
        "component_normalization": "complete-vector uncentered unit RMS",
        "post_blend_signed_power": POST_BLEND_POWER,
        "final_normalization": "complete-vector uncentered unit RMS",
        "centering": "none",
        "random_seed": spec.seed,
    }
    locked_configuration_sha256 = _sha256_json(locked_configuration)

    # Stage 3: only after the configuration is immutable do labels enter the
    # process.  Model fitting indexes target rows from months 0..58 exclusively.
    months, target = _load_labels()
    train_slice = _month_slice(months, *TRAIN_MONTHS)
    sealed_slice = _month_slice(months, *SEALED_MONTHS)
    train_indices = np.arange(train_slice.start, train_slice.stop, dtype=np.int64)
    sealed_indices = np.arange(sealed_slice.start, sealed_slice.stop, dtype=np.int64)
    if train_indices.size + sealed_indices.size != target.size:
        raise ValueError("training and sealed month blocks do not partition labels 0..70")

    features = materialize_feature_set("train", spec.feature_set)
    _validate_feature_contract(features.matrix, features.names, features.kinds)
    raw_feature_names = list(features.names)
    raw_feature_kinds = list(features.kinds)

    transform_started = time.perf_counter()
    preprocessor: RobustPreprocessor = _make_preprocessor(
        raw_feature_names, raw_feature_kinds
    )
    preprocessor.fit(features.matrix[train_slice])
    X_train = np.ascontiguousarray(preprocessor.transform(features.matrix[train_slice]))
    X_sealed = np.ascontiguousarray(
        preprocessor.transform(features.matrix[sealed_slice])
    )
    _smooth_clip_inplace(X_train, spec.smooth_clip_scale)
    _smooth_clip_inplace(X_sealed, spec.smooth_clip_scale)
    transform_seconds = time.perf_counter() - transform_started
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_sealed)):
        raise ValueError("TabM transformed feature matrix contains NaN or infinity")
    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    del features
    gc.collect()

    y_train = target[train_slice]
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if not np.isfinite(target_std) or target_std <= 0.0:
        raise ValueError("training target has invalid standard deviation")
    y_train_standard = ((y_train - target_mean) / target_std).astype(np.float32)

    _set_determinism(torch, spec.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = _build_tabm_mini(torch, X_train.shape[1], spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    X_tensor, y_tensor, data_device, cached_on_device = _cache_training_matrix(
        torch, X_train, y_train_standard, device
    )
    if cached_on_device:
        del X_train, y_train_standard
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, final_epoch + 1):
        epoch_started = time.perf_counter()
        loss = _run_epoch(
            torch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            data_device=data_device,
            device=device,
            spec=spec,
            use_amp=use_amp,
        )
        row = {
            "stage": "final_refit",
            "epoch": epoch,
            "train_individual_member_mse_standardized": loss,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"sealed blend TabM refit {epoch:03d}/{final_epoch:03d}: "
            f"member_mse={loss:.6f} seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started

    member_standard = _predict_members(
        torch,
        model,
        X_sealed,
        device=device,
        batch_size=spec.inference_batch_size,
        k=spec.k,
        use_amp=use_amp,
    )
    member_predictions = (
        member_standard.astype(np.float64) * target_std + target_mean
    )
    tabm_raw = member_predictions.mean(axis=1)
    if (
        tabm_raw.shape != sealed_indices.shape
        or member_predictions.shape != (sealed_indices.size, spec.k)
        or not np.all(np.isfinite(member_predictions))
        or np.ptp(tabm_raw) <= 0.0
    ):
        raise ValueError("TabM sealed prediction contract failed")

    # Stage 4: the already-existing GBDT prediction and sealed target are used
    # only after training and prediction are complete.  No branch below changes
    # locked_configuration.
    capacity_raw, capacity_records = _validate_capacity_artifact(
        frozen=frozen,
        months=months,
        target=target,
        expected_indices=sealed_indices,
    )
    views = _build_blend_views(tabm_raw, capacity_raw)
    capacity_q1p2 = rms_normalize(
        _signed_power(views["capacity_lightgbm_raw"], CAPACITY_BENCHMARK_POWER)
    )
    views["capacity_lightgbm_q1p2"] = capacity_q1p2
    sealed_target = target[sealed_slice]
    sealed_month_vector = months[sealed_slice]
    monthly, diagnostic_summary = _diagnostic_rows(
        sealed_target,
        sealed_month_vector,
        {
            "tabm_mini_raw": views["tabm_raw"],
            "capacity_lightgbm_raw": views["capacity_lightgbm_raw"],
            "capacity_lightgbm_q1p2": capacity_q1p2,
            "linear_blend": views["linear_blend_tabm_0p6_capacity_0p4"],
            "frozen_blend_final_q1p1": views["final_unit_rms"],
        },
    )
    frozen_gate = frozen["predeclared_promotion_gates"][
        "sealed_audit_safety_veto_only"
    ]
    maximum_deficit = float(frozen_gate["maximum_allowed_deficit_vs_capacity_q1p2"])
    minimum_positive_share = float(frozen_gate["minimum_positive_month_share"])
    excluding_month_66 = sealed_month_vector != 66
    blend_pooled = cosine_score(sealed_target, views["final_unit_rms"])
    capacity_pooled = cosine_score(sealed_target, capacity_q1p2)
    blend_excluding_66 = cosine_score(
        sealed_target[excluding_month_66], views["final_unit_rms"][excluding_month_66]
    )
    capacity_excluding_66 = cosine_score(
        sealed_target[excluding_month_66], capacity_q1p2[excluding_month_66]
    )
    blend_monthly = monthly[monthly["model"] == "frozen_blend_final_q1p1"]
    positive_month_share = float(np.mean(blend_monthly["cosine"].to_numpy() > 0.0))
    pooled_deficit = capacity_pooled - blend_pooled
    excluding_66_deficit = capacity_excluding_66 - blend_excluding_66
    safety_gate = {
        "status": "passed" if (
            pooled_deficit <= maximum_deficit
            and excluding_66_deficit <= maximum_deficit
            and positive_month_share >= minimum_positive_share
        ) else "vetoed",
        "passed": bool(
            pooled_deficit <= maximum_deficit
            and excluding_66_deficit <= maximum_deficit
            and positive_month_share >= minimum_positive_share
        ),
        "candidate": "frozen_blend_final_q1p1",
        "benchmark": "capacity_lightgbm_q1p2",
        "maximum_allowed_deficit": maximum_deficit,
        "minimum_positive_month_share": minimum_positive_share,
        "candidate_pooled_cosine": blend_pooled,
        "benchmark_pooled_cosine": capacity_pooled,
        "pooled_deficit": pooled_deficit,
        "candidate_excluding_month_66_cosine": blend_excluding_66,
        "benchmark_excluding_month_66_cosine": capacity_excluding_66,
        "excluding_month_66_deficit": excluding_66_deficit,
        "candidate_positive_month_share": positive_month_share,
        "failure_action": frozen_gate["failure_action"],
        "selection_scope": "predeclared safety veto only; no tuning",
    }
    member_cosines = np.asarray(
        [
            cosine_score(sealed_target, member_predictions[:, member])
            for member in range(spec.k)
        ],
        dtype=np.float64,
    )
    member_correlation = np.corrcoef(member_predictions, rowvar=False)
    mean_pairwise_correlation = float(
        (member_correlation.sum() - spec.k) / (spec.k * (spec.k - 1))
    )

    # Save the complete reproducibility bundle atomically.  Nothing here
    # replaces the existing v1/v2 canonical submission or any development file.
    checkpoint = {
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "input_dim": int(X_sealed.shape[1]),
        "target_mean": target_mean,
        "target_std": target_std,
        "selected_epoch": final_epoch,
        "epoch_rule": epoch_evidence["rule"],
        "inner_selected_epochs": epoch_evidence["inner_selected_epochs"],
        "spec": asdict(spec),
        "training_months": list(TRAIN_MONTHS),
        "locked_configuration_sha256": locked_configuration_sha256,
    }
    _atomic_torch_checkpoint(torch, CHECKPOINT_PATH, checkpoint)
    _atomic_joblib(PREPROCESSOR_PATH, preprocessor)
    _atomic_csv(TRAINING_HISTORY_PATH, pd.DataFrame(history))
    _atomic_npz(
        TABM_PREDICTIONS_PATH,
        row_indices=sealed_indices,
        months=sealed_month_vector,
        target=sealed_target,
        prediction=tabm_raw,
        member_predictions=member_predictions.astype(np.float32),
    )
    _atomic_npz(
        BLEND_VIEWS_PATH,
        row_indices=sealed_indices,
        months=sealed_month_vector,
        target=sealed_target,
        **views,
    )
    _atomic_csv(MONTHLY_PATH, monthly)

    summary: dict[str, Any] = {
        "status": "complete_one_time_sealed_audit_diagnostic_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_statement": (
            "No Dev3 outer score and no month 59..70 target influenced the model, "
            "epoch, weights, transform, or normalization. Sealed results are "
            "reported once for diagnosis and are not a selection set."
        ),
        "locked_configuration": locked_configuration,
        "locked_configuration_sha256": locked_configuration_sha256,
        "epoch_evidence": epoch_evidence,
        "training_rows": int(train_indices.size),
        "sealed_rows": int(sealed_indices.size),
        "raw_feature_count": EXPECTED_RAW_TABM_FEATURES,
        "transformed_feature_count": int(X_sealed.shape[1]),
        "raw_feature_names_sha256": _sha256_json(raw_feature_names),
        "raw_feature_kinds_sha256": _sha256_json(raw_feature_kinds),
        "transformed_feature_names_sha256": _sha256_json(transformed_feature_names),
        "device": str(device),
        "amp": use_amp,
        "training_cached_on_device": cached_on_device,
        "target_standardization_fit_months": list(TRAIN_MONTHS),
        "target_mean": target_mean,
        "target_std": target_std,
        "diagnostic_scores": diagnostic_summary,
        "predeclared_safety_gate": safety_gate,
        "tabm_members_diagnostic": {
            "cosine_mean": float(member_cosines.mean()),
            "cosine_min": float(member_cosines.min()),
            "cosine_max": float(member_cosines.max()),
            "mean_pairwise_correlation": mean_pairwise_correlation,
        },
        "timing_seconds": {
            "transform": transform_seconds,
            "training": training_seconds,
            "total": time.perf_counter() - started,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "capacity_inputs": capacity_records,
        "output_paths": {
            "tabm_predictions": _relative(TABM_PREDICTIONS_PATH),
            "blend_views": _relative(BLEND_VIEWS_PATH),
            "checkpoint": _relative(CHECKPOINT_PATH),
            "preprocessor": _relative(PREPROCESSOR_PATH),
            "training_history": _relative(TRAINING_HISTORY_PATH),
            "monthly": _relative(MONTHLY_PATH),
            "audit_ledger": _relative(AUDIT_LEDGER_PATH),
        },
    }
    _atomic_json(SUMMARY_PATH, summary)

    completed_outputs = {
        key: _file_record(path)
        for key, path in {
            "tabm_predictions": TABM_PREDICTIONS_PATH,
            "blend_views": BLEND_VIEWS_PATH,
            "checkpoint": CHECKPOINT_PATH,
            "preprocessor": PREPROCESSOR_PATH,
            "training_history": TRAINING_HISTORY_PATH,
            "monthly": MONTHLY_PATH,
            "summary": SUMMARY_PATH,
        }.items()
    }
    audit_ledger: dict[str, Any] = {
        "status": "complete_one_time_non_selective_sealed_audit",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stage_order": [
            "verified pre-Dev3 frozen contract and every recorded SHA-256",
            "derived median epoch from inner-validation traces only",
            "hashed and locked all model/blend configuration",
            "loaded labels and fit TabM on months 0..58 only",
            "predicted months 59..70 without using their targets",
            "loaded aligned capacity prediction and evaluated sealed labels diagnostically",
        ],
        "configuration_locked_before_labels_loaded": True,
        "locked_configuration_sha256": locked_configuration_sha256,
        "outer_validation_scores_used_for_configuration": False,
        "sealed_labels_used_for_configuration": False,
        "sealed_results_are_selection_eligible": False,
        "sealed_results_used_only_for_predeclared_safety_veto": True,
        "predeclared_safety_gate": safety_gate,
        "frozen_contract": preflight["frozen_file"],
        "verified_validation_inputs": preflight["verified_validation_inputs"],
        "verified_selection_artifacts": preflight["verified_selection_artifacts"],
        "development_epoch_evidence": epoch_evidence,
        "capacity_sealed_inputs": capacity_records,
        "outputs": completed_outputs,
    }
    _atomic_json(AUDIT_LEDGER_PATH, audit_ledger)

    del model, optimizer, scaler, X_tensor, y_tensor, X_sealed, member_standard
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        "sealed blend diagnostic complete: "
        f"cosine={diagnostic_summary['frozen_blend_final_q1p1']['pooled_cosine']:.6f}; "
        "result is one-time and non-selective",
        flush=True,
    )
    return summary


def _smoke_test() -> None:
    tabm = np.asarray([-2.0, -0.4, 0.3, 1.2, 2.5], dtype=np.float64)
    capacity = np.asarray([-1.0, 0.2, 0.8, 1.0, 1.5], dtype=np.float64)
    views = _build_blend_views(tabm, capacity)
    assert abs(rms(views["tabm_unit_rms"]) - 1.0) < 1e-12
    assert abs(rms(views["capacity_lightgbm_unit_rms"]) - 1.0) < 1e-12
    expected_linear = (
        TABM_WEIGHT * views["tabm_unit_rms"]
        + CAPACITY_WEIGHT * views["capacity_lightgbm_unit_rms"]
    )
    assert np.array_equal(
        views["linear_blend_tabm_0p6_capacity_0p4"], expected_linear
    )
    assert abs(rms(views["final_unit_rms"]) - 1.0) < 1e-12
    assert statistics.median([7, 6, 8]) == 7
    assert _sha256_json({"b": 2, "a": 1}) == _sha256_json({"a": 1, "b": 2})
    print("sealed TabM-blend synthetic smoke tests passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--confirm-one-time-sealed-audit",
        action="store_true",
        help="perform the frozen fit and one-time diagnostic sealed audit",
    )
    mode.add_argument(
        "--verify-frozen-only",
        action="store_true",
        help="verify the immutable freeze and its hashes without fitting",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="run synthetic blend/serialization-independent checks only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        _smoke_test()
    elif args.verify_frozen_only:
        result = verify_frozen_preflight()
        print(
            "frozen blend preflight passed: "
            f"{len(result['verified_validation_inputs'])} validation inputs and "
            f"{len(result['verified_selection_artifacts'])} selection artifacts"
        )
    else:
        run(confirm_one_time_sealed_audit=True)


if __name__ == "__main__":
    main()
