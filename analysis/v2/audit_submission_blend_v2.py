"""Replay both frozen blend components and audit the published v2 generation.

The audit resolves the stable generation pointer, verifies every recorded hash,
loads both saved estimators, and predicts the complete test set in bounded row
chunks.  It then independently reconstructs component RMS normalization, the
60/40 blend, signed power q=1.1, final RMS normalization, and CSV serialization.
The recorded Dev3 promotion and sealed safety-veto decisions must both pass.
"""

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
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
LOCAL_DEPS = PROJECT_ROOT / ".analysis_deps"
for dependency in (LOCAL_DEPS, ANALYSIS_ROOT, ANALYSIS_ROOT / "v2"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from feature_families import materialize_feature_set  # noqa: E402
from pipeline_config import DATA_ROOT, RANDOM_SEED, TEST_SAMPLES  # noqa: E402
from v2.run_sequence_experiment import SPECS  # noqa: E402
from v2.v2_features import materialize_v2_feature_set  # noqa: E402


GBDT_FEATURE_SET = "base_plus_sequence_all"
TABM_FEATURE_SET = "multiscale_mechanics_scale"
GBDT_SPEC_NAME = "capacity"
TABM_WEIGHT = 0.6
GBDT_WEIGHT = 0.4
POST_BLEND_POWER = 1.1
GBDT_RAW_FEATURES = 754
GBDT_TRANSFORMED_FEATURES = 1_326
TABM_RAW_FEATURES = 474
TABM_TRANSFORMED_FEATURES = 836
CHUNK_ROWS = 65_536
COMPONENT_RTOL = 1e-6
COMPONENT_ATOL = 1e-10
DERIVED_MAX_ABS = 2e-15

V2_ROOT = PROJECT_ROOT / "artifacts" / "v2"
POINTER_PATH = V2_ROOT / "models" / "final_blend_pointer.json"
FROZEN_PATH = V2_ROOT / "diagnostics" / "frozen_blend_before_dev3.json"
AUDIT_PATH = V2_ROOT / "diagnostics" / "submission_blend_audit.json"
CANONICAL_SUBMISSION_PATH = V2_ROOT / "submissions" / "submission_final.csv"
ROOT_ARTIFACT_SUBMISSION_PATH = (
    PROJECT_ROOT / "artifacts" / "submissions" / "submission_final.csv"
)
WORKSPACE_SUBMISSION_PATH = PROJECT_ROOT / "submission_final.csv"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve_relative(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError("artifact path must be a nonempty string")
    path = (PROJECT_ROOT / value).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    return path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _check(checks: dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


def _finite_chunked(values: np.ndarray) -> bool:
    if values.ndim not in (1, 2) or values.size == 0 or values.shape[0] == 0:
        return False
    return all(
        np.all(np.isfinite(values[left : min(left + CHUNK_ROWS, values.shape[0])]))
        for left in range(0, values.shape[0], CHUNK_ROWS)
    )


def _rms(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not _finite_chunked(vector):
        raise ValueError("RMS input is empty or non-finite")
    return float(np.sqrt(np.mean(np.square(vector, dtype=np.float64))))


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if not _finite_chunked(vector):
        raise ValueError("signed-power input is empty or non-finite")
    return np.sign(vector) * np.power(np.abs(vector), exponent)


def _max_error(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.shape != right_array.shape
        or left_array.ndim == 0
        or left_array.size == 0
        or left_array.shape[0] == 0
    ):
        return float("inf")
    maximum = 0.0
    for start in range(0, left_array.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, left_array.shape[0])
        difference = np.abs(left_array[start:stop] - right_array[start:stop])
        if not np.all(np.isfinite(difference)):
            return float("inf")
        maximum = max(maximum, float(np.max(difference)))
    return maximum


def _verify_record(
    checks: dict[str, bool], prefix: str, record: Any, expected: Path | None = None
) -> Path | None:
    if not isinstance(record, dict):
        _check(checks, f"{prefix}_record", False)
        return None
    try:
        path = _resolve_relative(record.get("path"))
    except (TypeError, ValueError):
        _check(checks, f"{prefix}_record", False)
        return None
    _check(checks, f"{prefix}_record", True)
    if expected is not None:
        _check(checks, f"{prefix}_path", path == expected.resolve())
    exists = path.is_file()
    _check(checks, f"{prefix}_exists", exists)
    if exists:
        _check(checks, f"{prefix}_bytes", path.stat().st_size == record.get("bytes"))
        _check(checks, f"{prefix}_sha256", _sha256_file(path).lower() == str(record.get("sha256", "")).lower())
    return path


def _walk_records(value: Any, prefix: str = "record") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield prefix, value
        else:
            for key, nested in value.items():
                yield from _walk_records(nested, f"{prefix}_{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_records(nested, f"{prefix}_{index}")


def _verify_frozen(checks: dict[str, bool], frozen: dict[str, Any]) -> None:
    _check(checks, "frozen_status", frozen.get("status") == "frozen_before_dev3_outer_labels")
    _check(checks, "frozen_selection_months", frozen.get("selection_months") == [23, 46])
    _check(checks, "frozen_confirmation_months", frozen.get("confirmation_months") == [47, 58])
    _check(checks, "frozen_dev3_unwritten", frozen.get("dev3_result") is None)
    _check(checks, "frozen_sealed_months", frozen.get("sealed_audit_months") == [59, 70])
    _check(checks, "frozen_sealed_unwritten", frozen.get("sealed_result") is None)
    components = frozen.get("components", {})
    gbdt = components.get("capacity_lightgbm", {})
    tabm = components.get("tabm_mini", {})
    expected_gbdt = {
        "feature_set": GBDT_FEATURE_SET,
        "raw_feature_count": GBDT_RAW_FEATURES,
        "preprocessor": {
            "class": "RobustPreprocessor",
            "clip": 8.0,
            "add_missing_indicators": True,
        },
        "spec": asdict(SPECS[GBDT_SPEC_NAME]),
    }
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
    _check(checks, "frozen_gbdt_exact", gbdt == expected_gbdt)
    _check(checks, "frozen_tabm_exact", tabm == expected_tabm)
    _check(checks, "frozen_gbdt_feature_set", gbdt.get("feature_set") == GBDT_FEATURE_SET)
    _check(checks, "frozen_gbdt_spec", gbdt.get("spec") == asdict(SPECS[GBDT_SPEC_NAME]))
    _check(checks, "frozen_tabm_feature_set", tabm.get("feature_set") == TABM_FEATURE_SET)
    for key, expected in {
        "k": 16,
        "hidden_size": 256,
        "hidden_layers": 2,
        "dropout": 0.1,
        "batch_size": 1024,
        "learning_rate": 0.002,
        "weight_decay": 0.0003,
        "inner_validation_months": 3,
        "smooth_clip_scale": 3.0,
        "random_seed": RANDOM_SEED,
    }.items():
        _check(checks, f"frozen_tabm_{key}", tabm.get(key) == expected)
    blend = frozen.get("blend_contract", {})
    _check(
        checks,
        "frozen_blend_exact",
        blend
        == {
            "tabm_weight": TABM_WEIGHT,
            "capacity_lightgbm_weight": GBDT_WEIGHT,
            "component_normalization": "uncentered unit RMS once over the complete evaluation vector for each component",
            "post_blend_transform": "sign(p) * abs(p)^1.1",
            "final_normalization": "uncentered unit RMS once over the complete transformed blend",
            "centering": "none",
        },
    )
    _check(checks, "frozen_tabm_weight", blend.get("tabm_weight") == TABM_WEIGHT)
    _check(checks, "frozen_capacity_weight", blend.get("capacity_lightgbm_weight") == GBDT_WEIGHT)
    _check(checks, "frozen_post_power", blend.get("post_blend_transform") == "sign(p) * abs(p)^1.1")
    _check(checks, "frozen_no_centering", blend.get("centering") == "none")
    _check(
        checks,
        "frozen_selection_grid",
        frozen.get("selection_grid")
        == {
            "tabm_weights": [value / 10.0 for value in range(11)],
            "post_blend_signed_power_exponents": [1.0, 1.1, 1.2, 1.3],
        },
    )
    _check(
        checks,
        "frozen_screen_score",
        np.isclose(
            float(frozen.get("selected_dev1_dev2_cosine", np.nan)),
            0.14445559865492444,
            rtol=0.0,
            atol=1e-15,
        ),
    )
    _check(
        checks,
        "frozen_promotion_gates",
        frozen.get("predeclared_promotion_gates")
        == {
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
        },
    )
    for section in ("validation_input_sha256", "selection_artifact_sha256"):
        records = frozen.get(section, {})
        _check(checks, f"frozen_{section}_present", isinstance(records, dict) and bool(records))
        if isinstance(records, dict):
            for relative, expected_hash in records.items():
                try:
                    path = _resolve_relative(relative)
                    matches = path.is_file() and _sha256_file(path).lower() == str(expected_hash).lower()
                except (TypeError, ValueError):
                    matches = False
                _check(checks, f"frozen_hash_{str(relative).replace('/', '_')}", matches)


def _build_tabm_replay(torch: Any, input_dim: int, spec: dict[str, Any]) -> Any:
    nn = torch.nn

    class MiniEnsemble(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_scale = nn.Parameter(torch.empty(int(spec["k"]), input_dim))
            layers: list[Any] = []
            in_features = input_dim
            for _ in range(int(spec["hidden_layers"])):
                layers.extend(
                    [
                        nn.Linear(in_features, int(spec["hidden_size"])),
                        nn.ReLU(),
                        nn.Dropout(float(spec["dropout"])),
                    ]
                )
                in_features = int(spec["hidden_size"])
            self.backbone = nn.Sequential(*layers)
            self.head_weight = nn.Parameter(torch.empty(int(spec["k"]), in_features))
            self.head_bias = nn.Parameter(torch.empty(int(spec["k"])))

        def forward(self, x: Any) -> Any:
            representation = x.unsqueeze(1) * self.input_scale.unsqueeze(0)
            representation = self.backbone(representation)
            return (
                (representation * self.head_weight.unsqueeze(0)).sum(dim=-1)
                + self.head_bias.unsqueeze(0)
            )

    return MiniEnsemble()


def _smooth_clip(matrix: np.ndarray, scale: float) -> None:
    for left in range(0, matrix.shape[0], CHUNK_ROWS):
        right = min(left + CHUNK_ROWS, matrix.shape[0])
        block = matrix[left:right]
        denominator = np.sqrt(1.0 + np.square(block / scale, dtype=np.float32))
        np.divide(block, denominator, out=block)


def _replay_gbdt(
    model: Any, preprocessor: Any, raw: np.ndarray, template_ids: np.ndarray
) -> np.ndarray:
    by_row = np.empty(raw.shape[0], dtype=np.float64)
    for left in range(0, raw.shape[0], CHUNK_ROWS):
        right = min(left + CHUNK_ROWS, raw.shape[0])
        transformed = preprocessor.transform(raw[left:right])
        if not _finite_chunked(transformed):
            raise ValueError("non-finite transformed GBDT replay chunk")
        by_row[left:right] = np.asarray(model.predict(transformed), dtype=np.float64)
        del transformed
    if not _finite_chunked(by_row):
        raise ValueError("non-finite GBDT replay prediction")
    return by_row[template_ids]


def _replay_tabm(
    torch: Any,
    model: Any,
    preprocessor: Any,
    raw: np.ndarray,
    template_ids: np.ndarray,
    spec: dict[str, Any],
    target_mean: float,
    target_std: float,
    device: Any,
    use_amp: bool,
) -> np.ndarray:
    by_row = np.empty(raw.shape[0], dtype=np.float64)
    model.eval()
    inference_batch = int(spec["inference_batch_size"])
    with torch.no_grad():
        for left in range(0, raw.shape[0], CHUNK_ROWS):
            right = min(left + CHUNK_ROWS, raw.shape[0])
            transformed = np.ascontiguousarray(preprocessor.transform(raw[left:right]))
            if not _finite_chunked(transformed):
                raise ValueError("non-finite transformed TabM replay chunk")
            _smooth_clip(transformed, float(spec["smooth_clip_scale"]))
            chunk_prediction = np.empty(right - left, dtype=np.float64)
            for inner_left in range(0, transformed.shape[0], inference_batch):
                inner_right = min(inner_left + inference_batch, transformed.shape[0])
                tensor = torch.as_tensor(
                    transformed[inner_left:inner_right],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    members_tensor = model(tensor)
                members = members_tensor.float().cpu().numpy().astype(np.float64)
                chunk_prediction[inner_left:inner_right] = (
                    members * target_std + target_mean
                ).mean(axis=1)
            by_row[left:right] = chunk_prediction
            del transformed, chunk_prediction
    if not _finite_chunked(by_row):
        raise ValueError("non-finite TabM replay prediction")
    return by_row[template_ids]


def audit(*, overwrite: bool = False) -> dict[str, Any]:
    if AUDIT_PATH.exists() and not overwrite:
        raise FileExistsError(f"{AUDIT_PATH} exists; pass --overwrite")
    pointer = _read_json(POINTER_PATH)
    checks: dict[str, bool] = {}
    _check(checks, "pointer_complete", pointer.get("status") == "complete")
    _check(checks, "pointer_not_multifile_transaction", pointer.get("multi_file_transactional") is False)
    _check(checks, "pointer_individual_atomic", pointer.get("individual_replacements_atomic") is True)
    manifest_path = _resolve_relative(pointer.get("generation_manifest"))
    _check(checks, "manifest_exists", manifest_path.is_file())
    _check(checks, "manifest_pointer_hash", _sha256_file(manifest_path).lower() == str(pointer.get("generation_manifest_sha256", "")).lower())
    manifest = _read_json(manifest_path)
    _check(checks, "manifest_complete", manifest.get("status") == "complete")
    _check(checks, "generation_id_matches", manifest.get("generation_id") == pointer.get("generation_id"))
    contract = manifest.get("pipeline_contract", {})
    _check(checks, "pipeline_hash", manifest.get("pipeline_sha256") == _sha256_json(contract))
    _check(checks, "gbdt_feature_set", contract.get("feature_sets", {}).get("capacity_lightgbm") == GBDT_FEATURE_SET)
    _check(checks, "tabm_feature_set", contract.get("feature_sets", {}).get("tabm_mini") == TABM_FEATURE_SET)
    _check(checks, "gbdt_spec", contract.get("capacity_spec") == asdict(SPECS[GBDT_SPEC_NAME]))
    _check(
        checks,
        "preprocessor_contract",
        contract.get("preprocessor") == {"clip": 8.0, "add_missing_indicators": True},
    )
    blend_contract = contract.get("blend", {})
    _check(checks, "blend_tabm_weight", blend_contract.get("tabm_weight") == TABM_WEIGHT)
    _check(checks, "blend_capacity_weight", blend_contract.get("capacity_weight") == GBDT_WEIGHT)
    _check(checks, "blend_power", blend_contract.get("signed_power") == POST_BLEND_POWER)
    _check(checks, "blend_no_centering", blend_contract.get("centering") == "none")
    _check(
        checks,
        "blend_component_normalization",
        blend_contract.get("component_normalization") == "complete-vector uncentered RMS",
    )
    _check(
        checks,
        "blend_final_normalization",
        blend_contract.get("final_normalization") == "complete-vector uncentered RMS",
    )
    _check(
        checks,
        "tabm_epoch_rule",
        contract.get("tabm_epoch_rule")
        == "median of Dev1, Dev2, Dev3 inner-selected epochs",
    )
    _check(checks, "training_months", contract.get("training_months") == [0, 70])
    _check(checks, "random_seed", contract.get("random_seed") == RANDOM_SEED)

    frozen = _read_json(FROZEN_PATH)
    _check(checks, "frozen_file_hash", _sha256_file(FROZEN_PATH).lower() == str(contract.get("frozen_config_sha256", "")).lower())
    _verify_frozen(checks, frozen)
    post_freeze = manifest.get("post_freeze_verification")
    _check(checks, "post_freeze_verification_present", isinstance(post_freeze, dict))
    if not isinstance(post_freeze, dict):
        post_freeze = {}
    _check(checks, "post_freeze_status", post_freeze.get("status") == "verified")
    selected_epochs = post_freeze.get("selected_epochs")
    selected_epochs_valid = (
        isinstance(selected_epochs, dict)
        and set(selected_epochs) == {"Dev1", "Dev2", "Dev3"}
        and all(
            not isinstance(selected_epochs[name], bool)
            and isinstance(selected_epochs[name], int)
            and 1 <= selected_epochs[name] <= 8
            for name in ("Dev1", "Dev2", "Dev3")
        )
    )
    _check(
        checks,
        "post_freeze_selected_epochs",
        selected_epochs_valid,
    )
    _check(
        checks,
        "post_freeze_final_epoch",
        post_freeze.get("final_epoch") == contract.get("tabm_final_epoch"),
    )
    _check(
        checks,
        "post_freeze_epoch_median",
        selected_epochs_valid
        and post_freeze.get("final_epoch")
        == sorted(selected_epochs.values())[1],
    )
    dev3_gate = post_freeze.get("dev3_promotion_gate")
    _check(checks, "dev3_gate_present", isinstance(dev3_gate, dict))
    if not isinstance(dev3_gate, dict):
        dev3_gate = {}
    _check(checks, "dev3_gate_passed", dev3_gate.get("passed") is True)
    _check(
        checks,
        "dev3_gate_cosine",
        np.isfinite(float(dev3_gate.get("cosine", np.nan)))
        and float(dev3_gate.get("cosine", np.nan))
        > float(
            frozen.get("predeclared_promotion_gates", {})
            .get("dev3", {})
            .get("candidate_cosine_must_exceed", np.inf)
        ),
    )
    _check(
        checks,
        "dev3_gate_positive_month_share",
        float(dev3_gate.get("positive_month_share", np.nan)) >= 1.0,
    )
    dev3_monthly = dev3_gate.get("monthly_cosines")
    _check(
        checks,
        "dev3_gate_all_months_positive",
        isinstance(dev3_monthly, dict)
        and set(dev3_monthly) == {str(month) for month in range(47, 59)}
        and all(
            np.isfinite(float(value)) and float(value) > 0.0
            for value in dev3_monthly.values()
        ),
    )
    _check(
        checks,
        "dev3_not_used_for_weight_or_power_selection",
        post_freeze.get("dev3_used_for_weight_or_power_selection") is False,
    )
    _check(
        checks,
        "dev3_used_for_predeclared_promotion_only",
        post_freeze.get("dev3_used_for_predeclared_promotion_gate") is True,
    )
    sealed_veto = post_freeze.get("sealed_safety_veto")
    _check(checks, "sealed_veto_present", isinstance(sealed_veto, dict))
    if not isinstance(sealed_veto, dict):
        sealed_veto = {}
    _check(checks, "sealed_veto_status", sealed_veto.get("status") == "verified_passed")
    _check(
        checks,
        "sealed_veto_decision",
        sealed_veto.get("decision") == "publish_frozen_tabm_capacity_blend",
    )
    full_veto = sealed_veto.get("full_months_59_70")
    excluding_66_veto = sealed_veto.get("excluding_month_66")
    if not isinstance(full_veto, dict):
        full_veto = {}
    if not isinstance(excluding_66_veto, dict):
        excluding_66_veto = {}
    _check(
        checks,
        "sealed_veto_thresholds_frozen",
        sealed_veto.get("maximum_allowed_deficit") == 0.001
        and sealed_veto.get("minimum_positive_month_share")
        == 0.9166666666666666,
    )
    _check(
        checks,
        "sealed_veto_full_passed",
        full_veto.get("passed") is True
        and np.isfinite(float(full_veto.get("deficit", np.nan)))
        and float(full_veto.get("deficit", np.nan)) <= 0.001,
    )
    _check(
        checks,
        "sealed_veto_excluding_66_passed",
        excluding_66_veto.get("passed") is True
        and np.isfinite(float(excluding_66_veto.get("deficit", np.nan)))
        and float(excluding_66_veto.get("deficit", np.nan)) <= 0.001,
    )
    _check(
        checks,
        "sealed_veto_positive_month_share_passed",
        sealed_veto.get("positive_month_share_passed") is True
        and float(sealed_veto.get("positive_month_share", np.nan))
        >= 0.9166666666666666,
    )
    source_hashes = contract.get("source_sha256", {})
    _check(checks, "source_hashes_present", isinstance(source_hashes, dict) and bool(source_hashes))
    if isinstance(source_hashes, dict):
        for relative, expected_hash in source_hashes.items():
            try:
                path = _resolve_relative(relative)
                matches = path.is_file() and _sha256_file(path).lower() == str(expected_hash).lower()
            except (TypeError, ValueError):
                matches = False
            _check(checks, f"source_{str(relative).replace('/', '_')}", matches)

    for name, record in _walk_records(manifest.get("frozen_verification", {}), "frozen_record"):
        _verify_record(checks, name, record)
    for name, record in _walk_records(manifest.get("post_freeze_verification", {}), "post_freeze_record"):
        _verify_record(checks, name, record)
    for name, record in _walk_records(manifest.get("input_artifacts", {}), "input_record"):
        _verify_record(checks, name, record)

    artifacts = manifest.get("artifacts", {})
    expected_artifact_names = {
        "capacity_model",
        "capacity_preprocessor",
        "tabm_preprocessor",
        "tabm_checkpoint",
        "tabm_spec",
        "tabm_training_history",
        "prediction_artifact",
        "generation_submission",
    }
    _check(checks, "artifact_set_exact", set(artifacts) == expected_artifact_names)
    artifact_paths: dict[str, Path] = {}
    for name in expected_artifact_names:
        path = _verify_record(checks, f"artifact_{name}", artifacts.get(name))
        if path is not None:
            artifact_paths[name] = path
    required_loaded = expected_artifact_names.issubset(artifact_paths)
    if not required_loaded:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"cannot replay because artifact checks failed: {failed}")

    published_paths = [
        CANONICAL_SUBMISSION_PATH,
        ROOT_ARTIFACT_SUBMISSION_PATH,
        WORKSPACE_SUBMISSION_PATH,
    ]
    _check(checks, "published_paths_exact", pointer.get("published_paths") == [_relative(path) for path in published_paths])
    for path in published_paths:
        _check(checks, f"published_exists_{path.name}_{len(str(path))}", path.is_file())
    canonical_hash = _sha256_file(CANONICAL_SUBMISSION_PATH)
    _check(checks, "pointer_submission_hash", canonical_hash.lower() == str(pointer.get("submission_sha256", "")).lower())
    _check(checks, "root_mirror_identical", _sha256_file(ROOT_ARTIFACT_SUBMISSION_PATH) == canonical_hash)
    _check(checks, "workspace_mirror_identical", _sha256_file(WORKSPACE_SUBMISSION_PATH) == canonical_hash)
    _check(checks, "generation_csv_identical", _sha256_file(artifact_paths["generation_submission"]) == canonical_hash)

    template_path = next(
        (path for path in (DATA_ROOT / "sample_submission.csv", DATA_ROOT / "submission.csv") if path.is_file()),
        None,
    )
    if template_path is None:
        raise FileNotFoundError("organizer submission template is missing")
    template = pd.read_csv(template_path)
    first_line = CANONICAL_SUBMISSION_PATH.open("r", encoding="utf-8-sig").readline().rstrip("\r\n")
    submission = pd.read_csv(CANONICAL_SUBMISSION_PATH, float_precision="round_trip")
    _check(checks, "header_exact", first_line == "sample_id,prediction")
    _check(checks, "columns_exact", submission.columns.tolist() == ["sample_id", "prediction"])
    _check(checks, "template_columns_exact", template.columns.tolist() == ["sample_id", "prediction"])
    _check(checks, "row_count_exact", len(submission) == TEST_SAMPLES)
    _check(checks, "template_row_count_exact", len(template) == TEST_SAMPLES)
    if submission.columns.tolist() != ["sample_id", "prediction"] or template.columns.tolist() != [
        "sample_id",
        "prediction",
    ]:
        raise ValueError("submission/template schema is invalid")
    if len(submission) != TEST_SAMPLES or len(template) != TEST_SAMPLES:
        raise ValueError("submission/template row count is invalid")
    _check(
        checks,
        "submitted_sample_id_integer",
        pd.api.types.is_integer_dtype(submission["sample_id"]),
    )
    _check(
        checks,
        "template_sample_id_integer",
        pd.api.types.is_integer_dtype(template["sample_id"]),
    )
    if not pd.api.types.is_integer_dtype(
        submission["sample_id"]
    ) or not pd.api.types.is_integer_dtype(template["sample_id"]):
        raise TypeError("submission/template sample_id must be integer-valued")
    template_ids = template["sample_id"].to_numpy(dtype=np.int64, copy=False)
    submitted_ids = submission["sample_id"].to_numpy(dtype=np.int64, copy=False)
    csv_prediction = submission["prediction"].to_numpy(dtype=np.float64, copy=False)
    _check(checks, "template_order_exact", np.array_equal(submitted_ids, template_ids))
    _check(checks, "sample_ids_complete", np.array_equal(np.sort(submitted_ids), np.arange(TEST_SAMPLES)))
    _check(checks, "template_ids_complete", np.array_equal(np.sort(template_ids), np.arange(TEST_SAMPLES)))
    _check(checks, "csv_prediction_shape", csv_prediction.shape == (TEST_SAMPLES,))
    _check(checks, "csv_prediction_finite", _finite_chunked(csv_prediction))
    _check(
        checks,
        "csv_prediction_nonconstant",
        _finite_chunked(csv_prediction)
        and np.ptp(csv_prediction) > 0.0
        and np.unique(csv_prediction).size > 1,
    )

    with np.load(artifact_paths["prediction_artifact"], allow_pickle=False) as saved:
        required_arrays = {
            "sample_id",
            "capacity_raw_prediction",
            "tabm_raw_prediction",
            "capacity_component_rms",
            "tabm_component_rms",
            "capacity_unit_rms_view",
            "tabm_unit_rms_view",
            "blend_pre_power",
            "powered_blend",
            "powered_blend_rms",
            "tabm_weight",
            "capacity_weight",
            "power_exponent",
            "prediction",
        }
        arrays_present = required_arrays.issubset(saved.files)
        _check(checks, "prediction_arrays_present", arrays_present)
        if not arrays_present:
            raise ValueError("prediction artifact is missing required arrays")
        arrays = {name: np.asarray(saved[name]) for name in required_arrays}
    _check(checks, "saved_id_integer", np.issubdtype(arrays["sample_id"].dtype, np.integer))
    _check(checks, "saved_id_shape", arrays["sample_id"].shape == (TEST_SAMPLES,))
    _check(checks, "saved_ids_exact", np.array_equal(arrays["sample_id"].astype(np.int64), submitted_ids))
    for name in (
        "capacity_raw_prediction",
        "tabm_raw_prediction",
        "capacity_unit_rms_view",
        "tabm_unit_rms_view",
        "blend_pre_power",
        "powered_blend",
        "prediction",
    ):
        _check(checks, f"{name}_shape", arrays[name].shape == (TEST_SAMPLES,))
        _check(checks, f"{name}_finite", _finite_chunked(arrays[name]))
    for name in (
        "capacity_component_rms",
        "tabm_component_rms",
        "powered_blend_rms",
        "tabm_weight",
        "capacity_weight",
        "power_exponent",
    ):
        _check(checks, f"{name}_scalar_shape", arrays[name].shape == (1,))
        _check(checks, f"{name}_scalar_finite", arrays[name].shape == (1,) and np.isfinite(float(arrays[name][0])))
    if any(
        arrays[name].shape != (1,) or not np.isfinite(float(arrays[name][0]))
        for name in (
            "capacity_component_rms",
            "tabm_component_rms",
            "powered_blend_rms",
            "tabm_weight",
            "capacity_weight",
            "power_exponent",
        )
    ):
        raise ValueError("prediction artifact scalar metadata is invalid")
    _check(checks, "saved_capacity_rms_positive", float(arrays["capacity_component_rms"][0]) > 0.0)
    _check(checks, "saved_tabm_rms_positive", float(arrays["tabm_component_rms"][0]) > 0.0)
    _check(checks, "saved_powered_rms_positive", float(arrays["powered_blend_rms"][0]) > 0.0)
    _check(checks, "saved_tabm_weight", float(arrays["tabm_weight"][0]) == TABM_WEIGHT)
    _check(checks, "saved_capacity_weight", float(arrays["capacity_weight"][0]) == GBDT_WEIGHT)
    _check(checks, "saved_power", float(arrays["power_exponent"][0]) == POST_BLEND_POWER)

    gbdt_model = joblib.load(artifact_paths["capacity_model"])
    gbdt_preprocessor = joblib.load(artifact_paths["capacity_preprocessor"])
    tabm_preprocessor = joblib.load(artifact_paths["tabm_preprocessor"])
    _check(checks, "gbdt_model_feature_count", int(gbdt_model.n_features_in_) == GBDT_TRANSFORMED_FEATURES)
    expected_gbdt_params = asdict(SPECS[GBDT_SPEC_NAME])
    expected_gbdt_params.update(
        force_col_wise=True,
        deterministic=True,
        bagging_seed=RANDOM_SEED,
        feature_fraction_seed=RANDOM_SEED,
    )
    observed_gbdt_params = gbdt_model.get_params()
    _check(
        checks,
        "gbdt_model_params_frozen",
        all(
            observed_gbdt_params.get(name) == value
            for name, value in expected_gbdt_params.items()
        ),
    )
    _check(checks, "gbdt_preprocessor_feature_count", int(gbdt_preprocessor.n_features_in_) == GBDT_RAW_FEATURES)
    _check(checks, "tabm_preprocessor_feature_count", int(tabm_preprocessor.n_features_in_) == TABM_RAW_FEATURES)
    schemas = manifest.get("feature_schemas", {})
    if not isinstance(schemas, dict):
        schemas = {}
    _check(checks, "gbdt_preprocessor_schema", gbdt_preprocessor.feature_names_in_.tolist() == schemas.get("capacity_raw_names"))
    _check(checks, "tabm_preprocessor_schema", tabm_preprocessor.feature_names_in_.tolist() == schemas.get("tabm_raw_names"))
    _check(
        checks,
        "gbdt_transformed_schema",
        gbdt_preprocessor.get_feature_names_out().tolist()
        == schemas.get("capacity_transformed_names")
        and len(schemas.get("capacity_transformed_names", []))
        == GBDT_TRANSFORMED_FEATURES,
    )
    _check(
        checks,
        "tabm_transformed_schema",
        tabm_preprocessor.get_feature_names_out().tolist()
        == schemas.get("tabm_transformed_names")
        and len(schemas.get("tabm_transformed_names", []))
        == TABM_TRANSFORMED_FEATURES,
    )

    print("replaying capacity LightGBM over the complete test cache", flush=True)
    gbdt_features = materialize_v2_feature_set("test", GBDT_FEATURE_SET)
    _check(checks, "gbdt_replay_raw_shape", gbdt_features.matrix.shape == (TEST_SAMPLES, GBDT_RAW_FEATURES))
    _check(checks, "gbdt_replay_schema", gbdt_features.names == schemas.get("capacity_raw_names") and gbdt_features.kinds == schemas.get("capacity_raw_kinds"))
    replay_gbdt = _replay_gbdt(gbdt_model, gbdt_preprocessor, gbdt_features.matrix, template_ids)
    del gbdt_features, gbdt_model, gbdt_preprocessor

    print("replaying TabM-mini over the complete test caches", flush=True)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to replay TabM-mini") from exc
    checkpoint = torch.load(artifact_paths["tabm_checkpoint"], map_location="cpu", weights_only=True)
    spec_record = _read_json(artifact_paths["tabm_spec"])
    spec = checkpoint.get("spec", {})
    _check(checks, "tabm_checkpoint_spec_matches_json", spec == spec_record.get("spec"))
    _check(checks, "tabm_checkpoint_spec_matches_contract", spec == contract.get("tabm_spec"))
    _check(checks, "tabm_checkpoint_epoch", int(checkpoint.get("final_epoch", 0)) == int(contract.get("tabm_final_epoch", -1)))
    _check(checks, "tabm_checkpoint_input_dim", int(checkpoint.get("input_dim", 0)) == TABM_TRANSFORMED_FEATURES)
    target_mean = float(checkpoint.get("target_mean", np.nan))
    target_std = float(checkpoint.get("target_std", np.nan))
    _check(checks, "tabm_target_scaling_finite", np.isfinite(target_mean) and np.isfinite(target_std) and target_std > 0.0)
    trained_with_amp = bool(spec_record.get("amp", False))
    replay_device = torch.device(
        "cuda" if trained_with_amp and torch.cuda.is_available() else "cpu"
    )
    replay_use_amp = trained_with_amp and replay_device.type == "cuda"
    _check(
        checks,
        "tabm_amp_replay_capability",
        not trained_with_amp or replay_device.type == "cuda",
    )
    tabm_model = _build_tabm_replay(torch, TABM_TRANSFORMED_FEATURES, spec).to(
        replay_device
    )
    tabm_model.load_state_dict(checkpoint["state_dict"], strict=True)
    tabm_features = materialize_feature_set("test", TABM_FEATURE_SET)
    _check(checks, "tabm_replay_raw_shape", tabm_features.matrix.shape == (TEST_SAMPLES, TABM_RAW_FEATURES))
    _check(checks, "tabm_replay_schema", tabm_features.names == schemas.get("tabm_raw_names") and tabm_features.kinds == schemas.get("tabm_raw_kinds"))
    replay_tabm = _replay_tabm(
        torch,
        tabm_model,
        tabm_preprocessor,
        tabm_features.matrix,
        template_ids,
        spec,
        target_mean,
        target_std,
        replay_device,
        replay_use_amp,
    )
    del tabm_features, tabm_model, tabm_preprocessor

    tolerances = manifest.get("replay_tolerances", {})
    expected_tolerances = {
        "component_rtol": COMPONENT_RTOL,
        "component_atol": COMPONENT_ATOL,
        "derived_max_abs": DERIVED_MAX_ABS,
    }
    _check(checks, "replay_tolerances_frozen", tolerances == expected_tolerances)
    component_rtol = COMPONENT_RTOL
    component_atol = COMPONENT_ATOL
    gbdt_replay_error = _max_error(replay_gbdt, arrays["capacity_raw_prediction"])
    tabm_replay_error = _max_error(replay_tabm, arrays["tabm_raw_prediction"])
    _check(checks, "gbdt_component_replay", np.allclose(replay_gbdt, arrays["capacity_raw_prediction"], rtol=component_rtol, atol=component_atol))
    _check(checks, "tabm_component_replay", np.allclose(replay_tabm, arrays["tabm_raw_prediction"], rtol=component_rtol, atol=component_atol))

    # Reconstruct every stored view from the saved raw components.  These are
    # deterministic array operations and therefore use a near-machine-precision
    # threshold independently of model replay tolerance.
    gbdt_raw = arrays["capacity_raw_prediction"].astype(np.float64)
    tabm_raw = arrays["tabm_raw_prediction"].astype(np.float64)
    gbdt_scale = _rms(gbdt_raw)
    tabm_scale = _rms(tabm_raw)
    expected_gbdt_unit = gbdt_raw / gbdt_scale
    expected_tabm_unit = tabm_raw / tabm_scale
    expected_blend = TABM_WEIGHT * expected_tabm_unit + GBDT_WEIGHT * expected_gbdt_unit
    expected_powered = _signed_power(expected_blend, POST_BLEND_POWER)
    powered_scale = _rms(expected_powered)
    expected_prediction = expected_powered / powered_scale
    derived_limit = DERIVED_MAX_ABS
    derived_errors = {
        "capacity_unit": _max_error(expected_gbdt_unit, arrays["capacity_unit_rms_view"]),
        "tabm_unit": _max_error(expected_tabm_unit, arrays["tabm_unit_rms_view"]),
        "blend_pre_power": _max_error(expected_blend, arrays["blend_pre_power"]),
        "powered_blend": _max_error(expected_powered, arrays["powered_blend"]),
        "prediction": _max_error(expected_prediction, arrays["prediction"]),
        "csv": _max_error(expected_prediction, csv_prediction),
    }
    for name, error in derived_errors.items():
        scale = max(1.0, float(np.max(np.abs(expected_prediction)))) if name in {"prediction", "csv"} else 1.0
        _check(checks, f"derived_{name}", error <= derived_limit * scale)
    _check(checks, "capacity_rms_scalar", abs(float(arrays["capacity_component_rms"][0]) - gbdt_scale) <= derived_limit)
    _check(checks, "tabm_rms_scalar", abs(float(arrays["tabm_component_rms"][0]) - tabm_scale) <= derived_limit)
    _check(checks, "powered_rms_scalar", abs(float(arrays["powered_blend_rms"][0]) - powered_scale) <= derived_limit)
    final_rms = _rms(expected_prediction)
    _check(checks, "final_unit_rms", abs(final_rms - 1.0) <= 1e-12)
    _check(checks, "final_finite", _finite_chunked(expected_prediction))
    _check(checks, "final_nonconstant", np.ptp(expected_prediction) > 0.0 and np.unique(expected_prediction).size > 1)

    ready = all(checks.values())
    payload: dict[str, Any] = {
        "status": "ready" if ready else "failed",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "generation_id": pointer.get("generation_id"),
        "generation_manifest": _relative(manifest_path),
        "generation_manifest_sha256": _sha256_file(manifest_path),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "component_replay": {
            "capacity_max_abs_error": gbdt_replay_error,
            "tabm_max_abs_error": tabm_replay_error,
            "rtol": component_rtol,
            "atol": component_atol,
        },
        "derived_max_abs_errors": derived_errors,
        "prediction": {
            "rows": TEST_SAMPLES,
            "rms": final_rms,
            "mean": float(np.mean(expected_prediction)),
            "std": float(np.std(expected_prediction, dtype=np.float64)),
            "min": float(np.min(expected_prediction)),
            "max": float(np.max(expected_prediction)),
            "zero_count": int(np.count_nonzero(expected_prediction == 0.0)),
        },
        "submission_sha256": canonical_hash,
    }
    _atomic_json(AUDIT_PATH, payload)
    if not ready:
        raise ValueError(f"frozen blend audit failed: {payload['failed_checks']}")
    print(
        f"READY blend generation {pointer.get('generation_id')}: "
        f"{TEST_SAMPLES:,} rows, RMS={final_rms:.12f}, sha256={canonical_hash}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing blend audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
