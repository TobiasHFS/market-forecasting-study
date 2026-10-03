"""Assemble the bounded v2 technical-report artifact without training a model.

The builder is deliberately read-only with respect to modeling inputs and
outputs.  It consumes completed JSON/CSV evidence, synthesizes small reviewed
datasets, validates the local artifact shape, and atomically writes the one
canonical report payload expected by the Data Analytics artifact reader.

Re-running the script after deployment-stress, final-training, or submission-
audit files appear upgrades the report from ``partial`` to ``ready`` without
changing any model or prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "v2" / "report" / "artifact.json"
TITLE = "Financial Market Forecasting  -  V2 Model Decision Report"
MAX_DATASET_ROWS = 250


PATHS = {
    "selection": "artifacts/v2/diagnostics/model_selection_summary.json",
    "fold_scores": "artifacts/v2/diagnostics/model_selection_fold_scores.csv",
    "monthly_scores": "artifacts/v2/diagnostics/model_selection_monthly_scores.csv",
    "sealed_experiment": (
        "artifacts/v2/experiments/"
        "sequence_base_plus_sequence_all_capacity_SealedAudit_summary.json"
    ),
    "postmortem": "artifacts/diagnostics/postmortem/postmortem_summary.json",
    "domain_shift": "artifacts/diagnostics/postmortem/domain_shift_summary.json",
    "raw_activity": (
        "artifacts/diagnostics/postmortem/domain_shift_raw_activity.csv"
    ),
    "tabm": "artifacts/v2/tabm_mini/tabm_mini_challenger_summary.json",
    "frozen_blend": "artifacts/v2/diagnostics/frozen_blend_before_dev3.json",
    "blend_comparison": (
        "artifacts/v2/diagnostics/tabm_capacity_blend_comparison.json"
    ),
    "blend_pooled": "artifacts/v2/diagnostics/tabm_capacity_pooled_report.json",
    "sealed_blend": "artifacts/v2/sealed_blend/sealed_blend_summary.json",
    "deployment_stress": (
        "artifacts/v2/diagnostics/deployment_stress_summary.json"
    ),
    "deployment_curve": (
        "artifacts/v2/diagnostics/deployment_stress_curve.csv"
    ),
    "blend_pointer": "artifacts/v2/models/final_blend_pointer.json",
    "blend_audit": "artifacts/v2/diagnostics/submission_blend_audit.json",
    "final_manifest": "artifacts/v2/models/final_training_manifest.json",
    "final_audit": "artifacts/v2/diagnostics/submission_audit.json",
    "final_submission": "artifacts/v2/submissions/submission_final.csv",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _path(root: Path, key: str) -> Path:
    return root / PATHS[key]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_project_path(root: Path, value: Any) -> tuple[Path | None, str | None]:
    """Resolve a producer-recorded relative path without escaping the project."""

    if not isinstance(value, str) or not value.strip():
        return None, "generation manifest path is missing"
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", value):
        return None, "generation manifest path is unsafe"
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, "generation manifest path escapes the project root"
    return candidate, None


def _first_value(obj: dict[str, Any] | None, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _dig(obj, *path)
        if value is not None:
            return value
    return None


def _final_publication(root: Path) -> dict[str, Any]:
    """Prefer the stable blend pointer while retaining the single-model contract."""

    pointer_path = _path(root, "blend_pointer")
    blend_audit_path = _path(root, "blend_audit")
    if pointer_path.is_file():
        pointer = _read_json(pointer_path)
        manifest_path, error = _safe_project_path(
            root, (pointer or {}).get("generation_manifest")
        )
        manifest = _read_json(manifest_path) if manifest_path and manifest_path.is_file() else None
        return {
            "mode": "blend",
            "pointer": pointer,
            "pointer_path": pointer_path,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "audit": _read_json(blend_audit_path),
            "audit_path": blend_audit_path,
            "resolution_error": error,
        }

    manifest_path = _path(root, "final_manifest")
    audit_path = _path(root, "final_audit")
    return {
        "mode": "single_model",
        "pointer": None,
        "pointer_path": pointer_path,
        "manifest": _read_json(manifest_path),
        "manifest_path": manifest_path,
        "audit": _read_json(audit_path),
        "audit_path": audit_path,
        "resolution_error": None,
    }


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


def _coerce_csv_value(value: str) -> str | int | float | bool | None:
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INT_RE.fullmatch(stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if _FLOAT_RE.fullmatch(stripped):
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        return number if math.isfinite(number) else None
    return stripped


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _dig(obj: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _fmt(value: Any, digits: int = 6, missing: str = "not available") -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if number is not None else missing


def _delta(a: Any, b: Any) -> float | None:
    left, right = _number(a), _number(b)
    return None if left is None or right is None else left - right


def _duckdb_source(
    source_id: str,
    label: str,
    relative_path: str,
    *,
    description: str,
    filters: Iterable[str] = (),
    metric_definitions: Iterable[str] = (),
) -> dict[str, Any]:
    suffix = PurePosixPath(relative_path).suffix.lower()
    if suffix == ".csv":
        sql = f"SELECT * FROM read_csv_auto('{relative_path}');"
    elif suffix == ".json":
        sql = f"SELECT * FROM read_json_auto('{relative_path}');"
    else:
        sql = None
    source: dict[str, Any] = {
        "id": source_id,
        "label": label,
        "path": relative_path,
    }
    if sql:
        source["query"] = {
            "engine": "DuckDB",
            "language": "SQL",
            "sql": sql,
            "description": description,
            "tables_used": [relative_path],
            "filters": list(filters),
            "metric_definitions": list(metric_definitions),
        }
    return source


def _web_source(source_id: str, label: str, href: str) -> dict[str, str]:
    return {"id": source_id, "label": label, "href": href}


def _fold_rows(
    rows: list[dict[str, Any]], *, final_is_blend: bool = False
) -> list[dict[str, Any]]:
    period_order = {"Dev1": 1, "Dev2": 2, "Dev3": 3, "Development pooled": 4}
    baseline = {
        str(row.get("period")): _number(row.get("cosine"))
        for row in rows
        if row.get("model") == "v1 slow raw"
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        period = str(row.get("period", ""))
        model = str(row.get("model", ""))
        cosine = _number(row.get("cosine"))
        if model == "v1 slow raw":
            family = "Archived v1"
        elif model == "v2 final q=1.2":
            family = "V2 capacity component" if final_is_blend else "V2 selected"
        else:
            family = "V2 ablation"
        result.append(
            {
                "period": period,
                "period_order": period_order.get(period, 99),
                "month_start": row.get("month_start"),
                "month_end": row.get("month_end"),
                "model": model,
                "model_family": family,
                "rows": row.get("rows"),
                "cosine": cosine,
                "v1_cosine_same_period": baseline.get(period),
                "uplift_vs_v1": _delta(cosine, baseline.get(period)),
                "selected_submission_view": (
                    model == "v2 final q=1.2" and not final_is_blend
                ),
            }
        )
    return result


def _tabm_rows(
    summary: dict[str, Any] | None,
    fold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not summary:
        return []
    v2_by_fold = {
        str(row["period"]): _number(row["cosine"])
        for row in fold_rows
        if row.get("model") == "v2 final q=1.2"
    }
    rows: list[dict[str, Any]] = []
    for fold in summary.get("folds", []):
        if not isinstance(fold, dict):
            continue
        fold_name = str(fold.get("fold", ""))
        raw = _number(fold.get("outer_ensemble_cosine_raw"))
        power = _number(fold.get("outer_ensemble_cosine_power"))
        selected_v2 = v2_by_fold.get(fold_name)
        rows.append(
            {
                "fold": fold_name,
                "device": fold.get("device"),
                "best_epoch": fold.get("best_epoch"),
                "outer_rows": fold.get("outer_validation_rows"),
                "tabm_raw_cosine": raw,
                "tabm_q1p2_cosine": power,
                "selected_v2_q1p2_cosine": selected_v2,
                "tabm_raw_minus_selected_v2": _delta(raw, selected_v2),
                "complete_development_oof": bool(
                    summary.get("complete_development_oof", False)
                ),
            }
        )
    return rows


def _sources(
    *,
    final_manifest_path: str = PATHS["final_manifest"],
    final_audit_path: str = PATHS["final_audit"],
    final_mode: str = "single_model",
) -> list[dict[str, Any]]:
    return [
        _duckdb_source(
            "v2_selection",
            "Frozen v2 model-selection summary",
            PATHS["selection"],
            description="Read the complete frozen model-selection and sealed-audit summary.",
            filters=(
                "Dev1-Dev2 selected the model and power transform.",
                "Dev3 was confirmation only; months 59-70 were a one-time sealed audit.",
            ),
            metric_definitions=(
                "Pooled cosine is dot(prediction,target)/(L2(prediction)*L2(target)) over all rows in the stated block.",
                "Bootstrap intervals resample circular three-month blocks and describe historical validation stability.",
            ),
        ),
        _duckdb_source(
            "v2_fold_scores",
            "Chronological fold and model scores",
            PATHS["fold_scores"],
            description="Read exact fold/model cosine values used in the v2 comparison.",
            filters=("Development folds cover labeled months 23-58.",),
            metric_definitions=(
                "Uplift versus v1 is candidate cosine minus the archived v1 cosine in the same fold.",
            ),
        ),
        _duckdb_source(
            "v2_monthly_scores",
            "Monthly v2 validation scores",
            PATHS["monthly_scores"],
            description="Read exact monthly cosine diagnostics for all compared models.",
            filters=("Monthly scores are diagnostic; pooled cosine is the primary metric.",),
        ),
        _duckdb_source(
            "v2_sealed_experiment",
            "V2 one-time sealed-audit experiment",
            PATHS["sealed_experiment"],
            description="Read the selected LightGBM specification and sealed-audit result.",
            filters=("Months 59-70; not used to select model or transform.",),
        ),
        _duckdb_source(
            "public_postmortem",
            "Archived v1 public-score postmortem",
            PATHS["postmortem"],
            description="Read the verified v1 public score, rank, validation gaps, and robustness notes.",
            metric_definitions=(
                "The public score is the leaderboard cosine reported by the competition after the archived v1 submission.",
            ),
        ),
        _duckdb_source(
            "domain_shift",
            "Train-test domain-shift audit",
            PATHS["domain_shift"],
            description="Read held-out domain-classifier and event-activity diagnostics.",
            filters=("Recent labeled comparison uses months 59-70.",),
            metric_definitions=(
                "Domain AUC is held-out ROC AUC for classifying labeled versus test feature rows; it does not establish concept drift.",
            ),
        ),
        _duckdb_source(
            "raw_activity",
            "Raw market, order, and trade activity shift",
            PATHS["raw_activity"],
            description="Read train/test rows-per-sample activity comparisons by raw event source.",
            metric_definitions=(
                "Activity change percent is 100*(test mean rows per sample/train mean rows per sample - 1).",
            ),
        ),
        _duckdb_source(
            "tabm_challenger",
            "Honest chronological TabM-mini challenger",
            PATHS["tabm"],
            description="Read completed TabM-mini fold diagnostics and run-completion status.",
            filters=(
                "Inner months select epoch only; each outer model is refit from scratch on all prior months.",
            ),
        ),
        _duckdb_source(
            "frozen_blend",
            "Frozen TabM-mini/capacity-LightGBM blend contract",
            PATHS["frozen_blend"],
            description="Read the weights, component normalization, post-blend transform, and predeclared gates frozen before Dev3 labels.",
            filters=("Dev1-Dev2 selected one candidate; Dev3 was confirmation only.",),
        ),
        _duckdb_source(
            "blend_comparison",
            "Frozen blend Dev3 confirmation and pooled report",
            PATHS["blend_comparison"],
            description="Read the independently reproduced screen result, one-shot Dev3 gate, and pooled development score.",
        ),
        _duckdb_source(
            "blend_pooled",
            "Frozen blend pooled development metrics",
            PATHS["blend_pooled"],
            description="Read reporting-only Dev1-Dev3 blend metrics and the median-epoch final-refit rule.",
        ),
        _duckdb_source(
            "sealed_blend",
            "Frozen blend one-time sealed diagnostic",
            PATHS["sealed_blend"],
            description="Read the months 59-70 diagnostics and predeclared keep/fallback safety-veto result for both components and the frozen blend.",
            filters=("Sealed results may only trigger the predeclared fallback; they cannot retune the blend.",),
        ),
        _duckdb_source(
            "v2_deployment_stress",
            "V2 repeated-origin deployment-stress summary",
            PATHS["deployment_stress"],
            description="Read fixed-model forecast-age diagnostics through 38 months.",
            filters=(
                "Only development months 0-58 are used; sealed months are untouched.",
                "Stress results are diagnostic and prohibited from model selection.",
            ),
        ),
        _duckdb_source(
            "v2_deployment_curve",
            "V2 forecast-age stress curve",
            PATHS["deployment_curve"],
            description="Read point and cumulative cosine by deployment horizon.",
            filters=("Horizons 1-38 months; repeated historical training origins.",),
        ),
        _duckdb_source(
            "v2_final_pointer",
            "Stable v2 final-publication pointer",
            PATHS["blend_pointer"],
            description="Resolve the immutable generation manifest and canonical submission hash for the published blend.",
        ),
        _duckdb_source(
            "v2_final_manifest",
            (
                "Frozen v2 blend generation manifest"
                if final_mode == "blend"
                else "Frozen v2 single-model final-training manifest"
            ),
            final_manifest_path,
            description="Read the immutable final pipeline contract and output hashes.",
        ),
        _duckdb_source(
            "v2_submission_audit",
            (
                "V2 frozen-blend submission integrity audit"
                if final_mode == "blend"
                else "V2 single-model submission integrity audit"
            ),
            final_audit_path,
            description="Read final CSV, component replay, checksum, and round-trip audit checks.",
            metric_definitions=(
                "Ready means every recorded schema, alignment, finiteness, hash, model-load, and transform-contract check passed.",
            ),
        ),
        _web_source(
            "tabm_paper",
            "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling",
            "https://arxiv.org/abs/2410.24210",
        ),
        _web_source(
            "realmlp_paper",
            "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data",
            "https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf",
        ),
        _web_source(
            "tabred_paper",
            "TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks",
            "https://proceedings.iclr.cc/paper_files/paper/2025/file/571799482291411607c54984153190b0-Paper-Conference.pdf",
        ),
        _web_source(
            "temporal_shift_paper",
            "Position: A Call for Better Tabular Benchmarks under Temporal Distribution Shift",
            "https://proceedings.mlr.press/v267/cai25j.html",
        ),
        _web_source(
            "ofi_paper",
            "The Price Impact of Order Book Events (order-flow imbalance)",
            "https://arxiv.org/abs/1011.6402",
        ),
        _web_source(
            "deeplob_paper",
            "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books",
            "https://arxiv.org/abs/1808.03668",
        ),
    ]


def _issue(
    issue_id: str,
    source_id: str,
    message: str,
) -> dict[str, str]:
    return {
        "id": issue_id,
        "scope": "report-completion",
        "sourceId": source_id,
        "message": message,
    }


def build_artifact(
    project_root: Path = PROJECT_ROOT,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build but do not write a v2 report artifact."""

    root = project_root.resolve()
    selection = _read_json(_path(root, "selection"))
    postmortem = _read_json(_path(root, "postmortem"))
    domain = _read_json(_path(root, "domain_shift"))
    tabm = _read_json(_path(root, "tabm"))
    frozen_blend = _read_json(_path(root, "frozen_blend"))
    blend_comparison = _read_json(_path(root, "blend_comparison"))
    blend_pooled = _read_json(_path(root, "blend_pooled"))
    sealed_blend = _read_json(_path(root, "sealed_blend"))
    stress = _read_json(_path(root, "deployment_stress"))
    publication = _final_publication(root)
    final_pointer = publication["pointer"]
    final_manifest = publication["manifest"]
    final_audit = publication["audit"]
    final_mode = str(publication["mode"])

    fold_data = _fold_rows(
        _read_csv(_path(root, "fold_scores")),
        final_is_blend=final_mode == "blend" or bool(blend_comparison),
    )
    activity_data = _read_csv(_path(root, "raw_activity"))
    stress_curve = _read_csv(_path(root, "deployment_curve"))
    tabm_data = _tabm_rows(tabm, fold_data)

    development = _dig(selection, "development") or {}
    sealed = _dig(selection, "sealed_audit_descriptive_only") or {}
    public = _dig(postmortem, "public_leaderboard") or {}
    domain_recent = _dig(domain, "recent_train_test_domain_classifier") or {}

    selected_dev = _number(development.get("v2_final_q1p2_cosine"))
    v1_dev = _number(development.get("v1_raw_cosine"))
    raw_uplift = _number(development.get("joint_minus_v1_raw"))
    sequence_increment = _number(development.get("sequence_increment_raw"))
    capacity_ci = _dig(development, "capacity_bootstrap_vs_v1") or {}
    sequence_ci = _dig(development, "sequence_bootstrap_vs_base_capacity") or {}
    sealed_final = _number(sealed.get("v2_final_q1p2_cosine"))
    sealed_v1 = _number(sealed.get("v1_raw_cosine"))
    recent_auc = _number(domain_recent.get("auc"))

    frozen_contract = _dig(frozen_blend, "blend_contract") or {}
    selected_blend = _dig(blend_comparison, "selected") or frozen_contract
    blend_dev3 = _number(_dig(blend_comparison, "dev3_audit", "cosine"))
    blend_dev3_gate = _dig(
        blend_comparison, "dev3_audit", "promotion_gate_passed"
    )
    blend_development = _number(
        _first_value(
            blend_pooled,
            ("pooled_dev1_dev2_dev3_cosine",),
            ("pooled_development_cosine",),
        )
    )
    if blend_development is None:
        blend_development = _number(
            _dig(blend_comparison, "pooled_reporting", "cosine")
        )
    sealed_blend_view = _dig(
        sealed_blend, "diagnostic_scores", "frozen_blend_final_q1p1"
    ) or {}
    blend_sealed = _number(
        sealed_blend_view.get("pooled_cosine")
        if isinstance(sealed_blend_view, dict)
        else None
    )
    blend_active = final_mode == "blend" or bool(blend_comparison)

    stress_views = _dig(stress, "prediction_views_diagnostic_only") or {}
    stress_power = stress_views.get("signed_power_q1p2", {}) if isinstance(stress_views, dict) else {}
    stress_pooled = _number(
        stress_power.get("all_forecasts_pooled_cosine")
        if isinstance(stress_power, dict)
        else None
    )
    final_rows = _number(
        _first_value(
            final_audit,
            ("rows",),
            ("row_count",),
            ("n_rows",),
            ("prediction", "rows"),
        )
    )
    final_hash = _first_value(
        final_audit,
        ("submission_sha256",),
        ("canonical_submission_sha256",),
        ("outputs", "submission_sha256"),
        ("submission", "sha256"),
    )
    manifest_submission_hash = _first_value(
        final_manifest,
        ("artifacts", "generation_submission", "sha256"),
        ("outputs", "canonical_submission", "sha256"),
        ("outputs", "submission", "sha256"),
        ("outputs", "submission_sha256"),
    )
    expected_submission_hash = (
        _first_value(final_pointer, ("submission_sha256",))
        if final_mode == "blend"
        else manifest_submission_hash
    )
    audit_checks = _dig(final_audit, "checks") or {}
    all_audit_checks_pass = bool(audit_checks) and all(
        value is True for value in audit_checks.values()
    )
    manifest_path = publication.get("manifest_path")
    manifest_file_hash = (
        _sha256_file(manifest_path) if isinstance(manifest_path, Path) else None
    )
    submission_file_hash = _sha256_file(_path(root, "final_submission"))
    publication_checks = {
        "manifest_complete": bool(final_manifest and final_manifest.get("status") == "complete"),
        "audit_ready": bool(final_audit and final_audit.get("status") == "ready"),
        "all_audit_checks_pass": all_audit_checks_pass,
        "row_count_recorded": final_rows is not None,
        "submission_exists": _path(root, "final_submission").is_file(),
        "submission_hash_matches_audit": bool(
            final_hash
            and submission_file_hash
            and str(final_hash).lower() == str(submission_file_hash).lower()
        ),
        "submission_hash_matches_publication": bool(
            final_hash
            and expected_submission_hash
            and str(final_hash).lower() == str(expected_submission_hash).lower()
        ),
    }
    if final_mode == "blend":
        publication_checks.update(
            {
                "pointer_complete": bool(
                    final_pointer and final_pointer.get("status") == "complete"
                ),
                "generation_manifest_hash_matches_pointer": bool(
                    manifest_file_hash
                    and _dig(final_pointer, "generation_manifest_sha256")
                    and str(manifest_file_hash).lower()
                    == str(_dig(final_pointer, "generation_manifest_sha256")).lower()
                ),
                "generation_manifest_hash_matches_audit": bool(
                    manifest_file_hash
                    and _dig(final_audit, "generation_manifest_sha256")
                    and str(manifest_file_hash).lower()
                    == str(_dig(final_audit, "generation_manifest_sha256")).lower()
                ),
                "generation_ids_match": bool(
                    _dig(final_pointer, "generation_id")
                    and _dig(final_pointer, "generation_id")
                    == _dig(final_manifest, "generation_id")
                    == _dig(final_audit, "generation_id")
                ),
                "generation_manifest_paths_match": bool(
                    _dig(final_pointer, "generation_manifest")
                    and _dig(final_pointer, "generation_manifest")
                    == _dig(final_audit, "generation_manifest")
                ),
                "generation_submission_hash_matches_pointer": bool(
                    manifest_submission_hash
                    and expected_submission_hash
                    and str(manifest_submission_hash).lower()
                    == str(expected_submission_hash).lower()
                ),
            }
        )
    final_ready = all(publication_checks.values())

    reported_development = (
        blend_development if blend_active and blend_development is not None else selected_dev
    )
    reported_sealed = (
        blend_sealed if blend_active and blend_sealed is not None else sealed_final
    )

    headline = {
        "selected_development_cosine": reported_development,
        "v1_development_cosine": v1_dev,
        "selected_minus_v1_development": _delta(reported_development, v1_dev),
        "raw_capacity_uplift": raw_uplift,
        "raw_capacity_ci_low": _number(capacity_ci.get("confidence_low")),
        "raw_capacity_ci_high": _number(capacity_ci.get("confidence_high")),
        "sequence_increment": sequence_increment,
        "sequence_ci_low": _number(sequence_ci.get("confidence_low")),
        "sequence_ci_high": _number(sequence_ci.get("confidence_high")),
        "sealed_cosine_q1p2": reported_sealed,
        "sealed_minus_v1": _delta(reported_sealed, sealed_v1),
        "blend_screen_cosine": _number(selected_blend.get("screen_cosine")),
        "blend_dev3_cosine": blend_dev3,
        "blend_development_cosine": blend_development,
        "blend_tabm_weight": _number(selected_blend.get("tabm_weight")),
        "blend_capacity_weight": _number(
            _first_value(
                selected_blend,
                ("capacity_lightgbm_weight",),
                ("capacity_weight",),
            )
        ),
        "blend_power": _number(
            _first_value(
                selected_blend,
                ("power_exponent",),
                ("signed_power",),
            )
        ),
        "public_v1_score": _number(public.get("score")),
        "public_rank": _number(public.get("rank")),
        "public_participants": _number(public.get("participants")),
        "recent_train_test_auc": recent_auc,
        "auc_above_chance": _delta(recent_auc, 0.5),
        "deployment_stress_q1p2": stress_pooled,
        "submission_rows": final_rows,
        "submission_ready": final_ready,
        "submission_sha256": str(final_hash) if final_hash is not None else None,
    }

    issues: list[dict[str, str]] = []
    required_now = [
        ("selection", "v2_selection", "Frozen v2 selection summary is missing."),
        ("fold_scores", "v2_fold_scores", "Fold/model score table is missing."),
        ("postmortem", "public_postmortem", "Archived public-score baseline is missing."),
        ("domain_shift", "domain_shift", "Domain-shift diagnostic is missing."),
    ]
    expected_later = [
        (
            "deployment_stress",
            "v2_deployment_stress",
            "V2 repeated-origin deployment stress has not completed yet.",
        ),
        (
            "deployment_curve",
            "v2_deployment_curve",
            "V2 forecast-age stress curve has not completed yet.",
        ),
        (
            "final_submission",
            "v2_submission_audit",
            "Canonical artifacts/v2/submissions/submission_final.csv is not present yet.",
        ),
    ]
    for key, source_id, message in required_now + expected_later:
        if not _path(root, key).is_file():
            issues.append(_issue(f"missing_{key}", source_id, message))
    if blend_active:
        for key, source_id, message in (
            (
                "frozen_blend",
                "frozen_blend",
                "The pre-Dev3 frozen blend contract is missing.",
            ),
            (
                "blend_comparison",
                "blend_comparison",
                "The frozen blend Dev3 confirmation artifact is missing.",
            ),
            (
                "blend_pooled",
                "blend_pooled",
                "The reporting-only pooled blend artifact is missing.",
            ),
        ):
            if not _path(root, key).is_file():
                issues.append(_issue(f"missing_{key}", source_id, message))
    if publication.get("resolution_error"):
        issues.append(
            _issue(
                "invalid_final_manifest_pointer",
                "v2_final_pointer",
                str(publication["resolution_error"]),
            )
        )
    if final_mode == "blend" and not publication["pointer_path"].is_file():
        issues.append(
            _issue(
                "missing_blend_pointer",
                "v2_final_pointer",
                "Stable v2 blend publication pointer is missing.",
            )
        )
    if final_manifest is None:
        issues.append(
            _issue(
                "missing_final_manifest",
                "v2_final_manifest",
                (
                    "The generation manifest referenced by the stable blend pointer is missing."
                    if final_mode == "blend"
                    else "Frozen v2 single-model final-training manifest has not been produced yet."
                ),
            )
        )
    if final_audit is None:
        issues.append(
            _issue(
                "missing_final_audit",
                "v2_submission_audit",
                (
                    "Frozen-blend submission integrity audit has not been produced yet."
                    if final_mode == "blend"
                    else "Final v2 single-model submission integrity audit has not passed yet."
                ),
            )
        )
    elif not final_ready:
        failed_publication_checks = [
            name for name, passed in publication_checks.items() if not passed
        ]
        issues.append(
            _issue(
                "final_audit_not_ready",
                "v2_submission_audit",
                "The final v2 publication chain is incomplete: "
                + ", ".join(failed_publication_checks),
            )
        )
    if final_manifest and final_manifest.get("status") != "complete":
        issues.append(
            _issue(
                "final_manifest_not_complete",
                "v2_final_manifest",
                "The final-training manifest exists but is not complete.",
            )
        )

    cards: list[dict[str, Any]] = []
    card_ids: list[str] = []

    def add_card(
        card_id: str,
        source_id: str,
        description: str,
        metrics: list[dict[str, Any]],
    ) -> None:
        if _number(headline.get(metrics[0]["field"])) is None:
            return
        cards.append(
            {
                "id": card_id,
                "description": description,
                "dataset": "headline",
                "sourceId": source_id,
                "metrics": metrics,
            }
        )
        card_ids.append(card_id)

    add_card(
        "selected_development",
        "blend_pooled" if blend_active else "v2_selection",
        (
            "Pooled uncentered cosine over Dev1-Dev3 for the frozen 60/40 TabM-mini/capacity-LightGBM blend."
            if blend_active
            else "Pooled uncentered cosine over Dev1-Dev3 after the frozen q=1.2 transform."
        ),
        [
            {"label": "Selected development cosine", "field": "selected_development_cosine", "format": "number"},
            {"label": "vs v1", "field": "selected_minus_v1_development", "format": "number", "signed": True},
        ],
    )
    add_card(
        "raw_capacity_uplift",
        "v2_selection",
        "Raw pooled gain from the higher-capacity base-plus-path LightGBM over archived v1.",
        [
            {"label": "Raw capacity uplift", "field": "raw_capacity_uplift", "format": "number", "signed": True},
            {"label": "95% block CI low", "field": "raw_capacity_ci_low", "format": "number", "signed": True},
            {"label": "95% block CI high", "field": "raw_capacity_ci_high", "format": "number", "signed": True},
        ],
    )
    add_card(
        "sequence_increment",
        "v2_selection",
        "Increment from the 280 fixed-time path features with model capacity held fixed.",
        [
            {"label": "Path-feature increment", "field": "sequence_increment", "format": "number", "signed": True},
            {"label": "95% block CI low", "field": "sequence_ci_low", "format": "number", "signed": True},
            {"label": "95% block CI high", "field": "sequence_ci_high", "format": "number", "signed": True},
        ],
    )
    add_card(
        "sealed_audit",
        "sealed_blend" if blend_active and blend_sealed is not None else "v2_selection",
        (
            "One-time months 59-70 audit used only for the predeclared keep/fallback safety veto; it did not retune the candidate."
            if blend_active and blend_sealed is not None
            else "One-time months 59-70 audit, recorded only after the model and transform were frozen."
        ),
        [
            {
                "label": "Sealed blend cosine" if blend_active and blend_sealed is not None else "Sealed cosine q=1.2",
                "field": "sealed_cosine_q1p2",
                "format": "number",
            },
            {"label": "vs sealed v1", "field": "sealed_minus_v1", "format": "number", "signed": True},
        ],
    )
    if blend_active:
        add_card(
            "blend_confirmation",
            "blend_comparison",
            "One frozen TabM-mini/capacity-LightGBM candidate was evaluated on Dev3 without a confirmation-set grid.",
            [
                {"label": "Dev3 blend cosine", "field": "blend_dev3_cosine", "format": "number"},
                {"label": "TabM weight", "field": "blend_tabm_weight", "format": "number"},
                {"label": "Post-blend q", "field": "blend_power", "format": "number"},
            ],
        )
    add_card(
        "public_v1",
        "public_postmortem",
        "Observed public leaderboard result for the archived v1 submission; not an estimate of v2.",
        [
            {"label": "Archived v1 public cosine", "field": "public_v1_score", "format": "number"},
            {"label": "Public rank", "field": "public_rank", "format": "number"},
            {"label": "Participants", "field": "public_participants", "format": "number"},
        ],
    )
    add_card(
        "recent_domain_auc",
        "domain_shift",
        "Held-out separability of recent labeled months 59-70 from test; this measures covariate shift, not target drift.",
        [
            {"label": "Recent-train vs test AUC", "field": "recent_train_test_auc", "format": "number"},
            {"label": "Above chance", "field": "auc_above_chance", "format": "number", "signed": True},
        ],
    )
    add_card(
        "deployment_stress",
        "v2_deployment_stress",
        (
            "Pooled q=1.2 capacity-component score over repeated historical fixed-model forecasts through 38 months; it is not a blend stress test."
            if blend_active
            else "Pooled q=1.2 score over repeated historical fixed-model forecasts through 38 months; diagnostic only."
        ),
        [
            {"label": "38-month stress cosine", "field": "deployment_stress_q1p2", "format": "number"},
        ],
    )
    add_card(
        "final_submission_rows",
        "v2_submission_audit",
        "Rows in the canonical v2 CSV after every integrity and reproducibility check passed.",
        [
            {"label": "Audited submission rows", "field": "submission_rows", "format": "number"},
        ],
    )

    charts: list[dict[str, Any]] = [
        {
            "id": "fold_model_cosine",
            "title": "Chronological validation cosine by fold and model",
            "subtitle": (
                "The q=1.2 capacity component is above archived v1 in Dev1, Dev2, and Dev3; the final blend is reported separately."
                if blend_active
                else "The frozen v2 q=1.2 view is above archived v1 in Dev1, Dev2, and Dev3."
            ),
            "type": "bar",
            "intent": "comparison",
            "question": "Did each modeling change improve all forward validation folds?",
            "rationale": "Grouped bars expose fold-level consistency and prevent the pooled score from hiding a localized regression.",
            "dataset": "fold_scores",
            "sourceId": "v2_fold_scores",
            "encodings": {
                "x": {"field": "period", "type": "ordinal", "label": "Validation period"},
                "y": {"field": "cosine", "type": "quantitative", "format": "number", "label": "Cosine"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "month_start", "type": "quantitative", "label": "Start month"},
                    {"field": "month_end", "type": "quantitative", "label": "End month"},
                    {"field": "rows", "type": "quantitative", "format": "compact", "label": "Rows"},
                    {"field": "uplift_vs_v1", "type": "quantitative", "format": "number", "label": "Uplift vs v1"},
                ],
            },
            "xAxisTitle": "Forward fold",
            "yAxisTitle": "Uncentered cosine",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 20,
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec", "title": "Model"},
            "labels": {"values": "endpoints"},
            "combinationRationale": "Color encodes the second categorical dimension, model, while the x-axis encodes fold.",
        }
    ]
    if activity_data:
        charts.append(
            {
                "id": "raw_activity_shift",
                "title": "Raw event activity change from train to test",
                "subtitle": "Order and trade streams contain roughly one-third more rows per sample in test.",
                "type": "bar",
                "intent": "comparison",
                "question": "How did raw observation density change between labeled and test data?",
                "rationale": "Three direct bars make the source-level activity change legible without implying that it causes target drift.",
                "dataset": "activity_shift",
                "sourceId": "raw_activity",
                "encodings": {
                    "x": {"field": "source", "type": "nominal", "label": "Raw source"},
                    "y": {"field": "activity_change_pct", "type": "quantitative", "format": "number", "unit": "%", "label": "Change in mean rows/sample"},
                    "tooltip": [
                        {"field": "train_mean_rows_per_sample", "type": "quantitative", "format": "number", "label": "Train mean"},
                        {"field": "test_mean_rows_per_sample", "type": "quantitative", "format": "number", "label": "Test mean"},
                        {"field": "test_to_train_activity_ratio", "type": "quantitative", "format": "number", "label": "Test/train ratio"},
                    ],
                },
                "xAxisTitle": "Raw event source",
                "yAxisTitle": "Test versus train activity change (%)",
                "valueFormat": "number",
                "unit": "%",
                "layout": "full",
                "maxRows": 10,
                "labels": {"values": "all"},
            }
        )
    if stress_curve:
        q_point_field = "signed_power_q1p2_point_cosine"
        q_cumulative_field = "signed_power_q1p2_cumulative_cosine"
        charts.append(
            {
                "id": "deployment_age_curve",
                "title": "V2 cosine across historical deployment horizons",
                "subtitle": "Repeated-origin diagnostics test whether a fixed model degrades as its training cutoff ages.",
                "type": "line",
                "intent": "trend",
                "question": (
                    "Does the capacity component retain signal through the 38-month test-like horizon?"
                    if blend_active
                    else "Does the selected model retain signal through the 38-month test-like horizon?"
                ),
                "rationale": "A line chart preserves horizon order and compares point versus cumulative q=1.2 cosine at the same scale.",
                "dataset": "deployment_stress_curve",
                "sourceId": "v2_deployment_curve",
                "encodings": {
                    "x": {"field": "horizon_months", "type": "ordinal", "label": "Deployment horizon (months)"},
                    "y": {"fields": [q_point_field, q_cumulative_field], "type": "quantitative", "format": "number", "label": "Cosine"},
                    "tooltip": [
                        {"field": "point_n_origins", "type": "quantitative", "label": "Origins"},
                        {"field": "point_n_forecasts", "type": "quantitative", "format": "compact", "label": "Point forecasts"},
                        {"field": "cumulative_n_forecasts", "type": "quantitative", "format": "compact", "label": "Cumulative forecasts"},
                    ],
                },
                "xAxisTitle": "Months since training cutoff",
                "yAxisTitle": "Uncentered cosine",
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 50,
                "legend": {"position": "bottom", "sort": "spec", "title": "Diagnostic view"},
                "labels": {"values": "endpoints"},
            }
        )

    tables: list[dict[str, Any]] = [
        {
            "id": "fold_model_table",
            "title": "Exact fold/model score ledger",
            "subtitle": "Pooled cosine is primary; fold rows show whether gains repeat chronologically.",
            "dataset": "fold_scores",
            "sourceId": "v2_fold_scores",
            "density": "dense",
            "defaultSort": {"field": "cosine", "direction": "desc"},
            "columns": [
                {"field": "period", "label": "Period", "type": "text"},
                {"field": "month_start", "label": "Start month", "type": "number", "format": "number"},
                {"field": "month_end", "label": "End month", "type": "number", "format": "number"},
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number", "format": "compact"},
                {"field": "cosine", "label": "Cosine", "type": "number", "format": "number"},
                {"field": "uplift_vs_v1", "label": "Uplift vs v1", "type": "number", "format": "number", "movement": True},
            ],
        }
    ]
    if tabm_data:
        tables.append(
            {
                "id": "tabm_challenger_table",
                "title": "TabM-mini chronological challenger",
                "subtitle": "Only completed folds are shown; an incomplete OOF run cannot select or blend a final model.",
                "dataset": "tabm_challenger",
                "sourceId": "tabm_challenger",
                "density": "dense",
                "defaultSort": {"field": "fold", "direction": "asc"},
                "columns": [
                    {"field": "fold", "label": "Fold", "type": "text"},
                    {"field": "device", "label": "Device", "type": "text"},
                    {"field": "best_epoch", "label": "Epoch", "type": "number", "format": "number"},
                    {"field": "outer_rows", "label": "Rows", "type": "number", "format": "compact"},
                    {"field": "tabm_raw_cosine", "label": "Raw cosine", "type": "number", "format": "number"},
                    {"field": "tabm_q1p2_cosine", "label": "q=1.2 cosine", "type": "number", "format": "number"},
                    {"field": "selected_v2_q1p2_cosine", "label": "Selected v2", "type": "number", "format": "number"},
                    {"field": "tabm_raw_minus_selected_v2", "label": "Raw minus v2", "type": "number", "format": "number", "movement": True},
                ],
            }
        )

    dev_gain = _delta(reported_development, v1_dev)
    if blend_active:
        sealed_sentence = (
            f"Its one-time sealed diagnostic cosine is **{_fmt(blend_sealed)}**, versus **{_fmt(sealed_v1)}** for archived v1. "
            if blend_sealed is not None
            else "The blend-specific one-time sealed diagnostic is not available in the report inputs; the earlier capacity-only sealed result must not be presented as the blend score. "
        )
        technical_summary = (
            "## Technical summary\n\n"
            f"The final frozen candidate is a **{_fmt(selected_blend.get('tabm_weight'), 1)} TabM-mini / "
            f"{_fmt(_first_value(selected_blend, ('capacity_lightgbm_weight',), ('capacity_weight',)), 1)} capacity-LightGBM** blend with post-blend signed power "
            f"**q={_fmt(_first_value(selected_blend, ('power_exponent',), ('signed_power',)), 1)}**. "
            f"It scores **{_fmt(blend_development)}** over pooled Dev1-Dev3 and **{_fmt(blend_dev3)}** on the one-shot Dev3 confirmation; "
            f"the recorded promotion gate is **{'passed' if blend_dev3_gate is True else 'not verified'}**. "
            + sealed_sentence
            + "The improvement is historical evidence, not a guarantee of hidden-test performance: the archived v1 public score was 0.124 and recent labeled rows remain distinguishable from test.\n\n"
            + (
                "The canonical v2 blend submission has passed the complete pointer, generation-manifest, replay-audit, and hash chain."
                if final_ready
                else "This report remains partial until the blend publication chain and any required diagnostics are complete."
            )
        )
    else:
        technical_summary = (
            "## Technical summary\n\n"
            f"The frozen v2 candidate improves pooled chronological development cosine from **{_fmt(v1_dev)}** to "
            f"**{_fmt(selected_dev)}** (**{_fmt(dev_gain, digits=6)}** absolute) and wins in every Dev1-Dev3 fold. "
            f"Its one-time sealed score is **{_fmt(sealed_final)}**, versus **{_fmt(sealed_v1)}** for archived v1. "
            "The improvement is meaningful offline but is not a guarantee of hidden-test performance: the archived v1 public score was 0.124 and recent labeled rows remain distinguishable from test.\n\n"
            + (
                "The canonical v2 submission has passed the full local integrity audit."
                if final_ready
                else "This report is currently partial because the final model/submission audit and any still-running deployment diagnostic have not all landed."
            )
        )

    capacity_low = _number(capacity_ci.get("confidence_low"))
    capacity_high = _number(capacity_ci.get("confidence_high"))
    seq_low = _number(sequence_ci.get("confidence_low"))
    seq_high = _number(sequence_ci.get("confidence_high"))
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {"id": "technical_summary", "type": "markdown", "body": technical_summary},
    ]
    if card_ids:
        blocks.append(
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": card_ids}
        )
    blocks.extend(
        [
            {
                "id": "validation_result",
                "type": "markdown",
                "sourceId": "v2_selection",
                "body": (
                    "## Higher tree capacity produced the main gain across every forward fold\n\n"
                    f"With the original features held available, the higher-capacity LightGBM adds **{_fmt(raw_uplift)}** pooled raw cosine versus v1. "
                    f"The three-month block bootstrap interval is **[{_fmt(capacity_low)}, {_fmt(capacity_high)}]**, with all three chronological folds positive. "
                    + (
                        "This was the first-stage capacity-component decision: q=1.2 and the v1 blend weight froze on Dev1-Dev2 before the later TabM-mini blend screen."
                        if blend_active
                        else "The q=1.2 signed-power transform was selected only on Dev1-Dev2, confirmed on Dev3, and the v1 blend weight froze at zero."
                    )
                ),
            },
            {"id": "fold_chart_block", "type": "chart", "chartId": "fold_model_cosine"},
            {"id": "fold_table_block", "type": "table", "tableId": "fold_model_table"},
            {
                "id": "path_increment",
                "type": "markdown",
                "sourceId": "v2_selection",
                "body": (
                    "## Fixed-time microstructure paths add a smaller but independently positive increment\n\n"
                    f"At identical LightGBM capacity, the 280 path features add **{_fmt(sequence_increment)}** pooled raw cosine over the base-only model. "
                    f"The three-month block-bootstrap interval is **[{_fmt(seq_low)}, {_fmt(seq_high)}]**. "
                    "This is modest, but every forward fold is positive; the result supports retaining the path representation without claiming that a deep sequence model is necessary."
                ),
            },
            {
                "id": "scope_definitions",
                "type": "markdown",
                "body": (
                    "## Scope, data, and score definition\n\n"
                    "The labeled period contains months 0-70; test follows in time and has no labels. Dev1, Dev2, and Dev3 are expanding chronological validation blocks covering months 23-58. Months 59-70 form a sealed audit opened once after model and transform choices were frozen. The competition score is uncentered cosine: `dot(prediction, target) / (||prediction||₂ ||target||₂)`. It measures directional alignment over all submitted observations, not percent accuracy.\n\n"
                    "Pooled block cosine is the primary validation metric because it reproduces the competition aggregation. Monthly median, lower-tail months, fold consistency, block-bootstrap intervals, and deployment age are robustness diagnostics."
                ),
            },
            {
                "id": "model_specification",
                "type": "markdown",
                "body": (
                    "## Frozen model specification and experimental design\n\n"
                    + (
                        "The final pipeline has two independently preprocessed components. Capacity-LightGBM uses 474 established summary features plus 280 fixed 6-second paths and the bounded 1,200-tree capacity specification. TabM-mini uses the 474 established summary features in a 16-member parameter-efficient ensemble with a shared two-layer 256-unit backbone; its final epoch is the median of the three inner-selected development-fold epochs.\n\n"
                        "Each complete component vector is normalized to uncentered unit RMS, then combined as 0.6 TabM-mini + 0.4 capacity-LightGBM. The blend applies `g(z)=sign(z)|z|^1.1` and one final uncentered RMS normalization. The weights and transform were selected on Dev1-Dev2 and frozen before Dev3 labels; Dev3 was one-shot confirmation, and sealed results were used only for the predeclared keep/fallback safety veto - not for retuning."
                        if blend_active
                        else "The selected pipeline combines 474 established summary features with 280 fixed 6-second market/order/trade path features. A robust median/IQR preprocessor clips transformed values at ±8 and adds missingness indicators, yielding 1,326 model inputs. The LightGBM regressor uses squared-error fitting, 1,200 trees, learning rate 0.02, 31 leaves, maximum depth 7, minimum child size 1,000, column fraction 0.75, and L2 penalty 30.\n\nMathematically, the tree ensemble estimates the conditional mean `f(x) = Σₜ η hₜ(x)` under squared loss. Final predictions apply `g(z)=sign(z)|z|^1.2`, followed by one global RMS normalization. Cosine is invariant to the final positive global scale; normalization exists for a stable submission contract. Feature/model selection used Dev1-Dev2, Dev3 was confirmation, and the sealed audit was descriptive only."
                    )
                ),
            },
            {
                "id": "domain_risk",
                "type": "markdown",
                "sourceId": "domain_shift",
                "body": (
                    "## The hidden test distribution remains the largest external risk\n\n"
                    f"A held-out classifier separates recent labeled months 59-70 from test at AUC **{_fmt(recent_auc)}**. "
                    "Raw order and trade rows per sample are also materially higher in test. These facts establish covariate shift, not a changed target relationship, but they explain why even honest historical gains can compress on the leaderboard."
                ),
            },
        ]
    )
    if activity_data:
        blocks.append(
            {"id": "activity_chart_block", "type": "chart", "chartId": "raw_activity_shift"}
        )

    robustness_text = (
        "## Robustness checks limit what can be claimed\n\n"
        f"The one-time sealed gain is positive even after removing historically dominant month 66: raw v2 minus v1 is **{_fmt(sealed.get('v2_minus_v1_raw_excluding_month_66'))}**. "
        "Both capacity and path-feature increments use three-month block bootstraps, which preserve some temporal dependence but remain historical stability diagnostics - not confidence intervals for future hidden targets. "
    )
    if stress_pooled is not None:
        robustness_text += (
            f"The repeated-origin 38-month q=1.2 {'capacity-component ' if blend_active else ''}stress cosine is **{_fmt(stress_pooled)}** and never touches sealed months. "
            + (
                "It is diagnostic only, did not reopen selection, and must not be described as a stress test of the final blend."
                if blend_active
                else "It is diagnostic only and did not reopen selection."
            )
        )
    else:
        robustness_text += (
            "The v2 repeated-origin 38-month deployment stress is not yet available, so long-horizon decay remains an explicit completion gap."
        )
    blocks.append(
        {"id": "robustness", "type": "markdown", "body": robustness_text}
    )
    if stress_curve:
        blocks.append(
            {"id": "stress_chart_block", "type": "chart", "chartId": "deployment_age_curve"}
        )

    if tabm_data or blend_active:
        complete_tabm = bool(tabm and tabm.get("complete_development_oof"))
        tabm_raw = _number(tabm_data[0].get("tabm_raw_cosine")) if tabm_data else None
        tabm_v2 = (
            _number(tabm_data[0].get("selected_v2_q1p2_cosine"))
            if tabm_data
            else None
        )
        tabm_blocks: list[dict[str, Any]] = [
            {
                "id": "neural_challenger",
                "type": "markdown",
                "body": (
                        (
                            "## TabM-mini was promoted only inside the frozen blend\n\n"
                            f"The 0.6 TabM-mini / 0.4 capacity-LightGBM candidate was selected on Dev1-Dev2 at **{_fmt(_number(selected_blend.get('screen_cosine')))}**, then evaluated once on Dev3 at **{_fmt(blend_dev3)}**. "
                            f"The predeclared Dev3 gate is recorded as **{'passed' if blend_dev3_gate is True else 'not verified'}**. TabM-mini is not published alone: each component is normalized independently before the frozen q=1.1 blend. Its inner window selects epoch only, and the final test refit uses the median inner-selected epoch across Dev1-Dev3."
                            if blend_active
                            else (
                                "## TabM-mini is promising but remains a bounded challenger\n\n"
                                f"The available TabM-mini run scores **{_fmt(tabm_raw)}** raw cosine on its first completed outer fold, compared with **{_fmt(tabm_v2)}** for selected v2 on that fold. "
                                + (
                                    "A complete development OOF exists, so it can be evaluated under the frozen promotion rule."
                                    if complete_tabm
                                    else "The run does not cover complete development OOF, so it cannot justify promotion or an ensemble weight."
                                )
                                + " Its inner window selects epoch only; a fresh network is then refit on every prior outer-training month, avoiding the common three-month training-lag error."
                            )
                    )
                ),
            }
        ]
        if tabm_data:
            tabm_blocks.append(
                {
                    "id": "tabm_table_block",
                    "type": "table",
                    "tableId": "tabm_challenger_table",
                }
            )
        blocks.extend(tabm_blocks)

    readiness_lines = [
        "## Final artifact readiness",
        "",
    ]
    if final_ready:
        readiness_lines.append(
            f"The canonical v2 CSV contains **{int(final_rows):,}** rows, all audit checks pass, and its SHA-256 is `{final_hash}`. "
            + (
                "The stable pointer, immutable generation manifest, capacity model/preprocessor, TabM checkpoint/preprocessor, prediction bundle, and all CSV mirrors are hash-verified."
                if final_mode == "blend"
                else "The training manifest, saved model, preprocessor, prediction array, root mirror, and workspace mirror are checksum-verified."
            )
        )
    else:
        readiness_lines.append(
            "Model selection is frozen, but the canonical final-training manifest and `status=ready` v2 submission audit are not both available. Do not describe the CSV as final until those files exist and this builder is rerun."
        )
    readiness_block: dict[str, Any] = {
        "id": "final_readiness",
        "type": "markdown",
        "body": "\n".join(readiness_lines),
    }
    if final_ready:
        readiness_block["sourceId"] = "v2_submission_audit"
    blocks.extend(
        [
            readiness_block,
            {
                "id": "recommended_next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Submit the audited v2 CSV once; do not tune repeatedly to the 49% public split.\n"
                    "2. Use any leaderboard result as one noisy external observation alongside the frozen chronological ledger.\n"
                    + (
                        "3. Treat the 60/40 TabM-mini/capacity-LightGBM weights and q=1.1 transform as immutable for this submission.\n"
                        if blend_active
                        else "3. Promote a neural challenger only with complete forward OOF and complementary, OOF-normalized ensemble gain.\n"
                    )
                    + "4. Prioritize regime-robust representations and density normalization over broader post-hoc calibration.\n\n"
                    "The design is consistent with evidence from [TabM](https://arxiv.org/abs/2410.24210), [RealMLP](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf), [temporal-shift benchmarking](https://proceedings.mlr.press/v267/cai25j.html), and classical [order-flow imbalance](https://arxiv.org/abs/1011.6402)."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- Does the v2 gain persist on the unseen private 51% rather than only the public split?\n"
                    "- Which path channels contribute stable incremental signal after controlling model capacity?\n"
                    + (
                        "- Does a separately frozen RealMLP candidate add complementary signal beyond the published TabM-mini blend?\n"
                        if blend_active
                        else "- Can a complete TabM/RealMLP OOF prediction add at least 0.001 after component RMS normalization?\n"
                    )
                    + "- How quickly does signal decay across the full 38-month horizon, and which features are most stable at long deployment ages?"
                ),
            },
        ]
    )

    manifest_path_value = publication.get("manifest_path")
    if isinstance(manifest_path_value, Path):
        try:
            manifest_source_path = manifest_path_value.relative_to(root).as_posix()
        except ValueError:
            manifest_source_path = PATHS["final_manifest"]
    else:
        manifest_source_path = PATHS["final_manifest"]
    audit_path_value = publication.get("audit_path")
    if isinstance(audit_path_value, Path):
        try:
            audit_source_path = audit_path_value.relative_to(root).as_posix()
        except ValueError:
            audit_source_path = PATHS["final_audit"]
    else:
        audit_source_path = PATHS["final_audit"]
    source_list = _sources(
        final_manifest_path=manifest_source_path,
        final_audit_path=audit_source_path,
        final_mode=final_mode,
    )
    status = "ready" if not issues else "partial"
    timestamp = generated_at or _utc_now()
    artifact: dict[str, Any] = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Technical selection, robustness, domain-shift, challenger, and submission-readiness report for the v2 forecasting pipeline.",
            "generatedAt": timestamp,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": source_list,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": status,
            "datasets": {
                "headline": [headline],
                "fold_scores": fold_data[:MAX_DATASET_ROWS],
                "activity_shift": activity_data[:MAX_DATASET_ROWS],
                "deployment_stress_curve": stress_curve[:MAX_DATASET_ROWS],
                "tabm_challenger": tabm_data[:MAX_DATASET_ROWS],
            },
            "accessIssues": issues,
        },
        "sources": source_list,
    }
    validate_artifact_shape(artifact)
    return artifact


def _assert_unique_ids(items: list[dict[str, Any]], path: str) -> None:
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError(f"{path} contains a missing or non-string id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate ids")


def _walk_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_finite(child, f"{path}[{index}]")


def validate_artifact_shape(artifact: dict[str, Any]) -> None:
    """Validate the locally checkable subset of the MCP report contract."""

    if artifact.get("surface") != "report":
        raise ValueError("surface must be report")
    manifest = artifact.get("manifest")
    snapshot = artifact.get("snapshot")
    if not isinstance(manifest, dict) or not isinstance(snapshot, dict):
        raise ValueError("manifest and snapshot must be objects")
    if manifest.get("version") != 1 or manifest.get("surface") != "report":
        raise ValueError("manifest must use version 1 and report surface")
    if snapshot.get("version") != 1:
        raise ValueError("snapshot must use version 1")
    if snapshot.get("status") not in {"ready", "partial", "blocked", "fixture"}:
        raise ValueError("invalid snapshot status")

    blocks = manifest.get("blocks")
    cards = manifest.get("cards", [])
    charts = manifest.get("charts", [])
    tables = manifest.get("tables", [])
    sources = manifest.get("sources", [])
    datasets = snapshot.get("datasets")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("report requires ordered blocks")
    if not isinstance(cards, list) or not isinstance(charts, list) or not isinstance(tables, list):
        raise ValueError("cards, charts, and tables must be arrays")
    if not isinstance(sources, list) or not isinstance(datasets, dict):
        raise ValueError("sources must be an array and datasets must be an object")
    for collection, name in (
        (blocks, "manifest.blocks"),
        (cards, "manifest.cards"),
        (charts, "manifest.charts"),
        (tables, "manifest.tables"),
        (sources, "manifest.sources"),
    ):
        _assert_unique_ids(collection, name)

    title = manifest.get("title")
    first = blocks[0]
    if first.get("type") != "markdown" or first.get("body") != f"# {title}":
        raise ValueError("first report block must be a matching visible # title")
    allowed_blocks = {"markdown", "metric-strip", "chart", "table", "html"}
    source_ids = {source["id"] for source in sources}
    card_ids = {card["id"] for card in cards}
    chart_ids = {chart["id"] for chart in charts}
    table_ids = {table["id"] for table in tables}
    for block in blocks:
        block_type = block.get("type")
        if block_type not in allowed_blocks:
            raise ValueError(f"unsupported block type: {block_type}")
        if block_type == "markdown":
            body = block.get("body")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"markdown block {block['id']} has no body")
            peer_h2 = sum(1 for line in body.splitlines() if line.startswith("## "))
            if block["id"] != "title" and peer_h2 != 1:
                raise ValueError(
                    f"markdown block {block['id']} must contain exactly one peer ## heading"
                )
        if block_type == "metric-strip" and not set(block.get("cardIds", [])).issubset(card_ids):
            raise ValueError(f"metric strip {block['id']} references an unknown card")
        if block_type == "chart" and block.get("chartId") not in chart_ids:
            raise ValueError(f"chart block {block['id']} references an unknown chart")
        if block_type == "table" and block.get("tableId") not in table_ids:
            raise ValueError(f"table block {block['id']} references an unknown table")
        if block.get("sourceId") and block["sourceId"] not in source_ids:
            raise ValueError(f"block {block['id']} references an unknown source")

    if not charts:
        raise ValueError("an MCP report requires at least one native chart")
    for dataset_id, rows in datasets.items():
        if not isinstance(dataset_id, str) or not isinstance(rows, list):
            raise ValueError("snapshot datasets must map string ids to row arrays")
        if len(rows) > MAX_DATASET_ROWS:
            raise ValueError(f"dataset {dataset_id} exceeds {MAX_DATASET_ROWS} reviewed rows")
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"dataset {dataset_id} contains a non-object row")

    for collection_name, collection in (
        ("cards", cards),
        ("charts", charts),
        ("tables", tables),
    ):
        for item in collection:
            dataset_id = item.get("dataset")
            if dataset_id not in datasets:
                raise ValueError(f"{collection_name}.{item['id']} references an unknown dataset")
            source_id = item.get("sourceId")
            if source_id not in source_ids and not isinstance(item.get("source"), dict):
                raise ValueError(f"{collection_name}.{item['id']} lacks canonical provenance")
    for table in tables:
        fields = {column.get("field") for column in table.get("columns", [])}
        sort = table.get("defaultSort")
        if not isinstance(sort, dict) or sort.get("field") not in fields:
            raise ValueError(f"table {table['id']} requires a declared default-sort field")
    for chart in charts:
        if chart.get("type") not in {
            "area", "bar", "boxPlot", "funnel", "heatmap", "histogram",
            "horizontalBar", "horizontalStackedBar", "horizontalStackedBar100",
            "leaderboard", "line", "pie", "scatter", "sparkline", "stackedArea",
            "stackedBar", "stackedBar100", "waterfall",
        }:
            raise ValueError(f"chart {chart['id']} has an unsupported native type")
        if not chart.get("intent") or not chart.get("question") or not chart.get("rationale"):
            raise ValueError(f"chart {chart['id']} lacks intent metadata")
        rows = datasets[chart["dataset"]]
        encodings = chart.get("encodings", {})
        plotted = set()
        for encoding in encodings.values():
            if isinstance(encoding, dict):
                if isinstance(encoding.get("field"), str):
                    plotted.add(encoding["field"])
                plotted.update(
                    field for field in encoding.get("fields", []) if isinstance(field, str)
                )
        available = set().union(*(row.keys() for row in rows)) if rows else set()
        if rows and not plotted.issubset(available):
            raise ValueError(f"chart {chart['id']} references absent encoding fields")
        if rows and available.issubset(plotted):
            raise ValueError(f"chart {chart['id']} dataset is not richer than its visible encodings")

    for source in sources:
        source_path = source.get("path")
        if source_path is not None:
            if not isinstance(source_path, str):
                raise ValueError(f"source {source['id']} path must be a string")
            pure = PurePosixPath(source_path)
            if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", source_path):
                raise ValueError(f"source {source['id']} uses an unsafe/noncanonical local path")
        href = source.get("href")
        if href is not None and urlparse(href).scheme not in {"http", "https"}:
            raise ValueError(f"source {source['id']} href must be http(s)")

    top_sources = artifact.get("sources")
    if top_sources != sources:
        raise ValueError("top-level sources must exactly match manifest.sources")
    issues = snapshot.get("accessIssues", [])
    if not isinstance(issues, list):
        raise ValueError("snapshot.accessIssues must be an array")
    if issues and snapshot.get("status") not in {"partial", "blocked"}:
        raise ValueError("access issues require partial or blocked status")
    _walk_finite(artifact)


def write_artifact(artifact: dict[str, Any], output: Path) -> None:
    validate_artifact_shape(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    if reloaded != artifact:
        raise RuntimeError("artifact JSON round-trip changed the payload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="artifact.json destination (default: artifacts/v2/report/artifact.json)",
    )
    parser.add_argument(
        "--strict-ready",
        action="store_true",
        help="exit nonzero when required later artifacts are still missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = build_artifact()
    write_artifact(artifact, args.output.resolve())
    status = artifact["snapshot"]["status"]
    print(
        f"wrote {args.output.resolve()} ({status}; "
        f"{len(artifact['manifest']['blocks'])} blocks, "
        f"{len(artifact['manifest']['charts'])} charts, "
        f"{len(artifact['manifest']['tables'])} tables)"
    )
    if args.strict_ready and status != "ready":
        missing = ", ".join(
            issue["id"] for issue in artifact["snapshot"].get("accessIssues", [])
        )
        raise SystemExit(f"report is {status}: {missing}")


if __name__ == "__main__":
    main()
