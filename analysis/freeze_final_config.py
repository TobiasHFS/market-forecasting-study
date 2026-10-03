"""Write the immutable pre-audit specification and its reproducibility hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from feature_families import FEATURE_SET_FAMILIES, materialize_feature_set
from pipeline_config import DIAGNOSTIC_ROOT, PROJECT_ROOT, RANDOM_SEED, ensure_artifact_directories
from run_gbdt_experiment import SPECS


FEATURE_SET = "multiscale_mechanics_scale"
MODEL_SPEC = "slow"
FROZEN_PATH = DIAGNOSTIC_ROOT / "frozen_pipeline.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frozen_payload() -> dict[str, object]:
    features = materialize_feature_set("train", FEATURE_SET)
    names_digest = hashlib.sha256(
        json.dumps(features.names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    kinds_digest = hashlib.sha256(
        json.dumps(features.kinds, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_files = (
        "analysis/market_features.py",
        "analysis/flow_features.py",
        "analysis/feature_families.py",
        "analysis/modeling.py",
        "analysis/run_gbdt_experiment.py",
    )
    payload: dict[str, object] = {
        "status": "frozen_before_sealed_audit",
        "feature_set": FEATURE_SET,
        "feature_families": list(FEATURE_SET_FAMILIES[FEATURE_SET]),
        "feature_count": len(features.names),
        "feature_names": features.names,
        "feature_kinds": features.kinds,
        "feature_names_sha256": names_digest,
        "feature_kinds_sha256": kinds_digest,
        "preprocessor": {
            "clip": 8.0,
            "add_missing_indicators": True,
            "fit_scope": "training rows only",
        },
        "model_family": "LightGBM histogram gradient boosting",
        "model_spec_name": MODEL_SPEC,
        "model_spec": asdict(SPECS[MODEL_SPEC]),
        "objective": "regression_l2",
        "history_weighting": "equal",
        "final_component_rule": "pure_gbdt",
        "ridge_weight": 0.0,
        "prediction_centering": "none",
        "prediction_normalization": "uncentered RMS once over complete prediction vector",
        "development_months": [23, 58],
        "sealed_audit_months": [59, 70],
        "random_seed": RANDOM_SEED,
        "rejected_candidates": {
            "path_satellite": "negative development delta",
            "half_life_48": "within one SE and negative latest-fold delta",
            "half_life_24": "within one SE and negative latest-fold delta",
            "ridge_blend": "best development grid weight was zero Ridge",
        },
        "source_sha256": {
            relative: _sha256_file(PROJECT_ROOT / relative) for relative in source_files
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["pipeline_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main() -> None:
    ensure_artifact_directories()
    if FROZEN_PATH.exists():
        raise FileExistsError(
            f"{FROZEN_PATH} already exists; refusing to silently change a frozen pipeline"
        )
    payload = build_frozen_payload()
    FROZEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(payload["pipeline_sha256"])


if __name__ == "__main__":
    main()

