"""Frozen constants for the competition modeling pipeline.

The raw event files do not expose absolute timestamps or instruments.  The
configuration therefore uses only per-sample physical lookback time and the
label month for chronological validation.  ``sample_id`` is an alignment key,
never a model feature.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "ms-capital-real-financial-market-forecasting"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
FEATURE_ROOT = ARTIFACT_ROOT / "features"
MODEL_ROOT = ARTIFACT_ROOT / "models"
DIAGNOSTIC_ROOT = ARTIFACT_ROOT / "diagnostics"
SUBMISSION_ROOT = ARTIFACT_ROOT / "submissions"

TRAIN_SAMPLES = 1_257_637
TEST_SAMPLES = 647_896
TRAIN_MONTH_MIN = 0
TRAIN_MONTH_MAX = 70

MARKET_WINDOWS = (10.0, 30.0, 60.0, 180.0, 600.0)
FLOW_WINDOWS = (2.0, 5.0, 15.0, 30.0, 60.0)
CORE_MARKET_WINDOWS = (10, 60, 600)
CORE_FLOW_WINDOWS = (5, 15, 60)

# Historical fold names retained for compatibility. Months 59--70 were
# inspected during development and are not an independent holdout.
DEVELOPMENT_FOLDS = (
    ("dev_1", 0, 22, 23, 34),
    ("dev_2", 0, 34, 35, 46),
    ("dev_3", 0, 46, 47, 58),
)
SEALED_AUDIT_FOLD = ("sealed_audit", 0, 58, 59, 70)

RANDOM_SEED = 20260823


def ensure_artifact_directories() -> None:
    """Create the small, explicit set of pipeline output directories."""

    for path in (
        ARTIFACT_ROOT,
        FEATURE_ROOT,
        MODEL_ROOT,
        DIAGNOSTIC_ROOT,
        SUBMISSION_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

