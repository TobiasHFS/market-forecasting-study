"""Diagnose fixed-model performance decay without opening the sealed audit.

This script is intentionally *not* a selector.  It is run only after
``frozen_pipeline.json`` exists and evaluates the already-frozen pipeline on
development labels through month 58.  Each origin fits once and is then kept
fixed for every later horizon, approximating the unusually long 38-month test
deployment.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from feature_families import materialize_feature_set
from modeling import RobustPreprocessor, cosine_score
from pipeline_config import DIAGNOSTIC_ROOT, RANDOM_SEED, ensure_artifact_directories
from run_gbdt_experiment import SPECS, _load_labels, _month_slice


FROZEN_PATH = DIAGNOSTIC_ROOT / "frozen_pipeline.json"
OUTPUT_CSV = DIAGNOSTIC_ROOT / "deployment_stress_curve.csv"
ORIGIN_CSV = DIAGNOSTIC_ROOT / "deployment_stress_origins.csv"
PREDICTION_PATH = DIAGNOSTIC_ROOT / "deployment_stress_predictions.npz"
SUMMARY_PATH = DIAGNOSTIC_ROOT / "deployment_stress_summary.json"

# Seven spaced origins make the near-horizon curve a genuine repeated-origin
# diagnostic while retaining one fully historical 38-month deployment.  No
# forecast or score is permitted to touch sealed months 59--70.
ORIGINS = (20, 22, 24, 26, 28, 30, 32)
MAX_HORIZON = 38
LAST_DEVELOPMENT_MONTH = 58
REFERENCE_HORIZONS = (1, 3, 6, 12, 18, 24, 30, 38)


def _sha256_canonical(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_and_verify_frozen() -> dict[str, object]:
    if not FROZEN_PATH.exists():
        raise FileNotFoundError("freeze the model before running deployment stress")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    supplied_hash = frozen.pop("pipeline_sha256", None)
    calculated_hash = _sha256_canonical(frozen)
    frozen["pipeline_sha256"] = supplied_hash
    if supplied_hash != calculated_hash:
        raise ValueError("frozen pipeline hash is invalid")
    if frozen.get("feature_set") != "multiscale_mechanics_scale":
        raise ValueError("unexpected frozen feature set")
    if frozen.get("model_spec_name") != "slow":
        raise ValueError("unexpected frozen model specification")
    if frozen.get("sealed_audit_months") != [59, 70]:
        raise ValueError("unexpected sealed-audit boundary")
    return frozen


def _fit_predict_origin(
    X: np.ndarray,
    names: list[str],
    kinds: list[str],
    target: np.ndarray,
    months: np.ndarray,
    origin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    train_slice = _month_slice(months, 0, origin)
    forecast_end = min(origin + MAX_HORIZON, LAST_DEVELOPMENT_MONTH)
    validation_slice = _month_slice(months, origin + 1, forecast_end)
    processor = RobustPreprocessor(
        feature_names=names,
        feature_kinds=kinds,
        clip=8.0,
        add_missing_indicators=True,
        output_dtype=np.float32,
    )
    started = time.perf_counter()
    processor.fit(X[train_slice])
    X_train = processor.transform(X[train_slice])
    X_validation = processor.transform(X[validation_slice])
    params = asdict(SPECS["slow"])
    params.update(
        {
            "force_col_wise": True,
            "deterministic": True,
            "bagging_seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
        }
    )
    model = LGBMRegressor(**params)
    model.fit(X_train, target[train_slice])
    prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
    forecast_months = months[validation_slice].astype(np.int16, copy=True)
    ages = (forecast_months.astype(np.int16) - origin).astype(np.int8)
    rows = np.arange(validation_slice.start, validation_slice.stop, dtype=np.int32)
    elapsed = time.perf_counter() - started
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"origin {origin} produced non-finite predictions")
    del model, processor, X_train, X_validation
    gc.collect()
    return rows, forecast_months, ages, prediction, elapsed


def run() -> dict[str, object]:
    ensure_artifact_directories()
    outputs = (OUTPUT_CSV, ORIGIN_CSV, PREDICTION_PATH, SUMMARY_PATH)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"deployment-stress output already exists: {existing}")
    frozen = _load_and_verify_frozen()
    months, target = _load_labels()
    if int(months.max()) != 70:
        raise ValueError("unexpected training chronology")
    materialized = materialize_feature_set("train", str(frozen["feature_set"]))
    if materialized.names != frozen["feature_names"]:
        raise ValueError("current feature names differ from frozen names")
    if materialized.kinds != frozen["feature_kinds"]:
        raise ValueError("current transform kinds differ from frozen kinds")

    row_parts: list[np.ndarray] = []
    month_parts: list[np.ndarray] = []
    age_parts: list[np.ndarray] = []
    origin_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    origin_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for origin in ORIGINS:
        rows, forecast_months, ages, prediction, elapsed = _fit_predict_origin(
            materialized.matrix,
            materialized.names,
            materialized.kinds,
            target,
            months,
            origin,
        )
        y = target[rows]
        origin_rows.append(
            {
                "origin": origin,
                "train_month_start": 0,
                "train_month_end": origin,
                "forecast_month_start": origin + 1,
                "forecast_month_end": int(forecast_months.max()),
                "maximum_age_months": int(ages.max()),
                "n_forecasts": int(rows.size),
                "pooled_cosine": cosine_score(y, prediction),
                "fit_predict_seconds": elapsed,
            }
        )
        row_parts.append(rows)
        month_parts.append(forecast_months)
        age_parts.append(ages)
        origin_parts.append(np.full(rows.size, origin, dtype=np.int8))
        prediction_parts.append(prediction)
        print(
            f"origin {origin}: through month {int(forecast_months.max())}, "
            f"cosine={origin_rows[-1]['pooled_cosine']:.6f}, {elapsed:.1f}s",
            flush=True,
        )

    rows = np.concatenate(row_parts)
    forecast_months = np.concatenate(month_parts)
    ages = np.concatenate(age_parts)
    origins = np.concatenate(origin_parts)
    predictions = np.concatenate(prediction_parts)
    y_all = target[rows]
    if int(forecast_months.max()) > LAST_DEVELOPMENT_MONTH:
        raise AssertionError("sealed audit labels were touched by deployment stress")

    curve_rows: list[dict[str, object]] = []
    for horizon in range(1, MAX_HORIZON + 1):
        point = ages == horizon
        cumulative = ages <= horizon
        if not np.any(point):
            raise ValueError(f"no forecast support at horizon {horizon}")
        curve_rows.append(
            {
                "horizon_months": horizon,
                "point_cosine": cosine_score(y_all[point], predictions[point]),
                "point_n_forecasts": int(point.sum()),
                "point_n_origins": int(np.unique(origins[point]).size),
                "cumulative_cosine": cosine_score(
                    y_all[cumulative], predictions[cumulative]
                ),
                "cumulative_n_forecasts": int(cumulative.sum()),
            }
        )
    curve = pd.DataFrame(curve_rows)
    per_origin = pd.DataFrame(origin_rows)
    curve.to_csv(OUTPUT_CSV, index=False)
    per_origin.to_csv(ORIGIN_CSV, index=False)
    np.savez_compressed(
        PREDICTION_PATH,
        row_indices=rows,
        months=forecast_months,
        age_months=ages,
        origins=origins,
        target=y_all,
        prediction=predictions,
    )
    reference = curve[curve["horizon_months"].isin(REFERENCE_HORIZONS)]
    summary: dict[str, object] = {
        "status": "diagnostic_only_not_used_for_selection",
        "pipeline_sha256": frozen["pipeline_sha256"],
        "feature_set": frozen["feature_set"],
        "model_spec_name": frozen["model_spec_name"],
        "origins": list(ORIGINS),
        "maximum_horizon_months": MAX_HORIZON,
        "last_scored_month": int(forecast_months.max()),
        "sealed_months_touched": False,
        "all_forecasts_pooled_cosine": cosine_score(y_all, predictions),
        "positive_point_horizon_share": float(np.mean(curve["point_cosine"] > 0.0)),
        "median_point_horizon_cosine": float(curve["point_cosine"].median()),
        "minimum_point_horizon_cosine_diagnostic_only": float(
            curve["point_cosine"].min()
        ),
        "reference_horizons": reference.to_dict(orient="records"),
        "per_origin": origin_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "curve_path": str(OUTPUT_CSV),
        "predictions_path": str(PREDICTION_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(
        "deployment stress complete: "
        f"pooled={summary['all_forecasts_pooled_cosine']:.6f}, "
        f"positive horizons={summary['positive_point_horizon_share']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
