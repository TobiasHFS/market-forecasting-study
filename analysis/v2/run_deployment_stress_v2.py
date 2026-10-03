"""Diagnose v2 fixed-model decay without opening the sealed audit.

This is a reporting diagnostic, never a model or calibration selector.  At
each repeated origin it fits the predeclared ``base_plus_sequence_all`` /
``capacity`` pipeline once, then keeps that fit fixed for every available
later development month.  All model inputs and scores are restricted to
months 0--58; sealed months 59--70 are excluded before feature fitting or
target scoring.

The signed-power q=1.2 view is reported beside the raw prediction only as a
post-hoc diagnostic.  Nothing produced here authorizes choosing that transform
or reopening feature, hyperparameter, or calibration selection.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = WORKSPACE_ROOT / "analysis"
LOCAL_DEPS = WORKSPACE_ROOT / ".analysis_deps"
V2_ROOT = ANALYSIS_ROOT / "v2"
for dependency_path in (LOCAL_DEPS, ANALYSIS_ROOT, V2_ROOT):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from modeling import RobustPreprocessor, cosine_score
from pipeline_config import RANDOM_SEED
from run_sequence_experiment import SPECS, _labels, _month_slice, _signed_power
from v2_features import materialize_v2_feature_set


OUTPUT_ROOT = WORKSPACE_ROOT / "artifacts" / "v2" / "diagnostics"
OUTPUT_CSV = OUTPUT_ROOT / "deployment_stress_curve.csv"
ORIGIN_CSV = OUTPUT_ROOT / "deployment_stress_origins.csv"
PREDICTION_PATH = OUTPUT_ROOT / "deployment_stress_predictions.npz"
SUMMARY_PATH = OUTPUT_ROOT / "deployment_stress_summary.json"

FEATURE_SET = "base_plus_sequence_all"
MODEL_SPEC_NAME = "capacity"
SIGNED_POWER_Q = 1.2
SIGNED_POWER_VIEW = "signed_power_q1p2"
DIAGNOSTIC_STATUS = "diagnostic_only_not_used_for_selection"

# Keep these identical to analysis/run_deployment_stress.py.  The earliest
# origin supplies the complete 38-month horizon; later origins provide genuine
# repeated-origin support wherever development labels remain available.
ORIGINS = (20, 22, 24, 26, 28, 30, 32)
MAX_HORIZON = 38
LAST_DEVELOPMENT_MONTH = 58
SEALED_MONTH_START = 59
SEALED_MONTH_END = 70
REFERENCE_HORIZONS = (1, 3, 6, 12, 18, 24, 30, 38)


def _effective_model_params() -> dict[str, object]:
    """Return the v2 capacity spec plus deterministic LightGBM controls."""

    params = asdict(SPECS[MODEL_SPEC_NAME])
    params.update(
        {
            "force_col_wise": True,
            "deterministic": True,
            "bagging_seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
        }
    )
    return params


def _assert_development_only(months: np.ndarray, *, context: str) -> None:
    observed = np.asarray(months).reshape(-1)
    if observed.size == 0:
        raise ValueError(f"{context} contains no months")
    if np.any((observed >= SEALED_MONTH_START) & (observed <= SEALED_MONTH_END)):
        raise AssertionError(f"{context} touched sealed months 59--70")
    if int(observed.min()) < 0 or int(observed.max()) > LAST_DEVELOPMENT_MONTH:
        raise AssertionError(f"{context} escaped the development range 0--58")


def _load_development_labels() -> tuple[np.ndarray, np.ndarray, int]:
    """Copy the development prefix so no sealed target can reach model code."""

    all_months, all_target = _labels()
    if all_months.shape != all_target.shape:
        raise ValueError("month and target arrays are misaligned")
    if all_months.ndim != 1 or np.any(all_months[1:] < all_months[:-1]):
        raise ValueError("labels must be one-dimensional and month-sorted")

    development_slice = _month_slice(all_months, 0, LAST_DEVELOPMENT_MONTH)
    if development_slice.start != 0:
        raise ValueError("development labels are not a contiguous row prefix")
    development_stop = int(development_slice.stop)
    if development_stop >= all_months.size:
        raise ValueError("sealed-audit boundary is missing from label chronology")
    if int(all_months[development_stop]) != SEALED_MONTH_START:
        raise ValueError("unexpected first month after the development prefix")

    # Copies are deliberate: downstream fitting receives arrays whose backing
    # storage ends at month 58 rather than views backed by the full label file.
    months = np.array(all_months[development_slice], dtype=np.int16, copy=True)
    target = np.array(all_target[development_slice], dtype=np.float64, copy=True)
    del all_months, all_target
    _assert_development_only(months, context="loaded labels")
    return months, target, development_stop


def _fit_predict_origin(
    X: np.ndarray,
    names: list[str],
    kinds: list[str],
    target: np.ndarray,
    months: np.ndarray,
    origin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if origin not in ORIGINS:
        raise ValueError(f"undeclared deployment-stress origin: {origin}")
    _assert_development_only(months, context=f"origin {origin} input")
    if X.shape[0] != months.size or target.shape != months.shape:
        raise ValueError(f"origin {origin} input rows are misaligned")

    forecast_end = min(origin + MAX_HORIZON, LAST_DEVELOPMENT_MONTH)
    if forecast_end >= SEALED_MONTH_START:
        raise AssertionError(f"origin {origin} forecast reaches the sealed audit")
    train_slice = _month_slice(months, 0, origin)
    validation_slice = _month_slice(months, origin + 1, forecast_end)
    _assert_development_only(
        months[train_slice], context=f"origin {origin} training slice"
    )
    _assert_development_only(
        months[validation_slice], context=f"origin {origin} forecast slice"
    )

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
    model = LGBMRegressor(**_effective_model_params())
    model.fit(X_train, target[train_slice])
    prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
    forecast_months = months[validation_slice].astype(np.int16, copy=True)
    ages = (forecast_months.astype(np.int16) - origin).astype(np.int8)
    rows = np.arange(validation_slice.start, validation_slice.stop, dtype=np.int32)
    elapsed = time.perf_counter() - started
    if prediction.shape != rows.shape or not np.all(np.isfinite(prediction)):
        raise ValueError(f"origin {origin} produced invalid predictions")
    if int(ages.min()) != 1 or int(ages.max()) > MAX_HORIZON:
        raise AssertionError(f"origin {origin} produced invalid forecast ages")
    _assert_development_only(
        forecast_months, context=f"origin {origin} completed forecast"
    )
    del model, processor, X_train, X_validation
    gc.collect()
    return rows, forecast_months, ages, prediction, elapsed


def _write_outputs_exclusively(
    curve: pd.DataFrame,
    per_origin: pd.DataFrame,
    prediction_payload: dict[str, np.ndarray],
    summary: dict[str, object],
) -> None:
    """Create every artifact with exclusive mode so races cannot overwrite it."""

    with OUTPUT_CSV.open("x", encoding="utf-8", newline="") as handle:
        curve.to_csv(handle, index=False)
    with ORIGIN_CSV.open("x", encoding="utf-8", newline="") as handle:
        per_origin.to_csv(handle, index=False)
    with PREDICTION_PATH.open("xb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    with SUMMARY_PATH.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _view_summary(curve: pd.DataFrame, prefix: str) -> dict[str, object]:
    point_column = f"{prefix}_point_cosine"
    cumulative_column = f"{prefix}_cumulative_cosine"
    return {
        "positive_point_horizon_share": float(np.mean(curve[point_column] > 0.0)),
        "median_point_horizon_cosine": float(curve[point_column].median()),
        "minimum_point_horizon_cosine_diagnostic_only": float(
            curve[point_column].min()
        ),
        "terminal_cumulative_cosine": float(curve[cumulative_column].iloc[-1]),
    }


def run() -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = (OUTPUT_CSV, ORIGIN_CSV, PREDICTION_PATH, SUMMARY_PATH)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"deployment-stress output already exists: {existing}")
    if LAST_DEVELOPMENT_MONTH + 1 != SEALED_MONTH_START:
        raise AssertionError("development and sealed-audit boundaries are inconsistent")

    months, target, development_stop = _load_development_labels()
    features = materialize_v2_feature_set("train", FEATURE_SET)
    if features.matrix.shape[0] < development_stop:
        raise ValueError("feature matrix is shorter than the development labels")
    # This row-limited view is the only feature matrix passed into fitting.
    X = features.matrix[:development_stop]
    if X.shape[0] != months.size:
        raise ValueError("development feature and label rows are misaligned")

    row_parts: list[np.ndarray] = []
    month_parts: list[np.ndarray] = []
    age_parts: list[np.ndarray] = []
    origin_parts: list[np.ndarray] = []
    raw_prediction_parts: list[np.ndarray] = []
    origin_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for origin in ORIGINS:
        rows, forecast_months, ages, prediction, elapsed = _fit_predict_origin(
            X,
            features.names,
            features.kinds,
            target,
            months,
            origin,
        )
        transformed = _signed_power(prediction, SIGNED_POWER_Q)
        if not np.all(np.isfinite(transformed)):
            raise ValueError(f"origin {origin} produced an invalid signed-power view")
        y = target[rows]
        origin_rows.append(
            {
                "diagnostic_only": True,
                "origin": origin,
                "train_month_start": 0,
                "train_month_end": origin,
                "forecast_month_start": origin + 1,
                "forecast_month_end": int(forecast_months.max()),
                "maximum_age_months": int(ages.max()),
                "n_forecasts": int(rows.size),
                # Preserve the v1 column as the raw-view score for simple
                # side-by-side comparisons; the row-level flag marks it as a
                # diagnostic rather than a selection result.
                "pooled_cosine": cosine_score(y, prediction),
                "raw_pooled_cosine": cosine_score(y, prediction),
                f"{SIGNED_POWER_VIEW}_pooled_cosine": cosine_score(y, transformed),
                "fit_predict_seconds": elapsed,
            }
        )
        row_parts.append(rows)
        month_parts.append(forecast_months)
        age_parts.append(ages)
        origin_parts.append(np.full(rows.size, origin, dtype=np.int8))
        raw_prediction_parts.append(prediction)
        print(
            f"origin {origin}: through month {int(forecast_months.max())}, "
            f"raw cosine={origin_rows[-1]['raw_pooled_cosine']:.6f}, "
            "q=1.2 cosine="
            f"{origin_rows[-1][SIGNED_POWER_VIEW + '_pooled_cosine']:.6f}, "
            f"{elapsed:.1f}s",
            flush=True,
        )

    rows = np.concatenate(row_parts)
    forecast_months = np.concatenate(month_parts)
    ages = np.concatenate(age_parts)
    origins = np.concatenate(origin_parts)
    raw_prediction = np.concatenate(raw_prediction_parts)
    transformed_prediction = _signed_power(raw_prediction, SIGNED_POWER_Q)
    y_all = target[rows]
    _assert_development_only(forecast_months, context="combined forecasts")
    if int(forecast_months.max()) != LAST_DEVELOPMENT_MONTH:
        raise AssertionError("deployment stress did not end at month 58")
    if not np.all(np.isfinite(transformed_prediction)):
        raise ValueError("combined signed-power prediction contains NaN or infinity")

    curve_rows: list[dict[str, object]] = []
    for horizon in range(1, MAX_HORIZON + 1):
        point = ages == horizon
        cumulative = ages <= horizon
        if not np.any(point):
            raise ValueError(f"no forecast support at horizon {horizon}")
        curve_rows.append(
            {
                "diagnostic_only": True,
                "horizon_months": horizon,
                "raw_point_cosine": cosine_score(
                    y_all[point], raw_prediction[point]
                ),
                "point_cosine": cosine_score(y_all[point], raw_prediction[point]),
                f"{SIGNED_POWER_VIEW}_point_cosine": cosine_score(
                    y_all[point], transformed_prediction[point]
                ),
                "point_n_forecasts": int(point.sum()),
                "point_n_origins": int(np.unique(origins[point]).size),
                "raw_cumulative_cosine": cosine_score(
                    y_all[cumulative], raw_prediction[cumulative]
                ),
                "cumulative_cosine": cosine_score(
                    y_all[cumulative], raw_prediction[cumulative]
                ),
                f"{SIGNED_POWER_VIEW}_cumulative_cosine": cosine_score(
                    y_all[cumulative], transformed_prediction[cumulative]
                ),
                "cumulative_n_forecasts": int(cumulative.sum()),
            }
        )
    curve = pd.DataFrame(curve_rows)
    per_origin = pd.DataFrame(origin_rows)
    reference = curve[curve["horizon_months"].isin(REFERENCE_HORIZONS)]
    raw_view_summary = _view_summary(curve, "raw")
    raw_view_summary["all_forecasts_pooled_cosine"] = cosine_score(
        y_all, raw_prediction
    )
    power_view_summary = _view_summary(curve, SIGNED_POWER_VIEW)
    power_view_summary["all_forecasts_pooled_cosine"] = cosine_score(
        y_all, transformed_prediction
    )

    summary: dict[str, object] = {
        "status": DIAGNOSTIC_STATUS,
        "diagnostic_only": True,
        "selection_use_prohibited": True,
        "feature_set": FEATURE_SET,
        "model_spec_name": MODEL_SPEC_NAME,
        "model_spec_source": "analysis/v2/run_sequence_experiment.py:SPECS[capacity]",
        "model_spec": asdict(SPECS[MODEL_SPEC_NAME]),
        "effective_model_params": _effective_model_params(),
        "random_seed": RANDOM_SEED,
        "origins": list(ORIGINS),
        "maximum_horizon_months": MAX_HORIZON,
        "last_scored_month": int(forecast_months.max()),
        "sealed_audit_months": [SEALED_MONTH_START, SEALED_MONTH_END],
        "sealed_months_touched": False,
        # These compatibility fields are the raw-view diagnostics.  They
        # remain explicitly under the diagnostic-only status above.
        "all_forecasts_pooled_cosine": raw_view_summary[
            "all_forecasts_pooled_cosine"
        ],
        "positive_point_horizon_share": raw_view_summary[
            "positive_point_horizon_share"
        ],
        "median_point_horizon_cosine": raw_view_summary[
            "median_point_horizon_cosine"
        ],
        "minimum_point_horizon_cosine_diagnostic_only": raw_view_summary[
            "minimum_point_horizon_cosine_diagnostic_only"
        ],
        "signed_power_diagnostic": {
            "status": DIAGNOSTIC_STATUS,
            "exponent_q": SIGNED_POWER_Q,
            "selection_use_prohibited": True,
            "note": (
                "The raw and q=1.2 views are post-hoc diagnostics only; this "
                "stress test does not select or promote a prediction transform."
            ),
        },
        "prediction_views_diagnostic_only": {
            "raw": raw_view_summary,
            SIGNED_POWER_VIEW: power_view_summary,
        },
        "reference_horizons_diagnostic_only": reference.to_dict(orient="records"),
        "per_origin_diagnostic_only": origin_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "curve_path": str(OUTPUT_CSV),
        "origin_path": str(ORIGIN_CSV),
        "predictions_path": str(PREDICTION_PATH),
    }
    prediction_payload = {
        "row_indices": rows,
        "months": forecast_months,
        "age_months": ages,
        "origins": origins,
        "target": y_all,
        "prediction": raw_prediction,
        "prediction_raw": raw_prediction,
        f"prediction_{SIGNED_POWER_VIEW}_diagnostic_only": transformed_prediction,
        "signed_power_q": np.asarray(SIGNED_POWER_Q, dtype=np.float64),
        "diagnostic_only": np.asarray(True),
    }
    _write_outputs_exclusively(curve, per_origin, prediction_payload, summary)
    return summary


def main() -> None:
    summary = run()
    raw = summary["prediction_views_diagnostic_only"]["raw"]
    power = summary["prediction_views_diagnostic_only"][SIGNED_POWER_VIEW]
    print(
        "v2 deployment stress complete (diagnostic only): "
        f"raw pooled={raw['all_forecasts_pooled_cosine']:.6f}, "
        f"q=1.2 pooled={power['all_forecasts_pooled_cosine']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
