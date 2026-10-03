"""Chronological LightGBM evaluation of stationary microstructure paths."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from lightgbm import LGBMRegressor

from modeling import (
    DEVELOPMENT_FOLDS,
    LightGBMSpec,
    RobustPreprocessor,
    SEALED_AUDIT_FOLD,
    cosine_score,
    monthly_diagnostics,
    summarize_monthly_diagnostics,
)
from pipeline_config import DATA_ROOT, RANDOM_SEED
from v2_features import VALID_FEATURE_SETS, materialize_v2_feature_set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "experiments"

SPECS = {
    "slow": LightGBMSpec(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=5,
        min_child_samples=1_500,
        reg_alpha=1.0,
        reg_lambda=30.0,
    ),
    "capacity": LightGBMSpec(
        n_estimators=1_200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=7,
        min_child_samples=1_000,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=30.0,
    ),
}


def _labels() -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather", columns=["sample_id", "month", "target"]
    )
    sample_id = table["sample_id"].to_numpy()
    if not np.array_equal(sample_id, np.arange(len(sample_id), dtype=sample_id.dtype)):
        raise ValueError("label/feature row alignment failed")
    return (
        table["month"].to_numpy().astype(np.int16, copy=False),
        table["target"].to_numpy().astype(np.float64, copy=False),
    )


def _month_slice(months: np.ndarray, start: int, end: int) -> slice:
    left = int(np.searchsorted(months, start, side="left"))
    right = int(np.searchsorted(months, end, side="right"))
    if left >= right or months[left] != start or months[right - 1] != end:
        raise ValueError(f"incomplete month slice: {start}..{end}")
    return slice(left, right)


def _signed_power(prediction: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(prediction) * np.power(np.abs(prediction), exponent)


def evaluate(
    feature_set: str,
    spec_name: str,
    *,
    folds: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> dict:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"sequence_{feature_set}_{spec_name}"
    if folds:
        stem += "_" + "-".join(folds)
    summary_path = OUTPUT_ROOT / f"{stem}_summary.json"
    prediction_path = OUTPUT_ROOT / f"{stem}_oof.npz"
    if (summary_path.exists() or prediction_path.exists()) and not overwrite:
        raise FileExistsError(f"experiment exists: {stem}")
    available_folds = DEVELOPMENT_FOLDS + (SEALED_AUDIT_FOLD,)
    selected_folds = tuple(
        fold
        for fold in available_folds
        if (folds is None and not fold.sealed) or (folds is not None and fold.name in folds)
    )
    if not selected_folds:
        raise ValueError("no folds selected")

    months, target = _labels()
    features = materialize_v2_feature_set("train", feature_set)
    X = features.matrix
    spec = SPECS[spec_name]
    fold_rows: list[dict] = []
    validation_indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    validation_months: list[np.ndarray] = []
    importance_rows: list[dict] = []
    started = time.perf_counter()

    for fold in selected_folds:
        train_slice = _month_slice(months, fold.train_start, fold.train_end)
        validation_slice = _month_slice(
            months, fold.validation_start, fold.validation_end
        )
        preprocessor = RobustPreprocessor(
            feature_names=features.names,
            feature_kinds=features.kinds,
            clip=8.0,
            add_missing_indicators=True,
            output_dtype=np.float32,
        )
        transform_started = time.perf_counter()
        preprocessor.fit(X[train_slice])
        X_train = preprocessor.transform(X[train_slice])
        X_validation = preprocessor.transform(X[validation_slice])
        transformed_names = preprocessor.get_feature_names_out().tolist()
        transform_seconds = time.perf_counter() - transform_started
        params = asdict(spec)
        params.update(
            {
                "force_col_wise": True,
                "deterministic": True,
                "bagging_seed": RANDOM_SEED,
                "feature_fraction_seed": RANDOM_SEED,
            }
        )
        model = LGBMRegressor(**params)
        fit_started = time.perf_counter()
        model.fit(X_train, target[train_slice])
        prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
        fit_seconds = time.perf_counter() - fit_started
        score = cosine_score(target[validation_slice], prediction)
        fold_rows.append(
            {
                "fold": fold.name,
                "train_rows": int(X_train.shape[0]),
                "validation_rows": int(X_validation.shape[0]),
                "raw_features": int(X.shape[1]),
                "transformed_features": int(X_train.shape[1]),
                "fold_cosine": score,
                "transform_seconds": transform_seconds,
                "fit_predict_seconds": fit_seconds,
            }
        )
        print(f"{fold.name}: cosine={score:.6f}", flush=True)
        gain = model.booster_.feature_importance(importance_type="gain")
        for name, value in zip(transformed_names, gain, strict=True):
            if value > 0.0:
                importance_rows.append(
                    {"fold": fold.name, "feature": name, "gain": float(value)}
                )
        validation_indices.append(
            np.arange(validation_slice.start, validation_slice.stop, dtype=np.int64)
        )
        predictions.append(prediction)
        targets.append(target[validation_slice])
        validation_months.append(months[validation_slice])
        del X_train, X_validation, model, preprocessor
        gc.collect()

    row_indices = np.concatenate(validation_indices)
    y = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    oof_months = np.concatenate(validation_months)
    order = np.argsort(row_indices)
    row_indices = row_indices[order]
    y = y[order]
    prediction = prediction[order]
    oof_months = oof_months[order]
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("validation folds overlap")

    power_rows = []
    for exponent in (0.8, 1.0, 1.1, 1.2, 1.3, 1.4):
        transformed = _signed_power(prediction, exponent)
        fold_scores = []
        for fold in selected_folds:
            mask = (oof_months >= fold.validation_start) & (
                oof_months <= fold.validation_end
            )
            fold_scores.append(cosine_score(y[mask], transformed[mask]))
        power_rows.append(
            {
                "exponent": exponent,
                "pooled_cosine": cosine_score(y, transformed),
                "minimum_fold_cosine": min(fold_scores),
                "fold_scores": fold_scores,
            }
        )
    power = pd.DataFrame(power_rows).sort_values(
        ["pooled_cosine", "minimum_fold_cosine"], ascending=False, ignore_index=True
    )
    diagnostics = monthly_diagnostics(y, prediction, oof_months)
    summary = summarize_monthly_diagnostics(
        diagnostics, cosine_score(y, prediction)
    ).to_dict()
    payload = {
        "feature_set": feature_set,
        "spec_name": spec_name,
        "spec": asdict(spec),
        "random_seed": RANDOM_SEED,
        "raw_feature_count": len(features.names),
        "folds": fold_rows,
        "summary": summary,
        "power_calibration": power.to_dict(orient="records"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez(
        prediction_path,
        row_indices=row_indices,
        months=oof_months,
        target=y,
        prediction=prediction,
    )
    diagnostics.to_csv(OUTPUT_ROOT / f"{stem}_monthly.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        OUTPUT_ROOT / f"{stem}_feature_importance.csv", index=False
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=sorted(VALID_FEATURE_SETS), required=True)
    parser.add_argument("--spec", choices=sorted(SPECS), default="slow")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.feature_set,
        args.spec,
        folds=tuple(args.folds) if args.folds else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
