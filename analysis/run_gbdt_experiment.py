"""Chronological LightGBM evaluation for the accepted feature hierarchy."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from lightgbm import LGBMRegressor

from feature_families import materialize_feature_set
from modeling import (
    DEVELOPMENT_FOLDS,
    LightGBMSpec,
    RobustPreprocessor,
    cosine_score,
    monthly_diagnostics,
    summarize_monthly_diagnostics,
)
from pipeline_config import DATA_ROOT, DIAGNOSTIC_ROOT, RANDOM_SEED, ensure_artifact_directories


SPECS: dict[str, LightGBMSpec] = {
    "shallow": LightGBMSpec(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=1_500,
        reg_alpha=1.0,
        reg_lambda=30.0,
    ),
    "balanced": LightGBMSpec(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=1_000,
        reg_alpha=1.0,
        reg_lambda=20.0,
    ),
    "slow": LightGBMSpec(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=5,
        min_child_samples=1_500,
        reg_alpha=1.0,
        reg_lambda=30.0,
    ),
}


def _load_labels() -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    sample_id = table["sample_id"].to_numpy()
    if not np.array_equal(sample_id, np.arange(len(sample_id), dtype=sample_id.dtype)):
        raise ValueError("label sample_id is not exact row alignment")
    months = table["month"].to_numpy().astype(np.int16, copy=False)
    target = table["target"].to_numpy().astype(np.float64, copy=False)
    if np.any(months[1:] < months[:-1]) or not np.all(np.isfinite(target)):
        raise ValueError("invalid label chronology or target")
    return months, target


def _month_slice(months: np.ndarray, start: int, end: int) -> slice:
    left = int(np.searchsorted(months, start, side="left"))
    right = int(np.searchsorted(months, end, side="right"))
    if left >= right or months[left] != start or months[right - 1] != end:
        raise ValueError(f"incomplete month slice {start}..{end}")
    return slice(left, right)


def evaluate(
    feature_set: str,
    spec_names: list[str],
    *,
    half_life_months: float | None = None,
    shuffle_within_month: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    ensure_artifact_directories()
    if half_life_months is not None and (
        not np.isfinite(half_life_months) or half_life_months <= 0.0
    ):
        raise ValueError("half_life_months must be finite and positive")
    suffix = (
        ""
        if half_life_months is None
        else "_hl" + f"{half_life_months:g}".replace(".", "p")
    )
    if shuffle_within_month:
        suffix += "_shuffle"
    prediction_path = DIAGNOSTIC_ROOT / f"gbdt_{feature_set}{suffix}_development_oof.npz"
    summary_path = DIAGNOSTIC_ROOT / f"gbdt_{feature_set}{suffix}_summary.json"
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"{prediction_path} exists; pass --overwrite")
    unknown = set(spec_names).difference(SPECS)
    if unknown:
        raise ValueError(f"unknown LightGBM specs: {sorted(unknown)}")

    months, target = _load_labels()
    fit_target = target.copy()
    if shuffle_within_month:
        rng = np.random.default_rng(RANDOM_SEED)
        for current_month in np.unique(months):
            positions = np.flatnonzero(months == current_month)
            fit_target[positions] = rng.permutation(fit_target[positions])
    materialized = materialize_feature_set("train", feature_set)
    X = materialized.matrix
    oof_slice = _month_slice(months, 23, 58)
    oof_rows = np.arange(oof_slice.start, oof_slice.stop, dtype=np.int64)
    oof_target = target[oof_slice]
    oof_months = months[oof_slice]
    predictions = np.full((oof_rows.size, len(spec_names)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for fold in DEVELOPMENT_FOLDS:
        train_slice = _month_slice(months, fold.train_start, fold.train_end)
        validation_slice = _month_slice(months, fold.validation_start, fold.validation_end)
        preprocessor = RobustPreprocessor(
            feature_names=materialized.names,
            feature_kinds=materialized.kinds,
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
        y_train = fit_target[train_slice]
        y_validation = target[validation_slice]
        if half_life_months is None:
            sample_weight = None
        else:
            age = fold.train_end - months[train_slice].astype(np.float64)
            sample_weight = np.exp(math.log(0.5) * age / half_life_months)
            sample_weight /= sample_weight.mean()
        destination = slice(
            validation_slice.start - oof_slice.start,
            validation_slice.stop - oof_slice.start,
        )

        for spec_index, spec_name in enumerate(spec_names):
            spec = SPECS[spec_name]
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
            model.fit(X_train, y_train, sample_weight=sample_weight)
            prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
            fit_seconds = time.perf_counter() - fit_started
            predictions[destination, spec_index] = prediction
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "spec": spec_name,
                    "fold": fold.name,
                    "train_rows": int(X_train.shape[0]),
                    "validation_rows": int(X_validation.shape[0]),
                    "raw_features": int(X.shape[1]),
                    "transformed_features": int(X_train.shape[1]),
                    "fold_cosine": cosine_score(y_validation, prediction),
                    "prediction_mean": float(prediction.mean()),
                    "prediction_rms": float(np.sqrt(np.mean(prediction * prediction))),
                    "transform_seconds": transform_seconds,
                    "fit_predict_seconds": fit_seconds,
                }
            )
            gain = model.booster_.feature_importance(importance_type="gain")
            for name, value in zip(transformed_names, gain, strict=True):
                if value > 0.0:
                    importance_rows.append(
                        {
                            "feature_set": feature_set,
                            "spec": spec_name,
                            "fold": fold.name,
                            "feature": name,
                            "gain": float(value),
                        }
                    )
            del model
            gc.collect()
        del X_train, X_validation, preprocessor
        gc.collect()

    if not np.all(np.isfinite(predictions)):
        raise ValueError("development OOF contains unfilled/non-finite predictions")
    ranking_rows: list[dict[str, object]] = []
    for index, spec_name in enumerate(spec_names):
        prediction = predictions[:, index]
        diagnostics = monthly_diagnostics(oof_target, prediction, oof_months)
        row = summarize_monthly_diagnostics(
            diagnostics, cosine_score(oof_target, prediction)
        ).to_dict()
        row["spec"] = spec_name
        ranking_rows.append(row)
        diagnostics.to_csv(
            DIAGNOSTIC_ROOT / f"gbdt_{feature_set}{suffix}_{spec_name}_monthly.csv",
            index=False,
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        "pooled_cosine", ascending=False, ignore_index=True
    )
    best_spec = str(ranking.iloc[0]["spec"])
    payload = {
        "feature_set": feature_set,
        "random_seed": RANDOM_SEED,
        "half_life_months": half_life_months,
        "shuffle_within_month": shuffle_within_month,
        "raw_feature_count": len(materialized.names),
        "feature_names": materialized.names,
        "feature_kinds": materialized.kinds,
        "specs": {name: asdict(SPECS[name]) for name in spec_names},
        "best_spec": best_spec,
        "ranking": ranking.to_dict(orient="records"),
        "folds": fold_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays = {"row_indices": oof_rows, "months": oof_months, "target": oof_target}
    arrays.update({name: predictions[:, index] for index, name in enumerate(spec_names)})
    np.savez(prediction_path, **arrays)
    pd.DataFrame(fold_rows).to_csv(
        DIAGNOSTIC_ROOT / f"gbdt_{feature_set}{suffix}_folds.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        DIAGNOSTIC_ROOT / f"gbdt_{feature_set}{suffix}_feature_importance.csv", index=False
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", default="multiscale_mechanics_scale")
    parser.add_argument("--specs", nargs="+", choices=tuple(SPECS), default=["balanced"])
    parser.add_argument("--half-life", type=float, default=None)
    parser.add_argument("--shuffle-within-month", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(
        args.feature_set,
        args.specs,
        half_life_months=args.half_life,
        shuffle_within_month=args.shuffle_within_month,
        overwrite=args.overwrite,
    )
    winner = payload["ranking"][0]
    print(
        f"{payload['feature_set']}: best GBDT={payload['best_spec']}, "
        f"pooled cosine={winner['pooled_cosine']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
