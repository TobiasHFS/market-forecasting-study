"""Chronological Ridge ablations over the declared feature hierarchy."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from threadpoolctl import threadpool_limits

from feature_families import FEATURE_SET_FAMILIES, materialize_feature_set
from modeling import (
    DEVELOPMENT_FOLDS,
    RIDGE_ALPHA_GRID,
    PerObservationRidge,
    RobustPreprocessor,
    cosine_score,
    monthly_diagnostics,
    summarize_monthly_diagnostics,
)
from pipeline_config import DATA_ROOT, DIAGNOSTIC_ROOT, RANDOM_SEED, ensure_artifact_directories


DEFAULT_SETS = (
    "market_core",
    "all_core",
    "multiscale",
    "multiscale_mechanics",
    "multiscale_scale",
    "multiscale_path",
)


def _penalty_label(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p").replace("-", "m")


def _load_labels() -> tuple[np.ndarray, np.ndarray]:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather",
        columns=["sample_id", "month", "target"],
    )
    sample_id = table["sample_id"].to_numpy()
    if not np.array_equal(sample_id, np.arange(len(sample_id), dtype=sample_id.dtype)):
        raise ValueError("label sample_id is not exact row alignment")
    months = table["month"].to_numpy().astype(np.int16, copy=False)
    if np.any(months[1:] < months[:-1]):
        raise ValueError("label months are not nondecreasing by row")
    target = table["target"].to_numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(target)):
        raise ValueError("target contains non-finite values")
    return months, target


def _month_slice(months: np.ndarray, start: int, end: int) -> slice:
    left = int(np.searchsorted(months, start, side="left"))
    right = int(np.searchsorted(months, end, side="right"))
    if left >= right:
        raise ValueError(f"empty month slice {start}..{end}")
    observed = months[left:right]
    if int(observed.min()) != start or int(observed.max()) != end:
        raise ValueError(f"incomplete month slice {start}..{end}")
    return slice(left, right)


def evaluate_feature_set(
    feature_set: str,
    *,
    penalties: tuple[float, ...] = RIDGE_ALPHA_GRID,
    overwrite: bool = False,
) -> dict[str, object]:
    ensure_artifact_directories()
    prediction_path = DIAGNOSTIC_ROOT / f"ridge_{feature_set}_development_oof.npz"
    summary_path = DIAGNOSTIC_ROOT / f"ridge_{feature_set}_summary.json"
    monthly_path = DIAGNOSTIC_ROOT / f"ridge_{feature_set}_monthly.csv"
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"{prediction_path} exists; pass --overwrite")

    months, target = _load_labels()
    materialized = materialize_feature_set("train", feature_set)
    X = materialized.matrix
    if X.shape[0] != target.size:
        raise ValueError("feature/label row mismatch")
    if any(name in {"sample_id", "month", "target"} for name in materialized.names):
        raise ValueError("identifier/label leaked into feature schema")

    oof_slice = _month_slice(months, 23, 58)
    oof_rows = np.arange(oof_slice.start, oof_slice.stop, dtype=np.int64)
    oof_target = target[oof_slice]
    oof_months = months[oof_slice]
    predictions = np.full((oof_rows.size, len(penalties)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for fold in DEVELOPMENT_FOLDS:
        train_slice = _month_slice(months, fold.train_start, fold.train_end)
        validation_slice = _month_slice(
            months, fold.validation_start, fold.validation_end
        )
        preprocessor = RobustPreprocessor(
            feature_names=materialized.names,
            feature_kinds=materialized.kinds,
            clip=8.0,
            add_missing_indicators=True,
            output_dtype=np.float32,
        )
        fold_started = time.perf_counter()
        preprocessor.fit(X[train_slice])
        X_train = preprocessor.transform(X[train_slice])
        X_validation = preprocessor.transform(X[validation_slice])
        y_train = target[train_slice]
        y_validation = target[validation_slice]
        destination = slice(
            validation_slice.start - oof_slice.start,
            validation_slice.stop - oof_slice.start,
        )

        for penalty_index, penalty in enumerate(penalties):
            model = PerObservationRidge(
                penalty=float(penalty), solver="lsqr", tol=1e-4, max_iter=2_000
            )
            fit_started = time.perf_counter()
            with threadpool_limits(limits=16):
                model.fit(X_train, y_train)
                prediction = np.asarray(
                    model.predict(X_validation), dtype=np.float64
                )
            predictions[destination, penalty_index] = prediction
            coefficient = np.asarray(model.model_.coef_, dtype=np.float64)
            iteration = getattr(model.model_, "n_iter_", None)
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold.name,
                    "penalty": float(penalty),
                    "absolute_alpha": float(model.alpha_),
                    "train_rows": int(X_train.shape[0]),
                    "validation_rows": int(X_validation.shape[0]),
                    "raw_features": int(X.shape[1]),
                    "transformed_features": int(X_train.shape[1]),
                    "fold_cosine": cosine_score(y_validation, prediction),
                    "prediction_mean": float(prediction.mean()),
                    "prediction_rms": float(np.sqrt(np.mean(prediction * prediction))),
                    "coefficient_l2": float(np.linalg.norm(coefficient)),
                    "iterations": int(np.asarray(iteration).reshape(-1)[0]) if iteration is not None else None,
                    "fit_seconds": time.perf_counter() - fit_started,
                    "fold_total_seconds": time.perf_counter() - fold_started,
                }
            )
        del X_train, X_validation, preprocessor
        gc.collect()

    if not np.all(np.isfinite(predictions)):
        raise ValueError("development OOF contains unfilled/non-finite predictions")

    penalty_rows: list[dict[str, object]] = []
    for penalty_index, penalty in enumerate(penalties):
        prediction = predictions[:, penalty_index]
        diagnostics = monthly_diagnostics(oof_target, prediction, oof_months)
        row = summarize_monthly_diagnostics(
            diagnostics, cosine_score(oof_target, prediction)
        ).to_dict()
        row.update({"feature_set": feature_set, "penalty": float(penalty)})
        penalty_rows.append(row)
    ranking = pd.DataFrame(penalty_rows).sort_values(
        ["pooled_cosine", "penalty"], ascending=[False, False], ignore_index=True
    )
    best_penalty = float(ranking.iloc[0]["penalty"])
    best_index = list(penalties).index(best_penalty)
    best_monthly = monthly_diagnostics(
        oof_target, predictions[:, best_index], oof_months
    )
    best_monthly.to_csv(monthly_path, index=False)

    payload = {
        "feature_set": feature_set,
        "families": list(FEATURE_SET_FAMILIES[feature_set]),
        "random_seed": RANDOM_SEED,
        "raw_feature_count": len(materialized.names),
        "feature_names": materialized.names,
        "feature_kinds": materialized.kinds,
        "penalties": [float(value) for value in penalties],
        "best_penalty": best_penalty,
        "ranking": ranking.to_dict(orient="records"),
        "folds": fold_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays = {
        "row_indices": oof_rows,
        "months": oof_months,
        "target": oof_target,
    }
    arrays.update(
        {
            _penalty_label(penalty): predictions[:, index]
            for index, penalty in enumerate(penalties)
        }
    )
    np.savez(prediction_path, **arrays)
    pd.DataFrame(fold_rows).to_csv(
        DIAGNOSTIC_ROOT / f"ridge_{feature_set}_folds.csv", index=False
    )
    del X, materialized, predictions
    gc.collect()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=tuple(FEATURE_SET_FAMILIES),
        default=list(DEFAULT_SETS),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for feature_set in args.sets:
        payload = evaluate_feature_set(feature_set, overwrite=args.overwrite)
        winner = payload["ranking"][0]
        print(
            f"{feature_set}: best lambda={payload['best_penalty']:g}, "
            f"pooled cosine={winner['pooled_cosine']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

