"""Postmortem adversarial validation for the frozen competition feature set.

This script never touches the fitted forecasting pipeline or submission.  It
samples the already-extracted feature memmaps directly, trains deterministic
LightGBM *domain* classifiers (train=0, test=1), and writes compact diagnostics
under ``artifacts/diagnostics/postmortem``.

The train sample is balanced across all 71 labeled months.  The test sample is
systematically spread across its complete row index, which is the only test
ordering information available.  Classifier holdouts are stratified and are
not used for fitting.  AUC near 0.5 means the two samples are hard to tell
apart; a high AUC is evidence of covariate shift, not by itself proof of
forecast-target concept drift.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from feature_families import FEATURE_ROOT, family_manifest
from pipeline_config import DATA_ROOT, DIAGNOSTIC_ROOT, RANDOM_SEED


FROZEN_PATH = DIAGNOSTIC_ROOT / "frozen_pipeline.json"
QUALITY_PATH = DIAGNOSTIC_ROOT / "engineered_feature_quality.csv"
FORECAST_IMPORTANCE_PATH = (
    DIAGNOSTIC_ROOT / "gbdt_multiscale_mechanics_scale_feature_importance.csv"
)
PROFILE_PATH = Path(__file__).resolve().parent / "output" / "competition_profile.json"
OUTPUT_ROOT = DIAGNOSTIC_ROOT / "postmortem"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def _read_frozen_contract() -> tuple[list[str], list[str], str]:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    names = frozen.get("feature_names")
    families = frozen.get("feature_family_by_feature")
    if not isinstance(names, list) or len(names) != 474:
        raise ValueError("frozen pipeline does not contain the expected 474 names")
    if len(names) != len(set(names)):
        raise ValueError("frozen feature names are not unique")
    if not isinstance(families, list) or len(families) != len(names):
        # Older frozen contracts use ``feature_families_per_feature``.
        families = frozen.get("feature_families_per_feature")
    if not isinstance(families, list) or len(families) != len(names):
        # Reconstruct below from the canonical family manifest.
        families = []
    feature_set = str(frozen.get("feature_set", ""))
    if feature_set != "multiscale_mechanics_scale":
        raise ValueError(f"unexpected frozen feature set: {feature_set!r}")
    return [str(value) for value in names], [str(value) for value in families], feature_set


def _selected_entries(
    frozen_names: list[str], frozen_families: list[str]
) -> tuple[list[tuple[str, int, str]], list[str]]:
    manifest = family_manifest()
    by_name: dict[str, tuple[str, int, str]] = {}
    for family, entries in manifest.items():
        for source, column, name in entries:
            if name in by_name:
                raise ValueError(f"ambiguous feature in family manifest: {name}")
            by_name[name] = (source, int(column), family)
    selected: list[tuple[str, int, str]] = []
    reconstructed_families: list[str] = []
    for position, name in enumerate(frozen_names):
        if name not in by_name:
            raise KeyError(f"frozen feature missing from family manifest: {name}")
        source, column, family = by_name[name]
        if frozen_families and frozen_families[position] != family:
            raise ValueError(
                f"family mismatch for {name}: frozen={frozen_families[position]}, "
                f"manifest={family}"
            )
        selected.append((source, column, family))
        reconstructed_families.append(family)
    return selected, reconstructed_families


def _load_months() -> np.ndarray:
    table = feather.read_table(
        DATA_ROOT / "train" / "label.feather", columns=["sample_id", "month"]
    )
    sample_id = table["sample_id"].to_numpy()
    if not np.array_equal(sample_id, np.arange(sample_id.size, dtype=sample_id.dtype)):
        raise ValueError("label sample_id is not exact row alignment")
    month = table["month"].to_numpy().astype(np.int16, copy=False)
    if np.any(month[1:] < month[:-1]) or set(np.unique(month)) != set(range(71)):
        raise ValueError("unexpected train month chronology")
    return month


def _balanced_train_indices(
    month: np.ndarray, per_month: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    sampled_months: list[np.ndarray] = []
    for value in range(71):
        positions = np.flatnonzero(month == value)
        if positions.size < per_month:
            chosen = positions
        else:
            chosen = np.sort(rng.choice(positions, size=per_month, replace=False))
        pieces.append(chosen.astype(np.int64, copy=False))
        sampled_months.append(np.full(chosen.size, value, dtype=np.int16))
    indices = np.concatenate(pieces)
    months = np.concatenate(sampled_months)
    order = np.argsort(indices, kind="stable")
    return indices[order], months[order]


def _spread_test_indices(n_rows: int, size: int, seed: int) -> np.ndarray:
    if size > n_rows:
        raise ValueError("test sample cannot exceed test rows")
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_rows, size + 1, dtype=np.int64)
    width = edges[1:] - edges[:-1]
    if np.any(width <= 0):
        raise ValueError("test sampling bins are empty")
    offsets = (rng.random(size) * width).astype(np.int64)
    result = edges[:-1] + offsets
    if np.any(result[1:] <= result[:-1]):
        raise ValueError("test systematic sample is not strictly increasing")
    return result


def _load_memmap(split: str, source: str) -> np.ndarray:
    path = FEATURE_ROOT / f"{split}_{source}.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.ndim != 2:
        raise ValueError(f"invalid feature memmap: {path}")
    return array


def _fill_sample_matrix(
    destination: np.ndarray,
    split: str,
    row_indices: np.ndarray,
    entries: list[tuple[str, int, str]],
) -> None:
    """Copy only sampled rows, one source at a time, from feature memmaps."""

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for destination_column, (source, source_column, _family) in enumerate(entries):
        grouped[source].append((destination_column, source_column))
    for source in sorted(grouped):
        mapping = grouped[source]
        destination_columns = np.asarray([item[0] for item in mapping], dtype=np.int64)
        source_columns = np.asarray([item[1] for item in mapping], dtype=np.int64)
        array = _load_memmap(split, source)
        # Rows are sampled first so the temporary is bounded by sample size,
        # never by the 1.26M/648k full matrices.
        sampled_source = np.asarray(array[row_indices, :], dtype=np.float32)
        destination[:, destination_columns] = sampled_source[:, source_columns]
        del sampled_source, array
        gc.collect()


def _classifier(seed: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=400,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=1.0,
        reg_lambda=20.0,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )


def _auc_interval(auc: float, n_negative: int, n_positive: int) -> tuple[float, float, float]:
    """Hanley--McNeil large-sample standard error and a clipped 95% interval."""

    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n_positive - 1) * (q1 - auc * auc)
        + (n_negative - 1) * (q2 - auc * auc)
    ) / (n_positive * n_negative)
    standard_error = math.sqrt(max(variance, 0.0))
    return standard_error, max(0.0, auc - 1.96 * standard_error), min(1.0, auc + 1.96 * standard_error)


def _fit_and_score(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    seed: int,
    n_estimators: int,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    all_indices = np.arange(y.size, dtype=np.int64)
    fit_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    model = _classifier(seed, n_estimators)
    started = time.perf_counter()
    model.fit(X[fit_indices], y[fit_indices])
    probability = np.asarray(model.predict_proba(X[holdout_indices])[:, 1], dtype=np.float64)
    elapsed = time.perf_counter() - started
    holdout_y = y[holdout_indices]
    auc = float(roc_auc_score(holdout_y, probability))
    n_positive = int(holdout_y.sum())
    n_negative = int(holdout_y.size - n_positive)
    standard_error, lower, upper = _auc_interval(auc, n_negative, n_positive)
    summary = {
        "label": label,
        "fit_rows": int(fit_indices.size),
        "holdout_rows": int(holdout_indices.size),
        "features": int(X.shape[1]),
        "auc": auc,
        "auc_standard_error": standard_error,
        "auc_95pct_lower": lower,
        "auc_95pct_upper": upper,
        "accuracy_at_0_5": float(accuracy_score(holdout_y, probability >= 0.5)),
        "log_loss": float(log_loss(holdout_y, probability, labels=[0, 1])),
        "elapsed_seconds": elapsed,
    }
    gain = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
    split = model.booster_.feature_importance(importance_type="split").astype(np.int64)
    gain_total = float(gain.sum())
    importance = pd.DataFrame(
        {
            "feature": names,
            f"{label}_gain": gain,
            f"{label}_gain_share": gain / gain_total if gain_total > 0 else 0.0,
            f"{label}_split_count": split,
        }
    )
    return summary, importance, holdout_indices, probability


def _raw_activity_frame() -> pd.DataFrame:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for source in ("market", "order", "transaction"):
        train = profile["files"][f"train/{source}"]["structure"]
        test = profile["files"][f"test/{source}"]["structure"]
        train_mean = float(train["mean_rows_per_observed_sample"])
        test_mean = float(test["mean_rows_per_observed_sample"])
        rows.append(
            {
                "source": source,
                "train_raw_rows": int(train["rows"]),
                "test_raw_rows": int(test["rows"]),
                "train_mean_rows_per_sample": train_mean,
                "test_mean_rows_per_sample": test_mean,
                "test_to_train_activity_ratio": test_mean / train_mean,
                "activity_change_pct": 100.0 * (test_mean / train_mean - 1.0),
                "train_median_rows_per_sample": float(train["rows_per_observed_sample"]["q50"]),
                "test_median_rows_per_sample": float(test["rows_per_observed_sample"]["q50"]),
                "train_p95_rows_per_sample": float(train["rows_per_observed_sample"]["q95"]),
                "test_p95_rows_per_sample": float(test["rows_per_observed_sample"]["q95"]),
            }
        )
    return pd.DataFrame(rows)


def _markdown_report(summary: dict[str, Any], top: pd.DataFrame, raw: pd.DataFrame) -> str:
    full = summary["train_test_domain_classifier"]
    recent = summary["recent_train_test_domain_classifier"]
    missing = summary["missingness_only_domain_classifier"]
    time_classifier = summary["early_late_train_classifier"]
    alignment = summary["forecast_domain_importance_alignment"]
    lines = [
        "# Postmortem: train-test covariate shift",
        "",
        "This is an unsupervised diagnostic of the frozen 474-feature representation. "
        "It does not alter the forecasting model or submission.",
        "",
        "## Headline evidence",
        "",
        f"- Full-feature train/test domain AUC: **{full['auc']:.4f}** "
        f"(95% large-sample interval {full['auc_95pct_lower']:.4f}-{full['auc_95pct_upper']:.4f}).",
        f"- Recent labeled months 59-70 vs test domain AUC: **{recent['auc']:.4f}**.",
        f"- Missingness-only train/test domain AUC: **{missing['auc']:.4f}**.",
        f"- Early-vs-late labeled-train domain AUC: **{time_classifier['auc']:.4f}**.",
        f"- Spearman alignment between all-train domain gain and development-forecast gain: "
        f"**{alignment['spearman']:.3f}**; for recent-train domain gain it is "
        f"**{alignment['recent_train_spearman']:.3f}**. Only "
        f"**{alignment['recent_train_top_50_overlap_count']}** features are top-50 in both "
        "the recent-domain and forecast rankings.",
        "- AUC substantially above 0.5 means covariates are distinguishable. It does not prove "
        "that the mapping from features to future returns changed.",
        "",
        "## Raw event activity",
        "",
        "| Source | Train rows/sample | Test rows/sample | Change |",
        "|---|---:|---:|---:|",
    ]
    for row in raw.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.train_mean_rows_per_sample:.2f} | "
            f"{row.test_mean_rows_per_sample:.2f} | {row.activity_change_pct:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Highest-gain train/test domain features",
            "",
            "| Feature | Family | Gain share | Missing-rate shift | Robust median shift (train IQR) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in top.head(15).itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.family} | {row.train_test_gain_share:.3%} | "
            f"{row.missing_rate_shift:+.3%} | {row.robust_median_shift_iqr:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and realistic remedies",
            "",
            "1. Treat domain AUC as a warning that ordinary chronological CV was not a complete "
            "simulation of the hidden test regimes. Quantify validation by deployment age and by "
            "adversarial-similarity strata, not only one pooled cosine.",
            "2. Re-run forecasting ablations that remove or normalize the strongest domain features, "
            "especially absolute activity/scale features, and keep changes only if they improve "
            "multiple forward folds. High domain importance alone is not a reason to delete a "
            "predictive feature.",
            "3. Test bounded density-normalized versions of count/volume features (per observed second, "
            "per book update, per trade, and cross-sectional ranks within inferred regimes).",
            "4. Use adversarial weights only with clipping and effective-sample-size reporting. Pure "
            "importance weighting is fragile when the test domain lies outside train support.",
            "5. Increase model diversity with independently validated linear, shallow-tree, and "
            "sequence/summary views. Blend only out-of-fold normalized predictions; the current single "
            "tree model likely leaves stable linear signal and local temporal shape unused.",
            "",
            "The public score shortfall cannot be assigned to covariate shift alone: domain shift can "
            "coexist with a stable conditional target relation, and leaderboard sampling noise or "
            "missing predictive structure can also explain the gap. Here the near-zero alignment "
            "between domain and forecast feature importance weakens the case that the most visible "
            "covariate changes were the dominant source of forecast degradation.",
            "",
        ]
    )
    return "\n".join(lines)


def run(sample_per_train_month: int, n_estimators: int, overwrite: bool) -> dict[str, Any]:
    if sample_per_train_month < 100:
        raise ValueError("sample_per_train_month must be at least 100")
    if n_estimators < 20:
        raise ValueError("n_estimators must be at least 20")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_summary = OUTPUT_ROOT / "domain_shift_summary.json"
    if output_summary.exists() and not overwrite:
        raise FileExistsError(f"{output_summary} exists; pass --overwrite")

    frozen_names, frozen_families, feature_set = _read_frozen_contract()
    entries, feature_families = _selected_entries(frozen_names, frozen_families)
    months = _load_months()
    train_indices, sampled_train_months = _balanced_train_indices(
        months, sample_per_train_month, RANDOM_SEED + 11
    )
    test_rows = _load_memmap("test", "market").shape[0]
    test_indices = _spread_test_indices(
        test_rows, train_indices.size, RANDOM_SEED + 29
    )

    domain_matrix = np.empty(
        (train_indices.size + test_indices.size, len(frozen_names)), dtype=np.float32
    )
    _fill_sample_matrix(domain_matrix[: train_indices.size], "train", train_indices, entries)
    _fill_sample_matrix(domain_matrix[train_indices.size :], "test", test_indices, entries)
    domain_target = np.concatenate(
        [
            np.zeros(train_indices.size, dtype=np.uint8),
            np.ones(test_indices.size, dtype=np.uint8),
        ]
    )

    full_summary, full_importance, full_holdout, full_probability = _fit_and_score(
        domain_matrix,
        domain_target,
        frozen_names,
        seed=RANDOM_SEED + 101,
        n_estimators=n_estimators,
        label="train_test",
    )

    # A stricter comparison against the final 12 labeled months avoids treating
    # old regimes that the model merely retained as if they represented the
    # deployment frontier.  The test side is thinned systematically to match.
    recent_mask = sampled_train_months >= 59
    recent_train = domain_matrix[: train_indices.size][recent_mask]
    recent_test_positions = np.linspace(
        0, test_indices.size - 1, recent_train.shape[0], dtype=np.int64
    )
    recent_matrix = np.empty(
        (2 * recent_train.shape[0], len(frozen_names)), dtype=np.float32
    )
    recent_matrix[: recent_train.shape[0]] = recent_train
    recent_matrix[recent_train.shape[0] :] = domain_matrix[train_indices.size :][
        recent_test_positions
    ]
    recent_target = np.concatenate(
        [
            np.zeros(recent_train.shape[0], dtype=np.uint8),
            np.ones(recent_train.shape[0], dtype=np.uint8),
        ]
    )
    recent_summary, recent_importance, recent_holdout, recent_probability = _fit_and_score(
        recent_matrix,
        recent_target,
        frozen_names,
        seed=RANDOM_SEED + 102,
        n_estimators=n_estimators,
        label="recent_train_test",
    )
    del recent_train, recent_matrix
    gc.collect()

    missing_matrix = (~np.isfinite(domain_matrix)).astype(np.uint8)
    missing_summary, missing_importance, missing_holdout, missing_probability = _fit_and_score(
        missing_matrix,
        domain_target,
        frozen_names,
        seed=RANDOM_SEED + 103,
        n_estimators=max(80, n_estimators // 2),
        label="missingness",
    )
    del missing_matrix
    gc.collect()

    # Natural labeled-period drift comparator using the exact same sampled rows.
    # Months 0--34 are early (0); months 35--70 are late (1).
    early_late_target = (sampled_train_months >= 35).astype(np.uint8)
    time_summary, time_importance, time_holdout, time_probability = _fit_and_score(
        domain_matrix[: train_indices.size],
        early_late_target,
        frozen_names,
        seed=RANDOM_SEED + 107,
        n_estimators=n_estimators,
        label="early_late_train",
    )

    importance = (
        full_importance.merge(recent_importance, on="feature", how="left")
        .merge(missing_importance, on="feature", how="left")
        .merge(time_importance, on="feature", how="left")
    )
    importance.insert(1, "family", feature_families)
    quality = pd.read_csv(QUALITY_PATH)
    selected_quality = quality[quality["feature"].isin(frozen_names)].copy()
    if selected_quality.shape[0] != len(frozen_names):
        missing = sorted(set(frozen_names).difference(selected_quality["feature"]))
        raise ValueError(f"quality artifact misses selected features: {missing[:5]}")
    selected_quality = selected_quality.drop(columns=["family"], errors="ignore")
    importance = importance.merge(selected_quality, on="feature", how="left")
    forecast_importance = pd.read_csv(FORECAST_IMPORTANCE_PATH)
    required_forecast_columns = {"spec", "feature", "gain"}
    if not required_forecast_columns.issubset(forecast_importance.columns):
        raise ValueError("development forecast-importance artifact has an invalid schema")
    forecast_importance = (
        forecast_importance.loc[forecast_importance["spec"].eq("slow")]
        .groupby("feature", as_index=False)["gain"]
        .mean()
        .rename(columns={"gain": "development_forecast_gain"})
    )
    importance = importance.merge(forecast_importance, on="feature", how="left")
    importance["development_forecast_gain"] = importance[
        "development_forecast_gain"
    ].fillna(0.0)
    importance = importance.sort_values(
        ["train_test_gain_share", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    importance["domain_gain_rank"] = importance["train_test_gain_share"].rank(
        ascending=False, method="min"
    )
    importance["recent_domain_gain_rank"] = importance[
        "recent_train_test_gain_share"
    ].rank(ascending=False, method="min")
    importance["development_forecast_gain_rank"] = importance[
        "development_forecast_gain"
    ].rank(ascending=False, method="min")
    importance_alignment = float(
        importance["train_test_gain_share"].corr(
            importance["development_forecast_gain"], method="spearman"
        )
    )
    recent_importance_alignment = float(
        importance["recent_train_test_gain_share"].corr(
            importance["development_forecast_gain"], method="spearman"
        )
    )
    top_both = importance.loc[
        (importance["domain_gain_rank"] <= 50)
        & (importance["development_forecast_gain_rank"] <= 50),
        [
            "feature",
            "family",
            "domain_gain_rank",
            "development_forecast_gain_rank",
            "train_test_gain_share",
            "development_forecast_gain",
        ],
    ].sort_values("development_forecast_gain_rank")
    recent_top_both = importance.loc[
        (importance["recent_domain_gain_rank"] <= 50)
        & (importance["development_forecast_gain_rank"] <= 50),
        [
            "feature",
            "family",
            "recent_domain_gain_rank",
            "development_forecast_gain_rank",
            "recent_train_test_gain_share",
            "development_forecast_gain",
        ],
    ].sort_values("development_forecast_gain_rank")

    missingness = quality[quality["feature"].isin(frozen_names)][
        [
            "source",
            "feature",
            "family",
            "train_missing_rate",
            "test_missing_rate",
            "missing_rate_shift",
            "robust_median_shift_iqr",
        ]
    ].copy()
    missingness["absolute_missing_rate_shift"] = missingness["missing_rate_shift"].abs()
    missingness = missingness.sort_values(
        ["absolute_missing_rate_shift", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    raw_activity = _raw_activity_frame()
    family_gain = (
        importance.groupby("family", as_index=False)
        .agg(
            features=("feature", "size"),
            train_test_gain_share=("train_test_gain_share", "sum"),
            recent_train_test_gain_share=("recent_train_test_gain_share", "sum"),
            missingness_gain_share=("missingness_gain_share", "sum"),
            early_late_train_gain_share=("early_late_train_gain_share", "sum"),
            mean_absolute_missing_shift=("missing_rate_shift", lambda x: float(np.mean(np.abs(x)))),
        )
        .sort_values("train_test_gain_share", ascending=False)
        .reset_index(drop=True)
    )

    summary: dict[str, Any] = {
        "diagnostic_only": True,
        "feature_set": feature_set,
        "feature_count": len(frozen_names),
        "random_seed": RANDOM_SEED,
        "sampling": {
            "train_rows": int(train_indices.size),
            "test_rows": int(test_indices.size),
            "train_rows_per_month": int(sample_per_train_month),
            "train_month_min": int(sampled_train_months.min()),
            "train_month_max": int(sampled_train_months.max()),
            "test_sampling": "one deterministic random position per equal-width row-index bin",
        },
        "classifier_spec": {
            "model": "LightGBM binary classifier",
            "n_estimators": int(n_estimators),
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 400,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 20.0,
            "holdout_fraction": 0.25,
        },
        "train_test_domain_classifier": full_summary,
        "recent_train_test_domain_classifier": recent_summary,
        "missingness_only_domain_classifier": missing_summary,
        "early_late_train_classifier": time_summary,
        "selected_feature_missingness": {
            "largest_absolute_shift": float(missingness["absolute_missing_rate_shift"].max()),
            "features_over_1pct": int((missingness["absolute_missing_rate_shift"] > 0.01).sum()),
            "features_over_5pct": int((missingness["absolute_missing_rate_shift"] > 0.05).sum()),
            "mean_absolute_shift": float(missingness["absolute_missing_rate_shift"].mean()),
        },
        "raw_activity": raw_activity.to_dict(orient="records"),
        "family_domain_gain": family_gain.to_dict(orient="records"),
        "forecast_domain_importance_alignment": {
            "spearman": importance_alignment,
            "recent_train_spearman": recent_importance_alignment,
            "top_50_overlap_count": int(top_both.shape[0]),
            "top_50_overlap_features": top_both.to_dict(orient="records"),
            "recent_train_top_50_overlap_count": int(recent_top_both.shape[0]),
            "recent_train_top_50_overlap_features": recent_top_both.to_dict(
                orient="records"
            ),
            "caveat": (
                "Gain importance is model-dependent and correlated features can share or exchange "
                "importance; this is a diagnostic, not causal attribution."
            ),
        },
        "interpretation": (
            "A high domain AUC establishes covariate distinguishability, not target-concept drift. "
            "The early/late labeled-period AUC provides context for how much temporal drift already "
            "existed inside training."
        ),
    }

    importance.to_csv(OUTPUT_ROOT / "domain_shift_top_features.csv", index=False)
    missingness.to_csv(OUTPUT_ROOT / "domain_shift_missingness.csv", index=False)
    raw_activity.to_csv(OUTPUT_ROOT / "domain_shift_raw_activity.csv", index=False)
    family_gain.to_csv(OUTPUT_ROOT / "domain_shift_family_gain.csv", index=False)
    np.savez_compressed(
        OUTPUT_ROOT / "domain_shift_holdout_predictions.npz",
        full_holdout_indices=full_holdout,
        full_holdout_y=domain_target[full_holdout],
        full_holdout_probability=full_probability,
        recent_holdout_indices=recent_holdout,
        recent_holdout_y=recent_target[recent_holdout],
        recent_holdout_probability=recent_probability,
        missing_holdout_indices=missing_holdout,
        missing_holdout_y=domain_target[missing_holdout],
        missing_holdout_probability=missing_probability,
        time_holdout_indices=time_holdout,
        time_holdout_y=early_late_target[time_holdout],
        time_holdout_probability=time_probability,
    )
    output_summary.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    (OUTPUT_ROOT / "domain_shift_report.md").write_text(
        _markdown_report(summary, importance, raw_activity), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-train-month", type=int, default=800)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run(args.sample_per_train_month, args.n_estimators, args.overwrite)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
