"""Postmortem diagnostics for the first public leaderboard submission.

This script is intentionally model-free: it only reads frozen development and
sealed predictions plus the labeled target table.  It quantifies the public
score gap, evaluates prediction-vector calibration with chronological
calibration/evaluation separation, and tests whether row order carries a
repeatable target pattern.  It never reads test predictions and never writes a
submission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "ms-capital-real-financial-market-forecasting"
DIAGNOSTIC_ROOT = ROOT / "artifacts" / "diagnostics"
OUTPUT_ROOT = DIAGNOSTIC_ROOT / "postmortem"
PUBLIC_SCORE = 0.124
RANDOM_SEED = 20260824


def cosine(y: np.ndarray, p: np.ndarray) -> float:
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    p64 = np.asarray(p, dtype=np.float64).reshape(-1)
    denominator = np.sqrt(np.dot(y64, y64) * np.dot(p64, p64))
    if y64.shape != p64.shape or denominator <= 0.0:
        raise ValueError("invalid cosine inputs")
    return float(np.dot(y64, p64) / denominator)


def rms(values: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values64 * values64)))


def _month_score_rows(
    y: np.ndarray, p: np.ndarray, months: np.ndarray, source: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month in np.unique(months):
        mask = months == month
        ym = y[mask]
        pm = p[mask]
        rows.append(
            {
                "source": source,
                "month": int(month),
                "n": int(mask.sum()),
                "cosine": cosine(ym, pm),
                "numerator": float(np.dot(ym, pm)),
                "target_ss": float(np.dot(ym, ym)),
                "prediction_ss": float(np.dot(pm, pm)),
                "target_mean": float(ym.mean()),
                "prediction_mean": float(pm.mean()),
                "target_rms": rms(ym),
                "prediction_rms": rms(pm),
            }
        )
    return rows


def _pooled_from_month_rows(frame: pd.DataFrame) -> float:
    return float(
        frame["numerator"].sum()
        / np.sqrt(frame["target_ss"].sum() * frame["prediction_ss"].sum())
    )


def load_evidence() -> dict[str, np.ndarray]:
    labels = feather.read_table(
        DATA_ROOT / "train" / "label.feather",
        columns=["month", "sample_id", "target"],
    )
    months_all = labels["month"].to_numpy().astype(np.int16, copy=False)
    sample_id = labels["sample_id"].to_numpy().astype(np.int64, copy=False)
    target_all = labels["target"].to_numpy().astype(np.float64, copy=False)
    if not np.array_equal(sample_id, np.arange(sample_id.size)):
        raise ValueError("training sample_id is not exact row order")

    with np.load(
        DIAGNOSTIC_ROOT
        / "gbdt_multiscale_mechanics_scale_development_oof.npz"
    ) as development:
        dev_rows = development["row_indices"].astype(np.int64, copy=True)
        dev_months = development["months"].astype(np.int16, copy=True)
        dev_target = development["target"].astype(np.float64, copy=True)
        dev_prediction = development["slow"].astype(np.float64, copy=True)
    if not np.array_equal(months_all[dev_rows], dev_months):
        raise ValueError("development month alignment failed")
    if not np.allclose(target_all[dev_rows], dev_target, rtol=0.0, atol=0.0):
        raise ValueError("development target alignment failed")

    with np.load(DIAGNOSTIC_ROOT / "sealed_audit_predictions.npz") as sealed:
        sealed_rows = sealed["row_index"].astype(np.int64, copy=True)
        sealed_months = sealed["month"].astype(np.int16, copy=True)
        sealed_prediction = sealed["prediction"].astype(np.float64, copy=True)
    sealed_target = target_all[sealed_rows]
    if not np.array_equal(months_all[sealed_rows], sealed_months):
        raise ValueError("sealed month alignment failed")

    return {
        "months_all": months_all,
        "sample_id": sample_id,
        "target_all": target_all,
        "dev_rows": dev_rows,
        "dev_months": dev_months,
        "dev_target": dev_target,
        "dev_prediction": dev_prediction,
        "sealed_rows": sealed_rows,
        "sealed_months": sealed_months,
        "sealed_target": sealed_target,
        "sealed_prediction": sealed_prediction,
    }


def score_gap_diagnostics(evidence: dict[str, np.ndarray]) -> dict[str, Any]:
    dev_rows = _month_score_rows(
        evidence["dev_target"],
        evidence["dev_prediction"],
        evidence["dev_months"],
        "development_oof",
    )
    sealed_rows = _month_score_rows(
        evidence["sealed_target"],
        evidence["sealed_prediction"],
        evidence["sealed_months"],
        "sealed_audit",
    )
    monthly = pd.DataFrame(dev_rows + sealed_rows).sort_values("month")
    monthly.to_csv(OUTPUT_ROOT / "historical_month_scores.csv", index=False)

    dev_score = cosine(evidence["dev_target"], evidence["dev_prediction"])
    sealed_score = cosine(evidence["sealed_target"], evidence["sealed_prediction"])
    stress_summary = json.loads(
        (DIAGNOSTIC_ROOT / "deployment_stress_summary.json").read_text(
            encoding="utf-8"
        )
    )
    stress_score = float(stress_summary["all_forecasts_pooled_cosine"])
    comparison_rows = []
    for name, score in (
        ("development_oof_months_23_58", dev_score),
        ("sealed_audit_months_59_70", sealed_score),
        ("historical_38_month_deployment_stress", stress_score),
    ):
        comparison_rows.append(
            {
                "reference": name,
                "reference_cosine": score,
                "public_cosine": PUBLIC_SCORE,
                "public_minus_reference": PUBLIC_SCORE - score,
                "relative_shortfall_fraction": (score - PUBLIC_SCORE) / score,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_ROOT / "score_gap_calibration.csv", index=False)

    rolling_rows: list[dict[str, Any]] = []
    for width in (3, 6, 12, 24, 38):
        for start in range(int(monthly["month"].min()), int(monthly["month"].max()) - width + 2):
            block = monthly[
                (monthly["month"] >= start)
                & (monthly["month"] <= start + width - 1)
            ]
            if block.shape[0] != width:
                continue
            rolling_rows.append(
                {
                    "width_months": width,
                    "start_month": start,
                    "end_month": start + width - 1,
                    "pooled_cosine": _pooled_from_month_rows(block),
                }
            )
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(OUTPUT_ROOT / "rolling_block_scores.csv", index=False)

    # Circular month-block bootstrap: an approximate regime-variation range,
    # not a confidence interval for the unknown test target.
    rng = np.random.default_rng(RANDOM_SEED)
    month_stats = monthly[["numerator", "target_ss", "prediction_ss"]].to_numpy()
    n_months = month_stats.shape[0]
    bootstrap_scores = np.empty(20_000, dtype=np.float64)
    for draw in range(bootstrap_scores.size):
        indices: list[int] = []
        while len(indices) < 38:
            start = int(rng.integers(0, n_months))
            indices.extend((start + offset) % n_months for offset in range(3))
        selected = month_stats[np.asarray(indices[:38], dtype=np.int64)]
        bootstrap_scores[draw] = selected[:, 0].sum() / np.sqrt(
            selected[:, 1].sum() * selected[:, 2].sum()
        )

    month_cosines = monthly["cosine"].to_numpy()
    rolling_12 = rolling.loc[rolling["width_months"] == 12, "pooled_cosine"]
    rolling_38 = rolling.loc[rolling["width_months"] == 38, "pooled_cosine"]
    sealed_monthly = monthly[monthly["source"] == "sealed_audit"].copy()
    highest_energy_index = sealed_monthly["target_ss"].idxmax()
    highest_energy = sealed_monthly.loc[highest_energy_index]
    sealed_without_highest_energy = sealed_monthly.drop(highest_energy_index)
    sealed_without_highest_energy_score = _pooled_from_month_rows(
        sealed_without_highest_energy
    )
    return {
        "public_score": PUBLIC_SCORE,
        "development_score": dev_score,
        "sealed_score": sealed_score,
        "deployment_stress_score": stress_score,
        "public_gap_from_development": PUBLIC_SCORE - dev_score,
        "public_gap_from_sealed": PUBLIC_SCORE - sealed_score,
        "public_gap_from_stress": PUBLIC_SCORE - stress_score,
        "development_monthly_median": float(
            monthly.loc[monthly["source"] == "development_oof", "cosine"].median()
        ),
        "sealed_monthly_median": float(sealed_monthly["cosine"].median()),
        "sealed_highest_target_energy_month": int(highest_energy["month"]),
        "sealed_highest_target_energy_month_cosine": float(
            highest_energy["cosine"]
        ),
        "sealed_highest_target_energy_month_target_ss_share": float(
            highest_energy["target_ss"] / sealed_monthly["target_ss"].sum()
        ),
        "sealed_highest_target_energy_month_numerator_share": float(
            highest_energy["numerator"] / sealed_monthly["numerator"].sum()
        ),
        "sealed_score_without_highest_target_energy_month": (
            sealed_without_highest_energy_score
        ),
        "public_gap_from_sealed_without_highest_energy_month": (
            PUBLIC_SCORE - sealed_without_highest_energy_score
        ),
        "historical_month_score_mean": float(month_cosines.mean()),
        "historical_month_score_std": float(month_cosines.std(ddof=1)),
        "historical_month_empirical_percentile_of_public": float(
            np.mean(month_cosines <= PUBLIC_SCORE)
        ),
        "historical_months_at_or_below_public": int(
            np.sum(month_cosines <= PUBLIC_SCORE)
        ),
        "historical_month_count": int(month_cosines.size),
        "rolling_12_month_min": float(rolling_12.min()),
        "rolling_12_month_median": float(rolling_12.median()),
        "rolling_38_month_min": float(rolling_38.min()),
        "rolling_38_month_median": float(rolling_38.median()),
        "block_bootstrap_38_month_q025": float(
            np.quantile(bootstrap_scores, 0.025)
        ),
        "block_bootstrap_38_month_median": float(np.median(bootstrap_scores)),
        "block_bootstrap_38_month_q975": float(
            np.quantile(bootstrap_scores, 0.975)
        ),
        "block_bootstrap_probability_at_or_below_public": float(
            np.mean(bootstrap_scores <= PUBLIC_SCORE)
        ),
        "block_bootstrap_draws": int(bootstrap_scores.size),
        "bootstrap_note": (
            "Approximate historical regime variation from 3-month circular blocks; "
            "not a confidence interval for future targets."
        ),
    }


@dataclass(frozen=True)
class CalibrationCandidate:
    name: str
    base: str = "raw"
    parameter: float | None = None
    affine: str | None = None


CALIBRATION_CANDIDATES = (
    CalibrationCandidate("raw"),
    CalibrationCandidate("center_batch", base="center_batch"),
    CalibrationCandidate("center_prior_unit", base="center_prior_unit"),
    CalibrationCandidate("affine_raw", affine="raw"),
    CalibrationCandidate("affine_unit_rms", affine="unit_rms"),
    CalibrationCandidate("winsor_0p1pct", base="winsor", parameter=0.001),
    CalibrationCandidate("winsor_0p5pct", base="winsor", parameter=0.005),
    CalibrationCandidate("winsor_1pct", base="winsor", parameter=0.01),
    CalibrationCandidate("winsor_2pct", base="winsor", parameter=0.02),
    CalibrationCandidate(
        "winsor_0p5pct_affine", base="winsor", parameter=0.005, affine="base"
    ),
    CalibrationCandidate(
        "winsor_1pct_affine", base="winsor", parameter=0.01, affine="base"
    ),
    CalibrationCandidate("power_0p50", base="power", parameter=0.50),
    CalibrationCandidate("power_0p75", base="power", parameter=0.75),
    CalibrationCandidate("power_1p25", base="power", parameter=1.25),
    CalibrationCandidate("power_1p50", base="power", parameter=1.50),
    CalibrationCandidate(
        "power_0p75_affine", base="power", parameter=0.75, affine="base"
    ),
    CalibrationCandidate(
        "power_1p25_affine", base="power", parameter=1.25, affine="base"
    ),
)


def _unit_rms(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / rms(values)


def _fit_base(
    candidate: CalibrationCandidate, prediction: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    p = np.asarray(prediction, dtype=np.float64)
    parameters: dict[str, float] = {}
    if candidate.base == "raw":
        transformed = p.copy()
    elif candidate.base == "center_batch":
        parameters["calibration_prediction_mean"] = float(p.mean())
        transformed = p - p.mean()
    elif candidate.base == "center_prior_unit":
        unit = _unit_rms(p)
        parameters["prior_unit_mean"] = float(unit.mean())
        transformed = unit - parameters["prior_unit_mean"]
    elif candidate.base == "winsor":
        unit = _unit_rms(p)
        q = float(candidate.parameter)
        parameters["lower"] = float(np.quantile(unit, q))
        parameters["upper"] = float(np.quantile(unit, 1.0 - q))
        transformed = np.clip(unit, parameters["lower"], parameters["upper"])
    elif candidate.base == "power":
        unit = _unit_rms(p)
        gamma = float(candidate.parameter)
        parameters["gamma"] = gamma
        transformed = np.sign(unit) * np.power(np.abs(unit), gamma)
    else:
        raise ValueError(f"unknown base transform {candidate.base}")
    return transformed, parameters


def _apply_base(
    candidate: CalibrationCandidate,
    prediction: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64)
    if candidate.base == "raw":
        return p.copy()
    if candidate.base == "center_batch":
        # Label-free centering of the complete evaluation vector.
        return p - p.mean()
    if candidate.base == "center_prior_unit":
        return _unit_rms(p) - parameters["prior_unit_mean"]
    if candidate.base == "winsor":
        return np.clip(
            _unit_rms(p), parameters["lower"], parameters["upper"]
        )
    if candidate.base == "power":
        unit = _unit_rms(p)
        return np.sign(unit) * np.power(np.abs(unit), parameters["gamma"])
    raise ValueError(f"unknown base transform {candidate.base}")


def _fit_candidate(
    candidate: CalibrationCandidate, prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    transformed, parameters = _fit_base(candidate, prediction)
    if candidate.affine is not None:
        affine_input = (
            _unit_rms(prediction)
            if candidate.affine == "unit_rms"
            else transformed
        )
        centered = affine_input - affine_input.mean()
        variance_sum = float(np.dot(centered, centered))
        slope = float(
            np.dot(centered, target - target.mean()) / variance_sum
        )
        intercept = float(target.mean() - slope * affine_input.mean())
        parameters["affine_slope"] = slope
        parameters["affine_intercept"] = intercept
    return parameters


def _apply_candidate(
    candidate: CalibrationCandidate,
    prediction: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    transformed = _apply_base(candidate, prediction, parameters)
    if candidate.affine is None:
        return transformed
    affine_input = (
        _unit_rms(prediction)
        if candidate.affine == "unit_rms"
        else transformed
    )
    return parameters["affine_slope"] * affine_input + parameters["affine_intercept"]


def rolling_calibration_diagnostics(
    evidence: dict[str, np.ndarray]
) -> dict[str, Any]:
    months = evidence["dev_months"]
    target = evidence["dev_target"]
    prediction = evidence["dev_prediction"]
    folds = (
        ("Dev2", 23, 34, 35, 46),
        ("Dev3", 23, 46, 47, 58),
    )
    result_rows: list[dict[str, Any]] = []
    applied: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for fold, calibration_start, calibration_end, evaluation_start, evaluation_end in folds:
        calibration_mask = (months >= calibration_start) & (months <= calibration_end)
        evaluation_mask = (months >= evaluation_start) & (months <= evaluation_end)
        p_cal = prediction[calibration_mask]
        y_cal = target[calibration_mask]
        p_eval = prediction[evaluation_mask]
        y_eval = target[evaluation_mask]
        raw_score = cosine(y_eval, p_eval)
        for candidate in CALIBRATION_CANDIDATES:
            parameters = _fit_candidate(candidate, p_cal, y_cal)
            adjusted = _apply_candidate(candidate, p_eval, parameters)
            score = cosine(y_eval, adjusted)
            result_rows.append(
                {
                    "fold": fold,
                    "calibration_months": f"{calibration_start}-{calibration_end}",
                    "evaluation_months": f"{evaluation_start}-{evaluation_end}",
                    "candidate": candidate.name,
                    "cosine": score,
                    "delta_from_raw": score - raw_score,
                    "calibration_rows": int(calibration_mask.sum()),
                    "evaluation_rows": int(evaluation_mask.sum()),
                    "adjusted_prediction_mean": float(adjusted.mean()),
                    "adjusted_prediction_rms": rms(adjusted),
                    "parameters_json": json.dumps(parameters, sort_keys=True),
                }
            )
            applied[(fold, candidate.name)] = (y_eval, adjusted)
    frame = pd.DataFrame(result_rows)
    frame.to_csv(OUTPUT_ROOT / "calibration_rolling_results.csv", index=False)

    pooled_rows: list[dict[str, Any]] = []
    for candidate in CALIBRATION_CANDIDATES:
        y_pooled = np.concatenate(
            [applied[(fold, candidate.name)][0] for fold, *_ in folds]
        )
        p_pooled = np.concatenate(
            [applied[(fold, candidate.name)][1] for fold, *_ in folds]
        )
        fold_scores = frame.loc[frame["candidate"] == candidate.name]
        pooled_rows.append(
            {
                "candidate": candidate.name,
                "rolling_pooled_cosine_dev2_dev3": cosine(y_pooled, p_pooled),
                "mean_fold_cosine": float(fold_scores["cosine"].mean()),
                "worst_fold_cosine": float(fold_scores["cosine"].min()),
                "mean_delta_from_raw": float(
                    fold_scores["delta_from_raw"].mean()
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values(
        "rolling_pooled_cosine_dev2_dev3", ascending=False
    )
    pooled.to_csv(OUTPUT_ROOT / "calibration_candidate_summary.csv", index=False)

    dev2 = frame[frame["fold"] == "Dev2"].sort_values(
        ["cosine", "candidate"], ascending=[False, True]
    )
    selected_name = str(dev2.iloc[0]["candidate"])
    selected_dev2 = float(dev2.iloc[0]["cosine"])
    selected_dev3 = float(
        frame.loc[
            (frame["fold"] == "Dev3") & (frame["candidate"] == selected_name),
            "cosine",
        ].iloc[0]
    )
    raw_dev3 = float(
        frame.loc[
            (frame["fold"] == "Dev3") & (frame["candidate"] == "raw"),
            "cosine",
        ].iloc[0]
    )
    center_rows = frame[frame["candidate"] == "center_batch"]
    affine_rows = frame[frame["candidate"] == "affine_unit_rms"]
    dev3 = frame[frame["fold"] == "Dev3"].sort_values(
        ["cosine", "candidate"], ascending=[False, True]
    )
    best_dev3_posthoc = dev3.iloc[0]
    return {
        "protocol": (
            "Fit transform/calibration parameters on Dev1 OOF then evaluate Dev2; "
            "fit on Dev1+Dev2 OOF then evaluate Dev3. Candidate choice is made "
            "using Dev2 only and audited on Dev3."
        ),
        "candidate_count": len(CALIBRATION_CANDIDATES),
        "dev2_selected_candidate": selected_name,
        "dev2_selected_cosine": selected_dev2,
        "strict_next_fold_dev3_cosine": selected_dev3,
        "strict_next_fold_dev3_raw_cosine": raw_dev3,
        "strict_next_fold_delta": selected_dev3 - raw_dev3,
        "best_dev3_candidate_posthoc_diagnostic_only": str(
            best_dev3_posthoc["candidate"]
        ),
        "best_dev3_delta_posthoc_diagnostic_only": float(
            best_dev3_posthoc["delta_from_raw"]
        ),
        "center_batch_dev2_delta": float(
            center_rows.loc[center_rows["fold"] == "Dev2", "delta_from_raw"].iloc[0]
        ),
        "center_batch_dev3_delta": float(
            center_rows.loc[center_rows["fold"] == "Dev3", "delta_from_raw"].iloc[0]
        ),
        "affine_unit_dev2_delta": float(
            affine_rows.loc[affine_rows["fold"] == "Dev2", "delta_from_raw"].iloc[0]
        ),
        "affine_unit_dev3_delta": float(
            affine_rows.loc[affine_rows["fold"] == "Dev3", "delta_from_raw"].iloc[0]
        ),
        "best_rolling_pooled_candidate_diagnostic_only": str(
            pooled.iloc[0]["candidate"]
        ),
        "best_rolling_pooled_cosine_diagnostic_only": float(
            pooled.iloc[0]["rolling_pooled_cosine_dev2_dev3"]
        ),
        "raw_rolling_pooled_cosine": float(
            pooled.loc[
                pooled["candidate"] == "raw",
                "rolling_pooled_cosine_dev2_dev3",
            ].iloc[0]
        ),
    }


def _pair_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    return float(np.dot(x_centered, y_centered) / denominator)


def _lag_rows(
    values: np.ndarray,
    months: np.ndarray,
    series: str,
    lags: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lag in lags:
        valid = months[lag:] == months[:-lag]
        left = values[:-lag][valid]
        right = values[lag:][valid]
        per_month = []
        for month in np.unique(months):
            xm = values[months == month]
            if xm.size > lag:
                per_month.append(_pair_correlation(xm[:-lag], xm[lag:]))
        rows.append(
            {
                "series": series,
                "lag_rows": lag,
                "n_pairs": int(left.size),
                "pooled_pearson": _pair_correlation(left, right),
                "pooled_uncentered_cosine": cosine(left, right),
                "monthly_median_pearson": float(np.median(per_month)),
                "monthly_p10_pearson": float(np.quantile(per_month, 0.10)),
                "monthly_p90_pearson": float(np.quantile(per_month, 0.90)),
            }
        )
    return rows


def _rank_fraction(months: np.ndarray) -> np.ndarray:
    result = np.empty(months.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(months[1:] != months[:-1]) + 1]
    ends = np.r_[starts[1:], months.size]
    for start, end in zip(starts, ends, strict=True):
        count = end - start
        result[start:end] = (np.arange(count, dtype=np.float64) + 0.5) / count
    return result


def _fit_bucket_prediction(
    fit_position: np.ndarray,
    fit_target: np.ndarray,
    evaluation_position: np.ndarray,
    bins: int,
) -> np.ndarray:
    fit_bin = np.minimum((fit_position * bins).astype(np.int64), bins - 1)
    evaluation_bin = np.minimum(
        (evaluation_position * bins).astype(np.int64), bins - 1
    )
    counts = np.bincount(fit_bin, minlength=bins).astype(np.float64)
    sums = np.bincount(fit_bin, weights=fit_target, minlength=bins)
    global_mean = float(fit_target.mean())
    means = np.divide(
        sums,
        counts,
        out=np.full(bins, global_mean, dtype=np.float64),
        where=counts > 0,
    )
    return means[evaluation_bin]


def row_order_diagnostics(evidence: dict[str, np.ndarray]) -> dict[str, Any]:
    months_all = evidence["months_all"]
    target_all = evidence["target_all"]
    row_position = _rank_fraction(months_all)
    lags = (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)
    autocorrelation_rows = _lag_rows(target_all, months_all, "target", lags)

    dev_months = evidence["dev_months"]
    dev_target = evidence["dev_target"]
    dev_prediction = evidence["dev_prediction"]
    # Remove the best no-intercept linear projection of the model prediction.
    residual = np.empty_like(dev_target)
    for start, end in ((23, 34), (35, 46), (47, 58)):
        mask = (dev_months >= start) & (dev_months <= end)
        y = dev_target[mask]
        p = dev_prediction[mask]
        beta = float(np.dot(y, p) / np.dot(p, p))
        residual[mask] = y - beta * p
    autocorrelation_rows.extend(
        _lag_rows(residual, dev_months, "development_model_residual", lags)
    )
    autocorrelation = pd.DataFrame(autocorrelation_rows)
    autocorrelation.to_csv(
        OUTPUT_ROOT / "row_order_autocorrelation.csv", index=False
    )

    baseline_rows: list[dict[str, Any]] = []
    folds = (
        ("Dev2", 0, 34, 35, 46),
        ("Dev3", 0, 46, 47, 58),
        ("Sealed", 0, 58, 59, 70),
    )
    month_counts = np.bincount(months_all.astype(np.int64))
    fixed_period = int(np.median(month_counts[month_counts > 0]))
    for fold, train_start, train_end, eval_start, eval_end in folds:
        fit_mask = (months_all >= train_start) & (months_all <= train_end)
        eval_mask = (months_all >= eval_start) & (months_all <= eval_end)
        local_eval_row = np.arange(int(eval_mask.sum()), dtype=np.float64)
        train_row = np.arange(int(fit_mask.sum()), dtype=np.float64)
        constant_prediction = np.full(
            int(eval_mask.sum()), float(target_all[fit_mask].mean())
        )
        baseline_rows.append(
            {
                "fold": fold,
                "method": "constant_training_mean_control",
                "bins": 0,
                "cosine": cosine(target_all[eval_mask], constant_prediction),
                "prediction_rms": rms(constant_prediction),
            }
        )
        for bins in (8, 16, 32, 64, 128, 256, 512):
            rank_prediction = _fit_bucket_prediction(
                row_position[fit_mask],
                target_all[fit_mask],
                row_position[eval_mask],
                bins,
            )
            baseline_rows.append(
                {
                    "fold": fold,
                    "method": "true_within_month_rank_not_directly_deployable",
                    "bins": bins,
                    "cosine": cosine(target_all[eval_mask], rank_prediction),
                    "prediction_rms": rms(rank_prediction),
                }
            )
            # A deployable approximation when test IDs restart at zero: use a
            # fixed estimated month length and no hidden month boundaries.
            fit_phase = (train_row % fixed_period + 0.5) / fixed_period
            eval_phase = (local_eval_row % fixed_period + 0.5) / fixed_period
            phase_prediction = _fit_bucket_prediction(
                fit_phase,
                target_all[fit_mask],
                eval_phase,
                bins,
            )
            baseline_rows.append(
                {
                    "fold": fold,
                    "method": "fixed_period_phase_deployable_approximation",
                    "bins": bins,
                    "cosine": cosine(target_all[eval_mask], phase_prediction),
                    "prediction_rms": rms(phase_prediction),
                }
            )
    baselines = pd.DataFrame(baseline_rows)
    baselines.to_csv(OUTPUT_ROOT / "row_order_baselines.csv", index=False)

    target_lag1 = autocorrelation[
        (autocorrelation["series"] == "target")
        & (autocorrelation["lag_rows"] == 1)
    ].iloc[0]
    residual_lag1 = autocorrelation[
        (autocorrelation["series"] == "development_model_residual")
        & (autocorrelation["lag_rows"] == 1)
    ].iloc[0]
    true_rank = baselines[
        baselines["method"] == "true_within_month_rank_not_directly_deployable"
    ]
    fixed_phase = baselines[
        baselines["method"] == "fixed_period_phase_deployable_approximation"
    ]
    best_true = true_rank.sort_values("cosine", ascending=False).iloc[0]
    best_fixed = fixed_phase.sort_values("cosine", ascending=False).iloc[0]
    fixed_dev2 = fixed_phase[fixed_phase["fold"] == "Dev2"].sort_values(
        ["cosine", "bins"], ascending=[False, True]
    )
    selected_fixed_bins = int(fixed_dev2.iloc[0]["bins"])
    selected_fixed = fixed_phase[fixed_phase["bins"] == selected_fixed_bins].set_index(
        "fold"
    )
    constant = baselines[
        baselines["method"] == "constant_training_mean_control"
    ].set_index("fold")
    return {
        "sample_id_definition": (
            "Train IDs are contiguous row indices 0..N-1; test IDs independently "
            "restart at 0. Therefore absolute train IDs cannot transfer to test."
        ),
        "test_month_boundaries_available": False,
        "target_lag1_pooled_pearson": float(target_lag1["pooled_pearson"]),
        "target_lag1_monthly_median_pearson": float(
            target_lag1["monthly_median_pearson"]
        ),
        "development_residual_lag1_pooled_pearson": float(
            residual_lag1["pooled_pearson"]
        ),
        "best_true_rank_baseline_cosine_diagnostic_only": float(
            best_true["cosine"]
        ),
        "best_true_rank_baseline_fold": str(best_true["fold"]),
        "best_true_rank_baseline_bins": int(best_true["bins"]),
        "best_fixed_phase_baseline_cosine": float(best_fixed["cosine"]),
        "best_fixed_phase_baseline_fold": str(best_fixed["fold"]),
        "best_fixed_phase_baseline_bins": int(best_fixed["bins"]),
        "fixed_phase_bins_selected_on_dev2": selected_fixed_bins,
        "fixed_phase_selected_dev2_cosine": float(selected_fixed.loc["Dev2", "cosine"]),
        "fixed_phase_selected_dev3_cosine": float(selected_fixed.loc["Dev3", "cosine"]),
        "fixed_phase_selected_sealed_cosine": float(
            selected_fixed.loc["Sealed", "cosine"]
        ),
        "fixed_phase_selected_minus_constant_dev2": float(
            selected_fixed.loc["Dev2", "cosine"] - constant.loc["Dev2", "cosine"]
        ),
        "fixed_phase_selected_minus_constant_dev3": float(
            selected_fixed.loc["Dev3", "cosine"] - constant.loc["Dev3", "cosine"]
        ),
        "fixed_phase_selected_minus_constant_sealed": float(
            selected_fixed.loc["Sealed", "cosine"]
            - constant.loc["Sealed", "cosine"]
        ),
        "fixed_period_rows": fixed_period,
        "interpretation_guardrail": (
            "Lag autocorrelation can motivate row-local feature aggregation or a "
            "sequence model, but using neighboring targets would be leakage. The "
            "true-rank baseline also requires hidden month boundaries unavailable "
            "in test and is not a valid submission feature as implemented."
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    gap = score_gap_diagnostics(evidence)
    calibration = rolling_calibration_diagnostics(evidence)
    row_order = row_order_diagnostics(evidence)

    historical_months = pd.read_csv(OUTPUT_ROOT / "historical_month_scores.csv")
    calibration_rows = pd.read_csv(
        OUTPUT_ROOT / "calibration_rolling_results.csv"
    )
    row_autocorrelation = pd.read_csv(
        OUTPUT_ROOT / "row_order_autocorrelation.csv"
    )
    validation_checks = {
        "historical_months_are_exactly_23_through_70": bool(
            np.array_equal(historical_months["month"].to_numpy(), np.arange(23, 71))
        ),
        "development_raw_score_reconciles": bool(
            np.isclose(
                gap["development_score"],
                cosine(evidence["dev_target"], evidence["dev_prediction"]),
                rtol=0.0,
                atol=1e-14,
            )
        ),
        "sealed_raw_score_reconciles": bool(
            np.isclose(
                gap["sealed_score"],
                cosine(evidence["sealed_target"], evidence["sealed_prediction"]),
                rtol=0.0,
                atol=1e-14,
            )
        ),
        "calibration_has_two_strict_chronological_evaluation_folds": bool(
            set(calibration_rows["fold"]) == {"Dev2", "Dev3"}
            and calibration_rows.shape[0] == 2 * len(CALIBRATION_CANDIDATES)
        ),
        "all_calibration_scores_finite": bool(
            np.all(np.isfinite(calibration_rows["cosine"]))
        ),
        "all_row_autocorrelations_finite": bool(
            np.all(np.isfinite(row_autocorrelation["pooled_pearson"]))
        ),
    }
    if not all(validation_checks.values()):
        raise AssertionError(f"postmortem validation failed: {validation_checks}")

    implications = [
        {
            "rank": 1,
            "finding": "The test regime is weaker than the historical regimes represented by validation.",
            "evidence": (
                f"Public {PUBLIC_SCORE:.6f} trails development by "
                f"{gap['development_score'] - PUBLIC_SCORE:.6f} and stress by "
                f"{gap['deployment_stress_score'] - PUBLIC_SCORE:.6f}; it is below "
                f"the minimum historical rolling 12-month score "
                f"{gap['rolling_12_month_min']:.6f}."
            ),
            "status": "strongest_supported_explanation_but_exact_test_targets_unavailable",
        },
        {
            "rank": 2,
            "finding": "The sealed headline overstated the ordinary validation level because one volatile month dominated it.",
            "evidence": (
                f"Month {gap['sealed_highest_target_energy_month']} supplied "
                f"{gap['sealed_highest_target_energy_month_target_ss_share']:.1%} "
                f"of sealed target energy at cosine "
                f"{gap['sealed_highest_target_energy_month_cosine']:.6f}; removing it "
                f"reduces sealed cosine from {gap['sealed_score']:.6f} to "
                f"{gap['sealed_score_without_highest_target_energy_month']:.6f}."
            ),
            "status": "verified_validation_concentration",
        },
        {
            "rank": 3,
            "finding": "Affine, centering, winsorization, and power transforms are not a material fix.",
            "evidence": (
                f"The transform selected on Dev2 changed Dev3 by "
                f"{calibration['strict_next_fold_delta']:+.6f}; even the best "
                f"post-hoc Dev3 transform changed it by only "
                f"{calibration['best_dev3_delta_posthoc_diagnostic_only']:+.6f}."
            ),
            "status": "verified_not_promising",
        },
        {
            "rank": 4,
            "finding": "Sample ID and row order do not provide a stable overlooked signal.",
            "evidence": (
                f"Within-month target lag-1 correlation is "
                f"{row_order['target_lag1_pooled_pearson']:.6f}; the fixed-phase "
                f"rule selected on Dev2 adds only "
                f"{row_order['fixed_phase_selected_minus_constant_dev3']:+.6f} "
                f"over a constant-mean control on Dev3 and "
                f"{row_order['fixed_phase_selected_minus_constant_sealed']:+.6f} "
                f"sealed."
            ),
            "status": "verified_unstable_and_high_leakage_risk",
        },
    ]
    payload = {
        "status": "complete_existing_artifacts_only_no_retraining",
        "public_leaderboard": {
            "score": PUBLIC_SCORE,
            "rank": 115,
            "participants": 151,
            "public_fraction_reported_by_competition": 0.49,
        },
        "score_gap": gap,
        "rolling_calibration": calibration,
        "row_order": row_order,
        "validation": {
            "assessment": "share_with_caveats",
            "checks": validation_checks,
            "reason": (
                "All reproducible calculations reconcile, but the exact cause of "
                "the public score cannot be verified without public test labels."
            ),
        },
        "ranked_implications": implications,
        "limitations": [
            "The public target and public/private membership are unavailable, so the exact causal decomposition of the leaderboard gap is impossible.",
            "The historical block bootstrap describes observed historical regime variation; it is not a future confidence interval.",
            "Postmortem transform comparisons are diagnostics and must be revalidated before any new submission.",
        ],
        "outputs": [
            "historical_month_scores.csv",
            "score_gap_calibration.csv",
            "rolling_block_scores.csv",
            "calibration_rolling_results.csv",
            "calibration_candidate_summary.csv",
            "row_order_autocorrelation.csv",
            "row_order_baselines.csv",
        ],
    }
    (OUTPUT_ROOT / "postmortem_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
