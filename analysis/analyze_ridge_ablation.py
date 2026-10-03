"""Paired chronological comparison of completed Ridge feature ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import cosine_score, monthly_diagnostics, paired_month_block_bootstrap
from pipeline_config import DIAGNOSTIC_ROOT


COMPARISONS = (
    ("market_core", "all_core", False),
    ("all_core", "multiscale", False),
    ("multiscale", "multiscale_mechanics", False),
    ("multiscale", "multiscale_scale", True),
    ("multiscale", "multiscale_path", True),
    ("multiscale_scale", "multiscale_mechanics_scale", False),
    ("multiscale_mechanics", "multiscale_mechanics_scale", True),
)


def _label(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p").replace("-", "m")


def _load_best(feature_set: str) -> dict[str, np.ndarray | float]:
    summary = json.loads(
        (DIAGNOSTIC_ROOT / f"ridge_{feature_set}_summary.json").read_text(
            encoding="utf-8"
        )
    )
    penalty = float(summary["best_penalty"])
    archive = np.load(
        DIAGNOSTIC_ROOT / f"ridge_{feature_set}_development_oof.npz",
        allow_pickle=False,
    )
    return {
        "penalty": penalty,
        "row_indices": archive["row_indices"],
        "months": archive["months"],
        "target": archive["target"],
        "prediction": archive[_label(penalty)],
    }


def compare(parent_name: str, candidate_name: str, strict: bool) -> dict[str, object]:
    parent = _load_best(parent_name)
    candidate = _load_best(candidate_name)
    for key in ("row_indices", "months", "target"):
        if not np.array_equal(parent[key], candidate[key]):
            raise ValueError(f"unaligned ablation results for {parent_name}/{candidate_name}")
    y = np.asarray(parent["target"])
    months = np.asarray(parent["months"])
    parent_prediction = np.asarray(parent["prediction"])
    candidate_prediction = np.asarray(candidate["prediction"])
    parent_score = cosine_score(y, parent_prediction)
    candidate_score = cosine_score(y, candidate_prediction)

    fold_deltas: dict[str, float] = {}
    for fold_name, start, end in (("Dev1", 23, 34), ("Dev2", 35, 46), ("Dev3", 47, 58)):
        mask = (months >= start) & (months <= end)
        fold_deltas[fold_name] = cosine_score(y[mask], candidate_prediction[mask]) - cosine_score(
            y[mask], parent_prediction[mask]
        )
    parent_monthly = monthly_diagnostics(y, parent_prediction, months)
    candidate_monthly = monthly_diagnostics(y, candidate_prediction, months)
    monthly_delta = (
        candidate_monthly["cosine"].to_numpy()
        - parent_monthly["cosine"].to_numpy()
    )
    bootstrap3 = paired_month_block_bootstrap(
        y,
        candidate_prediction,
        parent_prediction,
        months,
        block_length_months=3,
        n_bootstrap=4_000,
    )
    bootstrap6 = paired_month_block_bootstrap(
        y,
        candidate_prediction,
        parent_prediction,
        months,
        block_length_months=6,
        n_bootstrap=4_000,
    )
    required_positive_folds = 3 if strict else 2
    positive_folds = sum(value > 0.0 for value in fold_deltas.values())
    improvement = candidate_score - parent_score
    promoted = bool(
        improvement > 0.0
        and positive_folds >= required_positive_folds
        and float(np.median(monthly_delta)) >= 0.0
        and improvement > bootstrap3.bootstrap_standard_error
        and fold_deltas["Dev3"] >= -bootstrap3.bootstrap_standard_error
    )
    return {
        "parent": parent_name,
        "candidate": candidate_name,
        "strict_satellite_rule": strict,
        "parent_penalty": parent["penalty"],
        "candidate_penalty": candidate["penalty"],
        "parent_pooled_cosine": parent_score,
        "candidate_pooled_cosine": candidate_score,
        "pooled_delta": improvement,
        "fold_deltas": fold_deltas,
        "positive_fold_count": positive_folds,
        "median_monthly_delta": float(np.median(monthly_delta)),
        "positive_monthly_delta_share": float(np.mean(monthly_delta > 0.0)),
        "bootstrap_3m_se": bootstrap3.bootstrap_standard_error,
        "bootstrap_3m_ci_low": bootstrap3.confidence_low,
        "bootstrap_3m_ci_high": bootstrap3.confidence_high,
        "bootstrap_3m_probability_better": bootstrap3.probability_a_better,
        "bootstrap_6m_se": bootstrap6.bootstrap_standard_error,
        "bootstrap_6m_ci_low": bootstrap6.confidence_low,
        "bootstrap_6m_ci_high": bootstrap6.confidence_high,
        "promoted": promoted,
    }


def main() -> None:
    results = [compare(*comparison) for comparison in COMPARISONS]
    output = DIAGNOSTIC_ROOT / "ridge_feature_family_comparisons.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    flattened = []
    for result in results:
        row = {key: value for key, value in result.items() if key != "fold_deltas"}
        row.update(result["fold_deltas"])
        flattened.append(row)
    pd.DataFrame(flattened).to_csv(
        DIAGNOSTIC_ROOT / "ridge_feature_family_comparisons.csv", index=False
    )
    print(pd.DataFrame(flattened).to_string(index=False))


if __name__ == "__main__":
    main()
