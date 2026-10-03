"""Reproduce the bounded v2 model-selection and sealed-audit comparison.

The development selection uses only months 23--46.  Months 47--58 are a
confirmation block and months 59--70 are reported only as the already-opened
sealed audit.  This script never fits a model; it reads immutable predictions.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from modeling import cosine_score, paired_month_block_bootstrap
from pipeline_config import DATA_ROOT, RANDOM_SEED


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "artifacts" / "diagnostics"
EXPERIMENT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "experiments"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "v2" / "diagnostics"


def _load_v2(stems: tuple[str, ...]) -> dict[str, np.ndarray]:
    parts = [np.load(EXPERIMENT_ROOT / f"{stem}_oof.npz") for stem in stems]
    result = {
        key: np.concatenate([np.asarray(part[key]) for part in parts])
        for key in ("row_indices", "months", "target", "prediction")
    }
    order = np.argsort(result["row_indices"])
    result = {key: value[order] for key, value in result.items()}
    if len(np.unique(result["row_indices"])) != len(result["row_indices"]):
        raise ValueError("v2 OOF rows overlap")
    return result


def _power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), exponent)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _score_rows(
    y: np.ndarray, months: np.ndarray, predictions: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    periods = {
        "Dev1": (23, 34),
        "Dev2": (35, 46),
        "Dev3": (47, 58),
        "Development pooled": (23, 58),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        selected = (months >= start) & (months <= end)
        for name, prediction in predictions.items():
            rows.append(
                {
                    "period": period,
                    "month_start": start,
                    "month_end": end,
                    "model": name,
                    "rows": int(selected.sum()),
                    "cosine": cosine_score(y[selected], prediction[selected]),
                }
            )
    return rows


def _bootstrap_dict(value: object) -> dict[str, object]:
    payload = asdict(value)
    payload.pop("differences", None)
    return payload


def run(*, overwrite: bool = False) -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "model_selection_summary.json"
    fold_path = OUTPUT_ROOT / "model_selection_fold_scores.csv"
    monthly_path = OUTPUT_ROOT / "model_selection_monthly_scores.csv"
    if not overwrite and any(path.exists() for path in (summary_path, fold_path, monthly_path)):
        raise FileExistsError("v2 selection outputs already exist")

    joint = _load_v2(
        (
            "sequence_base_plus_sequence_all_capacity_Dev1-Dev2",
            "sequence_base_plus_sequence_all_capacity_Dev3",
        )
    )
    base_capacity = _load_v2(
        (
            "sequence_base_only_capacity_Dev1-Dev2",
            "sequence_base_only_capacity_Dev3",
        )
    )
    v1 = np.load(V1_ROOT / "gbdt_multiscale_mechanics_scale_development_oof.npz")
    if not (
        np.array_equal(joint["row_indices"], base_capacity["row_indices"])
        and np.array_equal(joint["row_indices"], v1["row_indices"])
        and np.array_equal(joint["months"], base_capacity["months"])
        and np.array_equal(joint["months"], v1["months"])
        and np.allclose(joint["target"], base_capacity["target"], rtol=0.0, atol=0.0)
        and np.allclose(joint["target"], v1["target"], rtol=0.0, atol=0.0)
    ):
        raise ValueError("development OOF artifacts are not exactly aligned")

    y = joint["target"].astype(np.float64, copy=False)
    months = joint["months"]
    p_joint = joint["prediction"].astype(np.float64, copy=False)
    p_base = base_capacity["prediction"].astype(np.float64, copy=False)
    p_v1 = np.asarray(v1["slow"], dtype=np.float64)
    screen = (months >= 23) & (months <= 46)
    confirmation = (months >= 47) & (months <= 58)

    power_rows: list[dict[str, object]] = []
    exponents = (0.8, 1.0, 1.1, 1.2, 1.3, 1.4)
    for exponent in exponents:
        transformed = _power(p_joint, exponent)
        power_rows.append(
            {
                "exponent": exponent,
                "screen_cosine": cosine_score(y[screen], transformed[screen]),
                "confirmation_cosine": cosine_score(
                    y[confirmation], transformed[confirmation]
                ),
                "development_cosine": cosine_score(y, transformed),
            }
        )
    selected_power = max(power_rows, key=lambda row: row["screen_cosine"])
    exponent = float(selected_power["exponent"])
    if exponent != 1.2:
        raise AssertionError(f"expected frozen exponent 1.2, observed {exponent}")

    # One constrained blend search, also restricted to Dev1+Dev2.  Components
    # are RMS-normalized before mixing so weights represent directional weight.
    z_joint = p_joint / _rms(p_joint[screen])
    z_v1 = p_v1 / _rms(p_v1[screen])
    blend_rows: list[dict[str, object]] = []
    for v1_weight in np.linspace(0.0, 1.0, 11):
        raw = (1.0 - v1_weight) * z_joint + v1_weight * z_v1
        prediction = _power(raw, exponent)
        blend_rows.append(
            {
                "v1_weight": float(v1_weight),
                "capacity_weight": float(1.0 - v1_weight),
                "screen_cosine": cosine_score(y[screen], prediction[screen]),
                "confirmation_cosine": cosine_score(
                    y[confirmation], prediction[confirmation]
                ),
                "development_cosine": cosine_score(y, prediction),
            }
        )
    selected_blend = max(blend_rows, key=lambda row: row["screen_cosine"])
    if float(selected_blend["v1_weight"]) != 0.0:
        raise AssertionError("frozen blend was expected to select pure v2 capacity")

    calibrated = _power(p_joint, exponent)
    predictions = {
        "v1 slow raw": p_v1,
        "v2 base-only capacity raw": p_base,
        "v2 base+path capacity raw": p_joint,
        "v2 final q=1.2": calibrated,
    }
    fold_rows = _score_rows(y, months, predictions)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    monthly_rows: list[dict[str, object]] = []
    for month in np.unique(months):
        selected = months == month
        for name, prediction in predictions.items():
            monthly_rows.append(
                {
                    "month": int(month),
                    "model": name,
                    "rows": int(selected.sum()),
                    "cosine": cosine_score(y[selected], prediction[selected]),
                }
            )
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)

    capacity_bootstrap = paired_month_block_bootstrap(
        y,
        p_joint,
        p_v1,
        months,
        block_length_months=3,
        n_bootstrap=8_000,
        random_state=RANDOM_SEED,
    )
    sequence_bootstrap = paired_month_block_bootstrap(
        y,
        p_joint,
        p_base,
        months,
        block_length_months=3,
        n_bootstrap=8_000,
        random_state=RANDOM_SEED + 1,
    )

    sealed_v2 = np.load(
        EXPERIMENT_ROOT
        / "sequence_base_plus_sequence_all_capacity_SealedAudit_oof.npz"
    )
    sealed_v1 = np.load(V1_ROOT / "sealed_audit_predictions.npz")
    labels = feather.read_table(
        DATA_ROOT / "train" / "label.feather", columns=["target"]
    )["target"].to_numpy().astype(np.float64, copy=False)
    if not (
        np.array_equal(sealed_v2["row_indices"], sealed_v1["row_index"])
        and np.array_equal(sealed_v2["months"], sealed_v1["month"])
    ):
        raise ValueError("sealed-audit prediction artifacts are not aligned")
    sealed_y = labels[sealed_v1["row_index"]]
    if not np.allclose(sealed_y, sealed_v2["target"], rtol=0.0, atol=0.0):
        raise ValueError("sealed-audit targets differ")
    sealed_p_v1 = np.asarray(sealed_v1["prediction"], dtype=np.float64)
    sealed_p_v2 = np.asarray(sealed_v2["prediction"], dtype=np.float64)
    exclude_66 = sealed_v2["months"] != 66

    payload: dict[str, object] = {
        "status": "complete_frozen_selection_and_one_time_sealed_audit",
        "selection_protocol": {
            "screen_months": [23, 46],
            "confirmation_months": [47, 58],
            "sealed_audit_months": [59, 70],
            "primary_metric": "pooled uncentered cosine over all rows",
            "power_candidates": list(exponents),
            "blend_candidates_v1_weight": [
                float(value) for value in np.linspace(0.0, 1.0, 11)
            ],
            "selected_feature_set": "base_plus_sequence_all",
            "selected_model_spec": "capacity",
            "selected_signed_power": exponent,
            "selected_v1_blend_weight": 0.0,
        },
        "power_selection": power_rows,
        "blend_selection": blend_rows,
        "selected_power_row": selected_power,
        "selected_blend_row": selected_blend,
        "development": {
            "v1_raw_cosine": cosine_score(y, p_v1),
            "v2_base_capacity_raw_cosine": cosine_score(y, p_base),
            "v2_joint_capacity_raw_cosine": cosine_score(y, p_joint),
            "v2_final_q1p2_cosine": cosine_score(y, calibrated),
            "joint_minus_v1_raw": cosine_score(y, p_joint)
            - cosine_score(y, p_v1),
            "sequence_increment_raw": cosine_score(y, p_joint)
            - cosine_score(y, p_base),
            "capacity_bootstrap_vs_v1": _bootstrap_dict(capacity_bootstrap),
            "sequence_bootstrap_vs_base_capacity": _bootstrap_dict(
                sequence_bootstrap
            ),
        },
        "sealed_audit_descriptive_only": {
            "v1_raw_cosine": cosine_score(sealed_y, sealed_p_v1),
            "v2_raw_cosine": cosine_score(sealed_y, sealed_p_v2),
            "v2_final_q1p2_cosine": cosine_score(
                sealed_y, _power(sealed_p_v2, exponent)
            ),
            "v2_minus_v1_raw": cosine_score(sealed_y, sealed_p_v2)
            - cosine_score(sealed_y, sealed_p_v1),
            "v1_raw_excluding_month_66": cosine_score(
                sealed_y[exclude_66], sealed_p_v1[exclude_66]
            ),
            "v2_raw_excluding_month_66": cosine_score(
                sealed_y[exclude_66], sealed_p_v2[exclude_66]
            ),
            "v2_minus_v1_raw_excluding_month_66": cosine_score(
                sealed_y[exclude_66], sealed_p_v2[exclude_66]
            )
            - cosine_score(sealed_y[exclude_66], sealed_p_v1[exclude_66]),
        },
        "outputs": {
            "fold_scores": str(fold_path),
            "monthly_scores": str(monthly_path),
        },
    }
    temporary = summary_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(payload["development"], indent=2), flush=True)
    print(json.dumps(payload["sealed_audit_descriptive_only"], indent=2), flush=True)
    return payload


if __name__ == "__main__":
    run()
