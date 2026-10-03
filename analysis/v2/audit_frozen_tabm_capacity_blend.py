"""Reproduce the frozen Dev1+Dev2 screen and audit it once on Dev3.

This program is deliberately separate from ``run_tabm_mini_challenger.py``.  The
selection contract was frozen before the TabM Dev3 outer labels were inspected,
and the challenger source hash is part of that contract.  Nothing in this file
can change the frozen weight, power exponent, component models, or epoch rule.

Normalization contract
----------------------
For every evaluation vector (screen, Dev3, or pooled reporting), each complete
component prediction is independently divided by its uncentered RMS.  The
frozen convex blend is then formed, transformed with signed power, and divided
by its own uncentered RMS.  No centering and no month-wise normalization occur.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_DIR = PROJECT_ROOT / "artifacts" / "v2" / "diagnostics"
FROZEN_PATH = DIAGNOSTIC_DIR / "frozen_blend_before_dev3.json"
TABM_DIR = PROJECT_ROOT / "artifacts" / "v2" / "tabm_mini"
CAPACITY_DIR = PROJECT_ROOT / "artifacts" / "v2" / "experiments"

SCREEN_GRID_PATH = DIAGNOSTIC_DIR / "tabm_capacity_screen_grid.csv"
DEV3_PREDICTION_PATH = (
    DIAGNOSTIC_DIR / "tabm_capacity_frozen_dev3_predictions.npz"
)
DEV3_AUDIT_PATH = DIAGNOSTIC_DIR / "tabm_capacity_frozen_dev3_audit.json"
POOLED_REPORT_PATH = DIAGNOSTIC_DIR / "tabm_capacity_pooled_report.json"
MONTHLY_PATH = DIAGNOSTIC_DIR / "tabm_capacity_monthly_scores.csv"
COMPARISON_PATH = DIAGNOSTIC_DIR / "tabm_capacity_blend_comparison.json"

TABM_PREDICTION_PATHS = {
    "Dev1": TABM_DIR / "dev1_predictions.npz",
    "Dev2": TABM_DIR / "dev2_predictions.npz",
    "Dev3": TABM_DIR / "dev3_predictions.npz",
}
TABM_SUMMARY_PATHS = {
    fold: TABM_DIR / f"{fold.lower()}_summary.json" for fold in TABM_PREDICTION_PATHS
}
CAPACITY_SCREEN_PATH = (
    CAPACITY_DIR
    / "sequence_base_plus_sequence_all_capacity_Dev1-Dev2_oof.npz"
)
CAPACITY_DEV3_PATH = (
    CAPACITY_DIR / "sequence_base_plus_sequence_all_capacity_Dev3_oof.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def rms(values: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values64))))


def unit_rms(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    scale = rms(values64)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Cannot RMS-normalize prediction with RMS={scale!r}")
    return values64 / scale


def cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.linalg.norm(target64) * np.linalg.norm(prediction64))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("Cosine denominator is zero or non-finite")
    return float(np.dot(target64, prediction64) / denominator)


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    return np.sign(values64) * np.power(np.abs(values64), exponent)


def frozen_prediction(
    capacity_prediction: np.ndarray,
    tabm_prediction: np.ndarray,
    tabm_weight: float,
    exponent: float,
) -> tuple[np.ndarray, dict[str, float]]:
    capacity_rms = rms(capacity_prediction)
    tabm_rms = rms(tabm_prediction)
    capacity_normalized = unit_rms(capacity_prediction)
    tabm_normalized = unit_rms(tabm_prediction)
    blended = (1.0 - tabm_weight) * capacity_normalized + tabm_weight * tabm_normalized
    transformed = signed_power(blended, exponent)
    final_prediction = unit_rms(transformed)
    scales = {
        "capacity_lightgbm_raw_rms": capacity_rms,
        "tabm_raw_rms": tabm_rms,
        "pre_final_normalization_blend_rms": rms(transformed),
        "final_prediction_rms": rms(final_prediction),
    }
    return final_prediction, scales


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as artifact:
        required = {"row_indices", "months", "target", "prediction"}
        missing = sorted(required.difference(artifact.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        result = {key: np.asarray(artifact[key]) for key in required}
    n_rows = result["target"].shape[0]
    for key, values in result.items():
        if values.ndim != 1 or values.shape[0] != n_rows:
            raise ValueError(f"Unexpected shape for {path}:{key}: {values.shape}")
    for key in ("target", "prediction"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Non-finite values in {path}:{key}")
    return result


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ("row_indices", "months", "target", "prediction")
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


def assert_aligned(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    label: str,
) -> None:
    for key in ("row_indices", "months", "target"):
        if not np.array_equal(left[key], right[key]):
            raise AssertionError(f"Alignment failure for {label}:{key}")


def assert_month_range(data: dict[str, np.ndarray], first: int, last: int, label: str) -> None:
    observed = np.unique(data["months"]).astype(int)
    expected = np.arange(first, last + 1)
    if not np.array_equal(observed, expected):
        raise AssertionError(
            f"Unexpected {label} months: observed={observed.tolist()}, "
            f"expected={expected.tolist()}"
        )


def monthly_rows(
    months: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month in np.unique(months):
        mask = months == month
        month_target = np.asarray(target[mask], dtype=np.float64)
        month_prediction = np.asarray(prediction[mask], dtype=np.float64)
        score = cosine(month_target, month_prediction)
        rows.append(
            {
                "split": split,
                "month": int(month),
                "n_rows": int(mask.sum()),
                "cosine": score,
                "prediction_rms": rms(month_prediction),
                "target_rms": rms(month_target),
                "prediction_mean": float(np.mean(month_prediction)),
                "target_mean": float(np.mean(month_target)),
                "positive_cosine": bool(score > 0.0),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    frozen = read_json(FROZEN_PATH)
    frozen_config_sha256 = sha256_file(FROZEN_PATH)
    if frozen.get("status") != "frozen_before_dev3_outer_labels":
        raise AssertionError("The blend contract is not in its pre-Dev3 frozen state")
    if frozen.get("dev3_result") is not None or frozen.get("sealed_result") is not None:
        raise AssertionError("Frozen contract unexpectedly contains post-freeze results")

    contract = frozen["blend_contract"]
    tabm_weight = float(contract["tabm_weight"])
    power_exponent = float(
        str(contract["post_blend_transform"]).rsplit("^", maxsplit=1)[-1]
    )
    if tabm_weight != 0.6 or power_exponent != 1.1:
        raise AssertionError(
            f"Unexpected frozen selection: weight={tabm_weight}, q={power_exponent}"
        )

    selection_hash_checks: dict[str, dict[str, Any]] = {}
    for relative_path, expected_hash in frozen["selection_artifact_sha256"].items():
        path = PROJECT_ROOT / relative_path
        actual_hash = sha256_file(path)
        passed = actual_hash.casefold() == str(expected_hash).casefold()
        selection_hash_checks[relative_path] = {
            "expected_sha256": str(expected_hash).upper(),
            "actual_sha256": actual_hash,
            "passed": passed,
        }
        if not passed:
            raise AssertionError(f"Frozen selection artifact changed: {relative_path}")

    tabm_dev1 = load_prediction(TABM_PREDICTION_PATHS["Dev1"])
    tabm_dev2 = load_prediction(TABM_PREDICTION_PATHS["Dev2"])
    tabm_screen = concatenate([tabm_dev1, tabm_dev2])
    capacity_screen = load_prediction(CAPACITY_SCREEN_PATH)
    assert_aligned(tabm_screen, capacity_screen, "Dev1+Dev2 screen")
    assert_month_range(tabm_screen, 23, 46, "screen")

    screen_grid: list[dict[str, Any]] = []
    tabm_screen_unit = unit_rms(tabm_screen["prediction"])
    capacity_screen_unit = unit_rms(capacity_screen["prediction"])
    for weight in frozen["selection_grid"]["tabm_weights"]:
        weight_float = float(weight)
        base = (1.0 - weight_float) * capacity_screen_unit + weight_float * tabm_screen_unit
        for exponent in frozen["selection_grid"][
            "post_blend_signed_power_exponents"
        ]:
            exponent_float = float(exponent)
            prediction = unit_rms(signed_power(base, exponent_float))
            score = cosine(tabm_screen["target"], prediction)
            screen_grid.append(
                {
                    "tabm_weight": weight_float,
                    "power_exponent": exponent_float,
                    "screen_cosine": score,
                }
            )
    screen_grid.sort(key=lambda row: row["screen_cosine"], reverse=True)
    for rank, row in enumerate(screen_grid, start=1):
        row["rank"] = rank
        row["selected"] = bool(
            row["tabm_weight"] == tabm_weight
            and row["power_exponent"] == power_exponent
        )
    screen_best = screen_grid[0]
    frozen_screen_cosine = float(frozen["selected_dev1_dev2_cosine"])
    if (
        screen_best["tabm_weight"] != tabm_weight
        or screen_best["power_exponent"] != power_exponent
        or abs(screen_best["screen_cosine"] - frozen_screen_cosine) > 1e-14
    ):
        raise AssertionError(
            "Recomputed Dev1+Dev2 winner does not exactly reproduce the frozen contract"
        )
    write_csv(
        SCREEN_GRID_PATH,
        screen_grid,
        ["rank", "selected", "tabm_weight", "power_exponent", "screen_cosine"],
    )

    # Dev3 starts only after every selection assertion above has passed.  No grid
    # or alternative weight is evaluated on Dev3.
    tabm_dev3 = load_prediction(TABM_PREDICTION_PATHS["Dev3"])
    capacity_dev3 = load_prediction(CAPACITY_DEV3_PATH)
    assert_aligned(tabm_dev3, capacity_dev3, "Dev3 audit")
    assert_month_range(tabm_dev3, 47, 58, "Dev3")
    dev3_prediction, dev3_scales = frozen_prediction(
        capacity_dev3["prediction"],
        tabm_dev3["prediction"],
        tabm_weight,
        power_exponent,
    )
    dev3_cosine = cosine(tabm_dev3["target"], dev3_prediction)
    dev3_monthly = monthly_rows(
        tabm_dev3["months"], tabm_dev3["target"], dev3_prediction, "Dev3"
    )
    dev3_positive_month_share = float(
        np.mean([row["positive_cosine"] for row in dev3_monthly])
    )
    gate = frozen["predeclared_promotion_gates"]["dev3"]
    gate_cosine_passed = bool(dev3_cosine > float(gate["candidate_cosine_must_exceed"]))
    gate_months_passed = bool(
        dev3_positive_month_share >= float(gate["minimum_positive_month_share"])
    )
    gate_passed = gate_cosine_passed and gate_months_passed

    np.savez_compressed(
        DEV3_PREDICTION_PATH,
        row_indices=tabm_dev3["row_indices"],
        months=tabm_dev3["months"],
        target=np.asarray(tabm_dev3["target"], dtype=np.float64),
        capacity_prediction_raw=np.asarray(capacity_dev3["prediction"], dtype=np.float64),
        tabm_prediction_raw=np.asarray(tabm_dev3["prediction"], dtype=np.float64),
        capacity_prediction_unit_rms=unit_rms(capacity_dev3["prediction"]),
        tabm_prediction_unit_rms=unit_rms(tabm_dev3["prediction"]),
        prediction=np.asarray(dev3_prediction, dtype=np.float64),
        blended_prediction=np.asarray(dev3_prediction, dtype=np.float64),
    )
    prediction_artifact_sha256 = sha256_file(DEV3_PREDICTION_PATH)
    dev3_source_hashes = {
        str(TABM_PREDICTION_PATHS["Dev3"].relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(
            TABM_PREDICTION_PATHS["Dev3"]
        ),
        str(CAPACITY_DEV3_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(
            CAPACITY_DEV3_PATH
        ),
    }
    dev3_audit = {
        "status": "complete_one_shot_dev3_audit",
        "frozen_config_sha256": frozen_config_sha256,
        "tabm_weight": tabm_weight,
        "capacity_lightgbm_weight": 1.0 - tabm_weight,
        "power_exponent": power_exponent,
        "dev3_cosine": dev3_cosine,
        "dev3_month_range": [47, 58],
        "dev3_rows": int(tabm_dev3["target"].shape[0]),
        "dev3_positive_month_share": dev3_positive_month_share,
        "promotion_gate": {
            "candidate_cosine_must_exceed": float(gate["candidate_cosine_must_exceed"]),
            "minimum_positive_month_share": float(gate["minimum_positive_month_share"]),
            "cosine_passed": gate_cosine_passed,
            "positive_month_share_passed": gate_months_passed,
            "passed": gate_passed,
        },
        "normalization": {
            "scope": "Dev3 complete prediction vector, independently per component",
            "centering": "none",
            **dev3_scales,
        },
        "prediction_artifact": str(DEV3_PREDICTION_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "dev3_source_sha256": dev3_source_hashes,
        "selection_hash_checks_all_passed": bool(
            all(item["passed"] for item in selection_hash_checks.values())
        ),
        "selection_hash_checks": selection_hash_checks,
        "guardrail": "One frozen Dev3 candidate only; no Dev3 grid or retuning.",
    }
    write_json(DEV3_AUDIT_PATH, dev3_audit)

    screen_prediction, screen_scales = frozen_prediction(
        capacity_screen["prediction"],
        tabm_screen["prediction"],
        tabm_weight,
        power_exponent,
    )
    screen_cosine = cosine(tabm_screen["target"], screen_prediction)
    if abs(screen_cosine - frozen_screen_cosine) > 1e-14:
        raise AssertionError("Frozen screen prediction score drifted")
    screen_monthly = monthly_rows(
        tabm_screen["months"], tabm_screen["target"], screen_prediction, "screen"
    )
    write_csv(
        MONTHLY_PATH,
        screen_monthly + dev3_monthly,
        [
            "split",
            "month",
            "n_rows",
            "cosine",
            "prediction_rms",
            "target_rms",
            "prediction_mean",
            "target_mean",
            "positive_cosine",
        ],
    )

    pooled_tabm = np.concatenate(
        [tabm_screen["prediction"], tabm_dev3["prediction"]]
    )
    pooled_capacity = np.concatenate(
        [capacity_screen["prediction"], capacity_dev3["prediction"]]
    )
    pooled_target = np.concatenate([tabm_screen["target"], tabm_dev3["target"]])
    pooled_prediction, pooled_scales = frozen_prediction(
        pooled_capacity, pooled_tabm, tabm_weight, power_exponent
    )
    pooled_cosine = cosine(pooled_target, pooled_prediction)

    selected_epochs: dict[str, int] = {}
    summary_hashes: dict[str, str] = {}
    for fold, path in TABM_SUMMARY_PATHS.items():
        summary = read_json(path)
        selected_epochs[fold] = int(summary["best_epoch"])
        summary_hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = sha256_file(path)
    final_refit_epoch = int(np.median(list(selected_epochs.values())))

    pooled_report = {
        "status": "reporting_only_no_selection",
        "frozen_config_sha256": frozen_config_sha256,
        "tabm_weight": tabm_weight,
        "power_exponent": power_exponent,
        "screen_dev1_dev2_cosine": screen_cosine,
        "dev3_cosine": dev3_cosine,
        "pooled_dev1_dev2_dev3_cosine": pooled_cosine,
        "pooled_month_range": [23, 58],
        "pooled_rows": int(pooled_target.shape[0]),
        "normalization": {
            "scope": "pooled months 23-58 complete vector, independently per component",
            "centering": "none",
            **pooled_scales,
        },
        "final_tabm_epoch_rule": "median of Dev1, Dev2, Dev3 inner-selected epochs",
        "selected_epochs": selected_epochs,
        "final_tabm_refit_epoch": final_refit_epoch,
        "tabm_summary_sha256": summary_hashes,
    }
    write_json(POOLED_REPORT_PATH, pooled_report)

    output_paths = [
        SCREEN_GRID_PATH,
        DEV3_PREDICTION_PATH,
        DEV3_AUDIT_PATH,
        POOLED_REPORT_PATH,
        MONTHLY_PATH,
    ]
    comparison = {
        "status": "frozen_dev3_audit_complete",
        "frozen_config": str(FROZEN_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "frozen_config_sha256": frozen_config_sha256,
        "selection_reproduced": True,
        "selection_hash_checks_all_passed": True,
        "selected": {
            "tabm_weight": tabm_weight,
            "capacity_lightgbm_weight": 1.0 - tabm_weight,
            "power_exponent": power_exponent,
            "screen_cosine": screen_cosine,
        },
        "dev3_audit": {
            "status": dev3_audit["status"],
            "cosine": dev3_cosine,
            "positive_month_share": dev3_positive_month_share,
            "promotion_gate_passed": gate_passed,
        },
        "pooled_reporting": {
            "cosine": pooled_cosine,
            "month_range": [23, 58],
        },
        "tabm_epoch_rule": {
            "selected_epochs": selected_epochs,
            "median_final_refit_epoch": final_refit_epoch,
        },
        "artifacts": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in output_paths
        },
        "source": str(Path(__file__).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(COMPARISON_PATH, comparison)

    print(
        json.dumps(
            {
                "screen_cosine": screen_cosine,
                "dev3_cosine": dev3_cosine,
                "dev3_positive_month_share": dev3_positive_month_share,
                "promotion_gate_passed": gate_passed,
                "pooled_cosine": pooled_cosine,
                "selected_epochs": selected_epochs,
                "final_tabm_refit_epoch": final_refit_epoch,
                "dev3_prediction_sha256": prediction_artifact_sha256,
                "comparison": str(COMPARISON_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
