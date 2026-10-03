"""Recompute retrospective v1/v2 evidence; never fit models or select recipes.

Requires only NumPy. Run from any directory:
    python analysis/v3_audit/recompute_evidence.py --notebook

All intervals condition on already selected predictions and exposed history.
They are neither selection-adjusted inference nor private-test forecasts.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/v3_audit"
LABEL = "conditional retrospective postselection uncertainty; not private-test inference"
SEED = 20260910
MODELS = {
    "v1_slow_raw": "v1 slow LightGBM, raw",
    "base_capacity_raw": "v2 base-only capacity LightGBM, raw",
    "path_capacity_raw": "v2 base + path capacity LightGBM, raw",
    "path_capacity_q1p2": "v2 base + path capacity LightGBM, signed power 1.2",
    "tabm_raw": "TabM-mini, raw",
    "blend_linear": "60% TabM / 40% path capacity, component unit RMS",
    "blend_q1p1": "final 60/40 component unit-RMS blend, signed power 1.1",
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cosine(y, p):
    denominator = np.linalg.norm(y) * np.linalg.norm(p)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid cosine denominator")
    return float(np.dot(y, p) / denominator)


def unit_rms(p):
    scale = float(np.sqrt(np.mean(np.square(p))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid RMS")
    return p / scale


def power(p, q):
    return np.sign(p) * np.abs(p) ** q


def load(paths, prediction_key, provenance, reference=None):
    parts = []
    for relative in paths:
        path = ROOT / relative
        with np.load(path, allow_pickle=False) as archive:
            checked = []
            for key in archive.files:
                value = archive[key]
                if np.issubdtype(value.dtype, np.number):
                    if not np.isfinite(value).all():
                        raise ValueError(f"Non-finite {relative}:{key}")
                    checked.append(key)
            row_key = "row_indices" if "row_indices" in archive.files else "row_index"
            month_key = "months" if "months" in archive.files else "month"
            part = {"row_indices": archive[row_key], "months": archive[month_key],
                    "prediction": archive[prediction_key].astype(np.float64)}
            if "target" in archive.files:
                part["target"] = archive["target"].astype(np.float64)
        n = len(part["prediction"])
        if any(v.shape != (n,) for v in part.values()):
            raise ValueError(f"Non-vector or unequal lengths: {relative}")
        for key in ("row_indices", "months"):
            if not np.issubdtype(part[key].dtype, np.integer):
                raise ValueError(f"Non-integer {relative}:{key}")
        provenance[relative] = {
            "sha256": sha256(path), "bytes": path.stat().st_size, "rows": n,
            "all_numeric_arrays_checked_finite": checked,
            "prediction_key": prediction_key,
        }
        parts.append(part)
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if np.any(np.diff(data["row_indices"]) <= 0):
        raise ValueError(f"Rows are duplicated or not strictly ordered: {paths}")
    if np.any(np.diff(data["months"]) < 0):
        raise ValueError(f"Months are not chronological: {paths}")
    if reference is not None:
        for key in ("row_indices", "months", "target"):
            if key in data and not np.array_equal(data[key], reference[key]):
                raise ValueError(f"Exact alignment failed for {paths}:{key}")
        if "target" not in data:
            data["target"] = reference["target"]
    return data


def assemble(provenance):
    prefix = "artifacts/v2/experiments/sequence_"
    dev = load(["artifacts/diagnostics/gbdt_multiscale_mechanics_scale_development_oof.npz"],
               "slow", provenance)
    dev_predictions = {"v1_slow_raw": dev["prediction"]}
    for name, stem in (("base_capacity_raw", "base_only"),
                       ("path_capacity_raw", "base_plus_sequence_all")):
        item = load([prefix + stem + "_capacity_Dev1-Dev2_oof.npz",
                     prefix + stem + "_capacity_Dev3_oof.npz"],
                    "prediction", provenance, dev)
        dev_predictions[name] = item["prediction"]
    tabm = load([f"artifacts/v2/tabm_mini/dev{i}_predictions.npz" for i in (1, 2, 3)],
                "prediction", provenance, dev)
    dev_predictions["tabm_raw"] = tabm["prediction"]
    late = load([prefix + "base_plus_sequence_all_capacity_SealedAudit_oof.npz"],
                "prediction", provenance)
    late_predictions = {"path_capacity_raw": late["prediction"]}
    for name, path in (
        ("v1_slow_raw", "artifacts/diagnostics/sealed_audit_predictions.npz"),
        ("tabm_raw", "artifacts/v2/sealed_blend/tabm_sealed_predictions.npz"),
    ):
        late_predictions[name] = load([path], "prediction", provenance, late)["prediction"]
    if not np.array_equal(np.unique(dev["months"]), np.arange(23, 59)):
        raise ValueError("Unexpected development coverage")
    if not np.array_equal(np.unique(late["months"]), np.arange(59, 71)):
        raise ValueError("Unexpected late-block coverage")
    if np.intersect1d(dev["row_indices"], late["row_indices"]).size:
        raise ValueError("Development and late-block rows overlap")
    return dev, dev_predictions, late, late_predictions


def views(raw):
    predictions = dict(raw)
    predictions["path_capacity_q1p2"] = power(raw["path_capacity_raw"], 1.2)
    linear = 0.6 * unit_rms(raw["tabm_raw"]) + 0.4 * unit_rms(raw["path_capacity_raw"])
    predictions["blend_linear"] = unit_rms(linear)
    predictions["blend_q1p1"] = unit_rms(power(linear, 1.1))
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Non-finite derived predictions")
    return predictions


def monthly_statistics(scope, months, y, predictions, excluded=()):
    calendar = np.arange(int(months.min()), int(months.max()) + 1)
    names = list(predictions)
    # Exact pooled cosine sufficient statistics: not a mean of monthly cosines.
    cross = np.zeros((len(calendar), len(names)))
    pred_sq = np.zeros_like(cross)
    target_sq = np.zeros(len(calendar))
    count = np.zeros(len(calendar), dtype=int)
    rows = []
    for i, month in enumerate(calendar):
        if month in excluded:
            continue  # Preserve an excluded month's position, with zero contribution.
        mask = months == month
        yy = y[mask]
        count[i] = int(mask.sum())
        target_sq[i] = np.dot(yy, yy)
        for j, name in enumerate(names):
            pp = predictions[name][mask]
            cross[i, j] = np.dot(yy, pp)
            pred_sq[i, j] = np.dot(pp, pp)
            rows.append({"scope": scope, "month": int(month), "model": name,
                         "rows": int(count[i]), "cosine": cosine(yy, pp),
                         "target_sum_squares": float(target_sq[i]),
                         "prediction_sum_squares": float(pred_sq[i, j]),
                         "target_prediction_sum": float(cross[i, j]),
                         "target_rms": float(np.sqrt(np.mean(yy ** 2))),
                         "prediction_rms": float(np.sqrt(np.mean(pp ** 2)))})
    return names, calendar, cross, pred_sq, target_sq, count, rows


def paired_bootstrap(scope, stats, replicates, seed, excluded):
    names, calendar, cross, pred_sq, target_sq, count, _ = stats
    n = len(calendar)
    point = cross.sum(0) / np.sqrt(target_sq.sum() * pred_sq.sum(0))
    pairs = [(name, "v1_slow_raw") for name in names if name != "v1_slow_raw"]
    pairs += [(a, b) for a, b in (
        ("path_capacity_raw", "base_capacity_raw"),
        ("path_capacity_q1p2", "path_capacity_raw"),
        ("tabm_raw", "path_capacity_q1p2"),
        ("blend_q1p1", "path_capacity_q1p2"),
        ("blend_q1p1", "tabm_raw"),
        ("blend_q1p1", "blend_linear"),
    ) if a in names and b in names]
    rows = []
    for length in (1, 3, 6):
        rng = np.random.default_rng(seed + length)
        # Ordinary noncircular overlapping moving blocks. The final sampled
        # block is truncated to n calendar positions, as in a standard MBB.
        starts = rng.integers(0, n - length + 1, size=(replicates, math.ceil(n / length)))
        selected = (starts[:, :, None] + np.arange(length)).reshape(replicates, -1)[:, :n]
        weights = np.zeros((replicates, n), dtype=np.int64)
        np.add.at(weights, (np.arange(replicates)[:, None], selected), 1)
        yy = weights @ target_sq
        pp = weights @ pred_sq
        yp = weights @ cross
        valid = (yy > 0) & (pp > 0).all(1)
        sampled = yp[valid] / np.sqrt(yy[valid, None] * pp[valid])
        for candidate, baseline in pairs:
            a, b = names.index(candidate), names.index(baseline)
            delta = sampled[:, a] - sampled[:, b]
            lower, median, upper = np.quantile(delta, [0.025, 0.5, 0.975])
            rows.append({"scope": scope, "candidate": candidate, "baseline": baseline,
                         "block_months": length, "calendar_months": n,
                         "scored_months": int(np.count_nonzero(count)),
                         "replicates": int(valid.sum()), "discarded_replicates": int((~valid).sum()),
                         "seed": seed + length, "point_delta_cosine": float(point[a] - point[b]),
                         "percentile_95_lower": float(lower), "bootstrap_median": float(median),
                         "percentile_95_upper": float(upper),
                         "bootstrap_fraction_delta_positive": float(np.mean(delta > 0)),
                         "uncertainty_label": LABEL,
                         "excluded_months": ",".join(map(str, excluded))})
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(replicates=10000):
    if replicates < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates")
    provenance = {}
    dev, dev_raw, late, late_raw = assemble(provenance)
    score_scopes = {}
    monthly_rows, bootstrap_rows = [], []
    constructed = {}
    specs = [("development_23_58", dev, dev_raw, 23, 58, ()),
             ("screen_23_46", dev, dev_raw, 23, 46, ()),
             ("dev3_47_58", dev, dev_raw, 47, 58, ()),
             ("late_59_70", late, late_raw, 59, 70, ()),
             ("late_59_70_excluding_66", late, late_raw, 59, 70, (66,))]
    for index, (scope, data, raw, first, last, excluded) in enumerate(specs):
        mask = (data["months"] >= first) & (data["months"] <= last)
        months, y = data["months"][mask], data["target"][mask]
        predictions = views({k: v[mask] for k, v in raw.items()})
        constructed[scope] = predictions
        selected = ~np.isin(months, excluded)
        stats = monthly_statistics(scope, months, y, predictions, excluded)
        names, calendar, cross, pred_sq, target_sq, count, rows = stats
        monthly_rows.extend(rows)
        scores = {name: cosine(y[selected], p[selected]) for name, p in predictions.items()}
        aggregate = cross.sum(0) / np.sqrt(target_sq.sum() * pred_sq.sum(0))
        np.testing.assert_allclose(aggregate, [scores[name] for name in names], rtol=0, atol=1e-13)
        score_scopes[scope] = {
            "rows": int(count.sum()), "scored_months": int(np.count_nonzero(count)),
            "month_min": first, "month_max": last, "excluded_months": list(excluded),
            "cosine": scores, "target_sum_squares": float(target_sq.sum()),
            "normalization": "components use complete scope vector; exclusion keeps full late-block scales",
            "largest_target_energy_month": int(calendar[np.argmax(target_sq)]),
            "largest_month_share_target_energy": float(target_sq.max() / target_sq.sum()),
        }
        bootstrap_rows.extend(paired_bootstrap(scope, stats, replicates, SEED + index * 100, excluded))

    # Independently verify recomputed transformations against saved final views.
    saved_late = load(["artifacts/v2/sealed_blend/sealed_blend_views.npz"],
                      "final_unit_rms", provenance, late)
    np.testing.assert_allclose(saved_late["prediction"], constructed["late_59_70"]["blend_q1p1"],
                               rtol=1e-12, atol=1e-12)
    dev3_mask = dev["months"] >= 47
    reference3 = {k: v[dev3_mask] for k, v in dev.items()}
    saved_dev3 = load(["artifacts/v2/diagnostics/tabm_capacity_frozen_dev3_predictions.npz"],
                      "prediction", provenance, reference3)
    np.testing.assert_allclose(saved_dev3["prediction"], constructed["dev3_47_58"]["blend_q1p1"],
                               rtol=1e-12, atol=1e-12)
    expected = {
        ("development_23_58", "v1_slow_raw"): 0.13323825194161495,
        ("development_23_58", "base_capacity_raw"): 0.13825987233313652,
        ("development_23_58", "path_capacity_raw"): 0.13901257218493554,
        ("development_23_58", "blend_q1p1"): 0.1452968982352971,
        ("screen_23_46", "blend_q1p1"): 0.14445559865492444,
        ("dev3_47_58", "blend_q1p1"): 0.14699994690096732,
        ("late_59_70", "tabm_raw"): 0.15339667922295144,
        ("late_59_70", "path_capacity_q1p2"): 0.15058665835251023,
    }
    for (scope, name), value in expected.items():
        np.testing.assert_allclose(score_scopes[scope]["cosine"][name], value, rtol=0, atol=1e-12)
    payload = {
        "status": "complete_saved_prediction_recomputation_no_training_or_selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "numpy_version": np.__version__,
        "uncertainty_label": LABEL, "model_definitions": MODELS,
        "bootstrap": {
            "method": "paired noncircular overlapping moving monthly blocks; percentile 95% interval",
            "block_lengths_months": [1, 3, 6], "replicates": replicates,
            "metric": "pooled uncentered cosine from summed per-month sufficient statistics",
            "normalization": "prediction recipe held fixed during bootstrap; no refit or rescaling",
            "month_66_exclusion": "resample original 12-month timeline, then zero month 66 contributions; never make months 65 and 67 adjacent",
            "fraction_positive_is_not": "a selection-adjusted p-value or probability of private-test improvement",
            "limitations": ["all evaluated months were previously exposed",
                           "intervals omit model/feature/calibration/search uncertainty",
                           "short 12-month late history cannot establish unseen regime robustness",
                           "noncircular moving blocks underweight calendar endpoints when block length > 1"],
        },
        "checks": {
            "all_loaded_numeric_arrays_finite": True, "row_ids_unique_and_strictly_increasing": True,
            "months_chronological_and_expected_ranges": True,
            "row_ids_months_targets_exactly_aligned": True,
            "development_late_row_intersection_empty": True,
            "v1_late_target_provenance": "v1 NPZ has no target; borrowed exact row/month-aligned v2 saved targets, cross-checked with TabM saved targets",
            "raw_source_labels_independently_reloaded": False,
            "direct_cosine_matches_monthly_sufficient_statistics": True,
            "late_and_dev3_saved_final_vectors_reproduced": True,
            "archived_headline_metrics_reproduced": len(expected),
        },
        "unavailable_comparisons": ["base-only capacity has no saved months 59-70 NPZ; late feature increment cannot be recomputed"],
        "scopes": score_scopes, "paired_comparisons": bootstrap_rows, "sources": provenance,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrospective_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(OUT / "retrospective_monthly.csv", monthly_rows)
    write_csv(OUT / "retrospective_bootstrap.csv", bootstrap_rows)
    return payload


def make_notebook():
    """Save a small notebook with outputs captured by executing its actual cells."""
    codes = [
        "from pathlib import Path\nimport json\nimport sys\n"
        "project = Path.cwd().resolve()\n"
        "while not (project / 'analysis/v3_audit/recompute_evidence.py').exists():\n"
        "    if project == project.parent: raise FileNotFoundError('Run inside this project')\n"
        "    project = project.parent\n"
        "sys.path.insert(0, str(project / 'analysis/v3_audit'))\n"
        "from recompute_evidence import run\n"
        "evidence = run()\n"
        "print(evidence['status'])\nprint(evidence['uncertainty_label'])",
        "for scope, result in evidence['scopes'].items():\n"
        "    print(scope, 'rows=', result['rows'], 'months=', result['scored_months'])\n"
        "    for model, score in result['cosine'].items():\n"
        "        print(f'  {model:26s} {score:.9f}')",
        "for row in evidence['paired_comparisons']:\n"
        "    if row['candidate'] == 'blend_q1p1' and row['baseline'] == 'path_capacity_q1p2':\n"
        "        print(row['scope'], 'block=', row['block_months'],\n"
        "              'delta=', round(row['point_delta_cosine'], 6),\n"
        "              '95% interval=', [round(row[k], 6) for k in ['percentile_95_lower', 'percentile_95_upper']])\n"
        "print('Exact alignment and finite checks:', evidence['checks'])",
    ]
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# Retrospective saved-prediction evidence\n",
        "This notebook executes the NumPy-only recomputation script; it does not retrain models. "
        "All periods were previously exposed. Intervals are conditional retrospective postselection uncertainty, "
        "not selection-adjusted confidence or a private-leaderboard forecast.\n\n",
        "Each displayed scope normalizes blend components over that complete scope. The month-66 exclusion "
        "retains the original late-block prediction scales and calendar positions during block resampling. "
        "Full definitions, input hashes, checks and limitations are in `artifacts/v3_audit/retrospective_metrics.json`.\n"
    ]}]
    namespace = {"__name__": "__notebook__"}
    for count, code in enumerate(codes, 1):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"retrospective_evidence.ipynb:cell{count}", "exec"), namespace)
        cells.append({"cell_type": "code", "metadata": {}, "source": code.splitlines(keepends=True),
                      "execution_count": count,
                      "outputs": [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(keepends=True)}]})
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": sys_version()},
                             "execution_method": "actual Python compile/exec of each cell, stdout captured; no training"}}
    path = ROOT / "analysis/v3_audit/retrospective_evidence.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def sys_version():
    import platform
    return platform.python_version()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--notebook", action="store_true")
    args = parser.parse_args()
    result = run(args.replicates)
    for name, scope in result["scopes"].items():
        print(name, scope["cosine"])
    if args.notebook:
        if args.replicates != 10000:
            parser.error("--notebook requires the default 10000 replicates for consistent saved outputs")
        make_notebook()
    print(f"Wrote evidence to {OUT}")
