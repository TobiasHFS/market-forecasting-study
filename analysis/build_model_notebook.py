"""Build the reader-facing final model report notebook.

The generated notebook is intentionally lightweight: it reads saved diagnostics and
audits instead of retraining models.  Execute it from any directory after the final
audit artifacts have been created; the first notebook cell locates the project root.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


WORKSPACE = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = WORKSPACE / "analysis" / "final_model_report.ipynb"


def markdown(text: str):
    """Return a normalized markdown cell."""

    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    """Return a normalized code cell."""

    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        r"""
        # Financial market forecasting: frozen model report

        Reproducible handoff for the final forecasting pipeline. This notebook reads the saved
        experiment, stress-test, sealed-audit, and submission-audit artifacts; it does **not**
        refit a model or use the sealed period to choose one.

        ## tl;dr

        The next cell renders the live headline values from the artifacts. A missing audit is
        displayed as pending rather than replaced with an assumed result.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Any

        import pandas as pd
        from IPython.display import Markdown, display


        def locate_project_root(start: Path | None = None) -> Path:
            # Find the repository from a root, analysis, or notebook launch directory.
            origin = (start or Path.cwd()).resolve()
            for candidate in (origin, *origin.parents):
                if (
                    (candidate / "analysis").is_dir()
                    and (candidate / "artifacts").is_dir()
                    and (candidate / "analysis" / "pipeline_config.py").exists()
                ):
                    return candidate
            raise FileNotFoundError(
                "Could not find the project root containing analysis/ and artifacts/. "
                "Launch the notebook from inside the Financial Modelling project."
            )


        project_root = locate_project_root()
        diagnostics_dir = project_root / "artifacts" / "diagnostics"

        artifact_paths = {
            "profile": project_root / "analysis" / "output" / "competition_profile.json",
            "activity_shift": project_root / "analysis" / "output" / "activity_shift.csv",
            "quality": diagnostics_dir / "engineered_feature_quality_summary.json",
            "ridge_ablation": diagnostics_dir / "ridge_feature_family_comparisons.json",
            "gbdt_multiscale": diagnostics_dir / "gbdt_multiscale_summary.json",
            "gbdt_multiscale_scale": diagnostics_dir / "gbdt_multiscale_scale_summary.json",
            "gbdt_final": diagnostics_dir / "gbdt_multiscale_mechanics_scale_summary.json",
            "gbdt_recency_24": diagnostics_dir / "gbdt_multiscale_mechanics_scale_hl24_summary.json",
            "gbdt_recency_48": diagnostics_dir / "gbdt_multiscale_mechanics_scale_hl48_summary.json",
            "shuffle_control": diagnostics_dir / "gbdt_multiscale_mechanics_scale_shuffle_summary.json",
            "frozen": diagnostics_dir / "frozen_pipeline.json",
            "deployment_stress": diagnostics_dir / "deployment_stress_summary.json",
            "sealed_audit": diagnostics_dir / "sealed_audit_summary.json",
            "submission_audit": diagnostics_dir / "submission_audit.json",
        }


        def load_json_optional(path: Path) -> dict[str, Any] | list[Any] | None:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))


        def nested_value(payload: Any, *candidate_paths: str, default: Any = None) -> Any:
            # Return the first non-null value at a dotted candidate path.
            if payload is None:
                return default
            for candidate_path in candidate_paths:
                value = payload
                found = True
                for key in candidate_path.split("."):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        found = False
                        break
                if found and value is not None:
                    return value
            return default


        def best_ranking_row(summary: dict[str, Any] | None) -> dict[str, Any]:
            if not summary:
                return {}
            best_spec = summary.get("best_spec")
            ranking = summary.get("ranking", [])
            for row in ranking:
                if row.get("spec") == best_spec:
                    return row
            return ranking[0] if ranking else {}


        def format_number(value: Any, digits: int = 6, missing: str = "pending") -> str:
            if value is None:
                return missing
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, int):
                return f"{value:,}"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)


        artifacts = {
            name: load_json_optional(path)
            for name, path in artifact_paths.items()
            if path.suffix == ".json"
        }
        frozen = artifacts["frozen"]
        final_development = artifacts["gbdt_final"]
        development_best = best_ranking_row(final_development)
        sealed_audit = artifacts["sealed_audit"]
        submission_audit = artifacts["submission_audit"]

        sealed_cosine = nested_value(
            sealed_audit,
            "sealed_cosine",
            "global_cosine",
            "pooled_cosine",
            "metrics.global_cosine",
            "metrics.cosine",
            "cosine",
            "score",
        )
        submission_status = nested_value(
            submission_audit, "status", "audit_status", "result", default="pending"
        )
        feature_families = nested_value(frozen, "feature_families", default=[])

        display(
            Markdown(
                "\\n".join(
                    [
                        f"- **Frozen candidate:** `{nested_value(frozen, 'feature_set', default='pending')}` "
                        f"with **{format_number(nested_value(frozen, 'feature_count'), 0)}** raw features "
                        f"across {', '.join(feature_families) if feature_families else 'pending families'}.",
                        f"- **Selected learner:** `{nested_value(frozen, 'model_family', default='pending')}` / "
                        f"`{nested_value(frozen, 'model_spec_name', default='pending')}`; "
                        f"Development 1-3 pooled cosine = **{format_number(development_best.get('pooled_cosine'))}**.",
                        f"- **Sealed audit (months 59-70):** cosine = "
                        f"**{format_number(sealed_cosine)}**. This value was not used for selection.",
                        f"- **Submission audit:** **{submission_status}**; rows = "
                        f"**{format_number(nested_value(submission_audit, 'rows', 'row_count', 'n_rows'), 0)}**.",
                    ]
                )
            )
        )
        print(f"Project root: {project_root}")
        """
    ),
    markdown(
        r"""
        ## Context & Methods

        The task is supervised prediction of a future return from the preceding order book,
        aggregate-trade bars, raw order flow, and raw transactions. Train contains months 0-70;
        the unlabeled test era follows it. The competition evaluates one prediction vector with
        **global uncentered cosine**:

        $$
        \operatorname{cos}(\hat y,y)
        =\frac{\sum_i \hat y_i y_i}
        {\sqrt{\sum_i \hat y_i^2}\sqrt{\sum_i y_i^2}}.
        $$

        Accordingly, the primary validation score is cosine after concatenating all rows in a
        validation block - not an average of monthly cosines. Monthly median, tail, worst-month, and
        positive-month share remain robustness diagnostics. Prediction scaling does not change a
        single model's cosine, but component scales would matter in a blend; candidate component
        vectors were normalized before blend-weight evaluation.

        ### Key Assumptions

        - Month is the trustworthy chronology. `sample_id` is only a join key and is excluded.
        - No label/source-window overlap across adjacent months has been demonstrated, so an
          arbitrary full-month gap is not imposed. If the hidden target horizon becomes known,
          purge only the actual overlapping information interval.
        - Preprocessing is fit on each fold's training rows only: robust centering/scaling,
          clipping at 8 robust units, and explicit missingness indicators.
        - The future 38-month test span creates model-aging risk. Deployment-age experiments are
          stress tests, not hyperparameter-selection folds.

        ### Temporal selection protocol

        | Role | Training months | Evaluation months | Permitted use |
        |---|---:|---:|---|
        | Development 1 | 0-22 | 23-34 | Selection |
        | Development 2 | 0-34 | 35-46 | Selection |
        | Development 3 | 0-46 | 47-58 | Selection |
        | Sealed audit | 0-58 | 59-70 | One-time audit only |

        Every feature-family, regularization, recency, and blending decision was frozen from the
        three development blocks before the sealed period was opened. The final test model may
        then use all labeled months 0-70 without feeding the sealed outcome back into model choice.
        """
    ),
    markdown(
        r"""
        ### Independently chosen feature hierarchy

        The hierarchy was chosen from data observability and market-microstructure invariance - not
        copied from the supplied examples:

        1. **Observability:** coverage, span, invalid-code/time fractions, and censoring flags so
           missing information is not confused with an economic zero.
        2. **Invariant core:** relative spread, queue imbalance, microprice displacement, returns,
           realized variation, depth-scaled order-flow imbalance, and signed order/trade pressure.
        3. **Multiscale dynamics:** the same mechanisms over fixed physical-time windows; physical
           seconds are preferable to event counts under the observed activity-rate drift.
        4. **Liquidity mechanics:** economically constrained agreements/differences among book,
           order, and trade flow, normalized by contemporaneous liquidity.
        5. **Scale satellite:** absolute depth, activity, and price-level context. It was admitted
           only after paired chronological ablation supported it.

        Path-shape features were quarantined because tied timestamps make some within-timestamp
        paths order-dependent and because their development delta was negative. Independently
        aggregated, unaligned L1/L2 cross-features were also kept out of the accepted hierarchy.

        Research anchors include order-flow imbalance ([Cont, Kukanov & Stoikov,
        2014](https://doi.org/10.1093/jjfinec/nbt003)), temporal financial validation and
        concept drift ([Lucchese, Pakkanen & Veraart,
        2024](https://doi.org/10.1016/j.ijforecast.2024.02.001)), and conservative model selection
        under finite validation data ([Cawley & Talbot,
        2010](https://jmlr.org/papers/v11/cawley10a.html)).
        """
    ),
    markdown("## Data"),
    code(
        """
        profile = artifacts["profile"]
        quality = artifacts["quality"]

        coverage_rows = []
        if profile:
            for source_key, payload in profile.get("files", {}).items():
                split, source = source_key.split("/", maxsplit=1)
                structure = payload.get("structure", {})
                coverage_rows.append(
                    {
                        "split": split,
                        "source": source,
                        "raw rows": structure.get("rows"),
                        "unique samples": structure.get("observed_unique_sample_ids"),
                        "missing samples": structure.get("missing_sample_ids"),
                        "mean rows/sample": structure.get("mean_rows_per_observed_sample"),
                        "median rows/sample": nested_value(
                            structure, "rows_per_observed_sample.q50"
                        ),
                    }
                )

        if coverage_rows:
            coverage = pd.DataFrame(coverage_rows)
            display(coverage)
        else:
            display(Markdown("**Coverage profile is missing.**"))

        data_summary = pd.DataFrame(
            [
                {
                    "labeled rows": nested_value(profile, "label.rows"),
                    "labeled months": nested_value(profile, "label.months"),
                    "train month min": nested_value(profile, "label.month_min"),
                    "train month max": nested_value(profile, "label.month_max"),
                    "test rows": nested_value(profile, "submission.rows"),
                }
            ]
        )
        display(data_summary)
        """
    ),
    code(
        """
        activity_path = artifact_paths["activity_shift"]
        if activity_path.exists():
            activity = pd.read_csv(activity_path)
            activity_pivot = activity.pivot(index="source", columns="split", values="mean_rows")
            activity_pivot["test vs train %"] = 100 * (
                activity_pivot["Test"] / activity_pivot["Train"] - 1
            )
            display(activity_pivot.reset_index())
        else:
            display(Markdown("**Activity-shift table is missing.**"))

        quality_checks = pd.DataFrame(
            [
                ("Profiled engineered features", nested_value(quality, "rows_profiled")),
                ("Any infinite values", nested_value(quality, "any_infinite_values")),
                ("Expected-bound violations", nested_value(quality, "total_bound_violations")),
                (
                    "Largest train/test missing-rate shift",
                    nested_value(quality, "largest_absolute_missing_shift"),
                ),
                (
                    "Features with >5 percentage-point missing shift",
                    nested_value(quality, "features_missing_shift_over_5pct"),
                ),
                (
                    "Features with robust median shift >1 IQR",
                    nested_value(quality, "features_robust_median_shift_over_1_iqr"),
                ),
                (
                    "Features with |sample-index correlation| >0.95",
                    nested_value(quality, "features_abs_index_correlation_over_0_95"),
                ),
                (
                    "Features with |target correlation| >0.5",
                    nested_value(quality, "features_abs_target_correlation_over_0_5"),
                ),
            ],
            columns=["quality check", "value"],
        )
        display(quality_checks)
        """
    ),
    markdown(
        """
        The event tables cover every sample, but test orders and trades are materially denser than
        train. That is why the core emphasizes fixed physical-time windows, ratios, relative
        prices, liquidity normalization, robust clipping, and explicit availability indicators.
        The missing-rate shift is retained as a caveat rather than hidden by global imputation.
        """
    ),
    markdown("## Results"),
    markdown("### Development-only learner and feature selection"),
    code(
        """
        ranking = pd.DataFrame((final_development or {}).get("ranking", []))
        ranking_columns = [
            "spec",
            "pooled_cosine",
            "median_month_cosine",
            "p10_month_cosine",
            "bottom_quartile_mean_cosine",
            "positive_month_share",
            "worst_month_cosine_diagnostic_only",
        ]
        if not ranking.empty:
            display(
                ranking.loc[:, [column for column in ranking_columns if column in ranking]]
                .sort_values("pooled_cosine", ascending=False)
            )
        else:
            display(Markdown("**Final development ranking is missing.**"))

        selected_spec = nested_value(final_development, "best_spec")
        fold_rows = pd.DataFrame((final_development or {}).get("folds", []))
        if not fold_rows.empty:
            fold_rows = fold_rows[fold_rows["spec"] == selected_spec]
            fold_columns = [
                "fold",
                "train_rows",
                "validation_rows",
                "raw_features",
                "transformed_features",
                "fold_cosine",
            ]
            display(
                fold_rows.loc[:, [column for column in fold_columns if column in fold_rows]]
            )
        """
    ),
    code(
        """
        gbdt_feature_rows = []
        for artifact_name in ("gbdt_multiscale", "gbdt_multiscale_scale", "gbdt_final"):
            summary = artifacts[artifact_name]
            if summary:
                row = best_ranking_row(summary)
                gbdt_feature_rows.append(
                    {
                        "feature set": summary.get("feature_set"),
                        "raw features": summary.get("raw_feature_count"),
                        "selected spec": summary.get("best_spec"),
                        "pooled cosine": row.get("pooled_cosine"),
                        "median month cosine": row.get("median_month_cosine"),
                        "positive month share": row.get("positive_month_share"),
                    }
                )
        if gbdt_feature_rows:
            display(
                pd.DataFrame(gbdt_feature_rows)
                .sort_values("pooled cosine")
            )

        ridge_ablation = artifacts["ridge_ablation"] or []
        ablation = pd.DataFrame(ridge_ablation)
        if not ablation.empty:
            ablation_columns = [
                "parent",
                "candidate",
                "pooled_delta",
                "positive_fold_count",
                "bootstrap_3m_se",
                "bootstrap_3m_ci_low",
                "bootstrap_3m_ci_high",
                "promoted",
            ]
            display(
                ablation.loc[:, [column for column in ablation_columns if column in ablation]]
            )
        """
    ),
    markdown(
        """
        The feature table above is hierarchical evidence, not a flat feature search. Invariant
        order/trade flow is compared with market-only features first; multiscale structure,
        liquidity mechanics, and scale context then have to earn admission on chronological
        out-of-fold predictions. Block-bootstrap uncertainty uses contiguous month blocks because
        millions of rows do not imply millions of independent validation units.
        """
    ),
    markdown("### Rejected alternatives and negative controls"),
    code(
        """
        rejected_rows = []

        if not ablation.empty:
            path_rows = ablation[ablation["candidate"] == "multiscale_path"]
            if not path_rows.empty:
                path_row = path_rows.iloc[0]
                rejected_rows.append(
                    {
                        "candidate": "Path-shape satellite",
                        "development cosine": path_row.get("candidate_pooled_cosine"),
                        "reference cosine": path_row.get("parent_pooled_cosine"),
                        "delta": path_row.get("pooled_delta"),
                        "latest-fold delta": nested_value(
                            path_row.to_dict(), "fold_deltas.Dev3", "Dev3"
                        ),
                        "decision": "Reject",
                        "reason": "Negative pooled delta; tied-timestamp path ordering is ambiguous",
                    }
                )

        equal_history_score = development_best.get("pooled_cosine")
        for artifact_name, label in (
            ("gbdt_recency_48", "48-month half-life"),
            ("gbdt_recency_24", "24-month half-life"),
        ):
            summary = artifacts[artifact_name]
            if summary:
                row = best_ranking_row(summary)
                score = row.get("pooled_cosine")
                rejected_rows.append(
                    {
                        "candidate": label,
                        "development cosine": score,
                        "reference cosine": equal_history_score,
                        "delta": (
                            score - equal_history_score
                            if score is not None and equal_history_score is not None
                            else None
                        ),
                        "latest-fold delta": None,
                        "decision": "Reject",
                        "reason": "Within uncertainty of equal weighting and weaker on latest fold",
                    }
                )

        rejected_rows.append(
            {
                "candidate": "Ridge + GBDT blend",
                "development cosine": None,
                "reference cosine": equal_history_score,
                "delta": None,
                "latest-fold delta": None,
                "decision": "Reject",
                "reason": nested_value(
                    frozen,
                    "rejected_candidates.ridge_blend",
                    default="Development grid selected zero Ridge weight",
                ),
            }
        )

        shuffle_summary = artifacts["shuffle_control"]
        if shuffle_summary:
            shuffle_score = best_ranking_row(shuffle_summary).get("pooled_cosine")
            rejected_rows.append(
                {
                    "candidate": "Within-month shuffled-label control",
                    "development cosine": shuffle_score,
                    "reference cosine": equal_history_score,
                    "delta": (
                        shuffle_score - equal_history_score
                        if shuffle_score is not None and equal_history_score is not None
                        else None
                    ),
                    "latest-fold delta": None,
                    "decision": "Control passed",
                    "reason": "Predictive score disappears after destroying feature-target pairing",
                }
            )

        rejected = pd.DataFrame(rejected_rows)
        display(rejected)
        """
    ),
    markdown(
        """
        The selected system is therefore one strongly regularized shallow LightGBM, with equal
        weight on all available historical rows. A two-model blend was not retained merely for
        diversity: after out-of-fold component normalization, the development optimum put zero
        weight on Ridge. The half-life candidates offered no uncertainty-adjusted gain and weakened
        the latest fold, so equal history weighting was frozen.
        """
    ),
    markdown("### Deployment-age stress (diagnostic, never a selector)"),
    code(
        """
        def scalar_leaves(payload: Any, prefix: str = "", max_depth: int = 4) -> list[dict[str, Any]]:
            # Create a bounded, readable table of scalar JSON leaves.
            leaves: list[dict[str, Any]] = []

            def visit(value: Any, path: str, depth: int) -> None:
                if depth > max_depth:
                    return
                if isinstance(value, dict):
                    for key, child in value.items():
                        visit(child, f"{path}.{key}" if path else str(key), depth + 1)
                elif isinstance(value, list):
                    if not value:
                        leaves.append({"metric": path, "value": "[]"})
                    elif all(not isinstance(item, (dict, list)) for item in value) and len(value) <= 12:
                        leaves.append({"metric": path, "value": value})
                else:
                    leaves.append({"metric": path, "value": value})

            visit(payload, prefix, 0)
            return leaves


        def display_record_tables(payload: Any, prefix: str = "", max_depth: int = 3) -> None:
            # Display bounded list-of-record tables nested in an audit JSON object.
            def visit(value: Any, path: str, depth: int) -> None:
                if depth > max_depth:
                    return
                if isinstance(value, dict):
                    for key, child in value.items():
                        visit(child, f"{path}.{key}" if path else str(key), depth + 1)
                elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    display(Markdown(f"**{path}**"))
                    display(pd.DataFrame(value).head(40))

            visit(payload, prefix, 0)


        deployment_stress = artifacts["deployment_stress"]
        if deployment_stress is None:
            display(Markdown("**Pending:** `deployment_stress_summary.json` has not been created."))
        else:
            stress_scalars = pd.DataFrame(scalar_leaves(deployment_stress)).head(40)
            display(stress_scalars)
            display_record_tables(deployment_stress)
        """
    ),
    markdown(
        """
        This check asks how a model trained at an earlier origin behaves as deployment age grows.
        It is intentionally separated from the Development 1-3 selection table: the historical
        38-month simulation has much less training history than the final model and therefore
        cannot choose feature families or hyperparameters without changing the estimand.
        """
    ),
    markdown("### One-time sealed audit"),
    code(
        """
        if sealed_audit is None:
            display(Markdown("**Pending:** `sealed_audit_summary.json` has not been created."))
        else:
            sealed_scalars = pd.DataFrame(scalar_leaves(sealed_audit)).head(50)
            display(sealed_scalars)
            display_record_tables(sealed_audit)
        """
    ),
    markdown(
        """
        The sealed score is an honest estimate for the already-frozen pipeline, not another row in
        the model search. No feature, model, history-weighting, or blend change is justified by
        this single block after it is opened. That separation is the main defense against adaptive
        validation overfit.
        """
    ),
    markdown("### Final submission audit"),
    code(
        """
        if submission_audit is None:
            display(Markdown("**Pending:** `submission_audit.json` has not been created."))
        else:
            submission_scalars = pd.DataFrame(scalar_leaves(submission_audit)).head(60)
            display(submission_scalars)
            display_record_tables(submission_audit)
        """
    ),
    markdown("## Takeaways"),
    code(
        """
        takeaway_lines = [
            f"1. The frozen model is `{nested_value(frozen, 'model_family', default='pending')}` "
            f"with the `{nested_value(frozen, 'model_spec_name', default='pending')}` specification "
            f"on `{nested_value(frozen, 'feature_set', default='pending')}`; its Development 1-3 "
            f"pooled cosine is **{format_number(development_best.get('pooled_cosine'))}**.",
            "2. The accepted hierarchy is observability → invariant core → multiscale dynamics → "
            "liquidity mechanics → scale context. Path features, recency weighting, and Ridge "
            "blending failed their predeclared development gates.",
            f"3. The one-time sealed cosine is **{format_number(sealed_cosine)}**. It is reported as "
            "an audit and was not recycled into selection.",
            f"4. Submission status is **{submission_status}**. The CSV should be uploaded only when "
            "the independent audit confirms exact sample-ID coverage, finite nonconstant values, "
            "the expected row count, and successful CSV round-trip.",
        ]
        display(Markdown("\\n".join(takeaway_lines)))
        """
    ),
    markdown(
        """
        ### Remaining caveats

        - The target horizon and absolute timestamps are undisclosed, so exact interval purging
          cannot be demonstrated; month-disjoint validation is the available safeguard.
        - Test spans 38 later months. No historical stress test can fully reproduce a model trained
          on all 71 labeled months and then deployed that far into the future.
        - Raw order and trade intensity shifts between train and test. Physical-time, normalized
          features reduce this risk but cannot prove invariant conditional relationships.
        - A positive chronological validation result is evidence of predictability, not a guarantee
          of leaderboard or live-trading performance; transaction costs and tradability are outside
          this competition metric.
        """
    ),
    markdown("### Artifact manifest"),
    code(
        """
        manifest_rows = []
        for name, path in artifact_paths.items():
            manifest_rows.append(
                {
                    "artifact": name,
                    "path relative to project root": path.relative_to(project_root).as_posix(),
                    "present": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else None,
                }
            )
        display(pd.DataFrame(manifest_rows))
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "project": {
            "report_type": "frozen-model audit",
            "project_root": WORKSPACE.name,
            "generated_by": Path(__file__).name,
        },
    },
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote unexecuted notebook to {NOTEBOOK_PATH}")
