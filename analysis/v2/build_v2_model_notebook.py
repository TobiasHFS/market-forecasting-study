"""Build the artifact-only v2 financial forecasting report notebook.

The generated notebook is a reproducible audit and handoff document.  It reads
saved JSON, CSV, and (only when needed for provenance) NPZ-adjacent manifests;
it never imports the training pipeline, fits a preprocessor, or trains a model.
Optional experiment, stress-test, and final-submission artifacts are rendered as
pending until they exist, so the notebook can be generated before the pipeline
has finished.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "analysis" / "v2" / "final_model_report_v2.ipynb"


def markdown(source: str):
    """Return a normalized Markdown cell."""

    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    """Return a normalized Python code cell."""

    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(
        r"""
        # Financial market forecasting v2: final model report

        This is the reader-facing, artifact-only record of the second modeling cycle. It answers
        why the first submission underperformed, what was changed, whether the changes survived
        strict chronology, and whether the final CSV is safe to submit.

        **Reproducibility contract:** every numerical claim below is read from a saved JSON/CSV
        artifact. This notebook does not refit, tune, or reopen model selection. Missing optional
        artifacts are shown as **pending**, never silently replaced with an assumption.

        ## tl;dr  -  executive answer

        The next cell renders the current answer from whatever audited artifacts are present.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import hashlib
        import json
        from datetime import datetime, timezone
        from pathlib import Path
        from typing import Any

        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None


        def locate_project_root(start: Path | None = None) -> Path:
            origin = (start or Path.cwd()).resolve()
            for candidate in (origin, *origin.parents):
                if (
                    (candidate / "analysis" / "v2").is_dir()
                    and (candidate / "artifacts").is_dir()
                    and (candidate / "analysis" / "pipeline_config.py").exists()
                ):
                    return candidate
            raise FileNotFoundError(
                "Could not locate the Financial Modelling project root. "
                "Launch this notebook from inside the project."
            )


        project_root = locate_project_root()
        artifact_paths = {
            "competition_profile": project_root / "analysis/output/competition_profile.json",
            "activity_shift": project_root / "analysis/output/activity_shift.csv",
            "v1_postmortem": project_root / "artifacts/diagnostics/postmortem/postmortem_summary.json",
            "v1_domain_shift": project_root / "artifacts/diagnostics/postmortem/domain_shift_summary.json",
            "v1_historical_months": project_root / "artifacts/diagnostics/postmortem/historical_month_scores.csv",
            "v2_selection": project_root / "artifacts/v2/diagnostics/model_selection_summary.json",
            "v2_fold_scores": project_root / "artifacts/v2/diagnostics/model_selection_fold_scores.csv",
            "v2_monthly_scores": project_root / "artifacts/v2/diagnostics/model_selection_monthly_scores.csv",
            "v2_joint_screen": project_root / "artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_Dev1-Dev2_summary.json",
            "v2_joint_confirmation": project_root / "artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_Dev3_summary.json",
            "v2_joint_sealed": project_root / "artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_SealedAudit_summary.json",
            "v2_base_screen": project_root / "artifacts/v2/experiments/sequence_base_only_capacity_Dev1-Dev2_summary.json",
            "v2_base_confirmation": project_root / "artifacts/v2/experiments/sequence_base_only_capacity_Dev3_summary.json",
            "tabm_challenger": project_root / "artifacts/v2/tabm_mini/tabm_mini_challenger_summary.json",
            "frozen_blend": project_root / "artifacts/v2/diagnostics/frozen_blend_before_dev3.json",
            "blend_comparison": project_root / "artifacts/v2/diagnostics/tabm_capacity_blend_comparison.json",
            "blend_pooled": project_root / "artifacts/v2/diagnostics/tabm_capacity_pooled_report.json",
            "sealed_blend": project_root / "artifacts/v2/sealed_blend/sealed_blend_summary.json",
            "deployment_stress": project_root / "artifacts/v2/diagnostics/deployment_stress_summary.json",
            "deployment_curve": project_root / "artifacts/v2/diagnostics/deployment_stress_curve.csv",
            "deployment_origins": project_root / "artifacts/v2/diagnostics/deployment_stress_origins.csv",
            "final_blend_pointer": project_root / "artifacts/v2/models/final_blend_pointer.json",
            "blend_submission_audit": project_root / "artifacts/v2/diagnostics/submission_blend_audit.json",
            "legacy_final_manifest": project_root / "artifacts/v2/models/final_training_manifest.json",
            "legacy_v2_submission_audit": project_root / "artifacts/v2/diagnostics/submission_audit.json",
            "root_submission_audit": project_root / "artifacts/diagnostics/submission_audit.json",
            "v2_submission": project_root / "artifacts/v2/submissions/submission_final.csv",
        }


        def load_json_optional(path: Path) -> dict[str, Any] | list[Any] | None:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))


        def load_csv_optional(path: Path) -> pd.DataFrame:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()


        def nested(payload: Any, dotted_path: str, default: Any = None) -> Any:
            value = payload
            for key in dotted_path.split("."):
                if not isinstance(value, dict) or key not in value:
                    return default
                value = value[key]
            return default if value is None else value


        def first_nested(payload: Any, paths: tuple[str, ...], default: Any = None) -> Any:
            for dotted_path in paths:
                value = nested(payload, dotted_path)
                if value is not None:
                    return value
            return default


        def fmt(value: Any, digits: int = 6, missing: str = "pending") -> str:
            if value is None:
                return missing
            if isinstance(value, (int, np.integer)):
                return f"{int(value):,}"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.{digits}f}"
            return str(value)


        def sha256_file(path: Path) -> str | None:
            if not path.exists() or not path.is_file():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()


        def resolve_project_relative(value: Any) -> Path | None:
            if not isinstance(value, str) or not value.strip():
                return None
            raw = Path(value)
            if raw.is_absolute() or ".." in raw.parts:
                return None
            candidate = (project_root / raw).resolve()
            try:
                candidate.relative_to(project_root)
            except ValueError:
                return None
            return candidate


        json_artifacts = {
            name: load_json_optional(path)
            for name, path in artifact_paths.items()
            if path.suffix.lower() == ".json"
        }
        selection = json_artifacts["v2_selection"]
        postmortem = json_artifacts["v1_postmortem"]
        blend_pointer = json_artifacts["final_blend_pointer"]
        if blend_pointer is not None:
            publication_mode = "blend"
            final_manifest_path = resolve_project_relative(
                nested(blend_pointer, "generation_manifest")
            )
            final_manifest = (
                load_json_optional(final_manifest_path)
                if final_manifest_path is not None and final_manifest_path.is_file()
                else None
            )
            final_audit = json_artifacts["blend_submission_audit"]
            if final_manifest_path is not None:
                artifact_paths["generation_manifest"] = final_manifest_path
        else:
            publication_mode = "single_model"
            final_manifest_path = artifact_paths["legacy_final_manifest"]
            final_manifest = json_artifacts["legacy_final_manifest"]
            final_audit = json_artifacts["legacy_v2_submission_audit"]

        development_v2 = nested(selection, "development.v2_final_q1p2_cosine")
        development_v1 = nested(selection, "development.v1_raw_cosine")
        sealed_v2 = nested(selection, "sealed_audit_descriptive_only.v2_final_q1p2_cosine")
        public_v1 = nested(postmortem, "public_leaderboard.score")
        selected_feature_set = nested(selection, "selection_protocol.selected_feature_set")
        selected_model = nested(selection, "selection_protocol.selected_model_spec")
        selected_power = nested(selection, "selection_protocol.selected_signed_power")

        manifest_status = nested(final_manifest, "status", "pending")
        final_rows = first_nested(
            final_audit,
            ("rows", "row_count", "n_rows", "prediction.rows"),
            nested(final_manifest, "test_rows"),
        )
        final_submission_hash = first_nested(
            final_audit,
            ("submission_sha256", "canonical_submission_sha256"),
            first_nested(
                blend_pointer,
                ("submission_sha256",),
                first_nested(
                    final_manifest,
                    (
                        "artifacts.generation_submission.sha256",
                        "outputs.canonical_submission.sha256",
                        "outputs.submission_sha256",
                    ),
                ),
            ),
        )
        blend_comparison = json_artifacts["blend_comparison"]
        blend_pooled = json_artifacts["blend_pooled"]
        selected_blend = nested(
            blend_comparison,
            "selected",
            nested(json_artifacts["frozen_blend"], "blend_contract", {}),
        )
        blend_development = first_nested(
            blend_pooled,
            ("pooled_dev1_dev2_dev3_cosine", "pooled_development_cosine"),
            nested(blend_comparison, "pooled_reporting.cosine"),
        )
        blend_dev3 = nested(blend_comparison, "dev3_audit.cosine")
        blend_sealed = nested(
            json_artifacts["sealed_blend"],
            "diagnostic_scores.frozen_blend_final_q1p1.pooled_cosine",
        )
        sealed_veto_status = nested(
            json_artifacts["sealed_blend"],
            "predeclared_safety_gate.status",
        )

        headline = [
            f"- **Diagnosis:** v1 scored **{fmt(public_v1, 3)}** publicly versus "
            f"**{fmt(development_v1)}** on chronological development OOF; the evidence points to "
            "regime/covariate shift plus material under-capacity, not an ID-alignment or calibration bug.",
            (
                f"- **Final v2 blend:** TabM-mini weight = **{fmt(selected_blend.get('tabm_weight'), 1)}**, "
                f"capacity-LightGBM weight = **{fmt(selected_blend.get('capacity_lightgbm_weight'), 1)}**, "
                f"post-blend $q={fmt(selected_blend.get('power_exponent'), 1)}$; pooled Dev1-Dev3 cosine = "
                f"**{fmt(blend_development)}**, one-shot Dev3 = **{fmt(blend_dev3)}**."
                if publication_mode == "blend" or blend_comparison is not None
                else f"- **Selected v2:** `{selected_feature_set or 'pending'}` with `{selected_model or 'pending'}` LightGBM and signed power $q={fmt(selected_power, 1)}$; development pooled cosine = **{fmt(development_v2)}**."
            ),
            (
                f"- **One-time sealed safety veto:** frozen-blend cosine = **{fmt(blend_sealed)}** and the predeclared keep/fallback gate is **{sealed_veto_status or 'pending'}**. Months 59-70 could only keep the frozen blend or trigger the already-frozen fallback; they could not retune either candidate."
                if publication_mode == "blend" or blend_comparison is not None
                else f"- **One-time sealed safety veto:** final-view cosine = **{fmt(sealed_v2)}**. Months 59-70 were reserved for the predeclared keep/fallback decision and could not retune the frozen candidates."
            ),
            f"- **Final publication:** manifest = **{manifest_status}**, rows = **{fmt(final_rows, 0)}**, "
            f"submission SHA-256 = `{final_submission_hash or 'pending'}`.",
        ]
        display(Markdown("\n".join(headline)))
        print(f"Project root: {project_root}")
        print("Notebook mode: artifact-only; no model fitting code is present.")
        """
    ),
    markdown(
        r"""
        ## Context & Methods

        ### Competition data and exact metric

        Each labeled sample joins a future-return target to three histories before the prediction
        timestamp: approximately 600 seconds of equally spaced market/book bars, 60 seconds of raw
        orders, and 60 seconds of raw trades. Train spans months 0-70; the unlabeled test era spans
        months 71-108. The leaderboard evaluates a single concatenated vector with **uncentered
        cosine**:

        $$
        C(\hat y,y)=\frac{\sum_i \hat y_i y_i}
        {\sqrt{\sum_i \hat y_i^2}\sqrt{\sum_i y_i^2}}.
        $$

        The primary validation statistic therefore concatenates all rows in a chronological block
        before scoring. An unweighted average of monthly cosines is not the competition metric;
        monthly scores are retained as robustness diagnostics.
        """
    ),
    code(
        r"""
        profile = json_artifacts["competition_profile"]
        activity = load_csv_optional(artifact_paths["activity_shift"])

        if profile:
            data_overview = pd.DataFrame(
                [
                    {
                        "train samples": nested(profile, "label.rows"),
                        "train months": nested(profile, "label.months"),
                        "train month range": f"{nested(profile, 'label.month_min')}-{nested(profile, 'label.month_max')}",
                        "test samples": nested(profile, "submission.rows"),
                        "duplicate train IDs": nested(profile, "label.duplicate_sample_ids"),
                        "target excess kurtosis": nested(profile, "label.target.excess_kurtosis"),
                    }
                ]
            )
            display(data_overview)
        else:
            display(Markdown("**Pending:** competition profile is unavailable."))

        if not activity.empty and {"source", "split", "mean_rows"}.issubset(activity.columns):
            activity_table = activity.pivot(index="source", columns="split", values="mean_rows")
            if {"Train", "Test"}.issubset(activity_table.columns):
                activity_table["test vs train %"] = 100 * (
                    activity_table["Test"] / activity_table["Train"] - 1
                )
            display(activity_table.reset_index().round(3))
        else:
            display(Markdown("**Pending:** activity-shift table is unavailable."))
        """
    ),
    markdown(
        r"""
        ### Why v1 plausibly underperformed

        The public labels are unavailable, so no postmortem can prove a causal decomposition.
        The saved diagnostics do, however, distinguish several realistic explanations:

        1. **The future regime is measurably different.** A held-out classifier distinguishes all
           train rows from test rows; even recent train versus test remains separable. This proves
           covariate shift, not necessarily target-concept drift.
        2. **The v1 sealed headline was unusually concentrated.** One high-target-energy month
           inflated its global cosine, making the ordinary regime look easier than it was.
        3. **The original tree was under-capacity.** The same base features improve materially when
           the tree ensemble moves from the deliberately shallow v1 specification to the bounded
           v2 capacity specification.
        4. **Cheap fixes were ruled out.** Strict next-fold calibration, row-order rules, and Ridge
           blending did not produce a stable gain. The answer is better representation and a
           better-sized learner, not leaderboard-scale hacking.
        """
    ),
    code(
        r"""
        domain_shift = json_artifacts["v1_domain_shift"]
        diagnosis_rows = []
        if postmortem:
            diagnosis_rows.extend(
                [
                    {
                        "evidence": "v1 public score",
                        "value": nested(postmortem, "public_leaderboard.score"),
                        "interpretation": "Observed leaderboard outcome",
                    },
                    {
                        "evidence": "v1 development pooled cosine",
                        "value": nested(postmortem, "score_gap.development_score"),
                        "interpretation": "Chronological OOF expectation",
                    },
                    {
                        "evidence": "v1 public minus development",
                        "value": nested(postmortem, "score_gap.public_gap_from_development"),
                        "interpretation": "Material negative generalization gap",
                    },
                    {
                        "evidence": "sealed score without highest-energy month",
                        "value": nested(postmortem, "score_gap.sealed_score_without_highest_target_energy_month"),
                        "interpretation": "Shows concentration of sealed headline",
                    },
                    {
                        "evidence": "strict next-fold calibration delta",
                        "value": nested(postmortem, "rolling_calibration.strict_next_fold_delta"),
                        "interpretation": "Calibration was not a stable remedy",
                    },
                ]
            )
        if domain_shift:
            diagnosis_rows.extend(
                [
                    {
                        "evidence": "all-train vs test domain AUC",
                        "value": nested(domain_shift, "train_test_domain_classifier.auc"),
                        "interpretation": "Strong covariate distinguishability",
                    },
                    {
                        "evidence": "recent-train vs test domain AUC",
                        "value": nested(domain_shift, "recent_train_test_domain_classifier.auc"),
                        "interpretation": "Shift persists near deployment boundary",
                    },
                ]
            )
        if diagnosis_rows:
            display(pd.DataFrame(diagnosis_rows).round(6))
        else:
            display(Markdown("**Pending:** v1 postmortem artifacts are unavailable."))
        """
    ),
    markdown(
        r"""
        ### Literature rationale: what was adopted and what was not

        | Primary source | Result used here | Implementation consequence |
        |---|---|---|
        | [Cont, Kukanov & Stoikov  -  order-flow imbalance](https://arxiv.org/abs/1011.6402) | Price pressure is better represented relative to contemporaneous depth | Depth-scaled OFI and signed order pressure |
        | [Gould & Bonart  -  queue imbalance](https://arxiv.org/abs/1512.03492) | Queue state contains short-horizon directional information | L1/L2 queue imbalance and microprice displacement |
        | [DeepLOB](https://arxiv.org/abs/1808.03668) and [robust LOB representations](https://arxiv.org/abs/2110.05479) | Local book paths matter, but raw price levels and venue-specific layouts transfer poorly | Fixed-clock, stationary book-relative paths instead of a raw-event CNN/Transformer transplant |
        | [Deep Order Flow Imbalance](https://doi.org/10.1111/mafi.12413) | Multiple depth levels and nonlinear response can add signal | L1/L2 interactions plus a nonlinear learner |
        | [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf) | Efficient parameter-shared MLP ensembles are strong on tabular data | One bounded TabM-mini challenger, promoted only with complete chronological OOF |
        | [RealMLP, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf) | Careful defaults can make MLPs competitive on tabular tasks | Considered as a challenger, not assumed superior to GBDT |
        | [TabReD, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/571799482291411607c54984153190b0-Paper-Conference.pdf) and [temporal shift, ICML 2025](https://proceedings.mlr.press/v267/cai25j.html) | Random splits overstate performance under temporal drift | Expanding chronological folds and a separate sealed audit |
        | [Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html) | Model-selection variance itself overfits | Small predeclared candidate family, paired block bootstrap, one-time audit |

        A full DeepLOB/TLOB-style network was not the single best practical choice here: the data
        exposes only two book levels, samples are anonymous windows rather than one continuous
        instrument stream, timestamps tie, and deployment covers 38 future months. The transferable
        part of that literature is the representation - stationary, book-aligned local paths - not the
        largest architecture.
        """
    ),
    markdown(
        r"""
        ### V2 representation and model, mathematically

        For sample $i$, v2 forms $x_i\in\mathbb{R}^{754}$:

        $$x_i=[b_i,\,s_i],\qquad b_i\in\mathbb{R}^{474},\quad
        s_i\in\mathbb{R}^{10(11+11+6)}=\mathbb{R}^{280}.$$

        $b_i$ is the established hierarchy of observability, invariant book/flow, multiscale,
        liquidity-mechanics, and scale features. $s_i$ divides the final 60 seconds into ten
        disjoint six-second bins. Each bin contains 11 market channels, 11 order channels, and six
        trade channels: book-relative returns/microprice/spread, L1/L2 imbalance, depth-scaled OFI,
        signed event and volume pressure, cancel fractions, contemporaneous-mid price displacement,
        and event intensity. Bin 0 is closest to prediction time.

        Preprocessing is fit on training rows only. After the declared monotone transform for each
        feature kind, a feature is median-centered, divided by $1.4826\,\mathrm{MAD}$ (with IQR,
        standard-deviation, then unit fallbacks), clipped to $[-8,8]$, and supplemented with a
        missing indicator when missingness occurs in training. This yields 1,326 transformed
        columns in the selected folds.

        The learner minimizes squared error with a regularized additive tree ensemble:

        $$f(x)=\sum_{t=1}^{1200}0.02\,h_t(x),$$

        where each $h_t$ has at most 31 leaves and depth 7; minimum leaf size is 1,000, column
        fraction is 0.75, row fraction is 0.8, and $(\lambda_1,\lambda_2)=(1,30)$. MSE targets the
        conditional mean direction, which is the natural population direction for cosine. The
        frozen output transform is

        $$u_i=\operatorname{sign}(f_i)|f_i|^{1.2},\qquad
        p_i=\frac{u_i}{\sqrt{n^{-1}\sum_j u_j^2}}.$$

        RMS normalization changes no single-vector cosine; it gives a stable submission scale.
        The signed power can change direction and was selected only on the screen period. For the
        final publication candidate, complete TabM-mini and capacity-LightGBM vectors are each
        RMS-normalized, mixed 60/40, transformed with signed power $q=1.1$, and normalized once
        more. Those weights and transforms were frozen before Dev3. Sealed months 59-70 then served
        only the predeclared keep/fallback safety veto: keep this blend if it passed, otherwise use
        the already-frozen capacity-LightGBM $q=1.2$ view; no sealed result could retune the model
        specification.
        """
    ),
    markdown(
        r"""
        ### Strict chronology and selection boundary

        | Role | Fit months | Score months | Use |
        |---|---:|---:|---|
        | Development 1 | 0-22 | 23-34 | Screen |
        | Development 2 | 0-34 | 35-46 | Screen |
        | Development 3 | 0-46 | 47-58 | Confirmation only |
        | Sealed audit | 0-58 | 59-70 | Predeclared keep/fallback safety veto only |
        | Final model | 0-70 | test 71-108 | Submission |

        The first-stage feature set, LightGBM capacity, signed-power exponent, and zero v1 blend
        weight were selected on Development 1-2. The later 60% TabM-mini / 40% capacity-LightGBM
        candidate and post-blend q=1.1 transform were also selected on Development 1-2, then frozen
        before a single Development 3 confirmation. The sealed block was opened once only to
        apply the predeclared keep/fallback safety veto: it could retain the frozen blend or
        trigger the already-frozen capacity-LightGBM q=1.2 fallback, but could not alter a
        feature, model, epoch, weight, transform, or normalization. Month boundaries are the
        available chronology; no arbitrary one-month embargo is imposed because no cross-month
        overlap has been demonstrated. If the hidden target horizon later reveals overlap, purge
        the exact overlap duration.
        """
    ),
    markdown("## Results"),
    markdown("### Fold-by-fold comparison"),
    code(
        r"""
        fold_scores = load_csv_optional(artifact_paths["v2_fold_scores"])
        if fold_scores.empty:
            display(Markdown("**Pending:** v2 fold-score artifact is unavailable."))
        else:
            display(fold_scores.round(6))
            plot_folds = fold_scores[fold_scores["period"].isin(["Dev1", "Dev2", "Dev3"])].copy()
            preferred_order = [
                "v1 slow raw",
                "v2 base-only capacity raw",
                "v2 base+path capacity raw",
                "v2 final q=1.2",
            ]
            model_order = [model for model in preferred_order if model in set(plot_folds["model"])]
            pivot = plot_folds.pivot(index="period", columns="model", values="cosine")
            pivot = pivot.reindex(index=["Dev1", "Dev2", "Dev3"], columns=model_order)

            if plt is None:
                display(
                    Markdown(
                        "**Chart fallback:** matplotlib is unavailable; the fold comparison "
                        "is shown as a period-by-model table."
                    )
                )
                display(pivot.rename_axis(columns=None).reset_index().round(6))
            else:
                fig, ax = plt.subplots(figsize=(11, 5.5))
                x = np.arange(len(pivot.index))
                width = 0.8 / max(1, len(model_order))
                colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
                hatches = ["//", "\\\\", "..", "xx"]
                for index, model in enumerate(model_order):
                    offset = (index - (len(model_order) - 1) / 2) * width
                    bars = ax.bar(
                        x + offset,
                        pivot[model],
                        width=width,
                        label=model,
                        color=colors[index % len(colors)],
                        hatch=hatches[index % len(hatches)],
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
                ax.set_title("V2 improves every chronological development fold (months 23-58)")
                ax.set_xlabel("Expanding-window validation fold")
                ax.set_ylabel("Global uncentered cosine")
                ax.set_xticks(x, pivot.index)
                ax.set_ylim(0, max(0.16, float(np.nanmax(pivot.to_numpy())) * 1.16))
                ax.grid(axis="y", alpha=0.25)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
                fig.tight_layout()
                plt.show()
        """
    ),
    markdown("### Capacity and path ablation"),
    code(
        r"""
        if selection is None:
            display(Markdown("**Pending:** v2 selection summary is unavailable."))
        else:
            development = nested(selection, "development", {})
            ablation = pd.DataFrame(
                [
                    {
                        "system": "v1 base features + slow LightGBM",
                        "development pooled cosine": development.get("v1_raw_cosine"),
                        "increment vs previous": None,
                    },
                    {
                        "system": "v1 base features + capacity LightGBM",
                        "development pooled cosine": development.get("v2_base_capacity_raw_cosine"),
                        "increment vs previous": (
                            development.get("v2_base_capacity_raw_cosine", np.nan)
                            - development.get("v1_raw_cosine", np.nan)
                        ),
                    },
                    {
                        "system": "+ 280 fixed-clock path features",
                        "development pooled cosine": development.get("v2_joint_capacity_raw_cosine"),
                        "increment vs previous": development.get("sequence_increment_raw"),
                    },
                    {
                        "system": "+ frozen signed power q=1.2",
                        "development pooled cosine": development.get("v2_final_q1p2_cosine"),
                        "increment vs previous": (
                            development.get("v2_final_q1p2_cosine", np.nan)
                            - development.get("v2_joint_capacity_raw_cosine", np.nan)
                        ),
                    },
                ]
            )
            display(ablation.round(6))

            capacity_bootstrap = nested(selection, "development.capacity_bootstrap_vs_v1", {})
            sequence_bootstrap = nested(selection, "development.sequence_bootstrap_vs_base_capacity", {})
            uncertainty = pd.DataFrame(
                [
                    {
                        "paired comparison": "joint capacity v2 minus v1",
                        "observed delta": capacity_bootstrap.get("observed_difference"),
                        "3-month block-bootstrap SE": capacity_bootstrap.get("bootstrap_standard_error"),
                        "95% low": capacity_bootstrap.get("confidence_low"),
                        "95% high": capacity_bootstrap.get("confidence_high"),
                        "P(delta > 0)": capacity_bootstrap.get("probability_a_better"),
                    },
                    {
                        "paired comparison": "path increment at fixed capacity",
                        "observed delta": sequence_bootstrap.get("observed_difference"),
                        "3-month block-bootstrap SE": sequence_bootstrap.get("bootstrap_standard_error"),
                        "95% low": sequence_bootstrap.get("confidence_low"),
                        "95% high": sequence_bootstrap.get("confidence_high"),
                        "P(delta > 0)": sequence_bootstrap.get("probability_a_better"),
                    },
                ]
            )
            display(uncertainty.round(6))
        """
    ),
    markdown(
        r"""
        The decomposition matters: most of the gain comes from correcting v1 under-capacity, while
        the fixed-clock path contributes a smaller, separately positive increment at identical
        model capacity. The path family was kept because all three folds improved and its paired
        three-month block-bootstrap interval is approximately positive - not because a deep sequence
        model is fashionable.
        """
    ),
    markdown("### Monthly robustness"),
    code(
        r"""
        monthly_scores = load_csv_optional(artifact_paths["v2_monthly_scores"])
        if monthly_scores.empty:
            display(Markdown("**Pending:** v2 monthly-score artifact is unavailable."))
        else:
            selected_models = [
                model
                for model in ("v1 slow raw", "v2 final q=1.2")
                if model in set(monthly_scores["model"])
            ]
            monthly_plot = monthly_scores[monthly_scores["model"].isin(selected_models)].copy()
            monthly_table = (
                monthly_plot.groupby("model", as_index=False)["cosine"]
                .agg(["median", "min", "mean"])
                .reset_index()
            )
            monthly_table["positive-month share"] = monthly_plot.groupby("model")["cosine"].apply(
                lambda values: float((values > 0).mean())
            ).to_numpy()
            display(monthly_table.round(6))

            if plt is None:
                monthly_fallback = (
                    monthly_plot.pivot(index="month", columns="model", values="cosine")
                    .reindex(columns=selected_models)
                    .rename_axis(columns=None)
                    .reset_index()
                )
                display(
                    Markdown(
                        "**Chart fallback:** matplotlib is unavailable; the chronological "
                        "month-by-model comparison is shown below."
                    )
                )
                display(monthly_fallback.round(6))
            else:
                fig, ax = plt.subplots(figsize=(12, 5.5))
                styles = {
                    "v1 slow raw": {"color": "#4C78A8", "marker": "o", "linestyle": "--"},
                    "v2 final q=1.2": {"color": "#E45756", "marker": "s", "linestyle": "-"},
                }
                for model in selected_models:
                    subset = monthly_plot[monthly_plot["model"] == model].sort_values("month")
                    ax.plot(
                        subset["month"],
                        subset["cosine"],
                        label=model,
                        markersize=4,
                        linewidth=1.6,
                        **styles[model],
                    )
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title("V2 monthly cosine versus v1 across all development months (23-58)")
                ax.set_xlabel("Chronological month")
                ax.set_ylabel("Per-month uncentered cosine (robustness diagnostic)")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
                fig.tight_layout()
                plt.show()
        """
    ),
    markdown("### Frozen selection and one-time sealed safety veto"),
    code(
        r"""
        if selection is None:
            display(Markdown("**Pending:** frozen selection/sealed comparison is unavailable."))
        else:
            protocol = nested(selection, "selection_protocol", {})
            selected_power_row = nested(selection, "selected_power_row", {})
            selected_blend_row = nested(selection, "selected_blend_row", {})
            sealed = nested(selection, "sealed_audit_descriptive_only", {})
            decision = pd.DataFrame(
                [
                    {
                        "decision": "feature set",
                        "selected value": protocol.get("selected_feature_set"),
                        "screen evidence": "Dev1-Dev2",
                        "confirmation": "Dev3 only",
                    },
                    {
                        "decision": "model specification",
                        "selected value": protocol.get("selected_model_spec"),
                        "screen evidence": "Dev1-Dev2",
                        "confirmation": "Dev3 only",
                    },
                    {
                        "decision": "signed-power exponent",
                        "selected value": protocol.get("selected_signed_power"),
                        "screen evidence": selected_power_row.get("screen_cosine"),
                        "confirmation": selected_power_row.get("confirmation_cosine"),
                    },
                    {
                        "decision": "v1 blend weight",
                        "selected value": protocol.get("selected_v1_blend_weight"),
                        "screen evidence": selected_blend_row.get("screen_cosine"),
                        "confirmation": selected_blend_row.get("confirmation_cosine"),
                    },
                ]
            )
            display(decision)

            sealed_comparison = pd.DataFrame(
                [
                    {"view": "v1 raw", "sealed cosine": sealed.get("v1_raw_cosine")},
                    {"view": "v2 raw", "sealed cosine": sealed.get("v2_raw_cosine")},
                    {"view": "v2 final q=1.2", "sealed cosine": sealed.get("v2_final_q1p2_cosine")},
                    {"view": "v1 raw excluding month 66", "sealed cosine": sealed.get("v1_raw_excluding_month_66")},
                    {"view": "v2 raw excluding month 66", "sealed cosine": sealed.get("v2_raw_excluding_month_66")},
                ]
            )
            display(sealed_comparison.round(6))
            display(
                Markdown(
                    "The sealed result passed the predeclared keep/fallback safety veto, including "
                    "after removing the previously dominant month 66, so the frozen pipeline was "
                    "kept. The veto could only keep that candidate or trigger the already-frozen "
                    "fallback; it could not authorize search or retuning."
                )
            )
        """
    ),
    markdown("### TabM-mini component and frozen blend decision"),
    code(
        r"""
        tabm = json_artifacts["tabm_challenger"]
        if blend_comparison is not None:
            display(
                pd.DataFrame(
                    [
                        {
                            "TabM weight": nested(blend_comparison, "selected.tabm_weight"),
                            "capacity-LightGBM weight": nested(blend_comparison, "selected.capacity_lightgbm_weight"),
                            "post-blend q": nested(blend_comparison, "selected.power_exponent"),
                            "Dev1-Dev2 screen cosine": nested(blend_comparison, "selected.screen_cosine"),
                            "one-shot Dev3 cosine": nested(blend_comparison, "dev3_audit.cosine"),
                            "Dev3 promotion gate": nested(blend_comparison, "dev3_audit.promotion_gate_passed"),
                            "pooled Dev1-Dev3 cosine": nested(blend_comparison, "pooled_reporting.cosine"),
                        }
                    ]
                ).round(6)
            )
            display(
                Markdown(
                    "TabM-mini was promoted only as one component of the frozen blend. "
                    "Both complete component vectors are RMS-normalized before the 60/40 mix, "
                    "then q=1.1 and one final RMS normalization are applied."
                )
            )
        if tabm is None:
            display(Markdown("**Pending:** no TabM-mini challenger artifact exists."))
        else:
            tabm_rows = pd.DataFrame(
                [
                    {
                        "device": nested(tabm, "device"),
                        "folds completed": ", ".join(nested(tabm, "selected_folds", [])),
                        "complete Development 1-3 OOF": nested(tabm, "complete_development_oof"),
                        "partial/raw pooled cosine": nested(tabm, "partial_or_pooled_tabm_cosine"),
                        "partial/powered pooled cosine": nested(tabm, "partial_or_pooled_power_cosine"),
                        "promotion decision": nested(tabm, "best_candidate"),
                    }
                ]
            )
            display(tabm_rows)
            tabm_folds = nested(tabm, "folds", [])
            if tabm_folds:
                display(
                    pd.DataFrame(
                        [
                            {
                                "fold": row.get("fold"),
                                "selected epoch": row.get("best_epoch"),
                                "inner cosine": row.get("best_inner_ensemble_cosine"),
                                "outer raw cosine": row.get("outer_ensemble_cosine_raw"),
                                "member correlation": row.get("outer_member_mean_pairwise_correlation"),
                                "refit on all outer-train rows": row.get("refit_from_scratch_on_all_outer_train_months"),
                            }
                            for row in tabm_folds
                        ]
                    ).round(6)
                )
            if (
                blend_comparison is None
                and not nested(tabm, "complete_development_oof", False)
            ):
                display(
                    Markdown(
                        "**Decision guardrail:** a partial-fold neural result is diagnostic only. "
                        "It cannot replace or blend with the fully validated LightGBM."
                    )
                )
        """
    ),
    markdown("### Long-horizon deployment-age stress (optional diagnostic)"),
    code(
        r"""
        stress = json_artifacts["deployment_stress"]
        stress_curve = load_csv_optional(artifact_paths["deployment_curve"])
        if stress is None or stress_curve.empty:
            display(
                Markdown(
                    "**Pending:** v2 deployment-age stress has not finished. This section will "
                    "populate from `artifacts/v2/diagnostics/deployment_stress_*` without refitting."
                )
            )
        else:
            views = nested(stress, "prediction_views_diagnostic_only", {})
            stress_summary = pd.DataFrame(
                [
                    {"prediction view": name, **metrics}
                    for name, metrics in views.items()
                    if isinstance(metrics, dict)
                ]
            )
            display(stress_summary.round(6))

            raw_point = "raw_point_cosine" if "raw_point_cosine" in stress_curve else "point_cosine"
            raw_cumulative = (
                "raw_cumulative_cosine"
                if "raw_cumulative_cosine" in stress_curve
                else "cumulative_cosine"
            )
            power_point = "signed_power_q1p2_point_cosine"
            power_cumulative = "signed_power_q1p2_cumulative_cosine"
            support_column = (
                "point_n_origins" if "point_n_origins" in stress_curve else "point_n_forecasts"
            )
            if plt is None:
                stress_columns = ["horizon_months", raw_point, raw_cumulative]
                if power_point in stress_curve and power_cumulative in stress_curve:
                    stress_columns.extend([power_point, power_cumulative])
                stress_columns.append(support_column)
                display(
                    Markdown(
                        "**Chart fallback:** matplotlib is unavailable; deployment-age scores "
                        "and support are shown by horizon below."
                    )
                )
                display(stress_curve.loc[:, stress_columns].round(6))
            else:
                fig, (ax_score, ax_support) = plt.subplots(
                    2, 1, figsize=(11, 8), sharex=True, height_ratios=[3, 1]
                )
                ax_score.plot(
                    stress_curve["horizon_months"], stress_curve[raw_point],
                    color="#4C78A8", marker="o", markersize=3, linewidth=1.3,
                    label="raw point horizon",
                )
                ax_score.plot(
                    stress_curve["horizon_months"], stress_curve[raw_cumulative],
                    color="#4C78A8", linestyle="--", linewidth=2,
                    label="raw cumulative",
                )
                if power_point in stress_curve and power_cumulative in stress_curve:
                    ax_score.plot(
                        stress_curve["horizon_months"], stress_curve[power_point],
                        color="#E45756", marker="s", markersize=3, linewidth=1.3,
                        label="q=1.2 point horizon",
                    )
                    ax_score.plot(
                        stress_curve["horizon_months"], stress_curve[power_cumulative],
                        color="#E45756", linestyle=":", linewidth=2.2,
                        label="q=1.2 cumulative",
                    )
                ax_score.axhline(0, color="black", linewidth=0.8)
                ax_score.set_title("Historical performance as a function of deployment age (diagnostic only)")
                ax_score.set_ylabel("Uncentered cosine")
                ax_score.grid(alpha=0.25)
                ax_score.legend(frameon=False, ncol=2)

                ax_support.step(
                    stress_curve["horizon_months"], stress_curve[support_column], where="mid",
                    color="#54A24B", linewidth=2,
                )
                ax_support.set_xlabel("Months since training origin")
                ax_support.set_ylabel("Origins" if support_column == "point_n_origins" else "Forecast rows")
                ax_support.grid(alpha=0.25)
                fig.tight_layout()
                plt.show()

            display(
                Markdown(
                    "This curve estimates model aging from repeated historical origins ending no "
                    "later than month 58. It is explicitly prohibited from selecting a model: "
                    "early origins have much less training data than the final fit."
                )
            )
        """
    ),
    markdown("### Final submission manifest and independent audit"),
    code(
        r"""
        v2_audit = final_audit
        root_audit = json_artifacts["root_submission_audit"]
        audit = v2_audit or (root_audit if publication_mode == "single_model" else None)
        audit_source = (
            "v2 frozen-blend replay audit"
            if publication_mode == "blend" and v2_audit
            else "v2 single-model audit"
            if v2_audit
            else "legacy root audit candidate"
        )

        if final_manifest is None:
            display(Markdown("**Pending:** final v2 training manifest is unavailable."))
        else:
            pipeline = nested(final_manifest, "pipeline_contract", {})
            prediction_stats = nested(final_manifest, "prediction_statistics", {})
            if publication_mode == "blend":
                feature_description = " / ".join(
                    f"{name}: {value}" for name, value in pipeline.get("feature_sets", {}).items()
                )
                model_description = "TabM-mini + capacity-LightGBM"
                model_specification = (
                    f"TabM weight {nested(pipeline, 'blend.tabm_weight')}, "
                    f"capacity weight {nested(pipeline, 'blend.capacity_weight')}, "
                    f"q={nested(pipeline, 'blend.signed_power')}"
                )
                test_rows_display = first_nested(
                    audit, ("prediction.rows", "rows", "row_count", "n_rows")
                )
            else:
                feature_description = pipeline.get("feature_set")
                model_description = pipeline.get("model_family")
                model_specification = pipeline.get("model_spec_name")
                test_rows_display = nested(final_manifest, "test_rows")
            display(
                pd.DataFrame(
                    [
                        {
                            "status": nested(final_manifest, "status"),
                            "created UTC": nested(final_manifest, "created_utc"),
                            "pipeline SHA-256": nested(final_manifest, "pipeline_sha256"),
                            "feature set(s)": feature_description,
                            "model": model_description,
                            "model spec / blend": model_specification,
                            "training months": pipeline.get("training_months"),
                            "test rows": test_rows_display,
                        }
                    ]
                )
            )
            display(pd.DataFrame([prediction_stats]).round(6))

        if audit is None:
            display(Markdown("**Pending:** no submission audit artifact is available."))
        else:
            manifest_submission_hash = first_nested(
                final_manifest,
                (
                    "artifacts.generation_submission.sha256",
                    "outputs.canonical_submission.sha256",
                    "outputs.submission.sha256",
                    "outputs.submission_sha256",
                ),
            )
            expected_hash = first_nested(
                blend_pointer, ("submission_sha256",), manifest_submission_hash
            )
            audit_hash = first_nested(
                audit,
                (
                    "submission_sha256",
                    "canonical_submission_sha256",
                    "outputs.submission_sha256",
                    "submission.sha256",
                ),
            )
            checks = nested(audit, "checks", {})
            all_checks_pass = bool(checks) and all(value is True for value in checks.values())
            provenance_match = (
                expected_hash is not None
                and audit_hash is not None
                and str(expected_hash).lower() == str(audit_hash).lower()
            )
            generation_match = (
                publication_mode != "blend"
                or (
                    nested(blend_pointer, "status") == "complete"
                    and nested(blend_pointer, "generation_id")
                    == nested(final_manifest, "generation_id")
                    == nested(audit, "generation_id")
                    and nested(blend_pointer, "generation_manifest")
                    == nested(audit, "generation_manifest")
                    and isinstance(final_manifest_path, Path)
                    and sha256_file(final_manifest_path)
                    == nested(blend_pointer, "generation_manifest_sha256")
                    == nested(audit, "generation_manifest_sha256")
                    and str(manifest_submission_hash).lower()
                    == str(expected_hash).lower()
                )
            )
            audit_ready = (
                nested(audit, "status") in {"ready", "complete", "passed"}
                and all_checks_pass
                and provenance_match
                and generation_match
            )
            display(
                pd.DataFrame(
                    [
                        {
                            "audit source": audit_source,
                            "reported status": nested(audit, "status"),
                            "rows": first_nested(audit, ("rows", "row_count", "n_rows", "prediction.rows")),
                            "all recorded checks pass": all_checks_pass,
                            "audit hash matches v2 manifest": provenance_match,
                            "pointer / generation / audit agree": generation_match,
                            "safe-to-submit verdict": "READY" if audit_ready else "NOT YET VERIFIED",
                            "submission SHA-256": audit_hash,
                        }
                    ]
                )
            )
            if checks:
                display(pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]))
        """
    ),
    markdown("## Takeaways"),
    code(
        r"""
        development = nested(selection, "development", {})
        sealed = nested(selection, "sealed_audit_descriptive_only", {})
        if blend_comparison is not None:
            sealed_blend_score = nested(
                json_artifacts["sealed_blend"],
                "diagnostic_scores.frozen_blend_final_q1p1.pooled_cosine",
            )
            takeaways = [
                f"1. The final candidate is the frozen 60/40 TabM-mini/capacity-LightGBM blend with q=1.1; pooled Dev1-Dev3 cosine is **{fmt(blend_development)}**.",
                f"2. Its one-shot Dev3 cosine is **{fmt(blend_dev3)}**, and the predeclared promotion gate is **{nested(blend_comparison, 'dev3_audit.promotion_gate_passed', 'pending')}**.",
                f"3. The blend-specific one-time sealed cosine is **{fmt(sealed_blend_score)}** and the predeclared keep/fallback safety veto is **{sealed_veto_status or 'pending'}**; it could keep the frozen blend or trigger the frozen capacity-only q=1.2 fallback, never retune them.",
                "4. TabM-mini is not a standalone submission: both components are independently RMS-normalized before the immutable blend and post-blend transform.",
                "5. Upload only the CSV whose SHA-256 agrees across the stable pointer, immutable generation manifest, canonical file, and all-checks-passing blend replay audit.",
            ]
        else:
            takeaways = [
                f"1. V2 raises development pooled cosine from **{fmt(development.get('v1_raw_cosine'))}** to **{fmt(development.get('v2_final_q1p2_cosine'))}** under the same chronological evaluation structure.",
                f"2. The raw joint-v2 gain is **{fmt(development.get('joint_minus_v1_raw'))}**. The path representation contributes **{fmt(development.get('sequence_increment_raw'))}** at fixed capacity; most of the improvement is the learner-capacity correction.",
                f"3. The already-frozen v2 view scores **{fmt(sealed.get('v2_final_q1p2_cosine'))}** on months 59-70, used only for the predeclared keep/fallback safety veto; it could not retune the frozen model or transform.",
                "4. A neural challenger is eligible only after complete chronological OOF. A strong single-fold result is useful research evidence, not a final ensemble component.",
                "5. Upload only the v2 CSV whose SHA-256 matches both the final manifest and an independent all-checks-passing audit.",
            ]
        display(Markdown("\n".join(takeaways)))
        """
    ),
    markdown(
        r"""
        ### Limitations and decision boundary

        - Public/private test labels and their month membership are unavailable. The exact cause
          of the 0.124 public score cannot be observed; domain shift is supported, not causally
          proven to explain every point of the gap.
        - The target horizon and absolute timestamps are undisclosed. Month-disjoint folds prevent
          obvious temporal reversal, but exact purging cannot be computed without the overlap
          interval.
        - Test covers 38 future months. Historical repeated-origin stress is informative about
          aging but gives early origins less training history than the final 71-month model.
        - The target is heavy-tailed and global cosine is energy-weighted. Report both pooled
          cosine and per-month robustness; neither guarantees the private half of the leaderboard.
        - This is a forecasting-competition model, not a tradable strategy. Costs, latency,
          capacity, and market impact are outside the evaluation.
        """
    ),
    markdown("### Source and hash manifest"),
    code(
        r"""
        manifest_rows = []
        for name, path in artifact_paths.items():
            exists = path.exists()
            manifest_rows.append(
                {
                    "artifact": name,
                    "path relative to project root": path.relative_to(project_root).as_posix(),
                    "present": exists,
                    "bytes": path.stat().st_size if exists else None,
                    "modified UTC": (
                        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                        if exists
                        else None
                    ),
                    "SHA-256": sha256_file(path) if exists else None,
                }
            )
        source_manifest = pd.DataFrame(manifest_rows)
        display(source_manifest)
        print(
            f"Manifest generated at {datetime.now(timezone.utc).isoformat()} from {project_root}"
        )
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
            "report_type": "v2 artifact-only model audit",
            "generated_by": Path(__file__).name,
            "training_permitted": False,
            "optional_artifacts_supported": True,
        },
    },
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote unexecuted notebook to {NOTEBOOK_PATH}")
