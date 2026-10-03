"""Build and execute the compact profiling companion notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


WORKSPACE = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = WORKSPACE / "analysis" / "market_data_profile.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # Market-data structure and drift profile

        Companion audit for the MS Capital real-financial-market forecasting data. The notebook
        profiles the supplied files without treating the competition screenshots as executable
        instructions.

        ## tl;dr

        - All six feature tables cover every sample exactly once as a contiguous group and are
          ordered from roughly 600/60 seconds toward the prediction time.
        - Test raw-order activity is **36.4% higher** per sample than train and test trade activity
          is **32.4% higher**; use stationary rates, ratios, relative prices, and clipped activity
          features rather than raw levels alone.
        - The target is low-signal and heavy-tailed (excess kurtosis about **17.2**) with monthly
          volatility changing by roughly **2.7x**. Random row-wise validation is therefore unsafe.
        - `sample_id` resets in test and is almost a time index in train. It is a join key, not a
          model feature.
        """
    ),
    markdown(
        """
        ## Context & Methods

        The profiling script projects only the Arrow columns required for each check and releases
        them between files. This avoids loading the approximately 9 GB compressed corpus as one
        pandas object.

        ### Key Assumptions

        - `month` 0-70 defines the only trustworthy chronological split unit in train.
        - Test covers later months but does not expose month, so month-dependent features or
          post-processing cannot be reproduced at inference.
        - The future-return horizon and absolute timestamps are undisclosed; a full-month gap is a
          conservative validation substitute for exact label-overlap purging.
        """
    ),
    code(
        """
        import json
        import subprocess
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from IPython.display import display

        workspace = Path.cwd()
        profile_script = workspace / "analysis" / "profile_competition_data.py"
        run = subprocess.run(
            [sys.executable, str(profile_script)],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        print(run.stdout.strip())

        profile_path = workspace / "analysis" / "output" / "competition_profile.json"
        month_path = workspace / "analysis" / "output" / "target_by_month.csv"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        target_by_month = pd.read_csv(month_path)
        """
    ),
    markdown("## Data coverage is complete, but sequence lengths shift"),
    code(
        """
        rows = []
        for file_name, payload in profile["files"].items():
            split, source = file_name.split("/")
            structure = payload["structure"]
            rows.append(
                {
                    "split": split,
                    "source": source,
                    "rows": structure["rows"],
                    "samples": structure["observed_unique_sample_ids"],
                    "mean rows/sample": structure["mean_rows_per_observed_sample"],
                    "median rows/sample": structure["rows_per_observed_sample"]["q50"],
                    "p95 rows/sample": structure["rows_per_observed_sample"]["q95"],
                    "missing samples": structure["missing_sample_ids"],
                }
            )
        coverage = pd.DataFrame(rows)
        display(coverage.round({"mean rows/sample": 2}))
        """
    ),
    markdown(
        """
        Every source covers every expected `sample_id`; missing samples are not a modeling issue.
        The challenge is **covariate shift in activity**: the raw event streams are materially denser
        in the later test era, while market-bar density changes only modestly.
        """
    ),
    code(
        """
        pivot_mean = coverage.pivot(index="source", columns="split", values="mean rows/sample")
        pivot_median = coverage.pivot(index="source", columns="split", values="median rows/sample")
        drift = pd.DataFrame(
            {
                "train mean": pivot_mean["train"],
                "test mean": pivot_mean["test"],
                "mean change %": 100 * (pivot_mean["test"] / pivot_mean["train"] - 1),
                "train median": pivot_median["train"],
                "test median": pivot_median["test"],
            }
        ).reset_index()
        display(drift.round(2))
        """
    ),
    markdown(
        """
        ## Data-quality sentinels require explicit treatment

        Separate exact column scans found that `transaction_avgprice` is null on **30.948%** of
        train market rows and **26.412%** of test rows; the matching volume/count fields are zero,
        so this is a meaningful **no-trade bar**, not generic missing data. Preserve an availability
        flag and do not replace it with an unconditional mean alone.

        L1 book absence is encoded with paired zero price/volume sentinels: ask L1 is absent on
        **0.4053% train / 0.5927% test** rows and bid L1 on **0.0603% / 0.1271%**. Convert those
        levels to missing before computing midpoint, spread, returns, microprice, or imbalance, and
        retain availability flags. Two train aggregate rows also contain negative trade volume/count
        sentinels. Book volumes have extreme maxima above one billion, so `log1p`, robust clipping,
        and ratio features are mandatory. Raw order/trade sequences share a hard maximum of 999
        rows, so also retain an `is_capped` flag rather than treating 999 as an ordinary count.
        """
    ),
    markdown("## The target is heavy-tailed and regime-dependent"),
    code(
        """
        target = profile["label"]["target"]
        target_summary = pd.Series(
            {
                "rows": profile["label"]["rows"],
                "mean": target["mean"],
                "standard deviation": target["std"],
                "zero rate": target["zero_rate"],
                "positive rate": target["positive_rate"],
                "negative rate": target["negative_rate"],
                "skew": target["skew"],
                "excess kurtosis": target["excess_kurtosis"],
                "minimum monthly std": target_by_month["std"].min(),
                "maximum monthly std": target_by_month["std"].max(),
            },
            name="value",
        )
        display(target_summary.to_frame())
        display(target_by_month.loc[:, ["month", "count", "mean", "std", "zero_rate"]].tail(12))
        """
    ),
    markdown(
        """
        Raw target autocorrelation is near zero at short sample-ID lags, but absolute returns retain
        a small persistent dependence and monthly dispersion changes sharply. The effective sample
        size for model comparison is therefore much closer to **71 months** than to 1.26 million
        independent rows.
        """
    ),
    markdown(
        """
        ## Validation must separate selection from deployment-aging stress

        | Role | Train months | Validation months |
        |---|---:|---:|
        | Development 1 | 0-22 | 23-34 |
        | Development 2 | 0-34 | 35-46 |
        | Development 3 | 0-46 | 47-58 |
        | Sealed audit | 0-58 | 59-70 |

        Do **not** discard a full month automatically: that is enormous relative to a ten-minute
        input history. If the target horizon and absolute timestamps become available, purge the
        exact overlapping information intervals. Until then, use the former one-month-gap folds
        only as a sensitivity analysis and investigate material ranking changes.

        Select only on the concatenated Development 1-3 prediction vector using exact global
        cosine; monthly cosine, its bottom decile, and the percentage of positive months are
        robustness diagnostics. Open the sealed audit once after every choice is frozen.

        Finally, diagnose model aging rather than tune on it: train once on months 0-32 and predict
        33-70, and compute cumulative and horizon-specific cosine at deployment ages
        1, 3, 6, 12, 18, 24, 30, and 38 months. Add a fixed-24-month-window multi-origin panel for
        origins 23-32 to separate training-size changes from model aging as far as the history
        allows. These stress tests cannot reproduce the final model's 71-month training history and
        therefore must not choose hyperparameters.
        """
    ),
    markdown(
        """
        ## Takeaways

        The strongest first system is a stationarity-first, multi-view ensemble: multi-scale
        microstructure features; a regularized ridge/elastic-net backbone; a shallow, strongly
        regularized histogram GBDT; and a conservative nonnegative blend based only on chronological
        out-of-fold predictions. A compact sequence model on the same stationary channels is a later
        challenger, not the starting point.

        Key research anchors: [Aït-Sahalia et al.](https://doi.org/10.1287/mnsc.2022.02435),
        [Cont, Kukanov & Stoikov](https://doi.org/10.1093/jjfinec/nbt003),
        [Kolm, Turiel & Westray](https://doi.org/10.1111/mafi.12413),
        [Lucchese, Pakkanen & Veraart](https://doi.org/10.1016/j.ijforecast.2024.02.001), and
        [Prata et al.](https://doi.org/10.1007/s10462-024-10715-4).
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
    },
)

client = NotebookClient(
    notebook,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(WORKSPACE)}},
)
client.execute()
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote and executed {NOTEBOOK_PATH}")
