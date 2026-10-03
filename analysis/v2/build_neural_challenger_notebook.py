"""Build and execute the compact v2 neural-challenger experiment notebook."""

from __future__ import annotations

import os
from pathlib import Path
import textwrap

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "v2" / "neural_challenger_experiment.ipynb"
JUPYTER_STATE_DIR = PROJECT_ROOT / "artifacts" / "v2" / "jupyter_runtime"


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def main() -> None:
    JUPYTER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    for variable, child in {
        "IPYTHONDIR": "ipython",
        "JUPYTER_CONFIG_DIR": "config",
        "JUPYTER_DATA_DIR": "data",
        "JUPYTER_RUNTIME_DIR": "runtime",
    }.items():
        directory = JUPYTER_STATE_DIR / child
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
            # V2 TabM-mini challenger: frozen chronological audit

            **Answer first.** The challenger is viable, and the pre-frozen
            60% TabM / 40% capacity-LightGBM blend passed the one-shot Dev3
            gate. This notebook is a read-only, executed companion to the
            machine-readable artifacts. It does not train, tune, or write a
            submission.

            Dev1 and Dev2 select the blend. Dev3 confirms it once. Months
            59-70 remain outside this notebook.
            """
        ),
        code(
            """
            from pathlib import Path
            import hashlib
            import json
            import numpy as np
            import pandas as pd
            from IPython.display import display

            root = Path.cwd()
            tabm_dir = root / 'artifacts' / 'v2' / 'tabm_mini'
            diag_dir = root / 'artifacts' / 'v2' / 'diagnostics'

            def load_json(path):
                with path.open('r', encoding='utf-8') as handle:
                    return json.load(handle)

            def sha256_file(path):
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                return digest.hexdigest().upper()

            summaries = {
                fold: load_json(tabm_dir / f'{fold.lower()}_summary.json')
                for fold in ('Dev1', 'Dev2', 'Dev3')
            }
            audit = load_json(diag_dir / 'tabm_capacity_frozen_dev3_audit.json')
            pooled = load_json(diag_dir / 'tabm_capacity_pooled_report.json')
            comparison = load_json(diag_dir / 'tabm_capacity_blend_comparison.json')
            grid = pd.read_csv(diag_dir / 'tabm_capacity_screen_grid.csv')
            monthly = pd.read_csv(diag_dir / 'tabm_capacity_monthly_scores.csv')
            """
        ),
        markdown("## Standalone TabM outer-fold results"),
        code(
            """
            fold_table = pd.DataFrame([
                {
                    'fold': fold,
                    'outer train': f"{summary['outer_train_months'][0]}-{summary['outer_train_months'][1]}",
                    'outer validation': f"{summary['outer_validation_months'][0]}-{summary['outer_validation_months'][1]}",
                    'selected epoch': summary['best_epoch'],
                    'inner cosine': summary['best_inner_ensemble_cosine'],
                    'outer raw cosine': summary['outer_ensemble_cosine_raw'],
                    'member mean cosine': summary['outer_member_cosine_mean'],
                    'member pair correlation': summary['outer_member_mean_pairwise_correlation'],
                    'elapsed seconds': sum(summary[key] for key in (
                        'selection_transform_seconds', 'selection_training_seconds',
                        'refit_transform_seconds', 'refit_training_seconds')),
                }
                for fold, summary in summaries.items()
            ])
            display(fold_table)
            """
        ),
        markdown(
            """
            Every outer score above is produced after discarding the inner
            checkpoint and refitting from scratch on the full outer-training
            window. Memberwise MSE is used during training; member predictions
            are averaged only at inference.
            """
        ),
        markdown("## Dev1+2 selection screen"),
        code(
            """
            display(grid.head(10))
            selected = grid.loc[grid['selected']].iloc[0]
            assert selected['tabm_weight'] == 0.6
            assert selected['power_exponent'] == 1.1
            assert abs(selected['screen_cosine'] - 0.14445559865492444) < 1e-14
            """
        ),
        markdown("## Frozen Dev3 audit and pooled reporting"),
        code(
            """
            result_table = pd.DataFrame([
                {'view': 'Dev1+2 screen', 'cosine': pooled['screen_dev1_dev2_cosine'],
                 'role': 'selection'},
                {'view': 'Dev3', 'cosine': audit['dev3_cosine'],
                 'role': 'one-shot confirmation'},
                {'view': 'Dev1+2+Dev3 pooled',
                 'cosine': pooled['pooled_dev1_dev2_dev3_cosine'],
                 'role': 'reporting only'},
            ])
            display(result_table)
            display(pd.DataFrame([{
                'status': audit['status'],
                'positive-month share': audit['dev3_positive_month_share'],
                'gate passed': audit['promotion_gate']['passed'],
                'final TabM epoch': pooled['final_tabm_refit_epoch'],
                'frozen config SHA-256': audit['frozen_config_sha256'],
                'prediction SHA-256': audit['prediction_artifact_sha256'],
            }]))
            """
        ),
        markdown("## Monthly robustness diagnostic"),
        code(
            """
            monthly_summary = monthly.groupby('split', sort=False).agg(
                months=('month', 'count'),
                median_cosine=('cosine', 'median'),
                worst_cosine=('cosine', 'min'),
                positive_month_share=('positive_cosine', 'mean'),
            ).reset_index()
            display(monthly_summary)
            display(monthly[['split', 'month', 'n_rows', 'cosine']])
            """
        ),
        markdown("## Integrity replay"),
        code(
            """
            prediction_path = diag_dir / 'tabm_capacity_frozen_dev3_predictions.npz'
            assert audit['status'] == 'complete_one_shot_dev3_audit'
            assert audit['selection_hash_checks_all_passed']
            assert audit['promotion_gate']['passed']
            assert audit['dev3_positive_month_share'] == 1.0
            assert sha256_file(prediction_path) == audit['prediction_artifact_sha256']

            with np.load(prediction_path, allow_pickle=False) as artifact:
                assert {'row_indices', 'months', 'target', 'prediction'} <= set(artifact.files)
                target = artifact['target'].astype(np.float64)
                prediction = artifact['prediction'].astype(np.float64)
                replay_cosine = float(
                    np.dot(target, prediction)
                    / (np.linalg.norm(target) * np.linalg.norm(prediction))
                )
                assert np.isfinite(prediction).all()
                assert abs(np.sqrt(np.mean(prediction ** 2)) - 1.0) < 1e-12
                assert abs(replay_cosine - audit['dev3_cosine']) < 1e-14

            pd.DataFrame([{
                'check': 'prediction artifact hash, schema, RMS, and cosine replay',
                'passed': True,
                'replayed Dev3 cosine': replay_cosine,
            }])
            """
        ),
        markdown(
            """
            ## Conclusion

            The complementary neural component is real: standalone TabM beat
            capacity LightGBM on the untouched Dev3 block, and the frozen blend
            improved further to 0.1469999469 while retaining positive cosine in
            all Dev3 months. The deployable recipe is therefore the frozen blend
            with a six-epoch full-data TabM refit. Sealed-audit behavior remains
            a separate safety-veto question and is not inspected here.
            """
        ),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT_PATH)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(notebook, OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
