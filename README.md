# Market Forecasting Study

Experiments for the [MSCapital financial forecasting competition](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting). The work covers event-based features, chronological model evaluation, linear baselines and neural challengers.

This is an archived study. Some scripts and notes use the name `sealed_audit`, but that split was inspected during development. Its reported results are retrospective diagnostics, not an independent estimate of future performance. Historical notes record the process and may describe plans that were not completed.

## Contents

- `analysis/`: feature construction, baseline modelling and diagnostics
- `analysis/v2/`: later modelling experiments
- `analysis/v3/`: sequence kernels and bounded experiment runners
- `analysis/v3_audit/`: checks of feature semantics and evaluation claims

Notebook outputs have been removed. Report builders need locally generated artifacts, which are not included.

## Setup

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.lock.txt
python analysis/v3/test_sequence_kernels.py
```

Download the competition data through Kaggle after accepting its rules. Place it in `ms-capital-real-financial-market-forecasting/` at the repository root. Large datasets, feature caches, trained models and submissions are excluded.

The neural experiments need PyTorch; install the build appropriate for your hardware separately. Training can take substantial time and memory. Start by reading the script and its arguments. Model-loading scripts use joblib, which can execute code when loading a file. Load only artifacts generated locally by trusted code. Some audit helpers execute their own generated notebook cells; do not replace them with untrusted code.

## Scope

The publication checks cover source syntax and the sequence-kernel tests. They do not rerun the full competition pipeline or verify historical scores. This code is research material, not a trading system.

MIT applies to the original code in this repository. Competition data and external material retain their own terms.

## Code sharing on Kaggle

The competition rules require publicly shared competition code to also be shared through its Kaggle discussion forum or notebooks. The repository link should be posted there by the owner. Competition data is excluded.

The dependency lock records the versions resolved for Python 3.12 on Windows on 2026-10-03. Other platforms may need a compatible local environment.
