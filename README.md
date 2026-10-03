# Market Forecasting Study

Experiments for the [MSCapital financial forecasting competition](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting), using market bars, orders and trades to predict price movements.

The work starts with linear and tree models in `analysis/`, adds neural models in `analysis/v2/`, and compares sequence features and model variants in `analysis/v3/`. Evaluation notes are in `analysis/v3_audit/`.

Validation uses expanding month splits. The split called `sealed_audit` was reused during development, so its scores should be read as historical comparisons.

## Setup

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.lock.txt
python analysis/v3/test_sequence_kernels.py
```

Download the competition data from Kaggle into `ms-capital-real-financial-market-forecasting/` at the repository root. Feature caches, models and reports are generated locally under `artifacts/`. The neural experiments also need a PyTorch build suited to your hardware.

Some scripts load joblib files. Only use model files you created yourself or otherwise trust.

[MIT license](LICENSE).
