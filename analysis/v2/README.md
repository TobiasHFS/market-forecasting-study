# V2 neural experiments

RealMLP-style and TabM-mini regressors using expanding month folds and the existing feature caches.

The scripts use CUDA when the installed PyTorch build supports it. Some also look for packages in the historical `.analysis_deps/` directory.

Results and model settings are described in `neural_challenger_benchmark.md`. `neural_challenger_experiment.ipynb` contains the analysis code. `audit_frozen_tabm_capacity_blend.py` replays the selected blend on the Dev3 period.

The evaluation periods were reused during development. See `../v3_audit/validation_audit.md` for the split history.
