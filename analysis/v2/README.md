# V2 neural challenger

This directory is isolated from the accepted v1 pipeline. It benchmarks two
predeclared neural challengers - a RealMLP-style regressor and TabM-mini - on the
same strict expanding-month folds and existing deterministic sample-level
feature caches. No file here is used by the current submission unless a later,
separately audited promotion step selects it.

The runtime is supplied through the workspace-local `.analysis_deps` directory
used by the v1 analysis. The v2 script prepends that dependency directory at
startup and remains device-aware, so it uses CUDA when the local Torch build
supports it.

The completed neural benchmark is documented in
`neural_challenger_benchmark.md` and the executed, read-only evidence notebook
`neural_challenger_experiment.ipynb`. The frozen blend replay and one-shot Dev3
audit are implemented separately in `audit_frozen_tabm_capacity_blend.py` so
the hash-frozen core TabM challenger is not changed after selection.
