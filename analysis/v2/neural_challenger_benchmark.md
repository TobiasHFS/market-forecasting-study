# V2 neural challenger benchmark

## Decision

Promote the pre-frozen blend of **60% TabM-mini and 40% capacity LightGBM**, not
standalone TabM.  Each complete component vector is divided by its uncentered
RMS, the components are blended without centering, signed power `q = 1.1` is
applied, and the complete result is divided by its uncentered RMS.  The Dev1+2
screen selected this recipe before Dev3 labels were used.  The one-shot Dev3
audit passed its predeclared gate.

The final TabM refit uses **6 epochs**, the median of the independently selected
Dev1/Dev2/Dev3 epoch counts `[7, 6, 6]`.

## Model and loss

The challenger is the official TabM MiniEnsemble pattern:

- 16 members;
- independently initialized member input scales;
- a shared two-layer, width-256 ReLU backbone;
- dropout 0.1 and member-specific linear heads;
- AdamW, learning rate 0.002, weight decay 0.0003;
- batch size 1024;
- robust training-fold preprocessing with missing indicators and smooth clip;
- standardized target during training.

For member prediction `z_ij`, target `y_i`, batch size `B`, and `K=16`, training
minimizes the mean individual-member loss

`L = (1/(B*K)) sum_i sum_j (z_ij - y_i)^2`.

This is intentionally not MSE of the mean prediction.  At inference only, the
prediction is `(1/K) sum_j z_ij` and is transformed back to target units.

## Honest chronology

For each outer fold, only the last three months inside its training window choose
the epoch count.  The checkpoint and preprocessor are then discarded.  A fresh
model and fresh training-fold preprocessor are fit from scratch on every outer
training month for exactly the selected count before producing outer predictions.

| Fold | Outer train | Inner select | Outer validation | Selected epoch | Raw TabM cosine | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Dev1 | 0-22 | 20-22 | 23-34 | 7 | 0.1456654904 | 1,061 s |
| Dev2 | 0-34 | 32-34 | 35-46 | 6 | 0.1391773578 | 1,110 s |
| Dev3 | 0-46 | 44-46 | 47-58 | 6 | 0.1460407145 | 1,830 s |

The Dev3 run was slower because another v2 process shared the CPU during its
scratch refit.  TabM itself remained stable at about 6.1 GiB working memory.

## Frozen screen and one-shot audit

| Evaluation | Capacity LightGBM | TabM-mini | Frozen 60/40 blend |
|---|---:|---:|---:|
| Dev1+2 screen, months 23-46 | 0.1393744789 | 0.1423835231 | **0.1444555987** |
| Dev3 audit, months 47-58 | 0.1399698558 raw / 0.1402706007 at q=1.1 | 0.1460407145 | **0.1469999469** |

The frozen blend had positive cosine in all 12 Dev3 months.  Its reporting-only
pooled cosine over months 23-58, with each component normalized once over that
whole pooled vector, was **0.1452968982**.  The full grid was evaluated only on
Dev1+2.  No Dev3 grid or retuning was performed.

## Reproduction commands

Run from the workspace root in PowerShell with your virtual environment activated:

```powershell
$env:PYTHONPATH='.analysis_deps'
$python = 'python'

& $python analysis\v2\smoke_neural_architectures.py
& $python analysis\v2\run_tabm_mini_challenger.py --folds Dev1 --max-epochs 8 --min-epochs 4 --patience 4 --batch-size 1024 --hidden-size 256 --overwrite
& $python analysis\v2\run_tabm_mini_challenger.py --folds Dev2 --max-epochs 8 --min-epochs 4 --patience 4 --batch-size 1024 --hidden-size 256 --overwrite
& $python analysis\v2\run_tabm_mini_challenger.py --folds Dev3 --max-epochs 8 --min-epochs 4 --patience 4 --batch-size 1024 --hidden-size 256 --overwrite
& $python analysis\v2\run_tabm_mini_challenger.py --folds Dev1 Dev2 Dev3 --max-epochs 8 --min-epochs 4 --patience 4 --batch-size 1024 --hidden-size 256
& $python analysis\v2\audit_frozen_tabm_capacity_blend.py
```

The tested runtime was official CPU Torch 2.13.0 on a 16-thread machine with
32 GiB RAM.  CUDA was unavailable to this Torch build, so automatic mixed
precision remained off.

## Audit artifacts

- `artifacts/v2/diagnostics/frozen_blend_before_dev3.json` is the immutable
  pre-Dev3 contract.
- `artifacts/v2/diagnostics/tabm_capacity_screen_grid.csv` contains only the
  Dev1+2 selection grid.
- `artifacts/v2/diagnostics/tabm_capacity_frozen_dev3_audit.json` is the
  one-candidate audit and records all frozen-selection hash checks as passing.
- `artifacts/v2/diagnostics/tabm_capacity_frozen_dev3_predictions.npz` contains
  the aligned Dev3 prediction vector.  SHA-256:
  `AC6B03EB8B95BADF1B4767C5C9CD8779BD2BB9969607F847F914DA767C899632`.
- `artifacts/v2/diagnostics/tabm_capacity_pooled_report.json` records the
  reporting-only pooled score and epoch rule.
- `artifacts/v2/diagnostics/tabm_capacity_blend_comparison.json` is the compact
  manifest with source and output hashes.

The core challenger source retained its frozen SHA-256
`332F253714CB2179FDB570D697C2329ADA07078816B32A4AC91AFD62793DA542`.

