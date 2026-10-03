# RealMLP-style challenger protocol

## Decision

Benchmark one transparent, predeclared neural family against the accepted
LightGBM OOF vector. This is a diversity test, not a broad neural architecture
search. The design follows the official standalone RealMLP-TD-S implementation
and NeurIPS 2024 paper:

- https://github.com/dholzmueller/realmlp-td-s_standalone
- https://arxiv.org/abs/2407.04491

## Mathematical construction

For fold-fitted, smoothly clipped features `x`, the model is

`h0 = s * x`

`h(l+1) = Mish((h(l) W(l))/sqrt(d(l)) + b(l)), l = 0,1,2`

`z = (h3 W3)/sqrt(256) + b3`

`prediction = target_mean_fit + target_std_fit * z`.

The output layer starts at zero. Adam uses beta `(0.9, 0.95)` and separate
learning-rate multipliers of `6`, `1`, and `0.1` for input scales, weights, and
biases. Training minimizes standardized-target MSE; epoch selection maximizes
cosine on a strictly earlier inner validation window. A four-cycle logarithmic
cosine schedule is used, matching the standalone recipe's schedule shape.

## Leakage controls

Each outer development fold reserves its final three training months for inner
epoch selection. The selected checkpoint itself is discarded: preprocessing
and the network are reinitialized, then refitted from scratch on **all** outer
training months for exactly the selected epoch count. The outer validation
months are touched exactly once after that refit. Feature imputation,
transformations, centers, scales, missing-indicator selection, and smooth
clipping are fitted or applied without outer-validation labels. `sample_id`,
`month`, and `target` are never model inputs.

The principal score is the cosine over the concatenated OOF vector, matching
the competition metric. Per-fold and per-month results are robustness
diagnostics. Prediction powers and LightGBM blend weights are selected on Dev1
+ Dev2; Dev3 is reported as a model-family audit rather than reused for that
choice. The sealed months 59--70 are not read.

## Scale adaptation and caveat

The original default uses 256 epochs and batch size 256 on datasets up to about
500k rows. This dataset has 1.26M labeled rows. The challenger therefore uses a
larger batch (2048), a 64-epoch cap, mixed precision on CUDA, and an honest
inner chronological holdout. It should be called **RealMLP-style**, not an exact
reproduction of RealMLP-TD-S. Promotion requires a stable OOF gain or useful
low-correlation blend gain across all three development folds.
