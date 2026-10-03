# TabM-mini challenger protocol

Primary references:

- https://github.com/yandex-research/tabm
- https://arxiv.org/abs/2410.24210

For each sample `x` and member `j = 1..16`, the model first forms a distinct
representation before mixing features:

`u_j = a_j * x`

where each input scale vector `a_j` is initialized independently from a normal
distribution. All members then pass through the same two-layer, width-256 ReLU
MLP with dropout 0.1. Each member has its own linear prediction head:

`z_j = head_j(shared_backbone(u_j))`.

Training minimizes the individual-member objective

`L = (1 / (B*K)) * sum_i sum_j (z_ij - y_i)^2`,

not the loss of the averaged prediction. Inference uses

`prediction_i = (1/K) * sum_j z_ij`.

This distinction is required by the official TabM training guidance. The
optimizer is AdamW with learning rate 0.002 and weight decay 0.0003. CUDA runs
use automatic mixed precision.

Validation follows the same honest protocol as the RealMLP-style challenger:
the last three months inside each outer training window select the epoch count;
then preprocessing and the model are reinitialized and retrained from scratch
on all outer-training months for exactly that count. Outer validation is never
used for checkpoint selection. Prediction power and the normalized LightGBM
blend weight are chosen on Dev1 + Dev2, then audited on Dev3. Promotion requires
stable month behavior and either a direct cosine gain or a robust blend gain.
