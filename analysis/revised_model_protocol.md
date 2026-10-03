# Revised model-selection protocol

This note adjudicates the five methodological proposals supplied after the initial report. The
screenshots are treated as critique, not as instructions. The feature hierarchy below is selected
independently for this dataset.

## Decisions on the proposed changes

1. **Remove the automatic one-month embargo.** Accept. The primary folds use adjacent whole
   months. A purge is imposed only if the target horizon, absolute timestamps, duplicated source
   windows, or another data-generation fact demonstrates cross-boundary information overlap. The
   former one-month-gap folds remain a sensitivity test.
2. **Keep the 38-month simulation diagnostic.** Accept and strengthen. It never tunes the model.
   Use both a single 33-month-training/38-month-deployment path and a multi-origin deployment-age
   curve.
3. **Normalize ensemble members before blending.** Accept with an important refinement: use
   uncentered whole-vector RMS/L2 scale, not automatic standard-deviation normalization or z-scoring,
   because the competition metric is uncentered cosine.
4. **Use a hierarchical feature program.** Accept, but replace the suggested two-tier split with
   the robustness ladder below.
5. **Use pooled cosine as primary and monthly metrics as diagnostics.** Accept. Replace the raw
   worst month as a selection statistic with a bottom-decile month and worst 12-month block; a
   single worst month is too noisy.

## Primary chronological validation

| Role | Training months | Validation months | Use |
|---|---:|---:|---|
| Development 1 | 0-22 | 23-34 | model selection |
| Development 2 | 0-34 | 35-46 | model selection |
| Development 3 | 0-46 | 47-58 | model selection |
| Sealed audit | 0-58 | 59-70 | opened once after freezing |

Each validation block is predicted by one frozen model. Model and feature choices use only the
concatenated Development 1-3 predictions and labels. This is a rolling-origin design in the sense
of [Tashman](https://doi.org/10.1016/S0169-2070(00)00065-0). The sealed audit protects against
second-level overfitting of the selection procedure, a risk emphasized by
[Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html).

### Conditional purge rule

If prediction time `t_i`, target end `e_i`, and instrument identity become available, define each
sample's information interval from the earliest feature time through its target end and purge only
training intervals that intersect validation intervals for the same instrument. If the target is a
future return of length `H`, a conservative boundary distance is at most roughly `H + 600 seconds`;
the exact value depends on whether duplicated feature windows as well as overlapping labels matter.
Dependent-data gap methods are justified by the dependence mechanism, not by an arbitrary calendar
unit; see [Racine](https://doi.org/10.1016/S0304-4076(00)00030-0).

With the currently hidden horizon/timestamps, zero-gap validation cannot be certified as perfectly
purged. Compare it with the former one-month-gap folds as a sensitivity check. A ranking reversal is
a boundary-dependence warning, not permission to choose whichever design scores better.

## Deployment-age diagnostics

After the whole pipeline is frozen, run the exact long stress path:

- train once on months 0-32;
- predict months 33-70 without updating;
- report horizon-specific and cumulative cosine at ages 1, 3, 6, 12, 18, 24, 30, and 38 months.

Add a multi-origin panel with origins `o = 23..32`. To reduce training-size confounding, use a fixed
trailing 24-month fit window `o-23..o`, freeze the fit, and score months `o+1..o+38`. At each age
`h`, concatenate predictions across origins before computing cosine. Also report cumulative
`C(â‰¤h)`. These overlapping pseudo-deployments remain correlated and still mix age with historical
regime, so they diagnose decay; they do not estimate the final 71-month-trained model's test score.
Inspecting the whole path is supported by instability-aware forecast comparison work such as
[Giacomini and Rossi](https://doi.org/10.1002/jae.1177).

## Chosen feature hierarchy

### Rung 0  -  Integrity and observability controls (always present)

Valid-row count, coverage span, newest-observation age, valid-book fraction, no-trade/VWAP-missing
flags, zero-price/invalid/crossed-book flags, duplicate-timestamp fraction, and 999-row-cap flags.
These prevent missing structure from being interpreted as economic zero. `sample_id` remains banned.

### Rung 1  -  Invariant Microstructure Core (about 60-80 features)

Use physical-time lookbacks: market `10/60/600s`; raw order/trade `5/15/60s`. Include relative
spread, L1 and cumulative-L2 queue imbalance, microprice displacement in spread units, L2/L1 depth
ratio, relative L2 gap, log-mid returns, robust realized variation, depth-normalized L1/L2 OFI,
signed new/cancel order pressure, aggressive signed trade flow, VWAP-to-mid displacement, and
depth-normalized execution intensity. The core follows the depth-scaled OFI evidence of
[Cont, Kukanov and Stoikov](https://doi.org/10.1093/jjfinec/nbt003) and the stationary-input evidence
of [Kolm, Turiel and Westray](https://doi.org/10.1111/mafi.12413).

### Rung 2  -  Multiscale Temporal Geometry (about 100-150 additions; default candidate)

Expand market windows to `10/30/60/180/600s` and raw-flow windows to `2/5/15/30/60s`. Add disjoint
shell aggregates, adjacent fast-minus-slow contrasts, latest-minus-time-weighted-mean, robust time
slopes, MAD/IQR, positive/negative semivariance, fast/slow ratios, and sign persistence/reversal.
Use clock time rather than last-N-event windows because test order/trade row density is 36.4%/32.4%
higher than train.

### Rung 3  -  Liquidity-Conditioned Mechanics (about 40-70 additions; must earn promotion)

Predeclare a small set of depth-scaled order/trade pressure, as-of-book price distance in
half-spreads, at-touch versus deeper pressure, cross-stream agreement/disagreement, L1/L2 imbalance
slope, replenishment after flow bursts, and within-input price-response concordance. Give ridge a
small fixed list of explicit products; let the GBDT learn most interactions.

### Quarantined satellites  -  disabled by default

- **Path shape:** tail quantiles, skew/kurtosis, concentration, entropy, run length, burst and
  interarrival features.
- **Absolute scale:** raw/log level, depth, volume, count, intensity, and tick proxies.

Evaluate each satellite separately against the last accepted rung. They must improve every
development block because they are the most plausible instrument/regime encoders under the observed
test-density shift.

All nonnegative scales use `log1p`; unbounded signed values use a fold-fitted `asinh` scale;
bounded imbalances stay bounded. Median/MAD scaling and clipping are fitted on training months only.
No monthwise normalization or rank transform is allowed unless it can be reproduced on hidden test
months.

## Cosine-aware blend

For base member `j`, normalize its prediction direction without centering:

`s_j = sqrt(mean(p_j^2))`, and `z_j = p_j / max(s_j, eps)`.

Do not subtract the prediction mean by default. Normalize each member once over the whole vector
that will be blended: the concatenated inner OOF vector when fitting a weight, the complete outer
validation block when evaluating it, the full 38-month stress vector, and all 647,896 test rows at
submission time. This batch normalization is label-free and reproducible. Never normalize by month,
deployment horizon, or public/private leaderboard partition.

Within Ridge, unit-RMS normalize and equally average the accepted view models, then renormalize the
family aggregate to obtain `r`. Do the same within histogram GBDT to obtain `g`. The only fitted
meta-parameter in the initial system is then `p(w) = w*r + (1-w)*g`, with
`w in {0, 0.25, 0.5, 0.75, 1}`. Default to `w=0.5`; retain another value only if fully nested outer
OOF improves repeatedly beyond the one-standard-error tie region. This avoids a high-dimensional
stacker built from correlated feature/view variants. Simple combinations are a strong baseline
because finite-sample weight estimation can erase theoretical gains; see
[Smith and Wallis](https://doi.org/10.1111/j.1468-0084.2008.00541.x).

## Selection hierarchy and robustness metrics

Primary statistic: exact cosine on the concatenated Development 1-3 vector. Report but do not
average monthly cosine. In fact pooled cosine is a norm-weighted sum of monthly cosines:
`C_pool = sum_m (||p_m||*||y_m||/(||p||*||y||))*C_m`. Diagnostics are median monthly cosine,
bottom-decile monthly cosine, positive-month share, each 12-month fold cosine, prediction RMS/mean,
base-prediction correlation, and monthly contribution to the global numerator.

For every feature-family addition, use paired differences on identical validation rows. Promote only
when pooled delta is positive, at least two of three development blocks improve, median monthly delta
is nonnegative, and the simpler parent is outside the candidate's one-standard-error region under a
paired 3-month moving-block bootstrap. When candidates are statistically tied, keep the lower rung.
The sealed audit is opened once; if an extension fails, revert only to the predeclared parent rather
than mining the audit.

The public leaderboard is a final pipeline smoke test, never a development loop.
