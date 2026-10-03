# MSCapital forecasting: model audit and next experiments

## Technical summary

The newer model improves the historical scores, but both the sealed period and the later confirmation period were inspected repeatedly during development. The fitting paths show no direct target leak; the reused evaluation periods limit what their scores establish.

**Neural signal and increased tree capacity explain most improvement; blending adds a smaller gain.** Recomputed development cosine is 0.133238 for v1, 0.143780 for raw TabM and 0.145297 for the final v2 blend. The blend adds 0.001517 over the stronger standalone model. Its improvement over the tree survives removing an unusually influential late month. The final power transform has much weaker support.

The first complete GPU selection/refit pilot on the RTX 3060 Ti took 78 seconds. The comparisons below include the model variants and a 38-month deployment-age test.

## What is being predicted and measured

There are **1,257,637 labeled samples across months 0-70**, and **647,896 test samples across months 71-108**. Market bars provide about 600 seconds of history; raw order and trade flows provide about 60 seconds. The return-generation method, target horizon, absolute timestamps and instrument identities are unavailable. Saved targets were independently checked against the original label file, with exact matches in all 12 target-bearing archives.

The [official metric](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/overview) is uncentered cosine: `sum(prediction × target) / sqrt(sum(prediction²) × sum(target²))`. A positive global rescale has no effect; relative amplitudes do. Pooled cosine is the selection metric, while monthly scores diagnose stability. Monthly scores must not simply be averaged and presented as the official score.

The [leaderboard](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/leaderboard) confirms approximately 49% public and 51% private. Their chronological ordering is not disclosed: the private partition cannot be assumed to be the last 51% of months. The public score remains a useful external check. Its sample fraction alone neither makes it reliable for selecting tiny improvements nor makes it irrelevant; market dependence and partition composition matter.

## The main validation flaw is repeated selection on exposed periods

The training implementation does several important things correctly: chronological outer splits; training-only robust preprocessing; neural epoch selection using only the last three months inside the training window; discarding that inner fit and retraining on the complete outer training slice. Using all labeled months for the final test refit is appropriate. Loading a full label array into memory is not itself leakage if only permitted slices enter fitting.

The independence claim fails at the project level. Months **59-70** were evaluated for v1, v2 slow trees, v2 capacity trees and the final neural blend. The saved tree summaries include six power-transform scores on that same period. Months **47-58** had already been used for earlier feature, capacity and calibration decisions. A new per-script freeze does not make these labels unseen again. A keep/fallback “safety veto” is also a bounded form of model selection.

Consequently, retain the scores but retire the descriptions “untouched Dev3” and “one-time sealed holdout.” The audit does not prove that any particular parameter was chosen by maximizing late-period scores; it proves that those periods were exposed. Confidence intervals below condition on already-chosen recipes and do not correct for the whole research search.

No precise overlap purge can be certified without timestamps, instrument IDs and target support. A one-month-gap sensitivity test is a conservative diagnostic, not proof of a correct purge and not a substitute for the missing metadata.

Evidence: [Source and holdout-provenance audit](<validation_audit.md>).

## Capacity and model diversity explain most of the gain

On the same 639,120 development rows, increasing tree capacity adds **0.005022** cosine, while adding the 280 bin features at fixed capacity adds **0.000753**. About 87% of the raw tree improvement came from capacity. The bins still contain ordered information: ten six-second slices, flattened into columns. They are coarsened temporal inputs, not a full raw-history neural model.

The following ledger separates architecture, input information and final calibration. These historical comparisons establish where the existing gain came from; they do not constitute independent model-selection trials. Relative to raw TabM, the final blend adds **0.001517**, with a conditional three-month-block interval **[0.000728, 0.002366]**. That is useful complementarity, but most of the gain over v1 was already present in TabM alone.

Evidence: [Recomputed aligned validation predictions](<../../artifacts/v3_audit/retrospective_metrics.json>).

| Model | Cosine | Rows |
| --- | --- | --- |
| V1 slow LightGBM | 0.133238 | 639120 |
| Capacity tree, base inputs | 0.13826 | 639120 |
| Capacity tree, base + bins | 0.139013 | 639120 |
| Original TabM, base inputs | 0.14378 | 639120 |
| Original 60/40 linear blend | 0.145137 | 639120 |
| Submitted blend, power 1.1 | 0.145297 | 639120 |

## The blend gain is broader than one favorable month

The chart compares monthly v1 and submitted-blend scores. It reveals the variation hidden by a pooled headline; it is not a forecast of test-month performance. For late months 59-70, the blend scores **0.155564**. Removing month 66 reduces it to **0.143779**. That month holds about a quarter of the late block’s target squared norm, so the higher late headline is a poor universal expectation.

Even excluding month 66, the blend improves over capacity q=1.2 by **0.004594**, with a conditional three-month-block interval of approximately **[0.001528, 0.006969]**. The blending gain has positive intervals across the tested 1-, 3- and 6-month block lengths. By contrast, the tiny power-1.1 improvement over the linear blend has intervals spanning zero. Block resampling respects month-level dependence better than independent-row intervals, but cannot remove model-selection bias or represent unseen regimes.

Evidence: [Recomputed aligned validation predictions](<../../artifacts/v3_audit/retrospective_metrics.json>).

## Four feature defects are reproducible, but score impact is unproven

1. **Invalid OFI predecessor:** a zero or crossed quote can become the previous book state. A valid → zero → unchanged-valid synthetic sequence produces false normalized imbalance +0.5 instead of zero.
2. **Missing reference encoded as zero:** undefined spread-normalized displacements become ordinary zeroes. A complete cached-row census found 5,627 / 1,257,637 training samples (**0.447%**) and 4,538 / 647,896 test samples (**0.700%**) with invalid terminal references. The corresponding populated flow bins all have zero VWAP displacement.
3. **Incorrect event-time interpretation:** “contemporaneous” mid-price is a bin average, sometimes using a quote later than the historical event or a terminal-mid fallback. This is look-ahead within supplied history, not access to data after the prediction timestamp. It may still encode valid subsequent price response, but it is not an as-of execution reference.
4. **Mismatched denominators:** invalid price/reference rows can be omitted from numerators while their volume remains in denominators, diluting averages.

New isolated NumPy helpers fix predecessor validity, missing references, backward as-of joins and matching-mask averages. **All 16 tests passed**, including future-quote invariance and sample isolation. These are reference implementations: they have not been integrated into large-scale extraction or used to claim a new submission score. Frozen v2 artifacts remain intact.

Evidence: [Feature-source audit and synthetic reproductions](<feature_audit.md>).

## Undefined references affect under 1% of samples, more often in test

The full-cache census independently reproduces the earlier 50,000-row probes and records all source hashes. Affected populated order/trade bins number **49,889 / 35,950 in training** and **41,931 / 32,914 in test**. This establishes frequency for the undefined-reference issue, not its predictive impact. The models also receive other reference and missingness information, so a malformed zero is not proof that the entire sample is unusable.

This census does not count raw invalid OFI transitions or every numerator/denominator mismatch; those require scanning the underlying event histories. Feature correctness is a justified engineering priority, but these relatively uncommon references should not be presented as the demonstrated main cause of the leaderboard gap.

Evidence: [Full-cache undefined-reference census](<../../artifacts/v3_audit/full_cache_reference_census.json>).

## Distribution shift is real; its effect on alpha is not identified

Existing diagnostics distinguish training from test with AUC **0.820**, and recent training from test with AUC **0.748**. Order-event density rises about **36.4%** and transaction density about **32.4%** in test. Yet early versus late training is even more distinguishable, at AUC **0.857**. A domain classifier demonstrates covariate differences; it does not prove target-concept drift or establish that every differing feature is harmful.

Prefer clock-time windows, relative price/depth quantities and explicit missingness. Keep useful activity context, but test its contribution. Removing every absolute-scale feature or weighting by an adversarial classifier without controlled evidence could discard signal.

Evidence: [Existing domain-shift diagnostics](<../../artifacts/diagnostics/postmortem/domain_shift_summary.json>).

## The full incumbent remains useful through one 38-month deployment

The original recipe was refitted once on months 0-20 and predicted months 21-58 without refresh. The submitted blend recipe scored **0.140529**, versus **0.135804** for the raw capacity tree and **0.137565** for raw TabM. The linear blend scored **0.140717**, slightly above power 1.1.

Complete-vector component scales were fixed before all age slices. The final 12 months score **0.138939** for the submitted recipe. This closes the earlier omission of the neural component from long-horizon diagnostics. It is one origin, with only 21 training months, and a new GPU training realization - not the original CPU weights and not an estimate of the 71-month-trained private-test score.

Evidence: [Incumbent recipe: one fit, 38 forecast months](<../../artifacts/v3/incumbent_age38/summary.json>).

## Controlled local comparisons

All 12 predeclared candidate/fold runs completed. The strongest new standalone candidate is **tabm_path**, with pooled cosine **0.141572**. The archived submitted blend is **0.145297** on those same rows.

Base TabM, full-input TabM and the temporal encoder use the same new seed, inner chronological epoch selection and outer refits. The outer fits are **0-22 → 23-34**, **0-34 → 35-46**, and **0-46 → 47-58**. TabM has 16 members, two shared width-256 layers and member-wise MSE; the temporal encoder has width-32 convolutions with dilations 1/2/4 plus a width-128 base-feature branch. The temporal encoder sees exactly the same ten-by-28 bins as full-input TabM plus the base branch. All appended missingness indicators enter its base branch. Its capacity and ensemble structure differ from TabM, so this compares complete training recipes at equal input information, not an isolated causal effect of convolution. A weak result would reject this small architecture/training recipe, not all temporal forecasting models. The native tree changes the whole missingness/tail-preprocessing package; it does not isolate clipping alone. One illustrative 20% challenger / 80% incumbent blend was specified before these results; its weights were not searched. The exact recipe is `0.8 × unit_RMS(incumbent) + 0.2 × unit_RMS(challenger)`, with each RMS calculated over the complete pooled evaluation vector and no centering. A single seed and repeatedly exposed folds limit claims about tiny differences. No candidate is automatically promoted.

Evidence: [Fixed GPU and native-tree comparisons](<../../artifacts/v3/bounded_experiments/comparison.json>).

## Separate input information, architecture and preprocessing effects

**Full-input TabM minus base TabM:** +0.001594 (conditional three-month-block interval [-0.000449, +0.003964]).

**Temporal encoder minus full-input TabM:** -0.009497 (conditional three-month-block interval [-0.011768, -0.007474]).

**Native tree minus the archived capacity tree:** -0.000287 (conditional three-month-block interval [-0.000870, +0.000223]).

These paired comparisons, rather than the highest isolated score, determine the next priority. Intervals describe variation across the observed months conditional on these fitted predictions. They do not include repeated research selection, new-seed uncertainty or absent market regimes.

Evidence: [Fixed GPU and native-tree comparisons](<../../artifacts/v3/bounded_experiments/comparison.json>).

| Candidate | Standalone cosine | 20% blend cosine | Correlation with incumbent |
| --- | --- | --- | --- |
| temporal_path | 0.132075 | 0.145208 | 0.8978 |
| native_tree | 0.138725 | 0.145071 | 0.9531 |
| tabm_base | 0.139978 | 0.145596 | 0.9417 |
| tabm_path | 0.141572 | 0.146104 | 0.9336 |

## The promising result is a small full-input TabM ensemble contribution

The tested temporal encoder and native-tree preprocessing package do not justify changing the incumbent. Full-input TabM improves over the new base-input TabM in each fold, but their pooled three-month-block interval still spans zero. The new full-input standalone score, 0.141572, also remains below the original raw TabM score, 0.143780. A different seed and inner epoch selection prevent attributing that historical difference solely to the added inputs.

The useful signal is complementarity: adding 20% full-input TabM to 80% of the incumbent raises retrospective pooled cosine from **0.145297 to 0.146104**, a gain of +0.000807 (conditional three-month-block interval [+0.000449, +0.001378]). Its paired interval is positive at all three tested block lengths. The equivalent base-input addition reaches only 0.145596, with a three-month interval spanning zero. This motivates the second-seed check; it does not establish a new private-leaderboard score.

Inner-selected epochs vary substantially: 9/16/4 for base TabM and 7/10/4 for full-input TabM. Three inner months can make stopping noisy. A bounded seed ensemble and a conservatively fixed training schedule deserve priority over a broad epoch or architecture search.

Evidence: [Fixed GPU and native-tree comparisons](<../../artifacts/v3/bounded_experiments/comparison.json>).

## The highest-value frontier is disciplined use of microstructure

**Keep modern tabular models as the anchor.** [TabM](https://arxiv.org/abs/2410.24210) is a credible efficient neural ensemble; [RealMLP](https://arxiv.org/abs/2407.04491) is a useful later diversity candidate. Benchmark results elsewhere do not imply that a larger neural model wins here.

**Prioritize stationary flow and correctly aligned event mechanics.** [Cont, Kukanov and Stoikov](https://arxiv.org/abs/1011.6402) motivate depth-scaled imbalance, largely from contemporaneous price-impact evidence. [Kolm, Turiel and Westray](https://doi.org/10.1111/mafi.12413) provide more direct forecasting evidence for stationary order-flow inputs. Our two-level bars and undisclosed target differ from their granular market data, so their gains cannot be transferred numerically.

**Let richer temporal information earn its compute.** If corrected cached-bin models warrant another round, extend to two compact streams: roughly 100 bins over 600 seconds of market history and 30-60 bins over 60 seconds of order/trade flow, plus age and validity masks. A width-32/64 TCN or GRU plus a small tabular branch is a sensible first test. [DeepLOB](https://arxiv.org/abs/1808.03668) supports learning temporal/book structure; copying its deeper, multi-level setup is not a faithful match here. The [TCN paper](https://arxiv.org/abs/1803.01271) supports an efficient architecture, not financial alpha.

The [official data restrictions](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/data) prohibit external data and models. These proposals train from random initialization on competition data. Pretrained forecasting models and external market datasets are excluded.

## Cosine rewards a conditional mean, not automatic risk normalization

Under a fixed population distribution with finite second moments, the optimal prediction direction is proportional to **E[target | features]**. This follows by conditioning the numerator and applying Cauchy-Schwarz. Ordinary MSE is therefore a principled baseline. Ranking predictions, centering them, using inverse-variance position sizing or optimizing noisy minibatch cosine can move away from the competition objective.

If training a volatility-normalized target, restore its known feature-based scale at inference. A later amplitude experiment should learn a strongly shrunk correction from earlier out-of-fold predictions and apply it to later months. Do not fit reliability weights on the same labels being evaluated. Given the weak power-transform evidence, this is lower priority than feature correctness and robust model diversity.

## Local compute is adequate; the previous runtime was the bottleneck

The machine has an **RTX 3060 Ti with 8 GB VRAM**, approximately **32 GB RAM**, and **8 cores / 16 threads**. The old PyTorch build was CPU-only. A separate workspace environment now runs official **PyTorch 2.11.0 + CUDA 12.8**, with real GPU arithmetic and training verified. The first base-TabM selection/refit pilot completed in **78 seconds**; the incumbent age diagnostic took **133 seconds** including tree training. These are observed workflow times, not controlled CPU/GPU speedup estimates because recipes and warm-up conditions differ.

The 754-column float32 training cache is about 3.79 GB before transformed copies and missing indicators. Run one training process at a time, stream batches where needed and retain memory-mapped caches. The 12-run comparison completed with all jobs successful; the serial queue took about **23.6 minutes**, plus the separately completed 78-second pilot. It enforced a shared six-hour ceiling. Data, model and source hashes are recorded; no existing submitted model or prediction file is overwritten.

Evidence: [Verified hardware, libraries and measured workflow times](<../../artifacts/v3/bounded_experiments/input_provenance.json>).

## Reproducibility and scope of the completed checks

All 12 new evaluation vectors were checked against the original labels and month/row identities, and their full cosine scores were recomputed. For each saved model, the first 2,048 evaluation rows were independently predicted from its checkpoint and preprocessor: all reproduced exactly. The frozen runner, modeling code, input arrays, schemas and raw-label hashes matched. The original submission hash also matched.

The isolated feature helpers passed 16 targeted tests. Their fixes are not yet connected to the production extractor. Consequently these experiments measure architecture, input-set and preprocessing differences on the existing cached features; they do not measure the benefit of corrected extraction.

Evidence: [Original-input hashes, label alignment and checkpoint replay](<../../artifacts/v3/bounded_experiments/verification.json>).

## What to do next, and what I need from you

1. **Use the measured comparison verdict**, with paired monthly gains and model complementarity; preserve the current submission unless an improvement is repeatable. Do not interpret a few extra decimal places as a reliable private-score gain.
2. **Integrate the corrected feature helpers into a versioned extractor**, quantify invalid-transition and denominator issues on the full data, and compare corrected versus original features on the same fixed folds. Keep retrospective response features clearly named if retained.
3. **Confirm useful candidates with a second seed or conservative fixed ensemble** before expanding architecture or calibration searches. Extend to richer 600/60-second tensors only when the controlled evidence justifies it.
4. **Maintain a project-wide exposure ledger.** All already-inspected validation months remain exposed. Any genuinely new labeled block would be valuable; a newly named subset of old data is not new evidence.

**No data re-upload is needed for the completed audit and comparisons.** The valuable missing information is instrument identity, absolute prediction timestamps and exact target support. If organizers make those available, they enable precise overlap purging and deduplication. Without them, residual boundary-overlap uncertainty must remain explicit. The private labels are unavailable by design; this work identifies the best-supported directions, not a guaranteed winning score.
