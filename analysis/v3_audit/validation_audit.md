# Validation and leakage audit of v1 and v2

Audit date: 2026-09-07. This review reads the saved source, manifests, notebooks and prediction artifacts. It does not retrain models, change the submitted prediction file, or claim that previously examined labels can be made unseen again.

## Main conclusion

**I found no direct train/validation target leak in the reviewed fitting paths. I did find extensive reuse of the supposed sealed and confirmation periods, so the final historical scores are retrospective evidence, not an untouched estimate of generalization.** The recorded gains are plausible and largely consistent across chronological periods. They should not be discarded, but the confidence implied by phrases such as “one-time sealed audit” is too strong at the project level.

The current canonical submission is the 60% TabM-mini / 40% capacity-LightGBM blend with signed-power exponent 1.1, trained on all labeled months 0-70. SHA-256 of both `submission_final.csv` and `artifacts/v2/submissions/submission_final.csv` is `7d3b92fbe13813ada0638d540789b484b013636d920c6f8a8e51ae6a5c41cf1a`, matching `artifacts/v2/models/final_blend_pointer.json`. All eight current source-file hashes in the final generation manifest match the files on disk. That establishes reproducible model identity; it cannot establish that the validation data had never influenced earlier research decisions.

## What is implemented correctly

1. **Chronological outer folds.** `analysis/modeling.py:71` defines training 0-22 / validation 23-34, training 0-34 / validation 35-46, training 0-46 / validation 47-58. The additional historical block trains 0-58 and validates 59-70. `fold_indices` checks chronology (`analysis/modeling.py:625`) and `_run_fold_set` creates and fits a fresh estimator per fold (`analysis/modeling.py:713`).
2. **Train-only preprocessing.** Robust centers, scales, missing-column indicators and nonlinear-transform scales are fitted from the supplied fit slice (`analysis/modeling.py:321`). The v2 tree evaluator fits preprocessing on `X[train_slice]`, transforms validation afterwards, and fits LightGBM on `target[train_slice]` (`analysis/v2/run_sequence_experiment.py:144`, `:162`). It uses a fixed tree count, with no outer early-stopping callback.
3. **Nested neural epoch selection and honest refit.** TabM's last three outer-training months are inner validation (`analysis/v2/run_tabm_mini_challenger.py:259`). Inner preprocessing and target standardization use only inner-fit observations (`:271`, `:283`); epoch selection uses only inner cosine (`:340`). The inner model is discarded, preprocessing is fitted afresh on all outer-training rows (`:392`), target scale is recomputed there (`:404`), and a fresh network trains for the selected epoch count (`:430`). Outer targets are used for reporting at `:474`, after prediction. Selected epochs are 7, 6 and 6; the final epoch is their median, 6.
4. **Final full-data refits are appropriate.** The deployed tree and neural model use months 0-70 after configuration selection (`analysis/v2/train_final_blend_and_submit_v2.py:1404`, `:1417`, `:1466`, `:1472`). Refitting on formerly held-out labeled months is normal when producing test predictions. It would only be leakage if those refits were then used to report held-out training performance; the reviewed code instead reports saved fold-model predictions.
5. **No identifier-as-feature path found in this review.** The feature materializer rejects `sample_id`, `month` and `target` (`analysis/feature_families.py:431`); the preprocessor also rejects these names (`analysis/modeling.py:229`, `:257`). This is evidence about the declared feature schema, not proof against every possible information-bearing derived feature or duplicated sample.
6. **Uncentered cosine and final prediction construction agree.** `analysis/modeling.py:98` uses the exact uncentered dot-product cosine. The final blend separately scales raw model predictions to unit RMS, combines them with 0.6/0.4 weights, applies signed power 1.1 and scales once again (`analysis/v2/run_tabm_blend_sealed_audit.py:494`). No target values enter these RMS scales.

Loading the complete target array into memory, as some development functions do, is weaker isolation than the v1 sealed loader, but is **not itself evidence of target leakage**. The relevant question is which rows are passed to fitting and selection. The reviewed calls pass the appropriate training slice.

## Confirmed validation limitations

### 1. Months 59-70 were reused across model generations

The v1 audit completes at 2026-08-23 22:46:59 UTC and reports 0.142998 cosine (`artifacts/diagnostics/sealed_audit_summary.json`). The v1 audit script's exclusive ledger and explicit delayed target loading are useful safeguards (`analysis/run_sealed_audit_once.py:302`, `:327`, `:412`). They apply to that script and ledger only.

Separate saved results show additional evaluations on the **same** months:

| Historical 59-70 model or view | Cosine |
|---|---:|
| v1 slow LightGBM, raw | 0.142998 |
| v2 slow LightGBM with path features, raw | 0.142943 |
| v2 capacity LightGBM with path features, raw | 0.149429 |
| v2 capacity LightGBM, q=1.2 | 0.150587 |
| TabM-mini, raw | 0.153397 |
| 60/40 blend, linear | 0.154977 |
| final 60/40 blend, q=1.1 | 0.155564 |

Sources: `artifacts/v2/experiments/sequence_base_plus_sequence_all_slow_SealedAudit_summary.json`, `sequence_base_plus_sequence_all_capacity_SealedAudit_summary.json`, and `artifacts/v2/sealed_blend/sealed_blend_summary.json`.

The generic tree evaluator explicitly includes `SEALED_AUDIT_FOLD` among selectable folds (`analysis/v2/run_sequence_experiment.py:111`) with no project-wide spent-holdout check. On any selected fold it reports and sorts six signed-power exponents (`:204`). **Both saved SealedAudit tree summaries contain the six exponent scores on months 59-70.** This is stronger evidence than merely finding an unused permissive code path.

The two v2 sealed tree summaries have filesystem modification times 2026-08-24 19:44:18 and 20:12:59 UTC, preceding the final blend contract's recorded creation at 20:37:23 UTC. File timestamps are corroborating evidence only; they are not tamper-proof provenance. Independently of timestamps, the final contract itself contains the already-known high-energy month 66 diagnostic and exact capacity benchmark result.

**Interpretation:** at least four distinct trained configurations and multiple transformed views were evaluated on this supposed holdout. I cannot prove from saved artifacts that an individual hyperparameter was selected by maximizing these late-block results. I can prove that the period was no longer unknown to the project and that calling its later use an untouched one-time holdout is inaccurate.

### 2. “Safety veto” is still a model-selection decision

`artifacts/v2/diagnostics/frozen_blend_before_dev3.json` fixes a rule that keeps the blend or falls back to capacity-LightGBM according to months 59-70 performance. The rule is implemented in `analysis/v2/run_tabm_blend_sealed_audit.py:803`, and final publication depends on reproducing that gate (`analysis/v2/train_final_blend_and_submit_v2.py:1221`).

This is a reasonable bounded decision rule. It is narrower than unrestricted tuning. Nevertheless, the deployed model depends on validation labels through the keep/fallback decision. Flags such as `sealed_results_are_selection_eligible: false` must be interpreted as “no parameter-grid tuning,” not “the validation results played no role in selecting the submitted pipeline.” A confidence interval for the finally retained candidate does not account for this selection automatically.

### 3. Dev3 was not an untouched project-level confirmation set

Dev3's months 47-58 were development data in v1, had multiple feature/capacity comparisons in v2, and had calibration alternatives exposed in postmortem work. The final blend contract includes the known capacity q=1.2 Dev3 cosine, 0.1401980574532448, as a promotion threshold.

The final TabM blend search itself is confined to Dev1+Dev2 in `analysis/v2/audit_frozen_tabm_capacity_blend.py:248`. Only the frozen combination is constructed when that script reaches Dev3 (`:287`), which is good local discipline. But `analysis/v2/analyze_v2_selection.py:142` already computes all six power alternatives on Dev3 and its following blend loop reports all eleven v1/capacity alternatives there. The earlier v1 postmortem also reports a 17-candidate calibration exercise and best post-hoc Dev3 alternatives (`artifacts/diagnostics/postmortem/postmortem_summary.json`). Consequently “first evaluation of this frozen combination on Dev3” is accurate; “previously untouched Dev3 labels” is not.

The TabM development runner additionally contains an all-fold comparison helper that would expose several Dev3 alternatives and use whole-OOF component RMS scales while screening Dev1+Dev2 (`analysis/v2/run_tabm_mini_challenger.py:558`). The saved final challenger summary is from a partial fold run and does not establish that this helper ran for the promoted pipeline. Treat this as a future workflow hazard, not a demonstrated cause of the current result.

### 4. Historical superiority is not a private-test confidence interval

The reported 3-month paired bootstrap intervals describe variation across the observed months conditional on the compared predictions and selected recipes. They do not include search over feature sets, neural architectures, powers, blending choices or leaderboard feedback. They do not capture regimes absent from the training history. Monthly-block resampling is preferable to pretending 600,000 rows are independent, but there are only 36 development months and 12 late audit months.

V1's high-energy month 66 alone contains 24.516% of the historical 59-70 target squared norm. Removing it drops v1 from 0.142998 to 0.132974. The final blend similarly moves from 0.155564 to 0.143779; the improvement over capacity survives exclusion (+0.004594), which supports a real comparative gain but also shows why the high late-block headline is an optimistic ordinary-regime anchor.

### 5. Long deployment age has not been validated for the complete final blend

`artifacts/v2/diagnostics/deployment_stress_summary.json` evaluates capacity-LightGBM only. It reports raw pooled cosine 0.136388 and q=1.2 cosine 0.137066. It does not test the final TabM component or the 60/40 blend over those long horizons.

The seven origins have overlapping forecast rows (`analysis/v2/run_deployment_stress_v2.py:276`), so the pooled 3,976,613 forecasts are not that many independent observations. At horizon 38 months only one origin and 17,660 rows remain. Repeated origins help at shorter horizons; the 38-month endpoint cannot establish robustness across seven independent future regimes. This is a central unresolved deployment question for the final model.

### 6. Boundary overlap cannot be ruled out with the current metadata

The outer month split has no purge or embargo (`analysis/modeling.py:71`). With sub-hour input/target windows, cross-boundary overlap could exist near a monthly cutoff if sampling spans the boundary. Absolute event timestamps and instrument identifiers are absent according to `analysis/pipeline_config.py:3`, so this review cannot verify event overlap, duplicate windows across folds, or a correct second-level purge. This is an unresolved possibility, **not a confirmed leak**. A coarse one-month gap is a conservative sensitivity experiment, not a principled replacement for precise overlap metadata.

## What the gains actually show

| Development months 23-58, same saved rows | Cosine | Increment |
|---|---:|---:|
| v1 slow LightGBM | 0.133238 |  -  |
| v2 capacity LightGBM, base features only | 0.138260 | +0.005022 |
| v2 capacity LightGBM, base + path features | 0.139013 | +0.000753 |
| capacity with q=1.2 | 0.139450 | +0.000437 |
| final TabM/capacity blend with q=1.1 | 0.145297 | +0.005847 |

Source: `artifacts/v2/diagnostics/model_selection_summary.json` and `tabm_capacity_pooled_report.json`. About 87% of the raw tree gain over v1 comes from the higher-capacity tree configuration; about 13% comes from adding path features at that capacity. The path experiment is promising but much smaller than the narrative emphasis on sequence features may suggest. These are engineered path statistics consumed by a tabular model, not a fitted raw-sequence neural model.

The frozen blend achieves 0.144456 on its screening months 23-46 and 0.147000 on months 47-58. Historical 59-70 improvement over capacity q=1.2 is +0.004977 and remains +0.004594 after removing month 66. These comparisons argue against the claim that the entire v2 improvement is a validation artifact. They do not establish a specific future leaderboard gain.

## Notebook and artifact provenance

The saved notebooks are executed reporting companions: v1 final report 11/11 code cells, v2 final report 12/12, neural challenger 6/6, with zero error outputs. Inspection found no fitting call or training subprocess in their code cells. The generator explicitly describes v2 as reading saved artifacts (`analysis/v2/build_v2_model_notebook.py:1`); neural reporting likewise declares that it does not train (`analysis/v2/build_neural_challenger_notebook.py:50`). They are not independent reruns of model training and should not be presented as proof that the historical research sequence complied with every no-peeking claim.

Some narrative language should be corrected when the report is next regenerated: “untouched Dev3” (`analysis/v2/build_neural_challenger_notebook.py:207`), “opened once” for months 59-70 (`analysis/v2/build_v2_model_notebook.py:496`), and selection-ineligible sealed flags next to a keep/fallback decision. Do not silently alter the archived v1/v2 evidence while performing a v3 audit.

## Recommended validation policy for the next round

1. Mark all months 23-70 as **previously exposed historical evaluation data**. Never reset the word “sealed” because a new model generation or ledger is introduced.
2. Before running a bounded next experiment, record the candidate recipes, training histories, primary metric, fixed deployment-age slices, and a small decision rule. Retain every result, including failures. Treat repeated retrospective selection honestly.
3. Evaluate the complete inference recipe, including component normalization, power, neural refitting and model age. New calibration weights should be learned using only earlier out-of-fold predictions and then applied to later blocks.
4. Use paired monthly results, block-length sensitivity and tail concentration to decide whether gains are broad. Confidence intervals must be labeled conditional retrospective intervals, not private-leaderboard forecasts.
5. Reserve a truly new evaluation block only if additional genuinely unexamined labeled history becomes available. A relabeled subset of already-examined months does not restore independence. Avoid public-leaderboard hill climbing.
6. Ask the organizer or obtain source metadata for instrument IDs, sample timestamps and precise label support if available. This would enable overlap deduplication and principled purging rather than guesswork.

The companion `analysis/v3_audit/recompute_evidence.py` recomputes aligned retrospective comparisons from saved NPZ files without model training. Its outputs under `artifacts/v3_audit` supplement this memo; they do not reopen an untouched holdout.
