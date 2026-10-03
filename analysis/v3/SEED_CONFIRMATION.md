# Second-seed follow-up

This isolated follow-up uses seed 20260912 for the existing base-feature and full-feature TabM candidates on Dev1, Dev2, and Dev3. It changes the random seed and output directory in memory after importing the immutable original runner. All other training settings are inherited: the inner validation consists of the last three training months, the inner maximum is 16 epochs, and the selected epoch is used for a fresh fit on all outer training data.

The queue has one shared 45-minute deadline and runs six jobs serially. The original experiment generation and canonical submission are unchanged. The protocol, source hashes, and feature and label identities are frozen before any second-seed training. Each job saves its log, checkpoint, preprocessing object, predictions, and summary in `artifacts/v3/seed_confirmation`.

This experiment was selected after seeing the first-seed results. The evaluation months were already used in previous modeling decisions. A second seed probes randomness sensitivity; it does not restore an untouched holdout or establish a private-leaderboard gain.

The fixed comparisons use both single seeds and equal-weight averages of unit-RMS out-of-fold predictions from the two seeds. Each candidate is evaluated alone and as a fixed 20 percent addition to the archived incumbent. The two-seed mean is itself normalized by RMS before receiving that 20 percent weight. The full-feature contribution is also compared directly with the base-feature contribution, so a generic seed-ensemble benefit is not mistaken for a feature benefit. No weights or powers are searched.

The existing month-block helper supplies 10000 paired bootstrap replicates for block lengths of one, three, and six months. These are conditional retrospective intervals for already selected recipes. The bootstrap does not rerun training or adjust for model selection.

Reproduce with the bundled Python from the workspace root:

```powershell
python -u analysis/v3/run_seed_confirmation.py --freeze-only
python -u analysis/v3/run_seed_confirmation.py
python -u analysis/v3/run_seed_confirmation.py --verify
python -u analysis/v3/compare_seed_confirmation.py
```

The queue refuses to overwrite an existing queue. Verification rechecks all frozen sources and input hashes, aligns the full saved prediction vectors to the original Feather labels, recomputes pooled and monthly scores, and replays the first 2048 evaluation rows of each checkpoint through its saved preprocessor. A completed result is only reviewable once `verification.json` reports `passed` and `comparison.json` exists.

## Completed results

All six second-seed fits completed successfully in 794.5 seconds (13.2 minutes). The 45-minute limit was not reached. Verification passed: each checkpoint reproduced its first 2048 evaluation predictions exactly, every pooled and monthly score was recomputed, all saved target vectors matched the original Feather labels, source and input hashes were unchanged, and the canonical submission was unchanged.

| Second-seed candidate | Dev1 | Dev2 | Dev3 | Pooled months 23-58 | Selected epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| Base features | 0.142145 | 0.137288 | 0.142986 | 0.140372 | 6, 15, 12 |
| Full features | 0.140853 | 0.141513 | 0.146396 | 0.142147 | 6, 7, 7 |

The full-feature model's second-seed pooled improvement over base was 0.001775, with a conditional 95% three-month block interval of [-0.000198, 0.003793]. The first fold changed sign relative to seed one. The two-seed standalone means scored 0.142407 for base and 0.143552 for full features; their difference of 0.001145 had an interval of [-0.000566, 0.003203]. The full-feature mean also remained below the archived original TabM's 0.143780. These results support a possible feature benefit but do not establish a superior standalone model.

The evidence is stronger for a small complementary ensemble contribution. The archived incumbent scores 0.145297 on the same 639120 rows. Adding the second-seed full-feature model at the fixed 20% weight scored 0.146204, an improvement of 0.000908 with an interval of [0.000630, 0.001301]. Adding the two-seed full-feature mean at that weight scored 0.146230, improving by 0.000933 with an interval of [0.000641, 0.001392]. All one-, three-, and six-month block intervals for these two improvements were positive.

The matched two-seed base-feature contribution scored 0.145727. The two-seed full-feature contribution therefore added 0.000503 beyond the base-feature contribution, with an interval of [0.000146, 0.000901]; this interval was also positive at all three block lengths. This comparison distinguishes complementary full-feature information from a generic benefit of averaging more seeds. It remains a retrospective result from previously exposed months and a follow-up selected after earlier results. It cannot certify a private-leaderboard improvement.

The stopping rule is materially sensitive to the seed: base Dev3 changed from four selected epochs in seed one to twelve in seed two, while the outer score changed only from 0.142726 to 0.142986. The experiment tests the complete seeded training recipe, including epoch selection, rather than isolating initialization variance alone. No settings were changed after observing this behavior.

Machine-readable evidence is in `artifacts/v3/seed_confirmation/comparison.json`, `verification.json`, `queue_status.json`, `protocol.json`, and `input_provenance.json`. The original runner and original experiment generation were not edited. There is no new submission file.
