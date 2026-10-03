# V3 audit and controlled experiments

This generation audits the existing models and runs a bounded, local comparison.
It does not publish a Kaggle submission or replace the V2 model.

All months 23-70 have already been inspected during earlier research. V3 calls
its scores **retrospective evaluation**, never a new sealed holdout. Training
windows always precede evaluation windows. Neural epoch selection uses the
last three months within each training window, followed by a fresh full-window
refit. No evaluation labels select the epoch count.

The fixed comparisons are base TabM, TabM with all 754 inputs, a compact temporal
encoder on the same ten six-second bins plus the base inputs, and LightGBM with
native missing values and uncompressed tails. The separate incumbent stress
experiment reproduces the existing 60/40 blend recipe at a single 38-month
forecast horizon. Both components are trained once; complete-vector RMS
normalization is fixed before reporting age slices.

The comparison is one seed per new neural candidate. The native tree experiment
changes missingness and tail handling together; it cannot attribute effects to
either one separately. Tiny improvements require another seed or a subsequent
bounded confirmation. An illustrative 20% challenger blend is a fixed diagnostic,
not a fitted blend weight or automatic promotion decision.

## Reproduction

From the project root in PowerShell:

```powershell
$modelPython = 'python'
& $modelPython analysis\v3_audit\recompute_evidence.py
& $modelPython -m unittest discover -s analysis\v3 -p test_sequence_kernels.py -v
& $modelPython analysis\v3\run_overnight_queue.py --hours 6
& $modelPython analysis\v3\compare_experiments.py
& $modelPython analysis\v3\verify_completed_experiments.py
& $modelPython analysis\v3_audit\full_cache_reference_census.py
& $modelPython analysis\v3_audit\build_audit_report.py
```

The queue skips completed experiments and refuses to duplicate a live identical
process. Check the actual process and `queue_status.json` before resuming. The
shared six-hour budget includes all jobs launched by that invocation; standalone
runner budgets are per job. Never run concurrent GPU training processes.

The runner deliberately rejects a changed source/protocol under the same output
directory. A materially changed experiment needs its own named generation.
Do not overwrite the frozen protocol to make that check pass.

Runtime folders `.v3_deps` and `.v3_torch` are isolated from the old CPU-only
`.analysis_deps`. The runtime and input hashes are recorded in
`artifacts/v3/bounded_experiments/input_provenance.json`.

## Feature-correction scope

`sequence_kernels.py` provides tested NumPy reference helpers. These correct
invalid quote predecessors, undefined normalization, backward as-of matching
and valid-only weighted means. They are **not** wired into production feature
extraction. The bounded architecture comparison intentionally uses the existing
cached inputs, preserving an equal-information comparison. A corrected extractor
must use a new cache generation, measure full-data prevalence and compare
corrected versus original inputs before making any score claim.

## Decision evidence

Read `analysis/v3_audit/assessment.md` for the integrated findings, and the source
audit, methods memo and executed evidence notebook in that directory for detail.
Results and prediction hashes are under `artifacts/v3` and `artifacts/v3_audit`.
The canonical `submission_final.csv` remains the original submitted V2 blend.

The first-round queue is complete: 12 candidate/fold fits. All prediction
vectors align exactly with the original labels; every saved checkpoint/model
reproduces the first 2,048 evaluation predictions exactly. The native-tree
package and the compact temporal recipe did not improve the pooled incumbent.
A 20% contribution from full-input TabM was promising in the first seed, so a
separate fixed second-seed follow-up was launched under
`artifacts/v3/seed_confirmation`. Its protocol was frozen after inspecting the
first round: it is a robustness check, not independent blind confirmation.
Read its `comparison.json`, `verification.json` and queue status before using
that follow-up. The wrapper refuses to relaunch an existing queue.

The full-cache reference census covers all 1,257,637 training and 647,896 test
rows. Undefined terminal spread references affect 0.447% and 0.700%, respectively.
That expands the earlier systematic probe, but does not count raw invalid OFI
transitions or establish a model-score benefit from correcting them.
