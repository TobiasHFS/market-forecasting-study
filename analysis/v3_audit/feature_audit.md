# Sequence-feature correctness audit

Audited 2026-09-10, with a live source-only check on 2026-09-11. Scope: frozen V1/V2 feature source, existing cache probes,
and new isolated NumPy correctness helpers. No V2 source, cache, model, or
prediction was changed, and no raw-data extraction or training was launched.

## Finding and practical interpretation

There are reproducible V2 feature defects. They concern invalid-state handling,
missingness, and event/quote alignment. These findings **do not demonstrate
target leakage or validation-label contamination**: the implicated quote and
event histories are supplied before each prediction timestamp. The separate
validation/selection audit must establish whether measured model improvements
generalize. Correcting these feature defects is justified, but an improvement
in the competition metric remains an experimental question.

### 1. Invalid quotes can become the OFI predecessor

- Current-row validity requires a positive bid and non-crossed book at
  `analysis/v2/sequence_features.py:200`.
- Previous-state assignment at `analysis/v2/sequence_features.py:271` checks
  finite prices and nonnegative volumes but omits positive prices and the
  non-crossed condition. It also ignores timestamp validity. A zero/crossed
  quote can therefore become the predecessor of the next valid quote.
- The next current-row OFI calculation at
  `analysis/v2/sequence_features.py:242` only checks finite previous prices.
  With valid (99/101, depth 10/10), invalid (0/0, depth 0/0), then the same
  valid quote, V2 produces +0.5 terminal-depth-scaled OFI instead of zero:
  `analysis/v3_audit/feature_probe_results.json:401`.
- V1 instead calls `_valid_l1` for the predecessor at
  `analysis/market_features.py:376` and skips invalid current states at
  `analysis/market_features.py:396`. Its quote-validity predicate is at
  `analysis/market_features.py:282`. The regression is therefore specific to
  the V2 path implementation for this case.

The synthetic example proves the code defect, not its frequency or score
impact. Existing cache probes do not count transitions across invalid states.
New V3 `ofi_l1_events` uses the same strict finite-price/depth book predicate
for both endpoints and excludes invalid times. Each sample's first valid
state has missing OFI because no previous observation exists; an observed
unchanged transition still has exactly zero OFI.

### 2. Undefined reference-normalized measurements become ordinary zeroes

- V2 initializes output to zero at
  `analysis/v2/sequence_features.py:557` and
  `analysis/v2/sequence_features.py:630`.
- `_valid_reference` at `analysis/v2/sequence_features.py:149` rejects absent,
  locked, or negligibly small reference spreads. Numerators are then skipped
  at lines 213, 223, 263, 371, and 453, but populated-bin finalization divides
  the initialized zeroes at lines 288, 294, 402, 407, 411, and 471.
- An invalid reference therefore becomes indistinguishable from a measured
  zero price displacement whenever there are qualifying events/volume.

The existing probe covers **50,000 evenly spaced rows of each split**, not a
random sample or a complete population census
(`analysis/v3_audit/feature_probe_results.json:2`). It finds:

| Existing cache probe | Train | Test |
|---|---:|---:|
| Samples with invalid terminal spread reference | 222 / 50,000 (0.444%) | 352 / 50,000 (0.704%) |
| Populated order bins with invalid reference | 2,004 | 3,293 |
| Such order bins with zero VWAP displacement | 2,004 | 3,293 |
| Populated trade bins with invalid reference | 1,488 | 2,614 |
| Such trade bins with zero VWAP displacement | 1,488 | 2,614 |
| Latest market bin without a valid snapshot | 4.954% | 3.470% |

Evidence: `analysis/v3_audit/feature_probe_results.json:192` and
`analysis/v3_audit/feature_probe_results.json:391`. The market invalid-reference
latest-bin counts are zero in this probe; do not generalize the demonstrated
flow-bin issue into a claim that these sampled market bins were also affected.
The locked-book synthetic example independently demonstrates market channels
returning zero for an undefined spread reference at
`analysis/v3_audit/feature_probe_results.json:405`.

V3 `spread_units` and `depth_units` return NaN for invalid references. Models
may impute these later using training-fitted preprocessing, but should retain
an explicit missingness/coverage indicator. Do not replace an unknown value
with a measured zero during extraction.

### 3. "Contemporaneous" mid is a six-second bin average

`_contemporaneous_mid` at `analysis/v2/sequence_features.py:304` reconstructs
the mean mid of the event's six-second market bin. This is not an event-time
backward as-of join. Quotes later than the event but within the bin influence
its reference. If the bin is empty, line 321 returns the terminal mid, which
can also be later than the event. Order/trade callers are at lines 372 and 454.

Example: an event at age 5 seconds and a quote at age 2 seconds occupy the
same bin. The quote happened three seconds *after* the event. If the latest
earlier quote has age 8 seconds, that is the backward as-of reference.
The independent live-source check in
`analysis/v3_audit/live_source_semantics_probe.json` uses quote mids 100 and
118 at those ages: V2 uses 118 for the event, whereas backward as-of uses
100. An order priced at 101 with terminal spread 2 therefore receives V2
displacement -8.5 instead of as-of displacement +0.5. This invokes freshly
read `_market_kernel`, `_contemporaneous_mid`, and `_order_kernel` function
bodies with synthetic inputs and does not read raw data or feature caches.

This is **look-ahead relative to the historical event**, not proof of access
to information after the prediction timestamp. The existing feature can
still be a legitimate retrospective summary of supplied history. Its naming
and economic interpretation as instantaneous execution/aggressiveness are
incorrect, and it mixes subsequent price response into the purported event
displacement. Retain a separately named retrospective-bin displacement only
if it earns its place in properly isolated validation.

V3 `asof_l1_quotes` joins within the same sample using
`quote_seconds_before_predict > event_seconds_before_predict` by default.
It exposes quote lag and returns no reference when no valid earlier quote
exists. Equal timestamps require explicit opt-in: timestamp ties do not
establish cross-stream ordering. `max_lag_seconds` optionally masks stale
quotes. There is no future/terminal fallback. The helper accepts arbitrary
query order but rejects quote-time inversions rather than silently using row
order as time.

### 4. Some means use a denominator broader than their numerator population

- Order displacement accumulation excludes invalid prices/references at
  `analysis/v2/sequence_features.py:371`, while line 402 divides by all event
  volume. The directional new/cancel versions use all new/cancel volume at
  lines 406 and 410. Trade displacement similarly uses lines 453 and 471.
- Market price-volume accumulation skips invalid bar price/reference at
  `analysis/v2/sequence_features.py:263`, but line 294 uses all positive bar
  volume collected at line 262.
- Market channel-specific depth/L2 conditions at lines 215, 230, and 239
  share the all-valid-book count denominator at line 288. V2's book predicate
  also admits zero total depth, unlike V1. Undefined channel observations
  therefore shrink that channel's purported mean toward zero.
- V1 flow VWAP explicitly accumulates a matching `vwap_volume` denominator:
  `analysis/flow_features.py:336`, `analysis/flow_features.py:378`,
  `analysis/flow_features.py:539`, and `analysis/flow_features.py:550`.

These are source-backed semantic defects. Frequency/score impact from invalid
price or channel-specific missingness has not been quantified. V3's
`weighted_bin_mean` includes precisely the same finite values and positive
weights in numerator and denominator, returns NaN when no value is observed,
and returns observed weight separately to preserve coverage information.
An independent synthetic call to the live V2 `_trade_kernel`, recorded in
`analysis/v3_audit/live_source_semantics_probe.json`, confirms the dilution:
valid displacement 1 with volume 10 plus undefined price with volume 90
produces 0.1. The matching-valid-volume mean is 1.

## Isolated corrected implementation and verification

New code: `analysis/v3/sequence_kernels.py`.

| Helper | Evidence |
|---|---|
| Strict, finite L1 quote validity | `analysis/v3/sequence_kernels.py:33` |
| Missing spread/depth reference handling | `analysis/v3/sequence_kernels.py:46` and `:66` |
| Consecutive-valid-state OFI | `analysis/v3/sequence_kernels.py:97` |
| Same-sample backward as-of quote join | `analysis/v3/sequence_kernels.py:135` |
| Matching-mask weighted bins | `analysis/v3/sequence_kernels.py:187` |

Verification command (run successfully on 2026-09-10, NumPy 2.3.5):

```powershell
python 'analysis/v3/test_sequence_kernels.py'
```

All **16 synthetic tests passed**. Tests cover expected OFI values for price
moves/depth changes, invalid zero/crossed/time states, missing references,
locked books, sample boundaries including empty groups, quote-time inversion
rejection, timestamp ties, stale quotes, invalid event times, correct backward
age direction, weighted-denominator alignment, and exact bin boundaries.
The future-invariance test at `analysis/v3/test_sequence_kernels.py:68`
appends and perturbs quotes after a historical event and confirms that its
as-of reference cannot change. This is a meaningful temporal correctness
property, not a test that merely copies the implementation's formula.

These functions are a tested reference implementation for a future extractor.
They are **not integrated into production extraction**, are not a regenerated
feature cache, and have not been shown to improve a model. Their NumPy arrays
and per-sample loops have not been benchmarked on hundreds of millions of
rows. Integrating them requires bounded streaming/Numba equivalents, including
retention of a valid pre-window seed and explicit channel coverage features.

## Remaining extraction checks before a V3 training claim

1. Run a deterministic raw-data pilot that counts invalid-state transitions,
   missing normalization, price-valid volume coverage, time ties, and stale
   as-of references. Record both sample and event/bin denominators. Read-only
   test-feature diagnostics do not justify training/selection on test labels.
2. Preserve existing V2 feature schemas/hashes; write new cache files and
   record source/code fingerprints. Direct cached replacement cannot repair
   true as-of alignment because a bin mean discards within-bin event times.
3. Compare existing bin-summary features and corrected as-of features under
   the same frozen validation recipe, with hyperparameter/tuning access
   recorded. Correctness alone supplies no reliable expected score delta.
4. Check remaining raw timing assumptions. For example V1's outside-window
   OFI seed branch at `analysis/market_features.py:374` receives a bucket that
   combines out-of-window and invalid timestamps, so invalid-time rows would
   require a separate guard. Existing latest-raw-row age diagnostics are not
   a substitute for the age of the selected latest *valid* reference.

No conclusion here assumes that a larger sequence model will cure incorrect
feature semantics or a biased validation procedure.

## Full-cache census update, 12 September 2026

The coordinator subsequently ran `full_cache_reference_census.py` over every
cached sample. This expands the original systematic probe; it does not scan
raw quote transitions or measure model-score impact. Original cache hashes
and the script hash are recorded in
`artifacts/v3_audit/full_cache_reference_census.json`.

| Full existing-cache population | Train | Test |
|---|---:|---:|
| Samples | 1,257,637 | 647,896 |
| Invalid terminal spread reference | 5,627 (0.447%) | 4,538 (0.700%) |
| Populated order bins with invalid reference | 49,889 | 41,931 |
| Such order bins with zero VWAP displacement | 49,889 | 41,931 |
| Populated trade bins with invalid reference | 35,950 | 32,914 |
| Such trade bins with zero VWAP displacement | 35,950 | 32,914 |
| Latest market bin without valid snapshot | 61,394 (4.882%) | 22,505 (3.474%) |

The exact original 50,000-row invalid-reference counts reproduce when the
full census is restricted to those probe indices. Every invalid-reference
sample contains at least one populated order bin and one populated trade bin.
No invalid-reference sample has a populated latest market bin, matching the
earlier probe's qualification. These are cache-defined validity conditions;
they do not imply that all information in the affected samples is unusable.
Other base features expose reference/missingness information to the models.
The higher test prevalence is established, but it is not evidence that this
under-1% issue explains the overall public-score gap.
