"""Fixed second-seed and two-seed comparisons on previously exposed months."""
from __future__ import annotations

import json
from pathlib import Path

import run_seed_confirmation as run
import compare_experiments as compare
import numpy as np

ROOT, OUT = run.ROOT, run.OUT


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if protocol != json.loads(json.dumps(run.protocol())):
        raise RuntimeError('Frozen source or protocol changed')
    data, sources, fold_rows = {}, {}, []
    for seed, generation in ((20260910, 'bounded_experiments'), (20260912, 'seed_confirmation')):
        for candidate in run.CANDIDATES:
            paths = []
            for fold in run.FOLDS:
                dest = ROOT / f'artifacts/v3/{generation}/{candidate}_{fold}'
                summary = json.loads((dest / 'summary.json').read_text())
                path = dest / 'predictions.npz'
                sha = run.runner.digest(path)
                if summary['status'] != 'complete' or sha != summary['predictions_sha256']:
                    raise AssertionError('Incomplete or changed model predictions')
                sources[str(path.relative_to(ROOT))] = sha
                paths.append(path.relative_to(ROOT))
                fold_rows.append({'seed': seed, 'candidate': candidate, 'fold': fold,
                    'cosine': summary['pooled_cosine'], 'selected_epoch': summary['selected_epoch'],
                    'seconds': summary['elapsed_seconds']})
            data[f'{candidate}_seed{seed}'] = compare.load(paths)
    capacity_paths = [Path('artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_Dev1-Dev2_oof.npz'),
                      Path('artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_Dev3_oof.npz')]
    tabm_paths = [Path(f'artifacts/v2/tabm_mini/dev{i}_predictions.npz') for i in (1, 2, 3)]
    capacity, tabm = compare.load(capacity_paths), compare.load(tabm_paths)
    for path in capacity_paths + tabm_paths:
        sources[str(path)] = run.runner.digest(ROOT / path)
    ref = next(iter(data.values())); y, months = ref['target'], ref['months']
    for values in [*data.values(), capacity, tabm]:
        for key in ('row_indices', 'months', 'target'):
            if not np.array_equal(values[key], ref[key]):
                raise AssertionError('Out-of-fold alignment mismatch')
        if not np.all(np.isfinite(values['prediction'])):
            raise AssertionError('Nonfinite predictions')
    lin = .6 * compare.unit(tabm['prediction']) + .4 * compare.unit(capacity['prediction'])
    incumbent = compare.unit(np.sign(lin) * np.abs(lin) ** 1.1)
    predictions = {name: values['prediction'] for name, values in data.items()}
    for candidate in run.CANDIDATES:
        predictions[f'{candidate}_mean2'] = .5 * (
            compare.unit(predictions[f'{candidate}_seed20260910']) +
            compare.unit(predictions[f'{candidate}_seed20260912']))
    comparisons = {}
    for name in ('seed20260910', 'seed20260912', 'mean2'):
        comparisons[f'tabm_path_minus_tabm_base_{name}'] = compare.paired(
            months, y, predictions[f'tabm_path_{name}'], predictions[f'tabm_base_{name}'])
    pooled, blends = [], {}
    for name, p in predictions.items():
        blend = .8 * incumbent + .2 * compare.unit(p)
        blends[name] = blend
        comparisons[f'fixed20pct_{name}_minus_incumbent'] = compare.paired(months, y, blend, incumbent)
        pooled.append({'candidate': name, 'pooled_cosine': compare.score(y, p),
            'fixed20pct_blend_cosine': compare.score(y, blend),
            'prediction_correlation_with_incumbent': float(np.corrcoef(p, incumbent)[0, 1])})
    for name in ('seed20260910', 'seed20260912', 'mean2'):
        comparisons[f'fixed20pct_path_minus_fixed20pct_base_{name}'] = compare.paired(
            months, y, blends[f'tabm_path_{name}'], blends[f'tabm_base_{name}'])
    result = {'status': 'complete', 'scope': 'Previously exposed months 23-58; second seed chosen after first-round results. Fixed recipes and conditional retrospective month-block intervals; not independent blind validation.',
        'rows': len(y), 'incumbent_pooled_cosine': compare.score(y, incumbent),
        'archived_tabm_pooled_cosine': compare.score(y, tabm['prediction']),
        'fold_rows': fold_rows, 'pooled': pooled, 'comparisons': comparisons, 'sources': sources,
        'recipe': 'Unit RMS each individual pooled prediction vector; equal two-seed average; unit RMS that average; 80 percent archived incumbent plus 20 percent candidate. No new power transform, weight search, or submission.',
        'bootstrap': '10000 paired moving month-block replicates, lengths 1, 3, and 6, from unchanged compare_experiments.paired helper. Fixed vectors are not refit or renormalized per replicate.'}
    run.runner.save(OUT / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
