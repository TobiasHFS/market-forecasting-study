"""Fixed-recipe incumbent age diagnostic on previously exposed months.

Train months 0--20 once; predict months 21--58 without updates. This recreates
the incumbent recipe using the original TabM and tree implementations. A new
GPU fit is not a bitwise reproduction of the original CPU-trained model.
Nothing here promotes a model or writes to an existing submission.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# Prepopulate every legacy import path so imported v2 scripts do not prepend
# .analysis_deps above the working v3 NumPy/PyArrow/PyTorch installations.
for entry in (ROOT / '.analysis_deps',):
    if str(entry) not in sys.path:
        sys.path.append(str(entry))
for entry in (ROOT / 'analysis', ROOT / 'analysis/v2', ROOT / '.v3_deps', ROOT / '.v3_torch'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pyarrow.feather as feather
import lightgbm
import joblib
from modeling import RobustPreprocessor
from run_sequence_experiment import SPECS
from run_tabm_mini_challenger import (
    TabMMiniSpec, _build_tabm_mini, _cache_training_matrix, _make_preprocessor,
    _predict_members, _run_epoch, _set_determinism, _smooth_clip_inplace,
)

OUT = ROOT / 'artifacts/v3/incumbent_age38'
FEATURES = ROOT / 'artifacts/v2/features'
SEED, EPOCHS, ORIGIN = 20260823, 6, 20
HORIZONS = (1, 3, 6, 12, 18, 24, 30, 38)
SPEC = TabMMiniSpec(max_epochs=8, min_epochs=4, patience=4, seed=SEED)
TREE_PARAMS = asdict(SPECS['capacity'])
TREE_PARAMS.update(force_col_wise=True, deterministic=True,
                   bagging_seed=SEED, feature_fraction_seed=SEED)
SOURCE_PATHS = (
    'analysis/v3/stress_incumbent.py', 'analysis/modeling.py', 'analysis/pipeline_config.py',
    'analysis/feature_families.py', 'analysis/v2/run_tabm_mini_challenger.py',
    'analysis/v2/run_realmlp_challenger.py', 'analysis/v2/run_sequence_experiment.py',
    'analysis/v2/v2_features.py', 'analysis/v2/sequence_features.py',
)
INPUT_PATHS = tuple(
    f'artifacts/v2/features/train_{stem}{suffix}'
    for stem in ('base_only', 'base_plus_sequence_all')
    for suffix in ('.npy', '.names.json', '.kinds.json')
) + ('ms-capital-real-financial-market-forecasting/train/label.feather',)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def file_record(relative):
    path = ROOT / relative
    stat = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    result = {'bytes': stat.st_size, 'sha256': h.hexdigest()}
    if path.suffix == '.npy':
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        result.update(shape=list(a.shape), dtype=str(a.dtype))
        del a
    return result


def protocol():
    return {
        'status': 'frozen_retrospective_incumbent_recipe_diagnostic',
        'training_months': [0, 20], 'forecast_months': [21, 58], 'origin': ORIGIN,
        'tabm_spec': asdict(SPEC), 'tabm_fixed_epochs': EPOCHS,
        'epoch_selection': 'none; six epochs inherited from incumbent final recipe',
        'tree_parameters': TREE_PARAMS,
        'component_features': {'tabm': 474, 'capacity': 754},
        'preprocessing': 'original train-only RobustPreprocessor clip8 and missing flags; TabM only also smooth clip3',
        'blend': {'tabm_weight': .6, 'capacity_weight': .4, 'signed_power': 1.1,
                  'component_rms_scope': 'entire 38-month forecast vector; raw component predictions',
                  'final_rms_scope': 'entire 38-month transformed blend; no centering',
                  'age_slice_rescaling': False},
        'diagnostic_views': ['tabm_raw', 'capacity_raw', 'capacity_q1p2', 'blend_linear', 'blend_q1p1'],
        'age_horizons': list(HORIZONS),
        'age_slices': 'cumulative age1..h and trailing min(12,h) months ending at h',
        'target_exposure': 'all evaluation history was previously exposed; retrospective diagnosis only',
        'runtime_qualification': 'new GPU training realization with original algorithm; not bitwise old CPU model',
        'scope_limit': 'one origin; cannot establish robustness across independent 38-month regimes',
        'decision': 'report only; no model, blend, power, epoch or submission selection',
        'source_files': {p: file_record(p) for p in SOURCE_PATHS},
        'input_files': {p: file_record(p) for p in INPUT_PATHS},
    }


def freeze():
    frozen = protocol()
    path = OUT / 'protocol.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['contract'] != frozen:
            raise RuntimeError('Frozen source/input/recipe changed; use a separate diagnostic generation')
    else:
        write_json(path, {'frozen_utc': utc(), 'contract': frozen})
    return frozen


def month_slice(months, first, last):
    start = int(np.searchsorted(months, first, side='left'))
    stop = int(np.searchsorted(months, last, side='right'))
    if not np.array_equal(np.unique(months[start:stop]), np.arange(first, last + 1)):
        raise ValueError(f'Incomplete months {first}..{last}')
    return slice(start, stop)


def load_inputs():
    table = feather.read_table(ROOT / INPUT_PATHS[-1], columns=['sample_id', 'month', 'target'])
    ids, months, target = (table[k].to_numpy(zero_copy_only=False) for k in ('sample_id', 'month', 'target'))
    if not np.array_equal(ids, np.arange(len(ids))) or np.any(np.diff(months) < 0):
        raise ValueError('Label row identity or chronology mismatch')
    if not np.isfinite(target).all():
        raise ValueError('Non-finite target')
    tr, va = month_slice(months, 0, 20), month_slice(months, 21, 58)
    if tr.stop != va.start:
        raise ValueError('Training and forecast rows must form disjoint chronological slices')
    n_total = len(target)
    # All downstream labels stop at 58. The six-epoch recipe never sees any
    # validation target during fitting or training-progress reporting.
    months = np.array(months[:va.stop], copy=True)
    target = np.array(target[:va.stop], dtype=np.float64, copy=True)
    del table, ids
    matrices, schemas = {}, {}
    for stem, width in (('base_only', 474), ('base_plus_sequence_all', 754)):
        matrix = np.load(FEATURES / f'train_{stem}.npy', mmap_mode='r', allow_pickle=False)
        names = json.loads((FEATURES / f'train_{stem}.names.json').read_text())
        kinds = json.loads((FEATURES / f'train_{stem}.kinds.json').read_text())
        if matrix.shape != (n_total, width) or matrix.dtype != np.float32:
            raise ValueError(f'Unexpected matrix schema for {stem}')
        if len(names) != width or len(kinds) != width or len(set(names)) != width:
            raise ValueError(f'Unexpected names/kinds for {stem}')
        if {'sample_id', 'month', 'target'} & set(names):
            raise ValueError('Forbidden feature')
        matrices[stem], schemas[stem] = matrix, (names, kinds)
    for index in (0, 1):
        if schemas['base_plus_sequence_all'][index][:474] != schemas['base_only'][index]:
            raise ValueError('Base schema differs between component caches')
    return matrices, schemas, months, target, tr, va


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Authorized wall-clock budget reached')


def status(stage, **kwargs):
    write_json(OUT / 'status.json', {'status': 'running', 'stage': stage, 'pid': os.getpid(),
                                     'updated_utc': utc(), **kwargs})


def fit_tabm(raw, schema, target, tr, va, deadline):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This frozen run expects an available CUDA GPU')
    device = torch.device('cuda')
    torch.set_num_threads(6)
    prep = _make_preprocessor(*schema)
    prep.fit(raw[tr])
    x_train = np.ascontiguousarray(prep.transform(raw[tr]))
    _smooth_clip_inplace(x_train, SPEC.smooth_clip_scale)
    mean, std = float(target[tr].mean()), float(target[tr].std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Degenerate training target')
    y_train = ((target[tr] - mean) / std).astype(np.float32)
    _set_determinism(torch, SEED)
    model = _build_tabm_mini(torch, x_train.shape[1], SPEC).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    xt, yt, data_device, cached = _cache_training_matrix(torch, x_train, y_train, device)
    transformed_features = x_train.shape[1]
    del x_train, y_train
    gc.collect()
    history = []
    for epoch in range(1, EPOCHS + 1):
        check_deadline(deadline)
        started = time.monotonic()
        loss = _run_epoch(torch, model=model, optimizer=optimizer, scaler=scaler,
                          X_tensor=xt, y_tensor=yt, data_device=data_device, device=device,
                          spec=SPEC, use_amp=True)
        if not np.isfinite(loss):
            raise FloatingPointError('Non-finite member training loss')
        row = {'epoch': epoch, 'member_mse': loss, 'seconds': time.monotonic() - started}
        history.append(row)
        print(json.dumps({'stage': 'tabm', **row}), flush=True)
        status('tabm', epoch=epoch, history=history)
    del xt, yt, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    # Chunked feature transformation keeps the 38-month forecast off GPU RAM.
    prediction = np.empty(va.stop - va.start, dtype=np.float64)
    for start in range(va.start, va.stop, 65536):
        check_deadline(deadline)
        stop = min(start + 65536, va.stop)
        xx = np.ascontiguousarray(prep.transform(raw[start:stop]))
        _smooth_clip_inplace(xx, SPEC.smooth_clip_scale)
        members = _predict_members(torch, model, xx, device=device,
                                   batch_size=SPEC.inference_batch_size, k=SPEC.k, use_amp=True)
        # Match the incumbent's inverse target scaling per member then mean.
        prediction[start-va.start:stop-va.start] = (members.astype(np.float64) * std + mean).mean(axis=1)
    torch.save({'state_dict': model.state_dict(), 'spec': asdict(SPEC), 'fixed_epochs': EPOCHS,
                'input_dim': transformed_features, 'target_mean': mean, 'target_std': std,
                'train_months': [0, 20], 'source': 'new GPU realization of incumbent recipe'}, OUT / 'tabm_checkpoint.pt')
    joblib.dump(prep, OUT / 'tabm_preprocessor.joblib')
    info = {'device': str(device), 'torch_version': torch.__version__, 'cuda': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(device), 'amp': True, 'cached_on_device': cached,
            'torch_module': str(Path(torch.__file__).resolve()),
            'target_mean': mean, 'target_std': std, 'transformed_features': transformed_features,
            'fixed_epochs': EPOCHS, 'history': history}
    del model, prep
    gc.collect()
    torch.cuda.empty_cache()
    return prediction, info


def fit_capacity(raw, schema, target, tr, va, deadline):
    check_deadline(deadline)
    status('capacity_preprocessing')
    prep = RobustPreprocessor(feature_names=schema[0], feature_kinds=schema[1], clip=8.,
                              add_missing_indicators=True, output_dtype=np.float32)
    prep.fit(raw[tr])
    x_train = prep.transform(raw[tr])
    model = lightgbm.LGBMRegressor(**TREE_PARAMS)
    def deadline_callback(_):
        check_deadline(deadline)
    model.fit(x_train, target[tr], callbacks=[deadline_callback])
    transformed_features = x_train.shape[1]
    del x_train
    gc.collect()
    prediction = np.empty(va.stop - va.start, dtype=np.float64)
    for start in range(va.start, va.stop, 65536):
        check_deadline(deadline)
        stop = min(start + 65536, va.stop)
        prediction[start-va.start:stop-va.start] = model.predict(prep.transform(raw[start:stop]))
    model.booster_.save_model(str(OUT / 'capacity_model.txt'))
    joblib.dump(prep, OUT / 'capacity_preprocessor.joblib')
    return prediction, {'lightgbm_version': lightgbm.__version__, 'parameters': TREE_PARAMS,
                        'transformed_features': transformed_features}


def rms(p):
    p = np.asarray(p, dtype=np.float64)
    result = float(np.sqrt(np.mean(p * p)))
    if not np.isfinite(result) or result <= 0:
        raise ValueError('Non-finite or zero component norm')
    return result


def make_views(tabm, capacity):
    if tabm.shape != capacity.shape or not np.isfinite(tabm).all() or not np.isfinite(capacity).all():
        raise ValueError('Invalid component prediction vectors')
    nt, nc = rms(tabm), rms(capacity)
    linear = .6 * (tabm / nt) + .4 * (capacity / nc)
    transformed = np.sign(linear) * np.abs(linear) ** 1.1
    scale = rms(transformed)
    return {'tabm_raw': tabm, 'capacity_raw': capacity,
            'capacity_q1p2': np.sign(capacity) * np.abs(capacity) ** 1.2,
            'blend_linear': linear / rms(linear), 'blend_q1p1': transformed / scale}, {
                'tabm_raw_rms': nt, 'capacity_raw_rms': nc,
                'transformed_blend_rms': scale, 'scope': 'entire months21..58 vector; fixed for all reported slices'}


def cosine(y, p):
    denominator = np.linalg.norm(y) * np.linalg.norm(p)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Invalid cosine inputs')
    return float(np.dot(y, p) / denominator)


def score_rows(months, target, predictions):
    rows = []
    for horizon in HORIZONS:
        for kind, first in (('cumulative', 1), ('trailing_up_to_12', max(1, horizon - 11))):
            mask = (months >= ORIGIN + first) & (months <= ORIGIN + horizon)
            for name, p in predictions.items():
                rows.append({'slice': kind, 'age_end': horizon, 'age_start': first,
                             'month_start': ORIGIN + first, 'month_end': ORIGIN + horizon,
                             'months': horizon-first+1, 'rows': int(mask.sum()), 'model': name,
                             'cosine': cosine(target[mask], p[mask]),
                             'prediction_rms': rms(p[mask]),
                             'target_sum_squares': float(np.dot(target[mask], target[mask]))})
    monthly = []
    for month in np.unique(months):
        mask = months == month
        for name, p in predictions.items():
            monthly.append({'month': int(month), 'age': int(month)-ORIGIN, 'rows': int(mask.sum()),
                            'model': name, 'cosine': cosine(target[mask], p[mask]),
                            'prediction_rms': rms(p[mask]), 'target_rms': rms(target[mask]),
                            'target_sum_squares': float(np.dot(target[mask], target[mask])),
                            'prediction_sum_squares': float(np.dot(p[mask], p[mask])),
                            'target_prediction_sum': float(np.dot(target[mask], p[mask]))})
    return rows, monthly


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    # Nonconstant norms catch accidental per-age normalizations in the scorer.
    months = np.repeat(np.arange(21, 59), 3)
    a = np.linspace(-2, 3, len(months))
    b = np.cos(np.arange(len(months)) / 7) + .4
    y = .2 * a - .1 * b + .05
    views, scales = make_views(a, b)
    np.testing.assert_allclose(rms(views['blend_q1p1']), 1., atol=1e-14)
    expected = .6 * a / rms(a) + .4 * b / rms(b)
    expected = np.sign(expected) * np.abs(expected) ** 1.1
    np.testing.assert_allclose(views['blend_q1p1'], expected / rms(expected), atol=1e-14)
    rows, monthly = score_rows(months, y, views)
    assert len(rows) == 80 and len(monthly) == 190
    terminal = [r for r in rows if r['age_end']==38 and r['slice']=='trailing_up_to_12']
    assert all(r['month_start']==47 and r['month_end']==58 and r['months']==12 for r in terminal)
    np.testing.assert_allclose([r['cosine'] for r in terminal if r['model']=='blend_q1p1'][0],
                               cosine(y[months>=47], views['blend_q1p1'][months>=47]), atol=1e-14)
    assert month_slice(months, 21, 58) == slice(0, len(months))
    print('Self-test passed: frozen recipe algebra, fixed-scale age slicing, complete month coverage.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--hours', type=float, default=3.)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 0 < args.hours <= 8:
        raise ValueError('Budget must be in (0,8] hours')
    if (OUT / 'summary.json').exists():
        raise FileExistsError('Completed incumbent age diagnostic already exists')
    frozen = freeze()
    if args.freeze_only:
        print('Frozen incumbent age38 source, input hashes and fixed recipe; no training.', flush=True)
        return
    # Exclusive creation prevents accidental concurrent training or automatic
    # reruns after partial failure. Inspect status/process before manual recovery.
    with (OUT / 'run_started.json').open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'started_utc': utc(), 'hours': args.hours}, handle)
    started = time.monotonic()
    deadline = started + args.hours * 3600
    try:
        status('loading_inputs')
        matrices, schemas, months, target, tr, va = load_inputs()
        tabm, tabm_info = fit_tabm(matrices['base_only'], schemas['base_only'], target, tr, va, deadline)
        np.save(OUT / 'tabm_raw.npy', tabm, allow_pickle=False)
        write_json(OUT / 'tabm_summary.json', tabm_info)
        capacity, capacity_info = fit_capacity(matrices['base_plus_sequence_all'],
                                               schemas['base_plus_sequence_all'], target, tr, va, deadline)
        predictions, scales = make_views(tabm, capacity)
        eval_months, eval_target = months[va], target[va]
        age_rows, monthly = score_rows(eval_months, eval_target, predictions)
        np.savez_compressed(OUT / 'predictions.npz', row_indices=np.arange(va.start, va.stop),
                            months=eval_months, target=eval_target, prediction=predictions['blend_q1p1'],
                            **predictions)
        write_csv(OUT / 'age_slices.csv', age_rows)
        write_csv(OUT / 'monthly.csv', monthly)
        summary = {
            'status': 'complete_retrospective_incumbent_recipe_age38_diagnostic', 'completed_utc': utc(),
            'elapsed_seconds': time.monotonic()-started, 'origin': ORIGIN,
            'training_months': [0, 20], 'forecast_months': [21, 58],
            'training_rows': tr.stop-tr.start, 'forecast_rows': va.stop-va.start,
            'pooled_cosine': {name: cosine(eval_target, p) for name, p in predictions.items()},
            'normalization': scales, 'age_slices': age_rows, 'tabm': tabm_info, 'capacity': capacity_info,
            'qualification': frozen['runtime_qualification'], 'scope_limit': frozen['scope_limit'],
            'target_exposure': frozen['target_exposure'], 'decision': frozen['decision'],
            'protocol': file_record('artifacts/v3/incumbent_age38/protocol.json'),
            'predictions': file_record('artifacts/v3/incumbent_age38/predictions.npz'),
            'python_version': platform.python_version(), 'numpy_version': np.__version__,
        }
        write_json(OUT / 'summary.json', summary)
        write_json(OUT / 'status.json', {'status': 'complete', 'completed_utc': utc()})
        print(json.dumps({'status': summary['status'], 'pooled_cosine': summary['pooled_cosine']}), flush=True)
    except BaseException as exc:
        write_json(OUT / 'status.json', {'status': 'failed', 'failed_utc': utc(), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
