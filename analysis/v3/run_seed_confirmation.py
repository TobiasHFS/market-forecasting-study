"""Bounded second-seed follow-up; leaves the original experiment generation intact."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_bounded_experiments as runner

ROOT = runner.ROOT
OUT = ROOT / 'artifacts/v3/seed_confirmation'
SEED = 20260912
CANDIDATES = ('tabm_base', 'tabm_path')
FOLDS = ('Dev1', 'Dev2', 'Dev3')
runner.OUT = OUT
runner.SEED = SEED


def protocol():
    paths = [Path(runner.__file__), Path(__file__),
             ROOT / 'analysis/v3/compare_experiments.py',
             ROOT / 'analysis/v3/compare_seed_confirmation.py']
    return {
        'version': 1, 'seed': SEED, 'candidates': CANDIDATES,
        'folds': {f: runner.FOLDS[f] for f in FOLDS},
        'max_queue_seconds': 2700,
        'runner_plan': {**runner.PLAN, 'seed': SEED},
        'sources': {str(p.relative_to(ROOT)): runner.digest(p) for p in paths},
        'selection_context': 'Follow-up selected after first-seed results. All evaluation months were previously exposed. This is a seed robustness check, not independent blind confirmation.',
        'comparison': 'Compare base and path in each seed and equal-weight means of the two seeds. Normalize each member by its RMS over the identical pooled out-of-fold vector before averaging. Normalize that mean by RMS before its fixed 20 percent contribution to the archived incumbent. No weight or power search.',
        'uncertainty': 'Use existing paired month-block helper: 10000 replicates at block lengths 1, 3, and 6. Intervals are conditional retrospective evidence, not selection-adjusted uncertainty.',
        'promotion': 'No submission generation or automatic promotion.'
    }


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    current = json.loads(json.dumps(protocol()))
    path = OUT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise RuntimeError('Frozen source or protocol changed')
    else:
        runner.save(path, current)
    inventory_path = OUT / 'input_provenance.json'
    prior = json.loads((ROOT / 'artifacts/v3/bounded_experiments/input_provenance.json').read_text())
    inputs = {}
    for name, reference in prior['inputs'].items():
        actual = runner.digest(ROOT / name)
        if actual != reference['sha256']:
            raise RuntimeError('Changed original input: ' + name)
        inputs[name] = reference
    for candidate in CANDIDATES:
        for fold in FOLDS:
            path = ROOT / f'artifacts/v3/bounded_experiments/{candidate}_{fold}/predictions.npz'
            inputs[str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': runner.digest(path)}
    if runner.digest(ROOT / 'submission_final.csv') != prior['submission_sha256']:
        raise RuntimeError('Canonical submission changed')
    if inventory_path.exists():
        old = json.loads(inventory_path.read_text())
        if old['inputs'] != inputs:
            raise RuntimeError('Frozen input identity changed')
    else:
        runner.save(inventory_path, {'recorded_unix': time.time(),
            'timing': 'Before all second-seed training jobs; first-seed results already observed.',
            'inputs': inputs, 'submission_sha256': prior['submission_sha256'],
            'software': prior['software'], 'hardware': prior['hardware']})


def verify():
    import joblib
    import numpy as np
    import torch
    freeze()
    checked = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for candidate in CANDIDATES:
        x, names, kinds, months, y = runner.load_inputs(candidate)
        for fold in FOLDS:
            dest = OUT / f'{candidate}_{fold}'
            summary = json.loads((dest / 'summary.json').read_text())
            if summary['status'] != 'complete':
                raise AssertionError('Incomplete experiment')
            if runner.digest(dest / 'predictions.npz') != summary['predictions_sha256']:
                raise AssertionError('Saved predictions changed')
            with np.load(dest / 'predictions.npz') as d:
                ids, target, p = d['row_indices'], d['target'], d['prediction']
                assert np.array_equal(target, y[ids])
                assert np.array_equal(d['months'], months[ids])
                assert np.all(np.isfinite(p))
                assert abs(runner.cosine(target, p) - summary['pooled_cosine']) < 1e-12
                for row in summary['monthly']:
                    ix = d['months'] == row['month']
                    assert int(ix.sum()) == row['rows']
                    assert abs(runner.cosine(target[ix], p[ix]) - row['cosine']) < 1e-12
                n = min(2048, len(ids)); first = int(ids[0]); expected = p[:n].copy()
            checkpoint = torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=True)
            prep = joblib.load(dest / 'preprocessor.joblib')
            matrix = runner.smooth(prep.transform(x[first:first+n]))
            model = runner.make_model(torch, candidate, checkpoint['input_dim']).to(device)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            actual = runner.predict(torch, model, matrix, device) * checkpoint['target_std'] + checkpoint['target_mean']
            error = float(np.max(np.abs(actual - expected)))
            if error > 1e-6:
                raise AssertionError(f'Checkpoint replay failed: {candidate}/{fold}: {error}')
            checked.append({'candidate': candidate, 'fold': fold, 'rows_replayed': n,
                            'max_abs_error': error, 'tolerance': 1e-6})
            del model, matrix, checkpoint, prep
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del x
    result = {'status': 'passed', 'model_replays': checked,
        'all_full_and_monthly_metrics_recomputed': True,
        'all_predictions_aligned_to_original_labels': True,
        'frozen_sources_and_input_hashes_verified': True,
        'submission_unchanged': True}
    runner.save(OUT / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--candidate', choices=CANDIDATES)
    parser.add_argument('--fold', choices=FOLDS)
    parser.add_argument('--remaining-seconds', type=float)
    args = parser.parse_args()
    if args.verify:
        verify(); return
    if args.candidate is not None:
        if args.fold is None or args.remaining_seconds is None or args.remaining_seconds <= 0:
            raise ValueError('Child requires a fold and positive remaining budget')
        if json.loads((OUT / 'protocol.json').read_text()) != json.loads(json.dumps(protocol())):
            raise RuntimeError('Frozen source or protocol changed')
        dest = OUT / f'{args.candidate}_{args.fold}'
        if (dest / 'summary.json').exists():
            raise FileExistsError('Completed experiment already exists')
        status = dest / 'status.json'
        runner.save(status, {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(), 'seed': SEED})
        try:
            runner.run(args.candidate, args.fold, time.monotonic() + args.remaining_seconds)
            runner.save(status, {'status': 'complete', 'completed_unix': time.time(), 'seed': SEED})
        except BaseException as exc:
            runner.save(status, {'status': 'failed', 'error': repr(exc), 'completed_unix': time.time()})
            raise
        return
    freeze()
    if args.freeze_only:
        print('Second-seed protocol and input provenance frozen before training.', flush=True)
        return
    queue = OUT / 'queue_status.json'
    if queue.exists():
        raise FileExistsError('Queue already exists; inspect it before starting another generation')
    started = time.monotonic(); deadline = started + 2700
    status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
              'max_seconds': 2700, 'completed_jobs': []}
    runner.save(queue, status)
    try:
        for fold in FOLDS:
            for candidate in CANDIDATES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Shared 45-minute queue deadline reached')
                dest = OUT / f'{candidate}_{fold}'; dest.mkdir(parents=True, exist_ok=True)
                print(f'Starting {candidate} {fold}, seed {SEED}.', flush=True)
                command = [sys.executable, '-u', str(Path(__file__)), '--candidate', candidate,
                           '--fold', fold, '--remaining-seconds', str(remaining)]
                with (dest / 'run.log').open('w', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                if result.returncode != 0:
                    raise RuntimeError(f'{candidate} {fold} failed with code {result.returncode}; inspect its run.log')
                summary = json.loads((dest / 'summary.json').read_text())
                status['completed_jobs'].append({'candidate': candidate, 'fold': fold,
                    'elapsed_seconds': summary['elapsed_seconds'], 'cosine': summary['pooled_cosine'],
                    'selected_epoch': summary['selected_epoch']})
                runner.save(queue, status)
                print(json.dumps(status['completed_jobs'][-1]), flush=True)
        status.update(status='complete', finished_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        runner.save(queue, status)
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), finished_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        runner.save(queue, status)
        raise


if __name__ == '__main__':
    main()
