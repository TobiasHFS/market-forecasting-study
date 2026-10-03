"""Fixed retrospective experiments; never publishes or changes a submission.

Uses the existing, previously exposed historical folds. These are controlled
experiments, not a newly sealed holdout. Run with bundled Python from the root.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / '.analysis_deps',):
    if str(entry) not in sys.path:
        sys.path.append(str(entry))
for entry in (ROOT / 'analysis', ROOT / '.v3_deps', ROOT / '.v3_torch'):
    sys.path.insert(0, str(entry))
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pyarrow.feather as feather
from modeling import RobustPreprocessor

OUT = ROOT / 'artifacts' / 'v3' / 'bounded_experiments'
SEED = 20260910
FOLDS = {'Dev1': (0, 22, 23, 34), 'Dev2': (0, 34, 35, 46),
         'Dev3': (0, 46, 47, 58), 'Age38': (0, 20, 21, 58)}
PLAN = {
    'version': 1, 'seed': SEED, 'folds': FOLDS,
    'candidates': ['native_tree', 'tabm_base', 'tabm_path', 'temporal_path'],
    'feature_source': 'existing v2 cached 474 base + 280 six-second bin features',
    'label_exposure': 'All evaluation months previously exposed; retrospective evidence only.',
    'metric': 'uncentered pooled cosine; raw predictions; no power search',
    'neural': {'inner_months': 3, 'max_epochs': 16, 'min_epochs': 4, 'patience': 4,
               'batch_size': 1024, 'lr': 0.002, 'weight_decay': 0.0003,
               'TabM': '16 member input scales, shared 2x256 ReLU, dropout0.1; member MSE',
               'temporal': 'three causal convolutions width32 dilation1/2/4, base MLP128'},
    'tree': {'n_estimators': 1200, 'learning_rate': .02, 'num_leaves': 31,
             'max_depth': 7, 'min_child_samples': 1000, 'subsample': .8,
             'subsample_freq': 1, 'colsample_bytree': .75, 'reg_alpha': 1.,
             'reg_lambda': 30., 'max_bin': 63, 'objective': 'regression_l2',
             'random_state': 20260823, 'n_jobs': 6, 'verbosity': -1},
    'promotion': 'No automatic promotion. Compare identical rows, each fold and month;'
                 ' inspect paired month-block intervals and age decay; retain current submission.'
}


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cosine(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
        raise ValueError('Invalid scoring vectors')
    den = np.sqrt(np.dot(y, y) * np.dot(p, p))
    return float(np.dot(y, p) / den) if den > 0 else 0.


def month_slice(months, start, end):
    a, b = np.searchsorted(months, [start, end], side='left')
    b = np.searchsorted(months, end, side='right')
    if a == b or months[a] != start or months[b-1] != end:
        raise ValueError('Incomplete chronological slice')
    return slice(int(a), int(b))


def load_inputs(candidate):
    stem = 'base_only' if candidate == 'tabm_base' else 'base_plus_sequence_all'
    path = ROOT / 'artifacts' / 'v2' / 'features' / f'train_{stem}.npy'
    x = np.load(path, mmap_mode='r', allow_pickle=False)
    names = json.loads(path.with_suffix('.names.json').read_text())
    kinds = json.loads(path.with_suffix('.kinds.json').read_text())
    t = feather.read_table(ROOT / 'ms-capital-real-financial-market-forecasting/train/label.feather')
    ids, months, y = (t[k].to_numpy() for k in ('sample_id', 'month', 'target'))
    if not np.array_equal(ids, np.arange(len(ids))) or len(x) != len(ids):
        raise ValueError('Label/feature alignment')
    if np.any(np.diff(months) < 0) or not np.all(np.isfinite(y)):
        raise ValueError('Invalid chronology/labels')
    if x.shape[1] != len(names) or len(names) != len(kinds):
        raise ValueError('Schema mismatch')
    if {'sample_id', 'month', 'target'} & set(names):
        raise ValueError('Forbidden feature')
    return x, names, kinds, months, y.astype(np.float64)


def preprocessor(x, names, kinds, train):
    p = RobustPreprocessor(feature_names=names, feature_kinds=kinds,
                           clip=8., add_missing_indicators=True)
    p.fit(x[train])
    return p


def smooth(x):
    for a in range(0, len(x), 8192):
        b = x[a:a+8192]
        b /= np.sqrt(1 + (b / 3.) ** 2)
    return x


def make_model(torch, candidate, dim):
    nn = torch.nn
    if candidate.startswith('tabm'):
        class TabM(nn.Module):
            def __init__(self):
                super().__init__()
                self.scales = nn.Parameter(torch.randn(16, dim))
                self.backbone = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(.1),
                                              nn.Linear(256, 256), nn.ReLU(), nn.Dropout(.1))
                self.head = nn.Parameter(torch.empty(16, 256))
                self.bias = nn.Parameter(torch.empty(16))
                nn.init.uniform_(self.head, -1/16, 1/16)
                nn.init.uniform_(self.bias, -1/16, 1/16)
            def forward(self, x):
                z = self.backbone(x[:, None, :] * self.scales[None, :, :])
                return (z * self.head).sum(-1) + self.bias
        return TabM()

    class Temporal(nn.Module):
        def __init__(self):
            super().__init__()
            if dim < 754:
                raise ValueError('Temporal input requires all 754 raw features')
            self.conv = nn.ModuleList([nn.Conv1d(28, 32, 3), nn.Conv1d(32, 32, 3, dilation=2),
                                      nn.Conv1d(32, 32, 3, dilation=4)])
            self.base = nn.Sequential(nn.Linear(dim-280, 128), nn.ReLU(), nn.Dropout(.1))
            self.head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))
        def forward(self, x):
            # Cache groups: market(10x11), order(10x11), trade(10x6), newest first.
            path = torch.cat((x[:,474:584].reshape(-1,10,11), x[:,584:694].reshape(-1,10,11),
                              x[:,694:754].reshape(-1,10,6)), dim=2).flip(1).transpose(1,2)
            for dilation, layer in zip((1,2,4), self.conv):
                path = torch.relu(layer(torch.nn.functional.pad(path, (2*dilation,0))))
            tab = self.base(torch.cat((x[:,:474], x[:,754:]), dim=1))
            return self.head(torch.cat((tab,path[:,:,-1],path.mean(-1)),dim=1))
    return Temporal()


def seed_all(torch):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict(torch, model, x, device):
    result = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for a in range(0,len(x),2048):
            z = torch.as_tensor(x[a:a+2048], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                out = model(z)
            result[a:a+len(z)] = out.float().mean(1).cpu().numpy()
    return result


def neural_fit(torch, candidate, x, y, device, epochs, deadline, validation=None, label=''):
    seed_all(torch)
    model = make_model(torch, candidate, x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    yy = np.asarray(y, dtype=np.float32)
    history, best, best_epoch, stale = [], -np.inf, 4, 0
    rng = np.random.default_rng(SEED)
    for epoch in range(1,epochs+1):
        if time.monotonic() > deadline:
            raise TimeoutError('Authorized wall-clock budget reached')
        model.train(); permutation = rng.permutation(len(x)); losses = []
        started = time.monotonic()
        for a in range(0,len(x),1024):
            ix = permutation[a:a+1024]
            xb = torch.as_tensor(x[ix], device=device)
            yb = torch.as_tensor(yy[ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                outputs = model(xb)
            loss = (outputs.float() - yb[:,None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        row = {'epoch':epoch, 'loss':float(np.mean(losses)), 'seconds':time.monotonic()-started}
        if validation is not None:
            xv,yv,mu,sd = validation
            row['inner_cosine'] = cosine(yv,predict(torch,model,xv,device)*sd+mu)
            if epoch >= 4:
                if row['inner_cosine'] > best:
                    best, best_epoch, stale = row['inner_cosine'], epoch, 0
                else:
                    stale += 1
        history.append(row)
        print(json.dumps({'stage':label,**row}),flush=True)
        if validation is not None and stale >= 4:
            break
    return model, best_epoch if validation is not None else epochs, history


def run(candidate, fold, deadline):
    import lightgbm
    import joblib
    started = time.monotonic()
    dest = OUT / f'{candidate}_{fold}'
    dest.mkdir(parents=True,exist_ok=True)
    x,names,kinds,months,y = load_inputs(candidate)
    begin,end,vbegin,vend = FOLDS[fold]
    tr,va = month_slice(months,begin,end),month_slice(months,vbegin,vend)
    extra = {}
    if candidate == 'native_tree':
        # Preserve undefined values for native missing routing, no clipping/imputation.
        xt = np.array(x[tr],dtype=np.float32); xv = np.array(x[va],dtype=np.float32)
        for matrix in (xt,xv):
            for a in range(0,len(matrix),8192):
                block=matrix[a:a+8192]; block[~np.isfinite(block)]=np.nan
            for j,kind in enumerate(kinds):
                if kind == 'log1p_nonnegative':
                    matrix[matrix[:,j]<0,j]=np.nan
        model = lightgbm.LGBMRegressor(**PLAN['tree'])
        model.fit(xt,y[tr]); p=model.predict(xv)
        model.booster_.save_model(str(dest/'model.txt'))
        extra['device']='cpu'
    else:
        import torch
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('DEVICE',str(device),torch.__version__,flush=True)
        inner_train=month_slice(months,begin,end-3); inner_val=month_slice(months,end-2,end)
        prep=preprocessor(x,names,kinds,inner_train)
        xt=smooth(prep.transform(x[inner_train])); xv=smooth(prep.transform(x[inner_val]))
        mu,sd=float(y[inner_train].mean()),float(y[inner_train].std())
        if sd<=0:raise ValueError('Degenerate target')
        model,epoch,inner_history=neural_fit(torch,candidate,xt,(y[inner_train]-mu)/sd,device,16,deadline,
                                            (xv,y[inner_val],mu,sd),f'{candidate}/{fold}/inner')
        del model,xt,xv,prep;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        # Discard inner model and preprocessors, then refit all outer training rows.
        prep=preprocessor(x,names,kinds,tr)
        xt=smooth(prep.transform(x[tr])); xv=smooth(prep.transform(x[va]))
        mu,sd=float(y[tr].mean()),float(y[tr].std())
        model,_,history=neural_fit(torch,candidate,xt,(y[tr]-mu)/sd,device,epoch,deadline,
                                    label=f'{candidate}/{fold}/refit')
        p=predict(torch,model,xv,device)*sd+mu
        torch.save({'state_dict':model.state_dict(),'candidate':candidate,'input_dim':xt.shape[1],
                    'target_mean':mu,'target_std':sd,'epoch':epoch},dest/'checkpoint.pt')
        joblib.dump(prep,dest/'preprocessor.joblib')
        extra.update(device=str(device),torch_version=torch.__version__,selected_epoch=epoch,
                     inner_history=inner_history,refit_history=history)
    selected_months=months[va]
    monthly=[{'month':int(m),'rows':int(np.sum(selected_months==m)),
              'cosine':cosine(y[va][selected_months==m],p[selected_months==m])}
             for m in np.unique(selected_months)]
    result={'status':'complete','candidate':candidate,'fold':fold,'bounds':FOLDS[fold],
            'rows':len(p),'pooled_cosine':cosine(y[va],p),'monthly':monthly,
            'prediction_rms':float(np.sqrt(np.mean(p*p))),
            'elapsed_seconds':time.monotonic()-started,'raw_features':len(names),**extra}
    np.savez_compressed(dest/'predictions.npz',row_indices=np.arange(va.start,va.stop),
                        months=selected_months,target=y[va],prediction=p)
    result['predictions_sha256']=digest(dest/'predictions.npz')
    save(dest/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('monthly','inner_history','refit_history')}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',choices=PLAN['candidates'])
    parser.add_argument('--fold',choices=FOLDS)
    parser.add_argument('--hours',type=float,default=8.)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if not 0<args.hours<=8:raise ValueError('Budget must be between zero and eight hours')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen={**PLAN,'source_sha256':digest(Path(__file__))}
    freeze=OUT/'protocol.json'
    if freeze.exists():
        if json.loads(freeze.read_text())!=json.loads(json.dumps(frozen)):
            raise RuntimeError('Frozen protocol/source changed; use a new experiment generation')
    else:save(freeze,frozen)
    if args.freeze_only:
        print('Protocol frozen',flush=True);return
    if args.candidate is None or args.fold is None:parser.error('Specify candidate and fold')
    summary=OUT/f'{args.candidate}_{args.fold}'/'summary.json'
    if summary.exists():raise FileExistsError('Completed experiment already exists')
    status=OUT/f'{args.candidate}_{args.fold}'/'status.json'
    save(status,{'status':'running','pid':os.getpid(),'started_unix':time.time(),'hours':args.hours})
    try:
        run(args.candidate,args.fold,time.monotonic()+args.hours*3600)
        save(status,{'status':'complete','completed_unix':time.time()})
    except BaseException as exc:
        save(status,{'status':'failed','error':repr(exc),'completed_unix':time.time()})
        raise


if __name__=='__main__':main()
