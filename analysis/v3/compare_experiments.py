"""Compare the fixed queue after completion; descriptive, with no promotion."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.v3_deps'))
import numpy as np

OUT=ROOT/'artifacts/v3/bounded_experiments'
CANDIDATES=('tabm_base','tabm_path','temporal_path','native_tree')


def score(y,p):
    return float(np.dot(y,p)/np.sqrt(np.dot(y,y)*np.dot(p,p)))


def unit(p):return p/np.sqrt(np.mean(p*p))


def load(paths,key='prediction'):
    arrays=[]
    for path in paths:
        with np.load(ROOT/path) as d:
            arrays.append({k:d[k] for k in ('row_indices','months','target')})
            arrays[-1]['prediction']=d[key]
    return {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}


def paired(months,y,p,q):
    # Fixed recipes, no bootstrap refitting or scale re-estimation.
    mm=np.unique(months)
    s=np.array([[np.dot(y[months==m],y[months==m]),
                 np.dot(p[months==m],p[months==m]),np.dot(y[months==m],p[months==m]),
                 np.dot(q[months==m],q[months==m]),np.dot(y[months==m],q[months==m])]
                for m in mm])
    answer={}
    for block in (1,3,6):
        rng=np.random.default_rng(20260911+block)
        starts=rng.integers(0,len(mm)-block+1,size=(10000,int(np.ceil(len(mm)/block))))
        ix=(starts[:,:,None]+np.arange(block)).reshape(10000,-1)[:,:len(mm)]
        a=s[ix].sum(axis=1)
        delta=a[:,2]/np.sqrt(a[:,0]*a[:,1])-a[:,4]/np.sqrt(a[:,0]*a[:,3])
        lo,hi=np.quantile(delta,[.025,.975])
        answer[str(block)]={'lower95':float(lo),'upper95':float(hi)}
    return {'delta':score(y,p)-score(y,q),'conditional_month_block_intervals':answer}


def main():
    sources={};data={};rows=[]
    for c in CANDIDATES:
        paths=[]
        for f in ('Dev1','Dev2','Dev3'):
            path=OUT/f'{c}_{f}'
            summary=json.loads((path/'summary.json').read_text())
            if summary['status']!='complete':raise ValueError('Incomplete queue')
            p=path/'predictions.npz'
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=summary['predictions_sha256']:raise ValueError('Prediction hash mismatch')
            sources[str(p.relative_to(ROOT))]=actual
            paths.append(p.relative_to(ROOT))
            rows.append({'candidate':c,'fold':f,'cosine':summary['pooled_cosine'],
                         'seconds':summary['elapsed_seconds'],'selected_epoch':summary.get('selected_epoch')})
        data[c]=load(paths)
    capacity=load([Path('artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_Dev1-Dev2_oof.npz'),
                   Path('artifacts/v2/experiments/sequence_base_plus_sequence_all_capacity_Dev3_oof.npz')])
    tabm=load([Path(f'artifacts/v2/tabm_mini/dev{i}_predictions.npz') for i in (1,2,3)])
    ref=data['tabm_base'];months=ref['months'];y=ref['target']
    for candidate in [*data.values(),capacity,tabm]:
        for key in ('row_indices','months','target'):
            if not np.array_equal(candidate[key],ref[key]):raise ValueError('Alignment mismatch')
        if not np.all(np.isfinite(candidate['prediction'])):raise ValueError('Nonfinite predictions')
    lin=.6*unit(tabm['prediction'])+.4*unit(capacity['prediction'])
    incumbent=unit(np.sign(lin)*np.abs(lin)**1.1)
    comparisons={
        'tabm_path_minus_tabm_base':paired(months,y,data['tabm_path']['prediction'],data['tabm_base']['prediction']),
        'temporal_path_minus_tabm_path':paired(months,y,data['temporal_path']['prediction'],data['tabm_path']['prediction']),
        'native_tree_minus_archived_capacity':paired(months,y,data['native_tree']['prediction'],capacity['prediction'])}
    pooled=[]
    for c,values in data.items():
        p=values['prediction']
        # This one illustrative 20% weight is fixed in this script before queue results.
        blend=.8*incumbent+.2*unit(p)
        comparisons[f'fixed20pct_{c}_minus_incumbent']=paired(months,y,blend,incumbent)
        pooled.append({'candidate':c,'pooled_cosine':score(y,p),
                       'correlation_with_incumbent':float(np.corrcoef(p,incumbent)[0,1]),
                       'fixed20pct_blend_cosine':score(y,blend),
                       'positive_month_share':float(np.mean([score(y[months==m],p[months==m])>0 for m in np.unique(months)]))})
    result={'status':'complete','scope':'previously exposed months23-58; one seed; descriptive postselection uncertainty',
            'incumbent_pooled_cosine':score(y,incumbent),'archived_capacity_pooled_cosine':score(y,capacity['prediction']),
            'fold_rows':rows,'pooled':pooled,'comparisons':comparisons,'sources':sources,
            'interpretation':'Native tree is a joint tail/missingness preprocessing ablation. New TabM uses a new seed and up to16 inner epochs; compare base/path with each other. Fixed20pct blends are illustrative, not selected weights. No new submission generated.'}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
