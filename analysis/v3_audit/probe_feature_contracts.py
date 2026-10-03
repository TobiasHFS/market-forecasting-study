"""Read-only source/cache audit; writes only this audit's JSON evidence."""
from pathlib import Path
import json
import sys
import ast
import math
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.analysis_deps'))
sys.path.insert(0, str(ROOT / 'analysis' / 'v2'))
import numpy as np
source = ast.parse((ROOT/'analysis'/'v2'/'sequence_features.py').read_text(encoding='utf-8'))
selected = []
for node in source.body:
    if isinstance(node,ast.FunctionDef) and node.name in {'_time_bin','_valid_reference','_market_kernel'}:
        node.decorator_list = []
        selected.append(node)
namespace = {'np':np,'math':math,'nb':SimpleNamespace(prange=range),'N_BINS':10,
             'BIN_WIDTH_SECONDS':6.0,'HORIZON_SECONDS':60.0,'N_MARKET_CHANNELS':11}
exec(compile(ast.Module(body=selected,type_ignores=[]),'<sequence_features pure Python audit>','exec'),namespace)
seq = SimpleNamespace(**namespace)


def stats(x):
    finite = np.isfinite(x)
    return {'nan_rate': float(np.mean(~finite)), 'positive_rate': float(np.mean(x > 0)),
            'mean': float(np.nanmean(x)), 'max': float(np.nanmax(x))}


result = {'scope': '50,000 evenly spaced cached samples per split; synthetic calls use Python kernel body'}
for split in ('train', 'test'):
    names = json.loads((ROOT/'artifacts'/'features'/f'{split}_market.names.json').read_text())
    m = np.load(ROOT/'artifacts'/'features'/f'{split}_market.npy', mmap_mode='r')
    ix = np.linspace(0, m.shape[0]-1, 50000, dtype=np.int64)
    m = np.asarray(m[ix])
    relevant = [n for n in names if n.startswith('ref_') or any(t in n for t in ('inversion','newest_age','nonpositive','crossed','book_valid_frac'))]
    result[split] = {'market': {n:stats(m[:, names.index(n)]) for n in relevant}}
    mid = m[:,names.index('ref_mid')]
    spread = m[:,names.index('ref_spread')]
    good = np.isfinite(mid)&(mid>0)&np.isfinite(spread)&(spread>np.maximum(1e-12,np.abs(mid)*1e-8))
    result[split]['invalid_terminal_reference_count'] = int((~good).sum())
    path = np.load(ROOT/'artifacts'/'v2'/'features'/f'{split}_market_path.npy',mmap_mode='r')
    path = np.asarray(path[ix])
    result[split]['invalid_ref_nonempty_bin_count'] = int(((path[:,0]>0)&~good).sum())
    result[split]['invalid_ref_mid_offset_zero_count'] = int(((path[:,0]>0)&~good&(path[:,1]==0)).sum())
    result[split]['invalid_ref_ofi_nonzero_count'] = int(((path[:,0]>0)&~good&(path[:,7]!=0)&np.isfinite(path[:,7])).sum())
    result[split]['empty_latest_6s_bin_rate'] = float(np.mean(path[:,0]==0))
    flow=np.load(ROOT/'artifacts'/'v2'/'features'/f'{split}_flow_path.npy',mmap_mode='r')
    flow=np.asarray(flow[ix])
    for source,offset,width,vwap in [('order',0,11,8),('trade',110,6,4)]:
        counts=flow[:,offset:offset+10*width:width]
        offsets=flow[:,offset+vwap:offset+10*width:width]
        impacted=(counts>0)&(~good[:,None])
        result[split][source+'_invalid_ref_populated_bins']=int(impacted.sum())
        result[split][source+'_invalid_ref_zero_vwap_bins']=int((impacted&(offsets==0)).sum())
    del flow
    del m, path

# Valid -> invalid zero quote -> same valid quote. Correct OFI must be zero
# when invalid quote rows are skipped. A zero predecessor creates false pressure.
f = np.float32
i = np.int32
output=np.zeros((1,seq.N_BINS*seq.N_MARKET_CHANNELS),dtype=f)
seq._market_kernel(np.array([0,3]), np.array([9,5,1],dtype=f),
    np.full(3,np.nan,dtype=f),np.zeros(3,dtype=i),np.zeros(3,dtype=i),
    np.array([101,0,101],dtype=f),np.array([10,0,10],dtype=i),
    np.array([99,0,99],dtype=f),np.array([10,0,10],dtype=i),
    np.ones(3,dtype=i),np.ones(3,dtype=i),np.array([100],dtype=f),
    np.array([2],dtype=f),np.array([20],dtype=f),output)
result['synthetic_zero_quote_predecessor']={'expected_latest_bin_ofi':0.0,'actual_latest_bin_ofi':float(output[0,7])}

# A locked terminal book invalidates spread-unit normalization, but current
# implementation leaves initialized zeros for those undefined measurements.
output=np.zeros((1,seq.N_BINS*seq.N_MARKET_CHANNELS),dtype=f)
seq._market_kernel(np.array([0,1]),np.array([1],dtype=f),
    np.array([100],dtype=f),np.array([2],dtype=i),np.array([1],dtype=i),
    np.array([100],dtype=f),np.array([10],dtype=i),np.array([100],dtype=f),
    np.array([10],dtype=i),np.array([5],dtype=i),np.array([5],dtype=i),
    np.array([100],dtype=f),np.array([0],dtype=f),np.array([20],dtype=f),output)
result['synthetic_locked_reference']={'mid_offset':float(output[0,1]), 'microprice_offset':float(output[0,2]),'trade_vwap_offset':float(output[0,8]),'expected':'NaN for undefined spread-unit features'}
target=Path(__file__).with_name('feature_probe_results.json')
target.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
