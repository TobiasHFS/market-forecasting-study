"""Count undefined-reference effects over all cached samples without training.

This is a census of existing cache semantics, not of raw invalid quote
transitions. It leaves all existing caches and the earlier sample probe intact.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.v3_deps'))
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.monotonic()
    previous=json.loads((ROOT/'analysis/v3_audit/feature_probe_results.json').read_text())
    result={'scope':'Full population of existing train/test cache rows; no raw-transition or model-score claim.',
            'script_sha256':digest(Path(__file__)),
            'invalid_reference_definition':'Nonfinite or nonpositive mid; nonfinite spread or spread <= max(1e-12, abs(mid)*1e-8).',
            'sources':{},'splits':{}}
    for split in ('train','test'):
        paths=[ROOT/'artifacts/features'/f'{split}_market.npy',
               ROOT/'artifacts/v2/features'/f'{split}_market_path.npy',
               ROOT/'artifacts/v2/features'/f'{split}_flow_path.npy']
        names_path=paths[0].with_suffix('.names.json')
        names=json.loads(names_path.read_text())
        market,path,flow=[np.load(p,mmap_mode='r',allow_pickle=False) for p in paths]
        assert len(market)==len(path)==len(flow)
        mid_col,spread_col=names.index('ref_mid'),names.index('ref_spread')
        invalid=np.empty(len(market),dtype=bool)
        stats={'rows':len(market),'invalid_terminal_reference_count':0,
               'empty_latest_6s_bin_count':0,'invalid_ref_latest_market_populated_bins':0}
        for source in ('order','trade'):
            for suffix in ('populated_bins','invalid_ref_populated_bins','invalid_ref_zero_vwap_bins','invalid_ref_samples_with_populated_bin'):
                stats[source+'_'+suffix]=0
        for a in range(0,len(market),8192):
            b=min(a+8192,len(market))
            mid=np.asarray(market[a:b,mid_col]);spread=np.asarray(market[a:b,spread_col])
            bad=~(np.isfinite(mid)&(mid>0)&np.isfinite(spread)&(spread>np.maximum(1e-12,np.abs(mid)*1e-8)))
            invalid[a:b]=bad
            stats['invalid_terminal_reference_count']+=int(bad.sum())
            stats['empty_latest_6s_bin_count']+=int((path[a:b,0]==0).sum())
            stats['invalid_ref_latest_market_populated_bins']+=int(((path[a:b,0]>0)&bad).sum())
            for source,offset,width,vwap in [('order',0,11,8),('trade',110,6,4)]:
                counts=flow[a:b,offset:offset+10*width:width]
                values=flow[a:b,offset+vwap:offset+10*width:width]
                populated=counts>0;impacted=populated&bad[:,None]
                stats[source+'_populated_bins']+=int(populated.sum())
                stats[source+'_invalid_ref_populated_bins']+=int(impacted.sum())
                stats[source+'_invalid_ref_zero_vwap_bins']+=int((impacted&(values==0)).sum())
                stats[source+'_invalid_ref_samples_with_populated_bin']+=int(impacted.any(axis=1).sum())
        ix=np.linspace(0,len(market)-1,50000,dtype=np.int64)
        sample_count=int(invalid[ix].sum())
        assert sample_count==previous[split]['invalid_terminal_reference_count']
        stats['original_50000_row_probe_reproduced']=True
        stats['invalid_terminal_reference_rate']=stats['invalid_terminal_reference_count']/len(market)
        stats['empty_latest_6s_bin_rate']=stats['empty_latest_6s_bin_count']/len(market)
        for source in ('order','trade'):
            assert stats[source+'_invalid_ref_zero_vwap_bins']<=stats[source+'_invalid_ref_populated_bins']<=stats[source+'_populated_bins']
            assert stats[source+'_invalid_ref_samples_with_populated_bin']<=stats['invalid_terminal_reference_count']
        result['splits'][split]=stats
        del market,path,flow
        for p in [*paths,names_path]:
            result['sources'][p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':digest(p)}
    result['elapsed_seconds']=time.monotonic()-started
    out=ROOT/'artifacts/v3_audit/full_cache_reference_census.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['splits'],indent=2))


if __name__=='__main__':main()
