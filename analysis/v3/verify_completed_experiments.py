"""Check source/input identity and replay saved models on evaluation rows."""
import gc
import json
from pathlib import Path

import run_bounded_experiments as runner
import numpy as np
import joblib
import lightgbm
import torch

ROOT=runner.ROOT
OUT=runner.OUT


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert runner.digest(Path(runner.__file__))==protocol['source_sha256']
    inventory=json.loads((OUT/'input_provenance.json').read_text())
    checked=[]
    for path,reference in inventory['inputs'].items():
        actual=runner.digest(ROOT/path)
        if actual!=reference['sha256']:raise AssertionError('Changed input: '+path)
        checked.append(path)
    assert runner.digest(ROOT/'submission_final.csv')==inventory['submission_sha256']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay=[]
    for candidate in runner.PLAN['candidates']:
        x,names,kinds,months,y=runner.load_inputs(candidate)
        for fold in ('Dev1','Dev2','Dev3'):
            dest=OUT/f'{candidate}_{fold}'
            summary=json.loads((dest/'summary.json').read_text())
            assert summary['status']=='complete'
            assert runner.digest(dest/'predictions.npz')==summary['predictions_sha256']
            with np.load(dest/'predictions.npz') as d:
                ids=d['row_indices'];target=d['target'];p=d['prediction']
                assert np.array_equal(target,y[ids])
                assert np.array_equal(d['months'],months[ids])
                assert abs(runner.cosine(target,p)-summary['pooled_cosine'])<1e-12
                first=int(ids[0]);n=min(2048,len(ids));expected=p[:n].copy()
            raw=x[first:first+n]
            if candidate=='native_tree':
                matrix=np.array(raw,dtype=np.float32);matrix[~np.isfinite(matrix)]=np.nan
                for j,k in enumerate(kinds):
                    if k=='log1p_nonnegative':matrix[matrix[:,j]<0,j]=np.nan
                model=lightgbm.Booster(model_file=str(dest/'model.txt'))
                actual=model.predict(matrix)
            else:
                checkpoint=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True)
                prep=joblib.load(dest/'preprocessor.joblib')
                matrix=runner.smooth(prep.transform(raw))
                model=runner.make_model(torch,candidate,checkpoint['input_dim']).to(device)
                model.load_state_dict(checkpoint['state_dict'],strict=True)
                actual=runner.predict(torch,model,matrix,device)*checkpoint['target_std']+checkpoint['target_mean']
            error=float(np.max(np.abs(actual-expected)))
            tolerance=1e-6 if candidate!='native_tree' else 1e-12
            if error>tolerance:raise AssertionError(f'{candidate}/{fold} replay error {error}')
            replay.append({'candidate':candidate,'fold':fold,'rows_replayed':n,'max_abs_error':error,'tolerance':tolerance})
            del model,matrix;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        del x
    result={'status':'passed','source_and_input_hashes_checked':checked,'model_replays':replay,
            'all_predictions_aligned_to_original_labels':True,'all_full_metrics_recomputed':True,
            'submission_unchanged':True,'scope':'All saved evaluation vectors verified; first2048 evaluation rows per model replayed from checkpoint. No independence claim.'}
    runner.save(OUT/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
