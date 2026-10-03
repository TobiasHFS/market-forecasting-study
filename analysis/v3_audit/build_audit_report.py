"""Build a source-backed technical assessment and canonical MCP report input."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import sqlite3

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/v3_audit'
TITLE='MSCapital forecasting: model audit and next experiments'


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    evidence=read('artifacts/v3_audit/retrospective_metrics.json')
    stress=read('artifacts/v3/incumbent_age38/summary.json')
    cp=ROOT/'artifacts/v3/bounded_experiments/comparison.json'
    comparison=json.loads(cp.read_text()) if cp.exists() else None
    verification=read('artifacts/v3/bounded_experiments/verification.json')
    if verification['status'] != 'passed':
        raise ValueError('Model/checkpoint verification must pass before report delivery')
    queue=read('artifacts/v3/bounded_experiments/queue_status.json')
    if queue['status'] != 'complete' or comparison is None:
        raise ValueError('The complete comparison queue is required for this final report')
    seed_path=ROOT/'artifacts/v3/seed_confirmation/comparison.json'
    seed_comparison=json.loads(seed_path.read_text()) if seed_path.exists() else None
    if seed_comparison:
        if read('artifacts/v3/seed_confirmation/verification.json')['status'] != 'passed':
            raise ValueError('Second-seed verification is required before reporting its results')
    generated=datetime.now(timezone.utc).isoformat()
    sources=[
        {'id':'retrospective','label':'Recomputed aligned validation predictions','path':'artifacts/v3_audit/retrospective_metrics.json',
         'query':{'description':'Recompute uncentered cosine, monthly sufficient statistics and paired moving-month-block intervals from archived predictions.',
                  'tables_used':['analysis/v3_audit/recompute_evidence.py','artifacts/v3_audit/retrospective_metrics.json','artifacts/v3_audit/retrospective_monthly.csv',*list(evidence['sources'])],
                  'transformation':{'language':'Python','script':'analysis/v3_audit/recompute_evidence.py','entrypoint':'run','description':'Align saved row indices/months/targets, compute prediction views, dot-product cosine and paired monthly resampling. Original NPZ hashes are recorded in retrospective_metrics.json.'},
                  'metric_definitions':['Cosine = sum(y*p)/sqrt(sum(y*y)*sum(p*p)). Intervals are conditional retrospective postselection uncertainty.']}},
        {'id':'validation','label':'Source and holdout-provenance audit','path':'analysis/v3_audit/validation_audit.md'},
        {'id':'features','label':'Feature-source audit and synthetic reproductions','path':'analysis/v3_audit/feature_audit.md'},
        {'id':'reference_census','label':'Full-cache undefined-reference census','path':'artifacts/v3_audit/full_cache_reference_census.json'},
        {'id':'raw_labels','label':'Independent original-label alignment','path':'artifacts/v3_audit/raw_label_alignment.json'},
        {'id':'stress','label':'Incumbent recipe: one fit, 38 forecast months','path':'artifacts/v3/incumbent_age38/summary.json',
         'query':{'description':'Train once on months0-20, forecast21-58. Preserve complete-vector RMS scales for every age slice.',
                  'tables_used':['artifacts/v3/incumbent_age38/summary.json','artifacts/v3/incumbent_age38/monthly.csv']}},
        {'id':'runtime','label':'Verified hardware, libraries and measured workflow times','path':'artifacts/v3/bounded_experiments/input_provenance.json',
         'query':{'description':'Hardware/runtime inventory plus recorded pilot and incumbent-stress elapsed times.',
                  'tables_used':['artifacts/v3/bounded_experiments/input_provenance.json','artifacts/v3/bounded_experiments/tabm_base_Dev1/summary.json','artifacts/v3/bounded_experiments/queue_status.json','artifacts/v3/incumbent_age38/summary.json']}},
        {'id':'experiments','label':'Fixed GPU and native-tree comparisons','path':'artifacts/v3/bounded_experiments/comparison.json'},
        {'id':'verification','label':'Original-input hashes, label alignment and checkpoint replay','path':'artifacts/v3/bounded_experiments/verification.json'},
        {'id':'shift','label':'Existing domain-shift diagnostics','path':'artifacts/diagnostics/postmortem/domain_shift_summary.json'},
        {'id':'rules','label':'Official competition data and restrictions','href':'https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/data'},
        {'id':'research','label':'Primary-source methods research','path':'analysis/v3_audit/methods_research.md'}]
    if seed_comparison:
        sources.append({'id':'seeds','label':'Second-seed and fixed two-seed comparisons','path':'artifacts/v3/seed_confirmation/comparison.json',
                        'query':{'description':'Fixed pooled-vector and two-seed comparisons with saved checkpoint replay.',
                                 'tables_used':['analysis/v3/compare_seed_confirmation.py','artifacts/v3/seed_confirmation/comparison.json','artifacts/v3/seed_confirmation/verification.json','artifacts/v3/seed_confirmation/protocol.json']}})
    blocks=[{'id':'title','type':'markdown','body':'# '+TITLE}]
    sections=[]
    datasets={};charts=[];tables=[]
    def query_rows(id, data, parent_source, order_by):
        # Execute the exact SQL recorded in provenance. Python computations
        # remain attributed to their original files and reproducible scripts.
        original=next(s for s in sources if s['id']==parent_source)
        sql=f'SELECT * FROM "{id}" ORDER BY "{order_by}"'
        with sqlite3.connect(':memory:') as db:
            db.row_factory=sqlite3.Row
            fields=list(data[0])
            db.execute(f'CREATE TABLE "{id}" ('+', '.join('"'+k+'"' for k in fields)+')')
            db.executemany(f'INSERT INTO "{id}" VALUES ('+','.join('?' for _ in fields)+')',
                           [tuple(row[k] for k in fields) for row in data])
            queried=[dict(row) for row in db.execute(sql)]
        source_id=id+'_source'
        sources.append({'id':source_id,'label':original['label'],'path':original.get('path'),
                        'query':{'engine':'SQLite','language':'SQL','sql':sql,
                                 'description':'Final display sorting only. Scores and uncertainty were calculated upstream in the cited Python pipeline from original saved predictions; this SQLite query is not the metric computation.',
                                 'tables_used':['main.'+id,original.get('path',''),'analysis/v3_audit/build_audit_report.py',*original.get('query',{}).get('tables_used',[])],
                                 'upstream_transformation':original.get('query',{}).get('transformation',{'description':'Read the original cited result and its recorded prediction/source hashes.'}),
                                 'metric_definitions':['Uncentered cosine over the indicated prediction/target vectors. Monthly or cumulative slices are descriptive.']}})
        return queried,source_id
    def paragraph(id,title,body,source=None):
        text='## '+title+'\n\n'+body
        entry={'id':id,'type':'markdown','body':text}
        if source:entry['sourceId']=source
        blocks.append(entry)
        if source:
            origin=next(s for s in sources if s['id']==source)
            target=(ROOT/origin['path']).as_posix() if origin.get('path') else origin.get('href')
            if target:
                text+='\n\nEvidence: ['+origin['label']+'](<'+target+'>).'
        sections.append(text)
    def table(id,title,data,columns,source,sort):
        data,source=query_rows(id,data,source,sort)
        datasets[id]=data
        tables.append({'id':id,'title':title,'dataset':id,'sourceId':source,'density':'spacious',
                       'defaultSort':{'field':sort,'direction':'asc'},
                       'columns':[{'field':f,'label':l,'type':t} for f,l,t in columns]})
        blocks.append({'id':id+'_block','type':'table','tableId':id})
        sections.append('| '+' | '.join(l for _,l,_ in columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+
                        '\n'.join('| '+' | '.join(str(r.get(f,'')) for f,_,_ in columns)+' |' for r in data))
    def line(id,title,data,x,y,group,source,subtitle):
        data,source=query_rows(id,data,source,x)
        datasets[id]=data
        charts.append({'id':id,'title':title,'subtitle':subtitle,'type':'line','intent':'trend','dataset':id,'sourceId':source,
                       'encodings':{'x':{'field':x,'type':'quantitative','label':'Month index'},
                                    'y':{'field':y,'type':'quantitative','label':'Uncentered cosine'},
                                    'color':{'field':group,'type':'nominal','label':'Model'}},
                       'layout':'full','legend':{'position':'bottom'},
                       'palette':{'kind':'categorical','colors':['#2563eb','#b7791f']}})
        blocks.append({'id':id+'_block','type':'chart','chartId':id})

    paragraph('summary','Technical summary',
        '**The newer model has credible incremental signal, but its historical validation was less independent than the earlier reports implied.** '
        'The supplied screenshot shows public scores **0.124 → 0.133**, not 1.333. I found no direct target leak in the reviewed training paths. '
        'I did find repeated inspection of both the purported sealed period and the later confirmation period. Their scores are retrospective research evidence.\n\n'
        '**Neural signal and increased tree capacity explain most improvement; blending adds a smaller gain.** Recomputed development cosine is 0.133238 for v1, '
        '0.143780 for raw TabM and 0.145297 for the final v2 blend. The blend adds 0.001517 over the stronger standalone model. '
        'Its improvement over the tree survives removing an unusually influential late month. The final power transform has much weaker support.\n\n'
        '**Your hardware is sufficient for disciplined, modern experiments.** CUDA now works on the RTX 3060 Ti. The first complete GPU selection/refit pilot took 78 seconds. '
        'The bounded comparisons and 38-month incumbent stress test below turn several plausible ideas into measured evidence. '
        'No result can certify a frontier private-leaderboard rank, and the existing submission is preserved.')
    paragraph('definition','What is being predicted and measured',
        'There are **1,257,637 labeled samples across months 0-70**, and **647,896 test samples across months 71-108**. '
        'Market bars provide about 600 seconds of history; raw order and trade flows provide about 60 seconds. '
        'The return-generation method, target horizon, absolute timestamps and instrument identities are unavailable. '
        'Saved targets were independently checked against the original label file, with exact matches in all 12 target-bearing archives.\n\n'
        'The [official metric](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/overview) is uncentered cosine: '
        '`sum(prediction × target) / sqrt(sum(prediction²) × sum(target²))`. A positive global rescale has no effect; relative amplitudes do. '
        'Pooled cosine is the selection metric, while monthly scores diagnose stability. Monthly scores must not simply be averaged and presented as the official score.\n\n'
        'The [leaderboard](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/leaderboard) confirms approximately 49% public and 51% private. '
        'Their chronological ordering is not disclosed: the private partition cannot be assumed to be the last 51% of months. '
        'The public score remains a useful external check. Its sample fraction alone neither makes it reliable for selecting tiny improvements nor makes it irrelevant; market dependence and partition composition matter.')
    paragraph('leakage','The main validation flaw is repeated selection on exposed periods',
        'The training implementation does several important things correctly: chronological outer splits; training-only robust preprocessing; '
        'neural epoch selection using only the last three months inside the training window; discarding that inner fit and retraining on the complete outer training slice. '
        'Using all labeled months for the final test refit is appropriate. Loading a full label array into memory is not itself leakage if only permitted slices enter fitting.\n\n'
        'The independence claim fails at the project level. Months **59-70** were evaluated for v1, v2 slow trees, v2 capacity trees and the final neural blend. '
        'The saved tree summaries include six power-transform scores on that same period. Months **47-58** had already been used for earlier feature, capacity and calibration decisions. '
        'A new per-script freeze does not make these labels unseen again. A keep/fallback “safety veto” is also a bounded form of model selection.\n\n'
        'Consequently, retain the scores but retire the descriptions “untouched Dev3” and “one-time sealed holdout.” '
        'The audit does not prove that any particular parameter was chosen by maximizing late-period scores; it proves that those periods were exposed. '
        'Confidence intervals below condition on already-chosen recipes and do not correct for the whole research search.\n\n'
        'No precise overlap purge can be certified without timestamps, instrument IDs and target support. A one-month-gap sensitivity test is a conservative diagnostic, '
        'not proof of a correct purge and not a substitute for the missing metadata.','validation')
    scoremap=evidence['scopes']['development_23_58']['cosine']
    names=[('v1_slow_raw','V1 slow LightGBM'),('base_capacity_raw','Capacity tree, base inputs'),
           ('path_capacity_raw','Capacity tree, base + bins'),('tabm_raw','Original TabM, base inputs'),
           ('blend_linear','Original 60/40 linear blend'),('blend_q1p1','Submitted blend, power 1.1')]
    paragraph('attribution','Capacity and model diversity explain most of the gain',
        'On the same 639,120 development rows, increasing tree capacity adds **0.005022** cosine, while adding the 280 bin features at fixed capacity adds **0.000753**. '
        'About 87% of the raw tree improvement came from capacity. The bins still contain ordered information: ten six-second slices, flattened into columns. '
        'They are coarsened temporal inputs, not a full raw-history neural model.\n\n'
        'The following ledger separates architecture, input information and final calibration. These historical comparisons establish where the existing gain came from; '
        'they do not constitute independent model-selection trials. Relative to raw TabM, the final blend adds **0.001517**, '
        'with a conditional three-month-block interval **[0.000728, 0.002366]**. That is useful complementarity, but most of the gain over v1 was already present in TabM alone.','retrospective')
    table('attribution_table','Development cosine, months 23-58',
          [{'order':i,'model':label,'cosine':round(scoremap[k],6),'rows':639120} for i,(k,label) in enumerate(names)],
          [('model','Model','text'),('cosine','Cosine','number'),('rows','Rows','number')],'retrospective','cosine')
    paragraph('monthly','The blend gain is broader than one favorable month',
        'The chart compares monthly v1 and submitted-blend scores. It reveals the variation hidden by a pooled headline; it is not a forecast of test-month performance. '
        'For late months 59-70, the blend scores **0.155564**. Removing month 66 reduces it to **0.143779**. '
        'That month holds about a quarter of the late block’s target squared norm, so the higher late headline is a poor universal expectation.\n\n'
        'Even excluding month 66, the blend improves over capacity q=1.2 by **0.004594**, with a conditional three-month-block interval of approximately **[0.001528, 0.006969]**. '
        'The blending gain has positive intervals across the tested 1-, 3- and 6-month block lengths. '
        'By contrast, the tiny power-1.1 improvement over the linear blend has intervals spanning zero. '
        'Block resampling respects month-level dependence better than independent-row intervals, but cannot remove model-selection bias or represent unseen regimes.','retrospective')
    chartrows=[]
    with (ROOT/'artifacts/v3_audit/retrospective_monthly.csv').open() as f:
        for r in csv.DictReader(f):
            if r['scope'] in ('development_23_58','late_59_70') and r['model'] in ('v1_slow_raw','blend_q1p1'):
                chartrows.append({'month':int(r['month']),'model':'V1' if r['model']=='v1_slow_raw' else 'Submitted V2 blend',
                                  'cosine':float(r['cosine']),'rows':int(r['rows']),
                                  'target_energy':float(r['target_sum_squares']),'prediction_rms':float(r['prediction_rms'])})
    line('monthly_chart','Monthly validation cosine',chartrows,'month','cosine','model','retrospective',
         'Previously exposed months 23-70; each fold was predicted by an earlier-trained model.')
    paragraph('feature_defects','Four feature defects are reproducible, but score impact is unproven',
        '1. **Invalid OFI predecessor:** a zero or crossed quote can become the previous book state. A valid → zero → unchanged-valid synthetic sequence produces false normalized imbalance +0.5 instead of zero.\n'
        '2. **Missing reference encoded as zero:** undefined spread-normalized displacements become ordinary zeroes. A complete cached-row census found 5,627 / 1,257,637 training samples (**0.447%**) and 4,538 / 647,896 test samples (**0.700%**) with invalid terminal references. The corresponding populated flow bins all have zero VWAP displacement.\n'
        '3. **Incorrect event-time interpretation:** “contemporaneous” mid-price is a bin average, sometimes using a quote later than the historical event or a terminal-mid fallback. This is look-ahead within supplied history, not access to data after the prediction timestamp. It may still encode valid subsequent price response, but it is not an as-of execution reference.\n'
        '4. **Mismatched denominators:** invalid price/reference rows can be omitted from numerators while their volume remains in denominators, diluting averages.\n\n'
        'New isolated NumPy helpers fix predecessor validity, missing references, backward as-of joins and matching-mask averages. '
        '**All 16 tests passed**, including future-quote invariance and sample isolation. These are reference implementations: '
        'they have not been integrated into large-scale extraction or used to claim a new submission score. Frozen v2 artifacts remain intact.','features')
    paragraph('census','Undefined references affect under 1% of samples, more often in test',
        'The full-cache census independently reproduces the earlier 50,000-row probes and records all source hashes. '
        'Affected populated order/trade bins number **49,889 / 35,950 in training** and **41,931 / 32,914 in test**. '
        'This establishes frequency for the undefined-reference issue, not its predictive impact. The models also receive other reference and missingness information, '
        'so a malformed zero is not proof that the entire sample is unusable.\n\n'
        'This census does not count raw invalid OFI transitions or every numerator/denominator mismatch; those require scanning the underlying event histories. '
        'Feature correctness is a justified engineering priority, but these relatively uncommon references should not be presented as the demonstrated main cause of the leaderboard gap.','reference_census')
    paragraph('shift','Distribution shift is real; its effect on alpha is not identified',
        'Existing diagnostics distinguish training from test with AUC **0.820**, and recent training from test with AUC **0.748**. '
        'Order-event density rises about **36.4%** and transaction density about **32.4%** in test. '
        'Yet early versus late training is even more distinguishable, at AUC **0.857**. '
        'A domain classifier demonstrates covariate differences; it does not prove target-concept drift or establish that every differing feature is harmful.\n\n'
        'Prefer clock-time windows, relative price/depth quantities and explicit missingness. Keep useful activity context, but test its contribution. '
        'Removing every absolute-scale feature or weighting by an adversarial classifier without controlled evidence could discard signal.','shift')
    age=stress['pooled_cosine']
    paragraph('age','The full incumbent remains useful through one 38-month deployment',
        f'The original recipe was refitted once on months 0-20 and predicted months 21-58 without refresh. '
        f'The submitted blend recipe scored **{age["blend_q1p1"]:.6f}**, versus **{age["capacity_raw"]:.6f}** for the raw capacity tree '
        f'and **{age["tabm_raw"]:.6f}** for raw TabM. The linear blend scored **{age["blend_linear"]:.6f}**, slightly above power 1.1.\n\n'
        'Complete-vector component scales were fixed before all age slices. The final 12 months score **0.138939** for the submitted recipe. '
        'This closes the earlier omission of the neural component from long-horizon diagnostics. It is one origin, with only 21 training months, '
        'and a new GPU training realization - not the original CPU weights and not an estimate of the 71-month-trained private-test score.','stress')
    agedata=[]
    with (ROOT/'artifacts/v3/incumbent_age38/monthly.csv').open() as f:
        stressmonthly=list(csv.DictReader(f))
    # The native chart uses exact recorded age slices; eight temporal anchors per model.
    for r in stress['age_slices']:
        if r['slice']=='cumulative' and r['model'] in ('blend_q1p1','capacity_raw'):
            agedata.append({'age_months':r['age_end'],'model':'Incumbent blend' if r['model']=='blend_q1p1' else 'Capacity tree',
                            'cosine':r['cosine'],'rows':r['rows'],'target_energy':r['target_sum_squares']})
    line('age_chart','Cumulative cosine by deployment age',agedata,'age_months','cosine','model','stress',
         'One fixed training origin; cumulative overlapping intervals, not independent experiments.')
    charts[-1]['encodings']['x']['label']='Months after training'
    if comparison:
        pooled=comparison['pooled'];best=max(pooled,key=lambda r:r['pooled_cosine'])
        def contrast(name):
            r=comparison['comparisons'][name];ci=r['conditional_month_block_intervals']['3']
            return f'{r["delta"]:+.6f} (conditional three-month-block interval [{ci["lower95"]:+.6f}, {ci["upper95"]:+.6f}])'
        paragraph('experiments','Controlled local comparisons',
            f'All 12 predeclared candidate/fold runs completed. The strongest new standalone candidate is **{best["candidate"]}**, '
            f'with pooled cosine **{best["pooled_cosine"]:.6f}**. The archived submitted blend is **{comparison["incumbent_pooled_cosine"]:.6f}** on those same rows.\n\n'
            'Base TabM, full-input TabM and the temporal encoder use the same new seed, inner chronological epoch selection and outer refits. '
            'The outer fits are **0-22 → 23-34**, **0-34 → 35-46**, and **0-46 → 47-58**. '
            'TabM has 16 members, two shared width-256 layers and member-wise MSE; the temporal encoder has width-32 convolutions with dilations 1/2/4 plus a width-128 base-feature branch. '
            'The temporal encoder sees exactly the same ten-by-28 bins as full-input TabM plus the base branch. '
            'All appended missingness indicators enter its base branch. Its capacity and ensemble structure differ from TabM, so this compares complete training recipes at equal input information, not an isolated causal effect of convolution. '
            'A weak result would reject this small architecture/training recipe, not all temporal forecasting models. '
            'The native tree changes the whole missingness/tail-preprocessing package; it does not isolate clipping alone. '
            'One illustrative 20% challenger / 80% incumbent blend was specified before these results; its weights were not searched. '
            'The exact recipe is `0.8 × unit_RMS(incumbent) + 0.2 × unit_RMS(challenger)`, with each RMS calculated over the complete pooled evaluation vector and no centering. '
            'A single seed and repeatedly exposed folds limit claims about tiny differences. No candidate is automatically promoted.','experiments')
        paragraph('experiment_contrasts','Separate input information, architecture and preprocessing effects',
            '**Full-input TabM minus base TabM:** '+contrast('tabm_path_minus_tabm_base')+'.\n\n'
            '**Temporal encoder minus full-input TabM:** '+contrast('temporal_path_minus_tabm_path')+'.\n\n'
            '**Native tree minus the archived capacity tree:** '+contrast('native_tree_minus_archived_capacity')+'.\n\n'
            'These paired comparisons, rather than the highest isolated score, determine the next priority. '
            'Intervals describe variation across the observed months conditional on these fitted predictions. '
            'They do not include repeated research selection, new-seed uncertainty or absent market regimes.','experiments')
        table('experiment_table','Pooled candidate results, months 23-58',
              [{'candidate':r['candidate'],'cosine':round(r['pooled_cosine'],6),
                'fixed20pct_blend':round(r['fixed20pct_blend_cosine'],6),
                'correlation':round(r['correlation_with_incumbent'],4)} for r in pooled],
              [('candidate','Candidate','text'),('cosine','Standalone cosine','number'),
               ('fixed20pct_blend','20% blend cosine','number'),('correlation','Correlation with incumbent','number')],
              'experiments','cosine')
        paragraph('experiment_decision','The promising result is a small full-input TabM ensemble contribution',
            'The tested temporal encoder and native-tree preprocessing package do not justify changing the incumbent. '
            'Full-input TabM improves over the new base-input TabM in each fold, but their pooled three-month-block interval still spans zero. '
            'The new full-input standalone score, 0.141572, also remains below the original raw TabM score, 0.143780. '
            'A different seed and inner epoch selection prevent attributing that historical difference solely to the added inputs.\n\n'
            'Lower prediction correlation is not enough to make a useful ensemble member: the temporal model is the least correlated with the incumbent (0.898), '
            'yet its fixed 20% addition does not improve pooled cosine. Its weaker signal offsets the apparent diversity.\n\n'
            'The useful signal is complementarity: adding 20% full-input TabM to 80% of the incumbent raises retrospective pooled cosine '
            'from **0.145297 to 0.146104**, a gain of '+contrast('fixed20pct_tabm_path_minus_incumbent')+'. '
            'Its paired interval is positive at all three tested block lengths. The equivalent base-input addition reaches only 0.145596, '
            'with a three-month interval spanning zero. This first-round result motivated the second-seed check below; it does not establish a new private-leaderboard score.\n\n'
            'Inner-selected epochs vary substantially: 9/16/4 for base TabM and 7/10/4 for full-input TabM. '
            'Three inner months can make stopping noisy. A bounded seed ensemble and a conservatively fixed training schedule deserve priority over a broad epoch or architecture search.','experiments')
    else:
        paragraph('experiments','Controlled comparisons are still running',
                  'The full candidate queue has not completed. No conclusion about the new model ranking is available. This report must be regenerated before final handoff.')
    if seed_comparison:
        def seed_contrast(name):
            r=seed_comparison['comparisons'][name];ci=r['conditional_month_block_intervals']['3']
            return f'{r["delta"]:+.6f}, conditional three-month-block interval [{ci["lower95"]:+.6f}, {ci["upper95"]:+.6f}]'
        paragraph('seed_confirmation','Second-seed robustness and the two-seed blend',
            'Six additional fits repeated base/full-input TabM on all three development folds with seed 20260912 and the same training recipe. '
            'This follow-up was chosen after seeing the first round; its code, input hashes, seed and averaging rules were frozen before these six fits. '
            'It tests sensitivity to training randomness on the same exposed months, not a new independent holdout.\n\n'
            '**Full-input minus base TabM in the second seed:** '+seed_contrast('tabm_path_minus_tabm_base_seed20260912')+'.\n\n'
            '**20% two-seed full-input ensemble minus incumbent:** '+seed_contrast('fixed20pct_tabm_path_mean2_minus_incumbent')+'.\n\n'
            '**20% two-seed base-input ensemble minus incumbent:** '+seed_contrast('fixed20pct_tabm_base_mean2_minus_incumbent')+'.\n\n'
            '**Direct comparison of the two 20% ensemble additions, full inputs minus base inputs:** '+seed_contrast('fixed20pct_path_minus_fixed20pct_base_mean2')+'.\n\n'
            'Each seed is RMS-normalized over its pooled prediction vector, the two seeds are averaged equally, and that mean is RMS-normalized before its fixed 20% contribution. '
            'No weights or powers were searched. All six saved checkpoints reproduce their first 2,048 evaluation rows, and all complete predictions and monthly metrics pass original-label alignment and recomputation checks.','seeds')
        seed_labels={'seed20260910':'First seed','seed20260912':'Second seed','mean2':'Mean of both seeds'}
        seed_rows=[]
        for r in seed_comparison['pooled']:
            candidate,suffix=r['candidate'].rsplit('_',1)
            seed_rows.append({'inputs':'Base + bins' if candidate=='tabm_path' else 'Base',
                              'seed':seed_labels[suffix],'cosine':round(r['pooled_cosine'],6),
                              'blend':round(r['fixed20pct_blend_cosine'],6)})
        table('seed_table','Base versus full inputs across two seeds, months 23-58',seed_rows,
              [('inputs','Inputs','text'),('seed','Training realization','text'),('cosine','Standalone cosine','number'),
               ('blend','20% addition to incumbent','number')],'seeds','blend')
    paragraph('methods','The highest-value frontier is disciplined use of microstructure',
        '**Keep modern tabular models as the anchor.** [TabM](https://arxiv.org/abs/2410.24210) is a credible efficient neural ensemble; '
        '[RealMLP](https://arxiv.org/abs/2407.04491) is a useful later diversity candidate. Benchmark results elsewhere do not imply that a larger neural model wins here.\n\n'
        '**Prioritize stationary flow and correctly aligned event mechanics.** '
        '[Cont, Kukanov and Stoikov](https://arxiv.org/abs/1011.6402) motivate depth-scaled imbalance, largely from contemporaneous price-impact evidence. '
        '[Kolm, Turiel and Westray](https://doi.org/10.1111/mafi.12413) provide more direct forecasting evidence for stationary order-flow inputs. '
        'Our two-level bars and undisclosed target differ from their granular market data, so their gains cannot be transferred numerically.\n\n'
        '**Let richer temporal information earn its compute.** If corrected cached-bin models warrant another round, extend to two compact streams: '
        'roughly 100 bins over 600 seconds of market history and 30-60 bins over 60 seconds of order/trade flow, plus age and validity masks. '
        'A width-32/64 TCN or GRU plus a small tabular branch is a sensible first test. '
        'For that richer sequence experiment, fit shared per-channel scales on training data and provide validity masks directly to the temporal branch. '
        'The current small encoder used the same per-column tabular preprocessing as TabM, with missingness indicators in the base branch; temporal-specific preprocessing has not been tested. '
        '[DeepLOB](https://arxiv.org/abs/1808.03668) supports learning temporal/book structure; copying its deeper, multi-level setup is not a faithful match here. '
        'The [TCN paper](https://arxiv.org/abs/1803.01271) supports an efficient architecture, not financial alpha.\n\n'
        'Stable instrument identities are absent, so sample IDs cannot justify stitching histories into a continuous market stream or building cross-asset relations. '
        'The [official data restrictions](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting/data) prohibit external data and models. '
        'These proposals train from random initialization on competition data. Pretrained forecasting models and external market datasets are excluded.')
    paragraph('objective','Cosine rewards a conditional mean, not automatic risk normalization',
        'Under a fixed population distribution with finite second moments, the optimal prediction direction is proportional to **E[target | features]**. '
        'This follows by conditioning the numerator and applying Cauchy-Schwarz. Ordinary MSE is therefore a principled baseline. '
        'Ranking predictions, centering them, using inverse-variance position sizing or optimizing noisy minibatch cosine can move away from the competition objective.\n\n'
        'If training a volatility-normalized target, restore its known feature-based scale at inference. '
        'A later amplitude experiment should learn a strongly shrunk correction from earlier out-of-fold predictions and apply it to later months. '
        'Do not fit reliability weights on the same labels being evaluated. Given the weak power-transform evidence, this is lower priority than feature correctness and robust model diversity.')
    paragraph('compute','Local compute is adequate; the previous runtime was the bottleneck',
        'The machine has an **RTX 3060 Ti with 8 GB VRAM**, approximately **32 GB RAM**, and **8 cores / 16 threads**. '
        'The old PyTorch build was CPU-only. A separate workspace environment now runs official **PyTorch 2.11.0 + CUDA 12.8**, with real GPU arithmetic and training verified. '
        'The first base-TabM selection/refit pilot completed in **78 seconds**; the incumbent age diagnostic took **133 seconds** including tree training. '
        'These are observed workflow times, not controlled CPU/GPU speedup estimates because recipes and warm-up conditions differ.\n\n'
        'The 754-column float32 training cache is about 3.79 GB before transformed copies and missing indicators. '
        'Run one training process at a time, stream batches where needed and retain memory-mapped caches. '
        f'The 12-run comparison completed with all jobs successful; the serial queue took about **{(queue["finished_unix"]-queue["started_unix"])/60:.1f} minutes**, plus the separately completed 78-second pilot. '
        'It enforced a shared six-hour ceiling. '
        'Data, model and source hashes are recorded; no existing submitted model or prediction file is overwritten.','runtime')
    paragraph('verification','Reproducibility and scope of the completed checks',
        'All 12 new evaluation vectors were checked against the original labels and month/row identities, and their full cosine scores were recomputed. '
        'For each saved model, the first 2,048 evaluation rows were independently predicted from its checkpoint and preprocessor: all reproduced exactly. '
        'The frozen runner, modeling code, input arrays, schemas and raw-label hashes matched. The original submission hash also matched.\n\n'
        'The isolated feature helpers passed 16 targeted tests. Their fixes are not yet connected to the production extractor. '
        'Consequently these experiments measure architecture, input-set and preprocessing differences on the existing cached features; '
        'they do not measure the benefit of corrected extraction.','verification')
    if seed_comparison:
        seed_queue=read('artifacts/v3/seed_confirmation/queue_status.json')
        paragraph('followup_compute','The second seed also fits comfortably within the local budget',
            f'The six follow-up fits completed serially in **{seed_queue["elapsed_seconds"]/60:.1f} minutes**, within their fixed 45-minute limit. '
            'Together with the first comparison queue, separate pilot and 38-month stress fit, the measured training workflows total about **40 minutes**. '
            'That excludes environment installation, code review, research and report preparation; it is not the elapsed time for the whole audit. '
            'The next expensive step is extracting and validating richer histories, not increasing the number of tiny cached-feature trials.','seeds')
    paragraph('next','What to do next, and what I need from you',
        '1. **Use the measured comparison verdict**, with paired monthly gains and model complementarity; preserve the current submission unless an improvement is repeatable. Do not interpret a few extra decimal places as a reliable private-score gain.\n'
        '2. **Integrate the corrected feature helpers into a versioned extractor**, quantify invalid-transition and denominator issues on the full data, and compare corrected versus original features on the same fixed folds. Keep retrospective response features clearly named if retained.\n'
        '3. **Confirm useful candidates with a second seed or conservative fixed ensemble** before expanding architecture or calibration searches. Extend to richer 600/60-second tensors only when the controlled evidence justifies it.\n'
        '4. **Maintain a project-wide exposure ledger.** All already-inspected validation months remain exposed. Any genuinely new labeled block would be valuable; a newly named subset of old data is not new evidence.\n\n'
        '**No data re-upload is needed for the completed audit and comparisons.** The valuable missing information is instrument identity, absolute prediction timestamps and exact target support. '
        'If organizers make those available, they enable precise overlap purging and deduplication. Without them, residual boundary-overlap uncertainty must remain explicit. '
        'The private labels are unavailable by design; this work identifies the best-supported directions, not a guaranteed winning score.')
    artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':TITLE,'generatedAt':generated,
              'blocks':blocks,'sources':sources,'charts':charts,'tables':tables,
              'description':'Technical audit of leakage, features, model evidence, compute and bounded next experiments.'},
              'snapshot':{'version':1,'status':'ready','generatedAt':generated,'datasets':datasets}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report_artifact.json').write_text(json.dumps(artifact,indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'analysis/v3_audit/assessment.md').write_text('# '+TITLE+'\n\n'+'\n\n'.join(sections)+'\n',encoding='utf-8')
    notes={'audience':'technical','delivery':'mcp-app','required_sections':'summary, definitions, findings, model/validation design, uncertainty, recommendations, open questions',
           'chart_contracts':{'monthly_chart':'48 month indices x2 models; two-root comparative line; inspect stability',
                              'age_chart':'8 cumulative age anchors x2 models; nested intervals explicitly marked'},
           'comparisons_complete':comparison is not None,'quantitative_table_rationale':'Exact model score lookup and attribution, alongside monthly shape charts.'}
    notes['second_seed_complete']=seed_comparison is not None
    (OUT/'report_build_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print(json.dumps({'artifact':str(OUT/'report_artifact.json'),'comparisons_complete':comparison is not None,'blocks':len(blocks),'charts':len(charts)}))


if __name__=='__main__':main()
