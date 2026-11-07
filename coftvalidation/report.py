import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def bootstrap(values):
    a=np.array(values,dtype=float); rng=np.random.default_rng(4242)
    estimates=np.mean(a[rng.integers(0,len(a),(2000,len(a)))],axis=1)
    return dict(mean=float(a.mean()),ci95=np.quantile(estimates,[.025,.975]).tolist(),n_scenes=len(a))

def report(out):
    out=Path(out)
    with (out/'factorial.csv').open() as f: rows=list(csv.DictReader(f))
    with (out/'baselines.csv').open() as f: baselines=list(csv.DictReader(f))
    test=[r for r in rows if r['split']=='test' and r['blocked']=='True']; ids=sorted({r['scene'] for r in test})
    if not ids:
        (out/'REPORT.md').write_text('Development smoke only; no held-out statistics.\n')
        return
    stats={}; differences=[]
    def value(sc,robot,mount,key):
        r=[x for x in test if x['scene']==sc and x['robot']==robot and x['mount']==mount]
        return np.array([float(x[key]) if key=='visible_fraction' else x[key]=='True' for x in r])
    for robot in ('panda','sr5'):
        for mount in ('mount_a','mount_b'):
            for key in ('ik_found','feasible_proxy','observable_proxy','conjunction_proxy'):
                stats[f'{robot}/{mount}/{key}']=bootstrap([value(s,robot,mount,key).mean() for s in ids])
            stats[f'{robot}/{mount}/conflict_with_alternative']=bootstrap([
                float(np.any(value(s,robot,mount,'feasible_proxy') & ~value(s,robot,mount,'observable_proxy')) and
                      np.any(value(s,robot,mount,'conjunction_proxy'))) for s in ids])
    for s in ids:
        ps=value(s,'panda','mount_a','feasible_proxy'); ts=value(s,'sr5','mount_a','feasible_proxy')
        pa=value(s,'panda','mount_a','conjunction_proxy').mean(); pb=value(s,'panda','mount_b','conjunction_proxy').mean()
        sa=value(s,'sr5','mount_a','conjunction_proxy').mean(); sb=value(s,'sr5','mount_b','conjunction_proxy').mean()
        differences.append(dict(scene=s,source_valid_target_invalid=int(np.sum(ps & ~ts)),reverse=int(np.sum(ts & ~ps)),
                                arm_feasibility_delta=float(ts.mean()-ps.mean()),factorial_interaction=float((sb-sa)-(pb-pa))))
    stats['arm_feasibility_delta']=bootstrap([d['arm_feasibility_delta'] for d in differences])
    stats['factorial_conjunction_interaction']=bootstrap([d['factorial_interaction'] for d in differences])
    stats['source_valid_target_invalid_candidates']=sum(d['source_valid_target_invalid'] for d in differences)
    stats['reverse_candidates']=sum(d['reverse'] for d in differences)
    # Independent experimental units are scenes; compare directions of mean paired difference.
    pos=sum(d['arm_feasibility_delta']>0 for d in differences); neg=sum(d['arm_feasibility_delta']<0 for d in differences)
    stats['exploratory_scene_sign_test']=dict(positive=pos,negative=neg,p=float(binomtest(pos,pos+neg).pvalue) if pos+neg else None)
    for robot in ('panda','sr5'):
        for mount in ('mount_a','mount_b'):
            names=sorted({r['baseline'] for r in baselines})
            for name in names:
                b=[r for r in baselines if r['split']=='test' and r['blocked']=='True' and r['robot']==robot and r['mount']==mount and r['baseline']==name]
                stats[f'baseline/{robot}/{mount}/{name}']=bootstrap([r['selected_conjunction_proxy']=='True' for r in b])
    failures={}
    for r in test:
        if r['mount']!='mount_a': continue
        for reason in filter(None,r['failures'].split('|')):
            k=f"{r['robot']}/{reason}"; failures[k]=failures.get(k,0)+1
    stats['failure_candidate_counts']=failures
    (out/'statistics.json').write_text(json.dumps(stats,indent=2))
    with (out/'paired_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(differences[0])); w.writeheader(); w.writerows(differences)
    # Independent split summaries: do not pool candidates as statistical replicates.
    grouped=[]
    for seed in sorted({r['seed'] for r in test}):
        for family in sorted({r['family'] for r in test}):
            for robot in ('panda','sr5'):
                a=[r for r in test if r['seed']==seed and r['family']==family and r['robot']==robot and r['mount']=='mount_a']
                grouped.append(dict(seed=seed,family=family,robot=robot,n_candidates=len(a),
                                    feasible_proxy_fraction=sum(r['feasible_proxy']=='True' for r in a)/len(a),
                                    conjunction_proxy_fraction=sum(r['conjunction_proxy']=='True' for r in a)/len(a)))
    with (out/'strata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(grouped[0])); w.writeheader(); w.writerows(grouped)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4)); labels=[]; values=[]
    for robot in ('panda','sr5'):
        for mount in ('mount_a','mount_b'):
            for metric in ('feasible_proxy','observable_proxy','conjunction_proxy'):
                labels.append(f'{robot}\n{mount}\n{metric.replace("_proxy", "")}'); values.append(stats[f'{robot}/{mount}/{metric}']['mean'])
    ax.bar(range(len(values)),values); ax.set_xticks(range(len(values)),labels,fontsize=7); ax.set_ylim(0,1)
    ax.set_ylabel('Candidate geometric proxy fraction'); ax.set_title('PROXY_ONLY; 27 held-out scene clusters; no task success outcomes')
    fig.tight_layout(); fig.savefig(out/'factorial.png',dpi=170); fig.savefig(out/'factorial.svg'); plt.close(fig)
    lines=['# v1 geometric diagnostics','', '**PROXY_ONLY — INSUFFICIENT_EVIDENCE**. No independent execution outcomes.', '', '| Metric | Mean | Scene bootstrap 95% CI |','|---|---:|---|']
    for key,v in stats.items():
        if isinstance(v,dict) and 'mean' in v:
            lines.append(f"| {key} | {v['mean']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
    lines+=['','Analytic AND, geometric planning and finite-menu MPC use the evaluation conjunction itself: these are coverage / selection consistency diagnostics, never independent algorithm success.','',f'Failures: `{json.dumps(failures)}`']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    import sys
    report(sys.argv[1])
