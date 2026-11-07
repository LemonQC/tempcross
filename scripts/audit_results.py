"""Post-run completeness and exploratory achieved-FK visibility audit.
No additional IK, parameter tuning, filtering or outcome fabrication.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coftvalidation.report import bootstrap

def audit(out):
    out=Path(out); data=json.loads((out/'trials.json').read_text() if (out/'trials.json').exists() else gzip.decompress((out/'trials.json.gz').read_bytes())); meta=json.loads((out/'manifest.json').read_text())
    expected=len(meta['seeds'])*meta['n_per_seed']*4
    assert len(data)==expected
    assert len({(d['scene'],d['blocked'],d['robot']) for d in data})==expected
    assert all(len(d['candidates'])==8 for d in data)
    assert all(r['independent_execution_outcome'] is None and r['continuous_collision_certificate'] is False for d in data for r in d['candidates'])
    for name,h in meta['source_hashes'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    indexed={(d['scene'],d['blocked'],d['robot']):d for d in data}
    transfers=[]; diagnostics=[]; failure_examples={}
    for d in data:
        for r in d['candidates']:
            for reason in r['failures']:
                key=d['robot']+'/'+reason
                failure_examples.setdefault(key,dict(scene=d['scene'],blocked=d['blocked'],candidate=r['candidate']))
        if d['robot']!='panda': continue
        other=indexed[(d['scene'],d['blocked'],'sr5')]
        for s,t in zip(d['candidates'],other['candidates']):
            for mount in ('mount_a','mount_b'):
                vs=(s['desired_visibility'][mount] or 0)>=.6; vt=(t['desired_visibility'][mount] or 0)>=.6
                assert vs==vt # Controlled desired optical paths do not depend on the joint chain.
                transfers.append(dict(scene=d['scene'],split=d['split'],blocked=d['blocked'],candidate=s['candidate'],mount=mount,
                                      source_ik=s['ik_found'],target_ik=t['ik_found'],source_feasible=s['feasible_proxy'],target_feasible=t['feasible_proxy'],
                                      source_valid_target_invalid=s['feasible_proxy'] and not t['feasible_proxy'],
                                      source_valid_target_invalid_reasons='|'.join(t['failures']) if s['feasible_proxy'] and not t['feasible_proxy'] else '',
                                      controlled_observable=vs))
    for d in data:
        for mount in ('mount_a','mount_b'):
            known=[r for r in d['candidates'] if r['actual_visibility'][mount] is not None]
            diagnostics.append(dict(scene=d['scene'],split=d['split'],blocked=d['blocked'],robot=d['robot'],mount=mount,
                                    known_actual_views=len(known),
                                    desired_visible_actual_invisible=sum((r['desired_visibility'][mount] or 0)>=.6 and r['actual_visibility'][mount]<.6 for r in known),
                                    feasible_achieved_conjunction=sum(r['feasible_proxy'] and r['actual_visibility'][mount]>=.6 for r in known)))
    for name,rows in [('transfers.csv',transfers),('achieved_visibility_audit.csv',diagnostics)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Pure-kinematic contrasts separate search/tracking from collision proxies.
    test=[d for d in data if d['split']=='test' and d['blocked']]
    ids=sorted({d['scene'] for d in test}); pure=[]; scene_sign=[]
    for s in ids:
        p=indexed[(s,True,'panda')]['candidates']; t=indexed[(s,True,'sr5')]['candidates']
        pure.append(sum(r['ik_found'] for r in t)/8-sum(r['ik_found'] for r in p)/8)
    result=dict(complete=True,scene_robot_obstacle_records=len(data),candidate_robot_obstacle_records=sum(len(d['candidates']) for d in data),
                optical_factorial_candidate_records=sum(len(d['candidates'])*2 for d in data),
                source_hashes_match=True,independent_execution_outcomes=0,
                pure_kinematic_search_delta_sr5_minus_panda=bootstrap(pure),failure_examples=failure_examples,
                actual_visibility_exploratory=True,
                visibility_note='Main factorial uses desired path to isolate mount from IK. Achieved FK+robot proxy occlusion is an exploratory geometric diagnostic, not an independent success endpoint.')
    (out/'integrity.json').write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__': print(json.dumps(audit(sys.argv[1]),indent=2))
