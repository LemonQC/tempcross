"""Offline only: no controller imports, connections or hardware commands."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .model import Arm, ROOT
from .scene import scenes, candidates, dense_path, interpolate
from .geometry import TCP, MOUNTS, collision, visibility
from coftprobe.geometry import inverse


def evaluate(arm,sc,can,blocked=True,joint_step=.04):
    boxes=sc['boxes'] if blocked else []
    path=dense_path(can); qs=[]; attempts=[]; previous=None
    for pose,carried in path:
        ik=arm.ik(pose@inverse(TCP),previous)
        attempts.append(ik)
        if ik['found']: previous=np.array(ik['q']); qs.append(previous)
        else: break
    found=len(qs)==len(path); reasons=set(); tracking=True; samples=0; length=0.; views={m:[] for m in MOUNTS}
    contact_views={m:[] for m in MOUNTS}
    target=np.array(sc['target']); actual_min={m:[] for m in MOUNTS}
    # Desired-pose visibility is independent of IK; unknown actual views stay unknown.
    first_contact=next(i for i,(pose,_) in enumerate(path) if np.linalg.norm(pose[:3,3]-target)<.002)
    for path_index,(pose,carried) in enumerate(path):
        if not carried and np.linalg.norm(pose[:3,3]-target)<.002:
            for m,mount in MOUNTS.items(): contact_views[m].append(visibility(pose@inverse(TCP),mount,sc,boxes)['fraction'])
        if path_index<first_contact and pose[2,3]-target[2]>.20:
            for m,mount in MOUNTS.items(): views[m].append(visibility(pose@inverse(TCP),mount,sc,boxes)['fraction'])
    if found:
        # Check all solved points and each interval; seed warm-start alone is not continuity validation.
        for i,q in enumerate(qs):
            start=qs[i-1] if i else q
            n=max(1,int(np.ceil(np.max(np.abs(q-start))/joint_step)))
            length+=float(np.linalg.norm(q-start))
            a=path[max(0,i-1)][0]; b=path[i][0]
            desired=[b] if i==0 else [t for t,u in interpolate(a,b,n)]
            for u,ref in zip(np.linspace(0,1,n+1)[1:],desired):
                qi=start*(1-u)+q*u; frames=arm.frames(qi); f=frames[-1]; tcp=f@TCP; samples+=1
                pos=np.linalg.norm(tcp[:3,3]-ref[:3,3]); ang=np.linalg.norm(Rotation.from_matrix(ref[:3,:3]@tcp[:3,:3].T).as_rotvec())
                if pos>.003 or ang>.04: tracking=False
                reasons.update(collision(frames,f,boxes,path[i][1],sc))
                if i<first_contact and ref[2,3]-target[2]>.20:
                    for m,mount in MOUNTS.items(): actual_min[m].append(visibility(f,mount,sc,boxes,frames)['fraction'])
    if not found: reasons.add('IK_SEARCH_FAILED_UNKNOWN_REACHABILITY')
    if found and not tracking: reasons.add('interpolated_tracking_error')
    feasible=found and tracking and not reasons
    record=dict(candidate=can['id'],ik_found=found,tracking_proxy=tracking if found else None,
                feasible_proxy=feasible,failures=sorted(reasons),joint_length_rad=length if found else None,
                joint_samples=samples,ik_attempts=attempts,joint_path=[q.tolist() for q in qs],
                desired_path=[dict(T_world_tcp=t.tolist(),carried=c) for t,c in path],
                desired_visibility={m:float(min(v)) if v else None for m,v in views.items()},
                contact_visibility={m:float(min(v)) if v else None for m,v in contact_views.items()},
                actual_visibility={m:float(min(v)) if v else None for m,v in actual_min.items()},
                continuous_collision_certificate=False,independent_execution_outcome=None)
    return record


def selectors(rows,mount):
    ids=list(range(len(rows)))
    def v(i): return rows[i]['desired_visibility'][mount] or 0.
    def cost(i): return rows[i]['joint_length_rad'] if rows[i]['joint_length_rad'] is not None else 1e6
    def f(i): return int(rows[i]['feasible_proxy'])
    joint=[i for i in ids if f(i) and v(i)>=.60]
    # All selectors see the same table; no label is an independent success endpoint.
    return {'ik_only':max(ids,key=lambda i:(rows[i]['ik_found'],-cost(i),-i)),
            'ik_collision':max(ids,key=lambda i:(f(i),-cost(i),-i)),
            'visibility_only':max(ids,key=lambda i:(v(i),-i)),
            'weighted_feasibility':max(ids,key=lambda i:(.5*f(i)+.5*v(i),-cost(i),-i)),
            'analytic_and':joint[0] if joint else None,
            'geometry_planning':min(joint,key=lambda i:(cost(i),i)) if joint else None,
            # Full-horizon deterministic MPC on a finite menu: exact solve, no stochastic handicap.
            'same_information_mpc':min(joint,key=lambda i:(cost(i),i)) if joint else None}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate_scene(sc):
    arms={name:Arm(name) for name in ('panda','sr5')}
    records=[]; baseline=[]; factorial=[]
    for blocked in (False,True):
        for robot,arm in arms.items():
            rows=[evaluate(arm,sc,c,blocked) for c in candidates(sc)]
            item=dict(scene=sc['id'],split=sc['split'],seed=sc['seed'],family=sc['family'],blocked=blocked,robot=robot,candidates=rows)
            records.append(item)
            for mount in MOUNTS:
                for r in rows:
                    vis=r['desired_visibility'][mount]
                    factorial.append(dict(scene=sc['id'],split=sc['split'],seed=sc['seed'],family=sc['family'],blocked=blocked,
                                          robot=robot,mount=mount,candidate=r['candidate'],ik_found=r['ik_found'],
                                          feasible_proxy=r['feasible_proxy'],visible_fraction=vis,
                                          observable_proxy=vis is not None and vis>=.6,
                                          conjunction_proxy=r['feasible_proxy'] and vis is not None and vis>=.6,
                                          failures='|'.join(r['failures'])))
                for name,idx in selectors(rows,mount).items():
                    selected=rows[idx] if idx is not None else None
                    baseline.append(dict(scene=sc['id'],split=sc['split'],seed=sc['seed'],family=sc['family'],blocked=blocked,
                                         robot=robot,mount=mount,baseline=name,selected=idx,
                                         selected_conjunction_proxy=bool(selected and selected['feasible_proxy'] and (selected['desired_visibility'][mount] or 0)>=.6),
                                         any_conjunction_proxy=any(r['feasible_proxy'] and (r['desired_visibility'][mount] or 0)>=.6 for r in rows),
                                         candidate_budget=len(rows),independent_execution_outcome='UNAVAILABLE'))
    return records,baseline,factorial,{k:a.calls for k,a in arms.items()}


def run(out,seeds=(314159,271828,161803,141421),n=9):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    from concurrent.futures import ProcessPoolExecutor
    dataset=[s for seed in seeds for s in scenes(seed,n)]
    (out/'scenes.json').write_text(json.dumps(dataset,indent=2))
    records=[]; baseline=[]; factorial=[]; calls={'panda':0,'sr5':0}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for sc,result in zip(dataset,pool.map(evaluate_scene,dataset)):
            rr,bb,ff,cc=result
            records.extend(rr); baseline.extend(bb); factorial.extend(ff)
            for k,v in cc.items(): calls[k]+=v
            (out/'trials.json').write_text(json.dumps(records,indent=1,allow_nan=False))
            print(sc['id'], 'completed all factorial cells',flush=True)
    write_csv(out/'factorial.csv',factorial); write_csv(out/'baselines.csv',baseline)
    files=[ROOT/'docs/VALIDATION_PROTOCOL.md',*sorted((ROOT/'configs/validation').glob('*.json')),
           *sorted((ROOT/'coftvalidation').glob('*.py')),*sorted((ROOT/'vendor').glob('*/*'))]
    manifest=dict(status='PROXY_ONLY',scientific_decision='INSUFFICIENT_EVIDENCE',seeds=list(seeds),n_per_seed=n,
                  seconds=time.monotonic()-start,python=platform.python_version(),numpy=np.__version__,
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
                  solver_nfev=calls,independent_execution_outcomes=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .report import report
    report(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='evidence/v1'); p.add_argument('--seeds',type=int,nargs='+',default=[314159,271828,161803,141421]); p.add_argument('--scenes',type=int,default=9)
    a=p.parse_args()
    if a.scenes<1: p.error('positive scenes required')
    run(a.out,a.seeds,a.scenes)
