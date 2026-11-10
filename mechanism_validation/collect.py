"""Audit and export the immutable first-attempt results, without physics."""
import csv
import json
import sys
from collections import Counter
import numpy as np
import h5py
import cv2
from mechanism_validation.common import *
from mechanism_validation.run import compare,validate_freeze


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def main():
    c,marker=validate_freeze()
    sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    from barx.evaluation_conditions import state_sha256
    complete=json.loads((ROOT/'completion.json').read_text());assert complete['status']=='COMPLETE'
    launches=[json.loads(x) for x in (ROOT/'physical_launches.jsonl').read_text().splitlines()]
    assert len(launches)==24 and len({(x['condition'],x['method']) for x in launches})==24
    rows=[];pairs=[];stage_rows=[];videos=[]
    for i in CASES:
        reference=None;reference_hand=None;initial_state=None;model_sha=None
        for method in c['methods']:
            directory=ROOT/f'condition-{i:03d}'/method
            row=json.loads((directory/'result.json').read_text());history=json.loads((directory/'history.json').read_text())
            assert row['status']=='COMPLETE' and len(history)==row['steps'] and row['steps']<=600
            assert not (directory/'technical_failure.json').exists()
            assert row['task_success']==any(h['official_task_success'] for h in history)
            with np.load(directory/'prefix.npz') as z:prefix={k:z[k] for k in z.files}
            hand=None
            if (directory/'handoff.npz').exists():
                with np.load(directory/'handoff.npz') as z:hand={k:z[k] for k in z.files}
                runtime=json.loads((directory/'handoff.json').read_text());assert runtime['policy_queue_length']==0
            if reference is None:reference=prefix;reference_hand=hand
            errors=compare(reference,prefix)
            assert (reference_hand is None)==(hand is None)
            if hand is not None:errors.update({'handoff::'+k:v for k,v in compare(reference_hand,hand).items()})
            assert max(errors.values(),default=0)<=1e-8
            initial=json.loads((directory/'initial.json').read_text())
            if model_sha is None:model_sha=initial['model_xml_sha256']
            assert model_sha==initial['model_xml_sha256']
            with h5py.File(row['telemetry_path']) as f:
                assert len(f['sim/qpos'])==row['steps']+1 and len(f['actions/environment'])==row['steps']
                for key in ['sim/qpos','sim/qvel','sim/body_xpos','sim/body_xquat','target/position','actions/environment']:
                    assert np.isfinite(f[key][:]).all(),key
                state=f['mechanism/restored_initial_state'][0]
                assert state_sha256(state)==initial['condition']['state_sha256']
                if initial_state is None:initial_state=state
                assert np.array_equal(initial_state,state)
                obj=f['target/position'][:];tcp=f['mechanism/post_action_tcp_world'][:]
                actions=f['actions/environment'][:]
                assert np.array_equal(actions[0:len(prefix['actions'])],prefix['actions'])
                assert np.all(actions[:,7:11]==0) and np.all(actions[:,-1]==-1)
                if row['handoff_confirmed'] and method=='M1':
                    assert len(f['mechanism/native_action'])==row['steps']-row['handoff_action_index']
                    assert f['mechanism/native_query_action_index'][0]==row['handoff_action_index']
            contacts=[json.loads(x) for x in Path(row['telemetry_path']).with_suffix('.jsonl').read_text().splitlines()]
            assert len(contacts)==row['steps']+1
            assert all(not x['telemetry_errors'] for x in contacts)
            cap=cv2.VideoCapture(row['video_path']);frames=0
            assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape==(180,320,3);frames+=1
            cap.release();assert frames==row['steps'] and fps==20
            videos.append(dict(condition=i,method=method,path=row['video_path'],decoded_frames=frames,fps=fps,sha256=digest(row['video_path'])))
            pairs.append(dict(condition=i,method=method,initial_state_bitwise_equal=True,model_xml_identical=True,prefix_actions= len(prefix['actions']),prefix_states=len(prefix['states']),handoff_present=hand is not None,maximum_numeric_error=max(errors.values(),default=0),trusted=True))
            for phase in dict.fromkeys(h['phase'] for h in history):
                hh=[h for h in history if h['phase']==phase]
                stage_rows.append(dict(condition=i,method=method,phase=phase,steps=len(hh),first_action=hh[0]['action_index'],last_action=hh[-1]['action_index'],
                    final_tcp_position_error_m=hh[-1]['tcp_position_error_m'],final_tcp_rotation_error_degrees=float(np.rad2deg(hh[-1]['tcp_rotation_error_rad'])),
                    minimum_tcp_position_error_m=min(h['tcp_position_error_m'] for h in hh),minimum_tcp_rotation_error_degrees=float(np.rad2deg(min(h['tcp_rotation_error_rad'] for h in hh))),
                    bilateral_contact_states=sum(h['bilateral_contact'] for h in hh),legacy_pad_contact_states=sum(h['legacy_pad_only_contact'] for h in hh),
                    maximum_rise_m=max(h['rise_m'] for h in hh),unwanted_contact_states=sum(h['unwanted_contacts']>0 for h in hh),joint_limit_states=sum(h['minimum_joint_margin'] is not None and h['minimum_joint_margin']<=.01 for h in hh)))
            rows.append(row)
    summary=[]
    for method in c['methods']:
        selected=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,episodes=8,approach=sum(r['approach_reached'] for r in selected),pregrasp=sum(r['pregrasp_reached'] for r in selected),
            bilateral_contact=sum(r['bilateral_contact'] for r in selected),stable_lift=sum(r['stable_lift'] for r in selected),handoff=sum(r['handoff_confirmed'] for r in selected),
            transport=sum(r['transport_10cm_in_contact'] for r in selected),placement=sum(r['placement'] for r in selected),official_success=sum(r['task_success'] for r in selected),
            failure_stages=dict(Counter(r['failure_stage'] for r in selected))))
    a,b,d=summary
    # Stable prefix is shared; do not let later BARX recovery change gate A.
    from p2_validation.offline_audit import longest_run
    common_stable=0
    for i in CASES:
        hh=json.loads((ROOT/f'condition-{i:03d}/M0/history.json').read_text())
        n=next(r['prefix_actions'] for r in pairs if r['condition']==i and r['method']=='M0')
        common_stable+=longest_run([h['bilateral_contact'] and h['rise_m']>=.03 for h in hh[:n]])>=10
    physical_pass=common_stable>=6 and max(a['official_success'],d['official_success'])>=4
    direction='PREGRASP' if physical_pass and b['official_success']>=4 else 'POSTGRASP' if physical_pass and b['official_success']<=2 and d['official_success']>=4 else None
    decision=dict(physical_recoverability='PASS' if physical_pass else 'NO_GO',common_prefix_stable_grasps=common_stable,ordinary_controller_best_official_success=max(a['official_success'],d['official_success']),
        selected_direction=direction,algorithm_development='PENDING_DISTINCT_MECHANISM_REVIEW' if direction else 'NO_GO_PHYSICAL_GATE' if not physical_pass else 'NO_GO_AMBIGUOUS',
        diagnostic_only=True,no_independent_generalization_claim=True,no_threshold_search=True,additional_physical_episodes=0)
    write_json(OUT/'analysis.json',dict(status='COMPLETE',frozen_commit=marker['remote_commit'],protocol_sha256=marker['protocol_sha256'],physical_episodes=24,summary=summary,decision=decision,episodes=rows))
    csv_write(OUT/'episodes.csv',rows);csv_write(OUT/'summary.csv',summary);csv_write(OUT/'stages.csv',stage_rows);csv_write(OUT/'prefix_audit.csv',pairs)
    write_json(OUT/'video_validation.json',dict(videos=videos,all24_decoded=True))
    write_json(OUT/'decision.json',decision);write_json(OUT/'frozen_publication.json',marker)
    checkpoint=Path(c['checkpoint'])
    write_json(OUT/'checkpoint_index.json',dict(role='UNCHANGED_ORIGINAL_BARX_JOINT_SOURCE_PRIOR',absolute_path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=c['checkpoint_sha256'],training_updates_this_round=0))
    bank=json.loads((OUT/'source_contact_bank.json').read_text())
    csv_write(OUT/'source_artifact_index.csv',[dict(condition=r['official_index'],kind=kind,absolute_path=r[key],bytes=Path(r[key]).stat().st_size,sha256=digest(r[key])) for r in bank for kind,key in [('SOURCE_SUCCESS_VIDEO','source_video'),('SOURCE_RAW_TELEMETRY','source_telemetry'),('SOURCE_EXTRACTED_FULL_TRACE','source_trace')]])
    csv_write(OUT/'external_files.csv',[dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['run.lock']])
    print(json.dumps(dict(summary=summary,decision=decision,validated_videos=len(videos)),indent=2))


if __name__=='__main__':main()
