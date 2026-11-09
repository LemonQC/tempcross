"""Audit closed rollout files without taking any new simulator step."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,h5py
from baselines.common.paths import ROOT,PROJECT,write_json,digest
from baselines.common.action_adapter import canonical_action


def validate_job(job_path):
 job_path=Path(job_path);job=json.loads(job_path.read_text());assert job['status']=='COMPLETE';directory=job_path.parent
 records=[json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()];assert len(records)==len(job['indices']);rows=[]
 frozen=json.loads((PROJECT/'configs/evaluation/baseline_conditions_60_99.json').read_text()) if job['role']=='FROZEN_FINAL_TEST' else None
 for r in records:
  assert type(r['success']) is bool
  path=Path(r['telemetry_path']);video=Path(r['video_path'])
  with h5py.File(path) as f:
   assert len(f['sim/qpos'])==r['steps']+1 and len(f['actions/environment'])==r['steps']
   for k in ['sim/qpos','sim/qvel','sim/body_xpos','sim/body_xquat','target/position','target/quaternion','actions/environment','baseline/native_action','baseline/pre_action_tcp_position','baseline/pre_action_tcp_rotation_matrix']:
    assert np.isfinite(f[k][:]).all(),(path,k)
   entry=json.loads(f.attrs['condition']);assert entry['condition_id']==r['condition_id'] and entry['episode']==r['official_condition_index'] and entry['seed']==r['seed']
   if frozen is not None:
    from barx.evaluation_conditions import state_sha256
    expected_state=next(e for e in frozen if e['task']==job['task'] and e['robot']==job['robot'] and e['official_index']==r['official_condition_index'])
    initial=f['baseline/restored_initial_state'][0]
    assert np.array_equal(initial,np.asarray(expected_state['initial_state'],dtype=np.float64))
    assert state_sha256(initial)==entry['state_sha256']==expected_state['condition']['state_sha256']
   actual=f['actions/environment'][:];native=f['baseline/native_action'][:]
   expected=np.stack([canonical_action(x,'barx_unnormalized' if job['kind']=='barx' else 'canonical_osc')[0] for x in native]);assert np.array_equal(actual,expected)
   tcp=f['baseline/pre_action_tcp_position'][:];motion=float(np.linalg.norm(tcp-tcp[0],axis=1).max())
   placement_available='baseline/post_action_placement' in f
   if placement_available:
    placement=f['baseline/post_action_placement'][:];retreat=f['baseline/post_action_retreat'][:]
    assert bool(np.any(placement & retreat))==r['success']
  cache=directory/f"video-validation-{r['official_condition_index']:03d}.json"
  if cache.exists():
   v=json.loads(cache.read_text());assert digest(video)==v['sha256'];frames=v['decoded_frames']
  else:
   cap=cv2.VideoCapture(str(video));frames=0
   while True:
    ok,_=cap.read()
    if not ok:break
    frames+=1
   cap.release();assert frames==r['steps'] and frames>0
   write_json(cache,dict(video_path=str(video),sha256=digest(video),decoded_frames=frames))
  rows.append(dict(condition_id=r['condition_id'],original_index=r['official_condition_index'],video_path=str(video),video_sha256=digest(video),decoded_frames=frames,telemetry_path=str(path),telemetry_sha256=digest(path),finite_states_and_actions=True,exact_native_action_adapter_match=True,maximum_tcp_excursion_m=motion,official_placement_subpredicates_recorded=placement_available))
 result=dict(status='COMPLETE',episodes=len(rows),all_verified=True,records=rows);write_json(directory/'validation.json',result);return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--formal',action='store_true');a=p.parse_args();rows=[]
 for path in sorted((ROOT/('formal' if a.formal else 'smoke')).rglob('job.json')):
  job=json.loads(path.read_text())
  if job['status']=='COMPLETE':
   r=validate_job(path);rows.append(dict(job=str(path),episodes=r['episodes'],all_verified=True));print(json.dumps(rows[-1]),flush=True)
 write_json(ROOT/('formal_validation_summary.json' if a.formal else 'smoke_validation_summary.json'),dict(status='COMPLETE',jobs=rows,episodes=sum(x['episodes'] for x in rows)))
if __name__=='__main__':main()
