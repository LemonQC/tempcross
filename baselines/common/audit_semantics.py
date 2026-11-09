"""Audit the shared physical action meaning without running a new rollout."""
import json,sys
from baselines.common.paths import ROOT,PROJECT,SOURCE,write_json

def main():
 sys.path.insert(0,str(SOURCE))
 from robocasa.utils.controller_utils import load_robocasa_controller_config
 rows=[]
 for robot in ['IIWAOmron','Kinova3Omron','UR5eOmron','PandaOmron']:
  c=load_robocasa_controller_config(robot=robot);right=c['body_parts']['right']
  fields={k:right.get(k) for k in ['type','input_ref_frame','input_type','input_min','input_max','output_min','output_max','kp','damping_ratio']}
  rows.append(dict(robot=robot,arm_controller=fields))
 assert all(r['arm_controller']==rows[0]['arm_controller'] for r in rows)
 splits={}
 for domain in ['source','target']:
  rows_data=json.loads((ROOT/f'data/{domain}_manifest.json').read_text());train={r['episode_id'] for r in rows_data if r['split']=='train'};val={r['episode_id'] for r in rows_data if r['split']=='validation'}
  assert train.isdisjoint(val);splits[domain]=dict(train=len(train),validation=len(val),overlap=0)
 result=dict(status='VERIFIED_IDENTICAL_CANONICAL_OSC_SEMANTICS',controllers=rows,splits=splits,source_raw_tfds_gripper='[-1,+1], unchanged',pretraining_stream_gripper='BARX learning transform later maps to [0,1]; not used for new raw-data labels',policy_inputs='RGB and language only',new_physics_steps=0)
 write_json(ROOT/'canonical_control_semantics.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
