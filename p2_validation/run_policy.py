"""Run unchanged official BARX evaluation with a validated condition window.

B1 intercepts only the executed gripper coordinate, using queued original
continuous gripper predictions; the other eleven simulator coordinates pass
through unchanged. The original official success and video writer are retained.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from p2_validation.controls import GripperCorrection, select_conditions


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--robot',choices=['ur5e','panda'],required=True);p.add_argument('--indices',required=True)
    p.add_argument('--method',choices=['original','b1'],default='original');p.add_argument('--threshold',type=float,default=.9)
    p.add_argument('--out',required=True);a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.out).resolve();checkpoint=Path(a.checkpoint).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    indices=[int(x) for x in a.indices.split(',')]
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit=='0f435089f48b1f16d225f868dbef855a5e75d1e0'
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain']).strip()
    job=dict(status='RUNNING',method=a.method,robot=a.robot,indices=indices,threshold=a.threshold if a.method=='b1' else None,
             hysteresis=.03,minimum_hold=8,checkpoint=str(checkpoint),official_commit=commit,
             privileged_information_used_by_controller=False)
    ledger=out/'job.json'
    def save():ledger.write_text(json.dumps(job,indent=2)+'\n')
    save();sys.path[:0]=[str(source),str(source/'policy')]
    import torch
    from barx.benchmark import EMBODIMENTS
    import barx.evaluation_conditions as conditions
    from robocasa.utils.robomimic import robomimic_env_utils as utils
    import experiments.robot.robot_utils as robot_utils
    from simulation_validation.observe import Recorder
    from p2_validation.offline_audit import sha256
    job['checkpoint_sha256']=sha256(checkpoint);job['gpu']=torch.cuda.get_device_name(0);save()
    torch.cuda.reset_peak_memory_stats()
    rec=Recorder(out/'telemetry');rec.install()
    original_load=conditions.load_bundle
    def load_window(*args,**kwargs):
        full,states,models=original_load(*args,**kwargs)
        arm=EMBODIMENTS[a.robot]
        conditions.validate_condition_bundle_for_evaluation(full,states,models,task='pnp_counter_to_sink',
            camera_name=arm.camera,image_width=320,image_height=180,count=len(states))
        selected=select_conditions(full,states,models,indices)
        job['conditions']=[{k:e[k] for k in ['episode','condition_id','seed','state_sha256','model_sha256','metadata_sha256']}
                           for e in selected[0]['episodes']];save()
        return selected
    conditions.load_bundle=load_window
    controller=GripperCorrection(a.threshold);queue=deque();active=False
    restore=conditions.restore_frozen_condition
    def restore_with_reset(*args,**kwargs):
        nonlocal active
        active=False;result=restore(*args,**kwargs);queue.clear();controller.reset();active=True
        return result
    conditions.restore_frozen_condition=restore_with_reset
    predict=robot_utils.get_action
    def predict_with_queue(*args,**kwargs):
        result=predict(*args,**kwargs)
        chunk=np.asarray(result['action'])
        if chunk.shape!=(8,7):raise ValueError(f'Unexpected action chunk {chunk.shape}')
        if queue:raise RuntimeError('Unconsumed original action predictions')
        queue.extend(chunk[:,6].copy().tolist())
        return result
    robot_utils.get_action=predict_with_queue
    load_model=robot_utils.get_model
    def load_with_memory(*args,**kwargs):
        model=load_model(*args,**kwargs)
        job['model_loaded_gpu_allocated_bytes']=torch.cuda.memory_allocated();save();return model
    robot_utils.get_model=load_with_memory
    create=utils.create_env
    def create_with_control(*args,**kwargs):
        env=create(*args,**kwargs);step=env.step
        def controlled_step(action,*ar,**kw):
            executed=np.asarray(action).copy()
            if active:
                if not queue:raise RuntimeError('No original gripper prediction for simulator action')
                raw=queue.popleft();before=executed[6]
                if a.method=='b1':executed[6]=controller.command(raw)
                rec.append('p2/raw_gripper',raw);rec.append('p2/original_gripper_command',before)
                rec.append('p2/executed_gripper_command',executed[6])
                rawenv=env.env;sim=rawenv.sim
                ids=[sim.model.geom_name2id(f'gripper0_right_{s}_fingerpad_collision') for s in ['left','right']]
                rec.append('p2/pre_action_pad_center_distance',np.linalg.norm(sim.data.geom_xpos[ids[0]]-sim.data.geom_xpos[ids[1]]))
            return step(executed,*ar,**kw)
        env.step=controlled_step
        return env
    utils.create_env=create_with_control
    arm=EMBODIMENTS[a.robot]
    sys.argv=[str(source/'policy/experiments/robot/robocasa_x/evaluate.py'),
       '--pretrained_checkpoint',str(checkpoint),'--embodiment',a.robot,'--robot',arm.robot,'--gripper_types',arm.gripper,
       '--camera',arm.camera,'--task','XPnPCounterToSink','--unnorm_key','mg_pnp_lite','--num_trials_per_task',str(len(indices)),
       '--start_seed','1000','--seed','7','--max_steps','600','--act_horizon','8',
       '--conditions_dir',str(source/'evaluation/conditions'),'--rollout_dir',str(out/'official')]
    os.chdir(source/'policy')
    try:
        runpy.run_path(sys.argv[0],run_name='__main__')
        summary_paths=list((out/'official').rglob('summary.json'));assert len(summary_paths)==1
        summary=json.loads(summary_paths[0].read_text());assert summary['status']=='complete' and summary['episodes']==len(indices)
        records=[json.loads(x) for x in (summary_paths[0].parent/'episodes.jsonl').read_text().splitlines()]
        assert [r['condition_id'] for r in records]==[r['condition_id'] for r in job['conditions']]
        assert all(type(r['success']) is bool for r in records)
        job.update(status='COMPLETE',summary=summary)
    except BaseException as error:
        job.update(status='FAILED',error=repr(error));raise
    finally:
        rec.close();job['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        job['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved();save()


if __name__=='__main__':main()
