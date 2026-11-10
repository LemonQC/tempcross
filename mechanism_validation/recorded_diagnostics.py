"""Read-only tracking/kinematic audit of the recorded interventions. No IK or step."""
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import h5py
from mechanism_validation.common import *
from mechanism_validation.collect import csv_write
from p3_validation.geometry import transform,site_definition


def jacobian(f,xml,indices):
    xml=xml.replace('<ROBOCASA>','__ROBOCASA__').replace('<ROBOSUITE>','__ROBOSUITE__')
    tree=ET.fromstring(xml);joints=[]
    for body in tree.iter('body'):
        for joint in body.findall('joint'):
            if re.fullmatch('robot0_joint[1-7]',joint.get('name','')):
                joints.append((joint.get('name'),body.get('name'),np.fromstring(joint.get('axis','0 0 1'),sep=' '),np.fromstring(joint.get('pos','0 0 0'),sep=' ')))
    joints.sort();assert len(joints)==7
    names=json.loads(f.attrs['body_names']);body,local=site_definition(xml);bid=names.index(body)
    tcp=np.stack([transform(p,q)@local for p,q in zip(f['sim/body_xpos'][indices,bid],f['sim/body_xquat'][indices,bid])])
    columns=[]
    for _,body,axis,pos in joints:
        bid=names.index(body)
        poses=np.stack([transform(p,q) for p,q in zip(f['sim/body_xpos'][indices,bid],f['sim/body_xquat'][indices,bid])])
        axes=np.einsum('nij,j->ni',poses[:,:3,:3],axis)
        anchors=np.einsum('nij,j->ni',poses[:,:3,:3],pos)+poses[:,:3,3]
        columns.append(np.concatenate([np.cross(axes,tcp[:,:3,3]-anchors),axes],axis=1))
    return np.stack(columns,axis=-1)


def main():
    with np.load(SOURCE/f'evaluation/conditions/{TASK}/panda.npz',allow_pickle=False) as z:models=z['model_xmls'].tolist()
    rows=[];checks=[]
    for i in CASES:
        for method in ['M0','M1','M2']:
            directory=ROOT/f'condition-{i:03d}'/method
            result=json.loads((directory/'result.json').read_text());history=json.loads((directory/'history.json').read_text())
            with h5py.File(result['telemetry_path']) as f:
                handoff=result['handoff_action_index']
                if handoff is not None:
                    estimated=jacobian(f,models[i],[handoff])[0]
                    with np.load(directory/'handoff.npz') as z:stored=z['controller::right/J_full']
                    delta=float(np.abs(estimated-stored).max());assert delta<1e-8
                    checks.append(dict(condition=i,method=method,analytic_vs_saved_controller_J_max_error=delta))
                for phase in dict.fromkeys(h['phase'] for h in history):
                    hh=[h for h in history if h['phase']==phase]
                    end=hh[-1]['action_index'];start=hh[max(0,len(hh)-20)]['action_index']
                    j=jacobian(f,models[i],[end+1])[0];sv=np.linalg.svd(j,compute_uv=False)
                    pos=f['mechanism/post_action_tcp_world'][start:end+1,:3,3]
                    obj=f['target/position'][start+1:end+2]
                    rows.append(dict(condition=i,method=method,phase=phase,official_task_success=result['task_success'],failure_stage=result['failure_stage'],
                        phase_steps=len(hh),terminal_position_error_m=hh[-1]['tcp_position_error_m'],terminal_rotation_error_degrees=float(np.rad2deg(hh[-1]['tcp_rotation_error_rad'])),
                        last20_tcp_displacement_m=float(np.linalg.norm(pos[-1]-pos[0])),last20_object_displacement_m=float(np.linalg.norm(obj[-1]-obj[0])),
                        local_jacobian_minimum_singular_value=float(sv[-1]),local_jacobian_condition_number=float(sv[0]/max(sv[-1],1e-15)),
                        terminal_minimum_joint_margin_rad=hh[-1]['minimum_joint_margin'],terminal_unwanted_contacts=hh[-1]['unwanted_contacts']))
    csv_write(OUT/'recorded_tracking_diagnostics.csv',rows)
    write_json(OUT/'jacobian_audit.json',dict(new_physics_steps=0,checks=checks,all_handoff_jacobians_verified=True,
        limitation='Mixed translation/rotation units; local Jacobian conditioning and tracking plateaus do not prove global pose infeasibility or collision causality. No IK search or alternate controller was run.'))
    print(json.dumps(dict(rows=len(rows),validated_handoffs=len(checks),maximum_J_error=max((r['analytic_vs_saved_controller_J_max_error'] for r in checks),default=None))))


if __name__=='__main__':main()
