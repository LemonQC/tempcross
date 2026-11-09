"""Recorded-state Jacobian audit: no new simulator, IK search, or stepping."""
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from p3_validation.common import ROOT,SOURCE,PROJECT
from p3_validation.geometry import transform,site_definition
from p3_validation.prepare_data import write_csv


def main():
    import h5py
    results=json.loads((ROOT/'diagnostics/results.json').read_text())['results'];rows=[];validation=[]
    with np.load(SOURCE/'evaluation/conditions/pnp_counter_to_sink/panda.npz',allow_pickle=False) as z:models=z['model_xmls'].tolist()
    for result in results:
        if result['method']!='A3':continue
        index=result['condition'];xml=models[index]
        for marker in ['ROBOCASA','ROBOSUITE']:xml=xml.replace('<'+marker+'>','__'+marker+'__')
        tree=ET.fromstring(xml);joints=[]
        for body in tree.iter('body'):
            for joint in body.findall('joint'):
                name=joint.get('name','')
                if re.fullmatch('robot0_joint[1-7]',name):
                    joints.append((name,body.attrib['name'],np.fromstring(joint.get('axis','0 0 1'),sep=' '),np.fromstring(joint.get('pos','0 0 0'),sep=' ')))
        joints.sort();assert len(joints)==7
        path=__import__('pathlib').Path(result['telemetry_path']);history=json.loads(__import__('pathlib').Path(result['history_path']).read_text())
        with h5py.File(path) as f:
            names=json.loads(f.attrs['body_names']);body,local=site_definition(xml);bid=names.index(body)
            tcp=np.stack([transform(p,q)@local for p,q in zip(f['sim/body_xpos'][:,bid],f['sim/body_xquat'][:,bid])])
            columns=[]
            for _,body,axis,pos in joints:
                bid=names.index(body);poses=np.stack([transform(p,q) for p,q in zip(f['sim/body_xpos'][:,bid],f['sim/body_xquat'][:,bid])])
                axes=np.einsum('nij,j->ni',poses[:,:3,:3],axis)
                anchors=np.einsum('nij,j->ni',poses[:,:3,:3],pos)+poses[:,:3,3]
                columns.append(np.concatenate([np.cross(axes,tcp[:,:3,3]-anchors),axes],axis=1))
            jacobian=np.stack(columns,axis=-1)
        with np.load(path.parents[1]/'branch_snapshot.npz') as z:
            reference=z['controller::right/J_full'];delta=float(np.abs(jacobian[result['branch_action_index']]-reference).max())
            assert delta<1e-8
        validation.append(dict(condition=index,analytic_vs_recorded_controller_jacobian_max_error=delta))
        singular=np.linalg.svd(jacobian,compute_uv=False)[:,-1]
        for phase in ['hold_grasp','lift','preplace','place','release','retreat']:
            selected=[h for h in history if h['phase']==phase]
            if not selected:continue
            indices=[h['action_index']+1 for h in selected]
            rows.append(dict(condition=index,phase=phase,recorded_states=len(selected),
                minimum_full_jacobian_singular_value=float(singular[indices].min()),median_full_jacobian_singular_value=float(np.median(singular[indices])),
                minimum_endpoint_position_error_m=min(h['endpoint_position_error_m'] for h in selected),
                minimum_endpoint_rotation_error_rad=min(h['endpoint_rotation_error_rad'] for h in selected),
                raw_unwanted_contact_state_steps=sum(h['unwanted_contacts']>0 for h in selected),
                joint_limit_state_steps=sum(h['minimum_joint_margin'] is not None and h['minimum_joint_margin']<=.01 for h in selected),
                official_task_success=result['task_success']))
    output=PROJECT/'evidence/p3/execution';output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'recorded_kinematic_diagnostics.csv',rows)
    (output/'jacobian_validation.json').write_text(json.dumps(dict(validation=validation,
       new_physics_steps=0,ik_search_performed=False,
       limitation='Singular values mix linear/angular units and indicate local conditioning only; they do not prove global reachability or establish collision causality.'),indent=2)+'\n')
    print(json.dumps(dict(validated_cases=len(validation),max_jacobian_error=max(x['analytic_vs_recorded_controller_jacobian_max_error'] for x in validation),phase_rows=len(rows)),indent=2))


if __name__=='__main__':main()
