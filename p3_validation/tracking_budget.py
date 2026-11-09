"""Audit scheduled Cartesian speed against recorded OSC response; no stepping."""
import json
from pathlib import Path
import numpy as np
from p3_validation.common import ROOT,PROJECT,SOURCE
from p3_validation.geometry import site_definition,transform
from p3_validation.prepare_data import write_csv


def main():
    import h5py
    c=json.loads((PROJECT/'configs/p3_protocol.json').read_text());results=json.loads((ROOT/'diagnostics/results.json').read_text())['results'];rows=[]
    with np.load(SOURCE/'evaluation/conditions/pnp_counter_to_sink/panda.npz',allow_pickle=False) as f:models=f['model_xmls'].tolist()
    for result in results:
        if result['method']!='A3':continue
        path=Path(result['telemetry_path']);hist=json.loads(Path(result['history_path']).read_text());idx=result['condition']
        with h5py.File(path) as f:
            names=json.loads(f.attrs['body_names']);body,local=site_definition(models[idx]);bid=names.index(body)
            tcp=np.stack([transform(p,q)@local for p,q in zip(f['sim/body_xpos'][:,bid],f['sim/body_xquat'][:,bid])])[:,:3,3]
            goals=f['p3/goal_tcp_world'][:,:3,3];actions=f['actions/environment'][:]
        with np.load(path.parents[1]/'branch_snapshot.npz') as b:
            kp=b['controller::right/kp'][:3];kd=b['controller::right/kd'][:3];scale=b['controller::right/action_scale'][:3]
        for phase in ['pregrasp','descend','lift','preplace','place','retreat']:
            chosen=[h for h in hist if h['phase']==phase]
            if not chosen:continue
            start,end=chosen[0]['action_index'],chosen[-1]['action_index']
            duration=(end-start+1)/20.;reference_start=goals[start-1] if start else goals[start]
            reference_end=goals[end];delta=reference_end-reference_start;distance=float(np.linalg.norm(delta))
            actual_delta=tcp[end+1]-tcp[start];unit=delta/distance if distance>1e-12 else np.zeros(3)
            increments=np.linalg.norm(actions[start:end+1,:3]*scale,axis=1)
            rows.append(dict(condition=idx,phase=phase,official_task_success=result['task_success'],steps=end-start+1,
                scheduled_duration_seconds=duration,scheduled_reference_displacement_m=distance,
                scheduled_reference_mean_speed_m_s=distance/duration,
                achieved_tcp_net_displacement_m=float(np.linalg.norm(actual_delta)),
                achieved_mean_speed_along_reference_m_s=float(actual_delta@unit/duration),
                capped_increment_fraction=float(np.mean(increments>=c['controls']['max_position_step_m']*(1-1e-6))),
                nominal_zero_acceleration_speed_at_cap_m_s=float(np.mean(kp/kd)*c['controls']['max_position_step_m']),
                final_endpoint_position_error_m=chosen[-1]['endpoint_position_error_m'],
                raw_unwanted_contact_state_steps=sum(h['unwanted_contacts']>0 for h in chosen)))
    out=PROJECT/'evidence/p3/execution';out.mkdir(parents=True,exist_ok=True);write_csv(out/'osc_tracking_budget.csv',rows)
    (out/'osc_tracking_budget_notes.json').write_text(json.dumps(dict(physics_steps=0,
       measured_control_frequency_hz=20,nominal_speed_formula='kp/kd * capped achieved-relative goal offset',
       limitations=['This linear steady-response heuristic is not a rigorous velocity/kinematic bound.',
                    'Scheduled phase speed and achieved motion are measured separately; controller/path timing is a confound, not evidence for network capacity.',
                    'No durations, gains or step caps were retuned after inspecting outcomes.']),indent=2)+'\n')
    print(json.dumps([r for r in rows if r['phase']=='preplace'],indent=2))


if __name__=='__main__':main()
