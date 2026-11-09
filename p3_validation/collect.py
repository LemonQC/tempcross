"""Read-only P3 evidence export; no policy training or physics stepping."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from p3_validation.common import ROOT,CACHE,PROJECT,SOURCE
from p3_validation.geometry import transform,error,site_definition
from p2_validation.offline_audit import sha256
from p3_validation.prepare_data import write_csv


def classify_contact(a,b,geom_bodies,fixtures,objects):
    def robot(name):return name.startswith(('robot0_','gripper0_'))
    ra,rb=robot(a),robot(b)
    if not (ra or rb):return 'nonrobot'
    if ra and rb:return 'robot_self'
    arm,other=(a,b) if ra else (b,a);body=geom_bodies.get(other,'') or ''
    owner=next((n for n in sorted(objects,key=len,reverse=True) if body==n or body.startswith(n+'_')),None)
    if owner=='obj' and arm.startswith('gripper0_'):return 'intended_gripper_target'
    if owner:return 'unwanted_arm_or_distractor_object'
    if any(body==n or body.startswith(n+'_') for n in fixtures):return 'unwanted_fixture'
    return 'support_or_unclassified'


def diagnostics(out,plots):
    import h5py
    import matplotlib.pyplot as plt
    results=json.loads((ROOT/'diagnostics/results.json').read_text());bank=json.loads((PROJECT/'evidence/p3/frozen/source_contact_bank.json').read_text())
    templates={r['official_index']:r for r in bank};rows=[];series={};pairs=[]
    with np.load(SOURCE/'evaluation/conditions/pnp_counter_to_sink/panda.npz',allow_pickle=False) as a:models=a['model_xmls'].tolist()
    for result in results['results']:
        row=dict(result);index=row['condition'];method=row['method'];template=templates[index];path=Path(row['telemetry_path'])
        history=json.loads(Path(row['history_path']).read_text())
        with h5py.File(path) as f:
            n=len(history);assert len(f['sim/qpos'])==n+1 and len(f['actions/environment'])==n
            assert all(np.isfinite(f[k][:]).all() for k in ['sim/qpos','sim/qvel','sim/body_xpos','sim/body_xquat','target/position','target/quaternion','actions/environment'])
            actions=f['actions/environment'][:];original=f['p3/original_environment_action'][:]
            assert np.array_equal(actions[:,7:],original[:,7:])
            if method=='A1':assert np.array_equal(actions[:,:6],original[:,:6])
            if method=='A2':assert np.array_equal(actions[:,6],original[:,6])
            names=json.loads(f.attrs['body_names']);meta=json.loads(f.attrs['ep_meta']);body,local=site_definition(models[index]);bid=names.index(body)
            tcp=np.stack([transform(p,q)@local for p,q in zip(f['sim/body_xpos'][:,bid],f['sim/body_xquat'][:,bid])])
            obj=np.stack([transform(p,q) for p,q in zip(f['target/position'],f['target/quaternion'])])
            expected=np.einsum('nij,jk->nik',obj,np.asarray(template['T_object_tcp']))
            position=np.linalg.norm(tcp[:,:3,3]-expected[:,:3,3],axis=1)
            angles=Rotation.from_matrix(expected[:,:3,:3]@np.swapaxes(tcp[:,:3,:3],1,2)).magnitude()
            geom_names=json.loads(f.attrs['geom_names']);geom_bodies={g:names[int(b)] for g,b in zip(geom_names,f['model/geom_bodyid'][:])}
            fixtures=set(meta['fixtures']);objects={o['name'] for o in meta['object_cfgs']}
            goals=f['p3/goal_tcp_world'][:]
        if method!='A1':
            peer=ROOT/f'diagnostics/condition-{index:03d}/A2/telemetry/episode-000.h5'
            if method=='A3' and peer.exists():
                with h5py.File(peer) as f:other=f['p3/goal_tcp_world'][:]
                length=min(len(other),len(goals));delta=float(np.abs(goals[:length]-other[:length]).max());assert delta==0
                pairs.append(dict(condition=index,A2_A3_shared_goal_prefix_steps=length,world_goal_max_error=delta))
        t_close=template['original_closest_action_index'] if method=='A1' else next((i for i in range(row['branch_action_index'],n) if actions[i,6]>0),None)
        row.update(first_postbranch_close_action_index=t_close,
             object_relative_tcp_position_error_at_close_m=float(position[t_close]) if t_close is not None else None,
             object_relative_tcp_rotation_error_at_close_deg=float(np.rad2deg(angles[t_close])) if t_close is not None else None,
             achieved_tcp_max_upward_excursion_while_forced_closed_m=float(tcp[t_close:min(n+1,row['branch_action_index']+276),2,3].max()-tcp[t_close,2,3]) if method=='A1' else None,
             state_values_all_finite=True,official_success_remains_separate_from_contact_lift=True)
        if method=='A3' and t_close is not None:
            gate=history[max(0,t_close-4):t_close+1]
            gate_valid=len(gate)==5 and all(h['grasp_position_error_m']<=.015 and h['grasp_rotation_error_rad']<=np.deg2rad(10) and h['phase'] in ['descend','hold_grasp'] for h in gate)
            assert gate_valid
            row.update(anchored_goal_gate_validated_at_close=True,
               anchored_goal_position_error_at_close_m=history[t_close]['grasp_position_error_m'],
               anchored_goal_rotation_error_at_close_deg=float(np.rad2deg(history[t_close]['grasp_rotation_error_rad'])),
               gate_note='Gate uses branch-anchored goal; separately reported object-relative error uses the current object pose')
        contact_rows=[json.loads(x) for x in path.with_suffix('.jsonl').read_text().splitlines()]
        counts=Counter();unwanted_states=0;support_states=0
        for record in contact_rows[row['branch_action_index']+1:]:
            classes=[classify_contact(c['geom1'] or '',c['geom2'] or '',geom_bodies,fixtures,objects) for c in record['contacts']]
            counts.update(classes);unwanted_states+=any(c.startswith('unwanted_') for c in classes);support_states+=('support_or_unclassified' in classes)
        row.update(corrected_fixture_or_object_contact_state_steps=unwanted_states,support_or_unclassified_contact_state_steps=support_states,
                   contact_classification_counts=dict(counts),collision_classification_note='Body-membership classification separates kitchen/object collisions from structural support; raw broad counts retained')
        rows.append(row);series[(index,method)]=dict(position=position,angles=np.rad2deg(angles),actions=actions[:,6],rise=np.array([h['rise_m'] for h in history]),contacts=np.array([h['bilateral_contact'] for h in history]))
        dest=out/'diagnostic_cases'/f'condition-{index:03d}'/method;dest.mkdir(parents=True,exist_ok=True)
        for name in ['result.json','branch_validation.json','world_goals.json']:
            shutil.copyfile(path.parents[1]/name,dest/name)
    summary=[]
    for method in ['A1','A2','A3']:
        subset=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,episodes=len(subset),stable_contact_lift=sum(r['stable_contact_lift'] for r in subset),official_task_successes=sum(r['task_success'] for r in subset),role='PRIVILEGED_DIAGNOSTIC_ONLY'))
    payload=dict(status=results['status'],results=rows,summary=summary,A2_A3_goal_checks=pairs,
       causal_limitations=['A1 keeps the original arm path and does not impose the A3 lift. Lack of A1 lift does not isolate static-pose error from post-contact arm behavior.',
                          'A2/A3 have identical pose goals, isolating gripper scheduling conditional on that retargeted path.',
                          'Developed cases selected using P2 outcomes; no independent generalization claim or fair vision-policy ranking.'])
    (out/'diagnostic_analysis.json').write_text(json.dumps(payload,indent=2)+'\n');write_csv(out/'diagnostic_results.csv',rows);write_csv(out/'diagnostic_summary.csv',summary)
    write_csv(out/'diagnostic_task_failures.csv',[r for r in rows if not r['task_success']])
    (out/'A2_A3_matched_world_goals.json').write_text(json.dumps(pairs,indent=2)+'\n')
    if plots:
        indices=[r['official_index'] for r in bank];fig,axes=plt.subplots(8,4,figsize=(18,24),sharex=True)
        colors={'A1':'tab:blue','A2':'tab:orange','A3':'tab:green'}
        for index,axis in zip(indices,axes):
            for method,color in colors.items():
                if (index,method) not in series:continue
                s=series[(index,method)];axis[0].plot(s['position']*1000,color=color,label=method,lw=.8)
                axis[1].plot(s['angles'],color=color,lw=.8);axis[2].plot(s['actions'],color=color,lw=.8)
                axis[3].plot(np.arange(len(s['rise']))+1,s['rise']*100,color=color,lw=.8)
                axis[3].scatter(np.flatnonzero(s['contacts'])+1,np.full(s['contacts'].sum(),-1-list(colors).index(method)),s=1,color=color)
            for ax,title in zip(axis,['Object-relative TCP error (mm)','Object-relative angle error (deg)','Executed gripper command','Object rise (cm), contact marks below']):
                ax.axvline(templates[index]['branch_action_index'],color='grey',ls=':',lw=.6);ax.set_xlim(0,600);ax.set_title(f'{index}: '+title,fontsize=8);ax.grid(alpha=.2)
            axis[0].set_ylabel('condition '+str(index));axis[0].legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel('action/state index')
        fig.suptitle('P3 privileged diagnostics — measured trajectories, not policy ranking');fig.tight_layout(rect=(0,0,1,.98))
        save_svg(fig,out/'plots/diagnostic_trajectories.svg');plt.close(fig)
    return payload


def save_svg(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,metadata={'Date':None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def formal_comparisons(out):
    data=json.loads((out/'analysis.json').read_text());rows=data['episodes'];complete={x['cell'] for x in data['cells'] if x['status']=='COMPLETE'}
    comparisons=[];paired=[]
    targets=['b1-joint_reps-panda','p2-6000-joint_reps-panda','p3-96000-joint_reps-panda']
    pairs=[(x,'original-joint_reps-panda') for x in targets]+[('p3-96000-joint_reps-panda','p2-6000-joint_reps-panda'),('p3-96000-joint_reps-panda','b1-joint_reps-panda')]
    for candidate,baseline in pairs:
        if candidate not in complete or baseline not in complete:continue
        before={r['condition_id']:r for r in rows if r['cell']==baseline};after={r['condition_id']:r for r in rows if r['cell']==candidate};assert before.keys()==after.keys()
        rescue=[k for k in before if not before[k]['success'] and after[k]['success']];regression=[k for k in before if before[k]['success'] and not after[k]['success']]
        comparisons.append(dict(candidate=candidate,baseline=baseline,episodes=len(before),rescued=len(rescue),regressed=len(regression),net_success_change=len(rescue)-len(regression),rescue_condition_ids=rescue,regression_condition_ids=regression))
        for key,b in before.items():
            a=after[key];paired.append(dict(candidate=candidate,baseline=baseline,condition_id=key,official_condition_index=b['official_condition_index'],baseline_success=b['success'],candidate_success=a['success'],baseline_stage=b['observed_stage'],candidate_stage=a['observed_stage'],baseline_contact_lift=b['sustained_contact_lift_proxy'],candidate_contact_lift=a['sustained_contact_lift_proxy']))
    (out/'paired_analysis.json').write_text(json.dumps(comparisons,indent=2)+'\n');write_csv(out/'paired_comparisons.csv',comparisons);write_csv(out/'paired_cases.csv',paired)
    return data,comparisons


def training_evidence(out,final):
    attempts=[];destination=out/'training';destination.mkdir(exist_ok=True)
    for status_file in sorted((ROOT/'training').glob('*/training_status.json')):
        directory=status_file.parent
        for _ in range(5):
            try:status=json.loads(status_file.read_text());break
            except json.JSONDecodeError:__import__('time').sleep(.1)
        target=destination/directory.name;target.mkdir(exist_ok=True)
        (target/'training_status.json').write_text(json.dumps(status,indent=2)+'\n')
        attempts.append(dict(label=directory.name,status=status['status'],batch=status['batch_size'],completed_optimizer_steps=status['completed_optimizer_steps'],sample_positions=status['sample_positions'],peak_gpu_allocated_bytes=status['peak_gpu_allocated_bytes'],peak_gpu_reserved_bytes=status['peak_gpu_reserved_bytes'],error=status.get('error')))
        for p in (directory/'runs').rglob('*'):
            if p.is_file() and p.name in ['config.json','config.yaml','adapter_config.json']:
                q=target/p.relative_to(directory);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        log=directory.with_suffix('.log')
        if log.exists():
            lines=log.read_text(errors='replace').splitlines()
            kept=[line.rstrip() for line in lines if not line.lstrip().startswith('=>> [Epoch')]
            (target/'console_without_progress_bars.txt').write_text('\n'.join(kept).rstrip()+'\n')
        execution=ROOT/'training_execution'/directory.name
        for name in ['execution.json','backward_losses.csv','initial_parameter_hashes.json']:
            p=execution/name
            if p.exists():shutil.copyfile(p,target/name)
        inputs=ROOT/'training_inputs'/directory.name
        for name in ['input_status.json','consumed_demonstration_coverage.csv']:
            p=inputs/name
            if p.exists():shutil.copyfile(p,target/name)
        metric_rows=[];metric_files=list((directory/'runs').glob('*/*.jsonl'))
        metric_files=[p for p in metric_files if p.name!='run-metrics.jsonl']
        if len(metric_files)==1:
            text=metric_files[0].read_text();lines=text.splitlines()
            if not text.endswith('\n'):lines=lines[:-1]
            metric_rows=[]
            for line in lines:
                m=json.loads(line)
                if 'VLA Train/Step' in m:
                    metric_rows.append(dict(step=m['VLA Train/Step'],loss=m['VLA Train/Loss'],raw_loss=m.get('VLA Train/Loss (Raw)'),learning_rate=m['VLA Train/Learning Rate'],step_time_seconds=m.get('VLA Train/Step Time')))
            if metric_rows:write_csv(target/'optimizer_metrics.csv',metric_rows)
        if final and directory.name.startswith('joint-reps-96000') and status['status']=='COMPLETE':
            assert status['sample_positions']==96000
            execution_state=json.loads((execution/'execution.json').read_text());input_state=json.loads((inputs/'input_status.json').read_text())
            assert execution_state['completed_backwards']==status['completed_optimizer_steps']
            assert execution_state['nonfinite_losses']==0
            assert input_state['samples_consumed']==96000 and input_state['consumed_trace_sha256']==sha256(CACHE/'sample_trace.csv')
            with (execution/'backward_losses.csv').open() as f:losses=list(csv.DictReader(f))
            assert len(losses)==status['completed_optimizer_steps'] and all(np.isfinite(float(x['raw_loss'])) for x in losses)
            assert len(metric_rows)==status['completed_optimizer_steps'] and all(np.isfinite(x['loss']) for x in metric_rows)
    write_csv(out/'training_attempts.csv',attempts)
    if (ROOT/'adapted/adapter_provenance.json').exists():
        for name in ['adapter_provenance.json','parameter_changes.json']:
            shutil.copyfile(ROOT/'adapted'/name,out/name)
    return attempts


def collect(out,final=False,plots=True):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='p3-20261109'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from p2_validation.collect import collect as policy_collect
    policy_collect(ROOT,out);data,comparisons=formal_comparisons(out);diag=diagnostics(out,plots)
    attempts=training_evidence(out,final)
    for name in ['frozen_publication.json','frozen_publication_initial.json','target_data_audit_40_59.json','training_suite.json']:
        p=ROOT/name
        if p.exists():shutil.copyfile(p,out/name)
    cache_out=out/'data';cache_out.mkdir(exist_ok=True)
    for name in ['complete.json','original_target_demonstrations.json','demonstration_coverage.csv']:
        if (CACHE/name).exists():shutil.copyfile(CACHE/name,cache_out/name)
    trace=CACHE/'sample_trace.csv';(cache_out/'sample_trace_index.json').write_text(json.dumps(dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha256(trace),rows=96000),indent=2)+'\n')
    if final:
        assert diag['status']=='COMPLETE' and len(diag['results'])==24
        suite=json.loads((ROOT/'training_suite.json').read_text());assert suite['status']=='COMPLETE'
        assert len(data['cells'])==5 and all(c['status']=='COMPLETE' and c['episodes']==20 for c in data['cells'])
        from p3_validation.validate import main as validate_completed
        validate_completed()
        verification=json.loads((ROOT/'validation/summary.json').read_text())
        assert verification['complete_independent_suite'] and verification['videos_verified']==124
        validated=set()
        for p in (ROOT/'validation').glob('*-videos.json'):
            for record in json.loads(p.read_text())['records']:validated.add(record['video_path'])
        assert validated=={str(p) for p in ROOT.rglob('*.mp4')}
        shutil.copytree(ROOT/'validation',out/'validation',dirs_exist_ok=True)
        files=[]
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':files.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=sha256(p)))
        write_csv(out/'external_files.csv',files)
    return dict(policy_cells=data['cells'],comparisons=comparisons,diagnostics=diag['summary'],training_attempts=attempts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(PROJECT/'evidence/p3/execution'));p.add_argument('--final',action='store_true');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    print(json.dumps(collect(a.out,a.final,not a.no_plots),indent=2))
