"""Validate recorded physical evidence and publish small, auditable artifacts.

This command performs no training, policy inference, or physics stepping.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def finalize(root,out,index_external=True):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from p2_validation.collect import collect,write_csv
    root=Path(root).resolve();out=Path(out).resolve();collect(root,out)
    data=json.loads((out/'analysis.json').read_text())
    frozen=json.loads((root/'script-freeze.json').read_text());versions=[]
    for name,old in frozen['scripts'].items():
        current=digest(Path(__file__).parent/name)
        versions.append(dict(file=name,initial_sha256=old,final_sha256=current,changed=current!=old))
    assert all(not r['changed'] for r in versions if r['file'] in ['controls.py','run_policy.py','execute.py'])
    (out/'execution_versions.json').write_text(json.dumps(dict(scripts=versions,
        experimental_rules_changed_after_formal_outcomes=False,
        notes=['Rule file frozen before formal launch; script hash snapshot captured 20 seconds after first process launch, during model loading.',
               'Post-snapshot B2 v3 adds full state/action telemetry to the same probes.',
               'Training cache preparation limits TensorFlow CPU threads; no change to sample/window/mixture settings.',
               'Training adapter adds exact completed-job reuse to support overlapping Joint training and No Reps evaluation; no additional updates.',
               'Matched budgets mean samples and optimizer updates, not exact token FLOPs or GPU seconds.']),indent=2)+'\n')
    audit=[];stage_counts=[]
    for cell in data['cells']:
        if cell['status']!='COMPLETE':continue
        rows=[r for r in data['episodes'] if r['cell']==cell['cell']]
        for label,count in sorted(Counter(r['observed_stage'] for r in rows).items()):
            stage_counts.append(dict(cell=cell['cell'],observed_stage=label,episodes=count))
        fig,axes=plt.subplots(5,4,figsize=(15,14),sharex=True,sharey=True)
        for ax,row in zip(axes.flat,rows):
            path=Path(row['telemetry_path'])
            with h5py.File(path,'r') as f:
                checks={}
                for key in ['sim/qpos','sim/qvel','sim/body_xpos','sim/body_xquat','target/position',
                            'obs/robot0_eef_pos','obs/robot0_gripper_qpos','actions/environment',
                            'p2/raw_gripper','p2/pre_action_pad_center_distance']:
                    array=f[key][:];checks[key]=bool(np.isfinite(array).all())
                assert all(checks.values()),str(path)
                assert len(f['sim/qpos'])==row['steps']+1
                assert len(f['actions/environment'])==row['steps']
                obj=f['target/position'][:];rise=obj[:,2]-obj[0,2]
                span=f['p2/pre_action_pad_center_distance'][:]
            history=[json.loads(x) for x in path.with_suffix('.jsonl').read_text().splitlines()]
            contact=np.array([r['grasp_contact'] is True for r in history])
            assert len(contact)==len(rise)
            audit.append(dict(cell=cell['cell'],condition_id=row['condition_id'],steps=row['steps'],
                              all_finite=True,checks=checks,complete_state_action_lengths=True))
            ax.plot(np.arange(len(rise)),rise*100,label='object rise (cm)')
            ax.plot(np.arange(len(span)),span*100,label='pad center span (cm)',alpha=.65)
            ax.scatter(np.flatnonzero(contact),np.full(contact.sum(),-2.),s=2,color='green',label='bilateral contact')
            ax.set_title(f"{row['official_condition_index']}: {'success' if row['success'] else row['observed_stage']}",fontsize=7)
            ax.axhline(3,color='grey',lw=.5,ls=':');ax.set_ylim(-5,35);ax.set_xlim(0,600);ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('executed action index')
        for ax in axes[:,0]:ax.set_ylabel('cm')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3)
        fig.suptitle(cell['cell']+' — recorded trajectories (contact/lift are diagnostic proxies)')
        fig.tight_layout(rect=(0,.025,1,.97))
        plots=out/'plots';plots.mkdir(exist_ok=True)
        destination=plots/(cell['cell']+'.svg');fig.savefig(destination);plt.close(fig)
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    (out/'state_integrity.json').write_text(json.dumps(dict(episodes=len(audit),all_verified=bool(audit),records=audit),indent=2)+'\n')
    write_csv(out/'stage_counts.csv',stage_counts)
    # Exact frozen identity is checked independently from the evaluation wrapper.
    identities=[]
    for path in sorted((root/'formal').glob('*/job.json')):
        job=json.loads(path.read_text())
        if job['status']!='COMPLETE':continue
        arm=job['robot'];bundle=Path('/home/users/lemon/coft-p2/barx/evaluation/conditions/pnp_counter_to_sink')
        payload=json.loads((bundle/(arm+'.json')).read_text())
        expected=[{k:e[k] for k in ['episode','condition_id','seed','state_sha256','model_sha256','metadata_sha256']}
                  for e in payload['episodes'][20:40]]
        assert job['indices']==list(range(20,40)) and job['conditions']==expected
        identities.append(dict(cell=path.parent.name,exact_official_entries_20_39=True,
                               dev_indices_disjoint=True,condition_ids=[e['condition_id'] for e in expected]))
    (out/'condition_integrity.json').write_text(json.dumps(identities,indent=2)+'\n')
    train_out=out/'training';train_out.mkdir(exist_ok=True)
    training_audit=[];curves=[]
    for label in ['full-step-probe','lora-step-probe','no_reps-lora','joint_reps-lora']:
        directory=root/'training'/label
        if not (directory/'training_status.json').exists():continue
        destination=train_out/label;destination.mkdir(exist_ok=True)
        shutil.copyfile(directory/'training_status.json',destination/'training_status.json')
        for p in (directory/'runs').rglob('*'):
            if p.is_file() and p.suffix in ['.json','.jsonl','.yaml']:
                target=destination/p.relative_to(directory);target.parent.mkdir(parents=True,exist_ok=True)
                # Repeated auxiliary-metric keys make the raw training stream
                # large. Keep it on the server (indexed below), commit every
                # optimizer step as compact CSV plus console/hyperparameter logs.
                if p.suffix=='.jsonl' and p.name!='run-metrics.jsonl':
                    if target.exists():target.unlink()
                    continue
                shutil.copyfile(p,target)
        actual_log=root/'training'/(label+'-actual.log')
        log=actual_log if actual_log.exists() else root/'training'/(label+'.log')
        if log.exists():
            # Preserve all text, while making tqdm carriage-return output readable.
            (destination/'console.log').write_text(log.read_text(errors='replace'))
        status=json.loads((directory/'training_status.json').read_text())
        if label in ['no_reps-lora','joint_reps-lora'] and status['status']=='COMPLETE':
            metrics=list((directory/'runs').rglob(label.replace('-lora','--lora')+'.jsonl'))
            assert len(metrics)==1
            logs=[json.loads(line) for line in metrics[0].read_text().splitlines()]
            assert len(logs)==status['completed_optimizer_steps']==3000
            assert [x['VLA Train/Step'] for x in logs]==list(range(1,3001))
            assert all(np.isfinite(x['VLA Train/Loss']) and x['VLA Train/Learning Rate']==2e-5 for x in logs)
            compact=[dict(step=x['VLA Train/Step'],loss=x['VLA Train/Loss'],
                          learning_rate=x['VLA Train/Learning Rate'],action_l1=x['VLA Train/L1 Loss']) for x in logs]
            write_csv(destination/'optimizer_metrics.csv',compact)
            curves.append((status['method'],compact))
            training_audit.append(dict(method=status['method'],optimizer_steps=3000,finite_loss_every_step=True,
                                       sample_positions=status['sample_positions'],sample_trace_sha256=status['sample_trace_sha256'],
                                       mean_first_100_loss=float(np.mean([r['loss'] for r in compact[:100]])),
                                       mean_last_100_loss=float(np.mean([r['loss'] for r in compact[-100:]]))))
    (out/'training_integrity.json').write_text(json.dumps(training_audit,indent=2)+'\n')
    if curves:
        fig,axes=plt.subplots(len(curves),2,figsize=(12,4*len(curves)),squeeze=False)
        for (method,rows),axis in zip(curves,axes):
            for ax,key in zip(axis,['loss','action_l1']):
                y=np.array([r[key] for r in rows]);x=np.array([r['step'] for r in rows])
                ax.plot(x,y,alpha=.2,lw=.5)
                if len(y)>=100:ax.plot(x[99:],np.convolve(y,np.ones(100)/100,mode='valid'),lw=1.)
                ax.set_title(method+' '+key);ax.set_xlabel('completed optimizer steps');ax.grid(alpha=.2)
        fig.suptitle('Restricted adaptation: raw and 100-step means; composite losses differ by method')
        fig.tight_layout(rect=(0,0,1,.96));destination=out/'plots/training-curves.svg'
        fig.savefig(destination);plt.close(fig)
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    suite=json.loads((root/'adaptation_suite.json').read_text())
    checkpoints=[]
    for model in suite.get('completed_models',[]):
        checkpoint=Path(model['checkpoint']);assert checkpoint.is_file()
        record=dict(**model,checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=digest(checkpoint))
        provenance=checkpoint.parents[1]/'adapter_provenance.json'
        if provenance.exists():
            record['provenance']=json.loads(provenance.read_text())
            adapter=Path(record['provenance']['adapter']);weight=adapter/'adapter_model.safetensors'
            from safetensors import safe_open
            with safe_open(weight,framework='pt',device='cpu') as tensors:
                bkeys=[key for key in tensors.keys() if 'lora_B' in key]
                norms=[float(tensors.get_tensor(key).float().norm()) for key in bkeys]
                assert bkeys and max(norms)>0
            record.update(adapter_path=str(adapter),adapter_sha256=digest(weight),
                          lora_b_tensors=len(bkeys),nonzero_lora_b_tensors=sum(n>0 for n in norms))
        checkpoints.append(record)
    (out/'checkpoint_index.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
    for name in ['development-video-validation.json','source-video-validation.json',
                 'target-video-validation.json','adapted-video-validation.json','b2-video-validation.json']:
        p=root/name
        if p.exists():shutil.copyfile(p,out/name)
    cache=root.parent/'matched-samples-6000'
    shutil.copyfile(cache/'sample_trace.csv',out/'matched_sample_trace.csv')
    b2=root/'b2-diagnostics-v3'
    b2_rows=[]
    for p in sorted(b2.glob('*-original.json'))+sorted(b2.glob('*-close32.json')):
        history=json.loads(p.read_text())
        assert all(np.isfinite(h['qpos']).all() and np.isfinite(h['qvel']).all() and
                   np.isfinite(h['full_environment_action']).all() for h in history)
        b2_rows.append(dict(path=str(p),steps=len(history),all_finite=True))
    (out/'b2_telemetry_integrity.json').write_text(json.dumps(b2_rows,indent=2)+'\n')
    probes=json.loads((b2/'results.json').read_text())['probes']
    fig,axes=plt.subplots(len(probes),3,figsize=(15,18),sharex=True)
    mechanisms=[]
    for axis,probe in zip(axes,probes):
        method,_,index=probe['case_id'].split('/')
        stem=f"{method}-{int(index):03d}-step-{probe['branch_action_index']}"
        for label,color in [('original','tab:blue'),('close32','tab:orange')]:
            history=json.loads((b2/f'{stem}-{label}.json').read_text())
            t=np.array([h['step'] for h in history])
            axis[0].plot(t,[h['gripper_command'] for h in history],label=label,color=color,alpha=.8)
            axis[1].plot(t,[h['pad_center_distance_m']*100 for h in history],color=color,alpha=.8)
            axis[2].plot(t,[h['rise_m']*100 for h in history],color=color,alpha=.8)
            contact=np.array([h['bilateral_contact'] for h in history])
            axis[2].scatter(t[contact],np.full(contact.sum(),-1 if label=='original' else -2),s=1,color=color)
        for ax in axis:
            ax.axvspan(probe['branch_action_index']+1,probe['branch_action_index']+32,color='grey',alpha=.15)
            ax.grid(alpha=.2)
        axis[0].set_ylabel(probe['case_id']+'\nstep '+str(probe['branch_action_index']),fontsize=8)
        for result in probe['outcomes']:
            mechanisms.append(dict(case_id=probe['case_id'],branch_action_index=probe['branch_action_index'],
                                   positive_control=probe['positive_control'],branch_trusted=probe['branch_trusted'],**result))
    axes[0,0].set_title('Executed gripper command');axes[0,1].set_title('Pad center span (cm)')
    axes[0,2].set_title('Object rise (cm); contact marks at bottom')
    for ax in axes[-1]:ax.set_xlabel('executed action index (1-based)')
    axes[0,0].legend();fig.suptitle('B2 physical replay pairs — seven probes, not seven independent failure cases')
    fig.tight_layout(rect=(0,0,1,.97));destination=out/'plots/b2-causal-probes.svg'
    fig.savefig(destination);plt.close(fig)
    destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    write_csv(out/'b2_mechanism.csv',mechanisms)
    # All heavy artifacts stay on the host, indexed by absolute path and digest.
    files=[]
    if index_external:
        for p in sorted(root.rglob('*')):
            if p.is_file():files.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
        write_csv(out/'external_files.csv',files)
    print(json.dumps(dict(verified_episodes=len(audit),indexed_files=len(files),stages=stage_counts),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--skip-external-index',action='store_true',help='For intermediate validation while artifacts are still being written')
    a=p.parse_args();finalize(a.root,a.out,index_external=not a.skip_external_index)
