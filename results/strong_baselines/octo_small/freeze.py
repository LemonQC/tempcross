"""Freeze once after real update/native-data/full-task smoke acceptance."""
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,out,write,read,digest,resource,now,TASKS

def main():
    directory=out('octo_small');run=ROOT/'octo_small'
    probe=read(run/'probe/batch-32/result.json');assert probe['status']=='PASS' and probe['actual_training_updates']==3
    assert probe['changed_tensors']>0 and probe['frozen_language_changed_tensors']==0
    audit=read(run/'native_data_audit.json');assert audit['status']=='PASS' and audit['cross_task_token_collisions']==0
    for task in TASKS:
        r=read(directory/f'smoke_source_target_{task}.json');assert r['status']=='PASS' and r['episodes']==1
    path=directory/'protocol.json'
    if path.exists():
        assert read(directory/'freeze.json')['protocol_sha256']==digest(path)
        print('ALREADY_FROZEN_NO_CHANGE');return
    code=['data.py','model.py','runtime.py','train.py','launch.py','pack.py']
    native=Path('/home/users/lemon/baseline-benchmark/vendor/octo')
    protocol=dict(method='octo_small',setting='source_target',algorithm='official Octo-Small1.5 full finetuning',
        official_source='241fb3514b7c40957a86d869fecb7c7fc353f540',weights_revision='dc9aa3019f764726c770814b27e4ab0fc6e32a58',
        native_recipe='scripts/configs/finetune_config.py:full,language_conditioned',
        core_architecture_modified=False,pretrained_weights=probe['checkpoint'].split('/probe/')[0]+'/pretrained/300000/default/checkpoint',
        real_update_gate='PASS',full_task_smoke_gate='PASS',native_data_gate='PASS',
        official_default_batch_size=256,official_default_training_updates=50000,
        batch_size=32,training_updates=24000,anchors_consumed=768000,seed=20261113,
        source_train_demonstrations=540,source_validation_demonstrations=60,
        target_train_demonstrations=90,target_validation_demonstrations=10,target_train_per_task=45,
        domain_and_task_sampling='equal four groups, uniform full-demo ID then uniform frame',
        information=['primary RGB256','original instruction, official T5-base max16 tokens'],
        extra_robot_state=False,real_wrist_camera=False,goal_image=False,missing_modalities='zero padding, false native masks',
        history=2,action_horizon=4,execute_actions=4,action_dimensions=7,
        canonical_interface='original OSC7->12;20Hz;0.05m/0.5rad controller limits;positive-close canonical gripper',
        native_action='base-frame metre/radian delta and0..1 openness; arm Gaussian normalized with train-only stats',
        optimizer=dict(name='native AdamW',learning_rate='native warmup cosine',peak=3e-4,warmup_updates=2000,
            end=0.0,weight_decay=.01,clip_gradient=1.0,mu_dtype='bfloat16',frozen_keys=['*hf_model*']),
        log_interval=100,save_interval=500,validation_interval=1000,
        validation_anchors=256,validation_rng='PRNGKey(17000+batch_offset)',
        checkpoint_selection='minimum native DDPM validation loss at1000..24000; earliest tie',
        stopping_rule='fixed24000 updates; no performance-dependent extension; halt on nonfinite/error/disk/wall cap',
        minimum_free_bytes=25*1024**3,maximum_training_seconds=14400,
        formal_evaluation_role='EXPOSED_REFERENCE',formal_conditions=list(range(60,100)),formal_episodes_per_task=40,
        formal_success='official placement AND retreat',diffusion_seed_each_episode=7,
        smoke_used_for_model_selection=False,exposed_reference_used_for_model_selection=False,
        discarded_probe_optimizer_updates=6,discarded_probe_anchors=120,
        data_files=[dict(path=str(run/'data'/name),sha256=digest(run/'data'/name))
            for name in ['manifest.json','action_statistics.json','train.npy','validation.npy']],
        code_files=[dict(path=str(directory/name),sha256=digest(directory/name)) for name in code],
        official_core_files=[dict(path=str(native/name),sha256=digest(native/name)) for name in
            ['octo/model/octo_model.py','octo/model/octo_module.py','octo/model/components/action_heads.py',
             'octo/utils/train_utils.py','octo/data/traj_transforms.py','octo/data/obs_transforms.py']],
        pack_gate_before_training='PASS640 original training/target-validation native RGB caches; no new simulation')
    write(path,protocol)
    write(directory/'freeze.json',dict(utc=now(),protocol_sha256=digest(path),resources=resource(),
        real_probe_sha256=digest(run/'probe/batch-32/result.json'),native_data_audit_sha256=digest(run/'native_data_audit.json'),
        smoke_acceptances=[dict(path=str(directory/f'smoke_source_target_{task}.json'),
            sha256=digest(directory/f'smoke_source_target_{task}.json')) for task in TASKS]))
    write(directory/'status.json',dict(status='PARTIAL',method='octo_small',phase='FROZEN_WAITING_LOSSLESS_CACHE',
        actual_training_updates=0,discarded_probe_optimizer_updates=6,physical_evaluation_episodes=0,
        physical_smoke_episodes=2,success_rate=None,ordinary_repairs_allowed=True,formal_budget_frozen=True))
    print('FROZEN',digest(path),flush=True)

if __name__=='__main__':main()
