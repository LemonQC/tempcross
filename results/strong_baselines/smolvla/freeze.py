"""One post-gate budget freeze. Never tune using physical reference success."""
from pathlib import Path
from baselines.strong.common import ROOT,out,read,write,digest,resource,now,TASKS

def main():
    run=ROOT/'smolvla';directory=out('smolvla');path=directory/'protocol.json'
    if path.exists():
        assert read(directory/'freeze.json')['protocol_sha256']==digest(path)
        print('ALREADY_FROZEN_NO_CHANGE');return
    assert read(run/'acquisition.json')['status']=='PASS'
    assert read(run/'data/ready.json')['status']=='PASS_IDENTITIES_AND_REAL_ROBOT_STATE'
    assert read(run/'data/pack_index.json')['completed']==640
    probe=read(run/'probe/batch-32-padmask-correction/result.json')
    assert probe['status']=='PASS' and probe['actual_training_updates']==3 and probe['frozen_tensors_changed']==0
    assert probe['native_future_padding_mask_preserved']
    for name in ['queue_audit.json','rpc_dry_run.json','validation_probe.json']:
        assert read(run/name)['status'].startswith('PASS')
    for task in TASKS:
        assert read(directory/f'smoke_state_action_{task}.json')['status']=='PASS'
    launches=[__import__('json').loads(x) for x in (run/'physical/physical_launches.jsonl').read_text().splitlines()]
    assert len(launches)==2 and all(x['role']=='DEVELOPMENT_SMOKE' for x in launches)
    native=Path('/home/users/lemon/baseline-benchmark/vendor/lerobot-smolvla')
    protocol=dict(method='smolvla',setting='source_target_state',information_tier='RGB_PLUS_NATIVE_PROPRIOCEPTION',
        official_source='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',official_tag='v0.4.4',license='Apache-2.0',
        pretrained_revision='5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6',
        pretrained_sha256='7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb',
        backbone_metadata_revision=read(run/'acquisition.json')['backbone_revision'],
        core_network_modified=False,core_flow_matching_modified=False,
        strict_entire_checkpoint_loading=True,redundant_base_weight_download_skipped=True,
        real_update_gate='PASS',full_task_smoke_gate='PASS',readonly_replay_gate='PASS',native_queue_gate='PASS',
        dataset_source_train=540,dataset_source_validation=60,dataset_target_train=90,dataset_target_validation=10,
        target_train_per_task=45,training_data_added=0,generated_reference_data_used_for_training=0,
        input_fields=['one primary raw RGB180x320','original language, native48 tokens','arm joint qpos6/7','two Robotiq joint qpos'],
        state_feature_dimension=9,source_padding='UR5e seventh arm slot zero before native normalization',
        state_source='verified qpos from original saved full state; misnamed joint_states qvel never substituted',
        deployment_state='named simulator joint qpos, verified against every saved pre-action qpos row',
        privileged_object_state_input=False,extra_real_cameras=False,image_processing='native aspect-preserving512 resize/pad and[-1,1] normalization',
        optional_dataset_image_augmentation=False,normalization='native mean/std from original630 train demos only',
        action='original canonical OSC7; positive grip close; native Gaussian normalize all7',
        action_interface='unchanged OSC7->12,20Hz,original controller limits/gains',
        n_obs_steps=1,chunk_size=50,n_action_steps=50,physical_chunk_seconds=2.5,
        max_action_dim=32,max_state_dim=32,flow_integration_steps=10,rtc_enabled=False,
        train_expert_only=True,freeze_vlm=True,train_state_proj=True,
        actual_total_parameters=probe['total_parameters'],actual_trainable_parameters=probe['trainable_parameters'],
        batch_size=32,training_updates=16000,anchors_consumed=512000,seed=20261113,
        sampler='immutable original balanced source/target x two-task anchors; uniform demo then frame',
        optimizer='native get_optimizer_preset().build',learning_rate_peak=1e-4,betas=[.9,.95],epsilon=1e-8,
        weight_decay=1e-10,gradient_clip_norm=10,use_amp=False,
        scheduler='native get_scheduler_preset().build(num_training_steps=16000)',
        preset_warmup_updates=1000,preset_decay_updates=30000,effective_native_scaled_warmup_updates=533,
        decay_learning_rate=2.5e-6,training_loss='unchanged native forward,32 dimensions including native padded dimensions',
        padding_flag_adapter='preserve real actions_id_pad across native processor transition conversion',
        validation_anchors=256,validation_interval=1000,validation_rng='fork/restore Torch CPU+CUDA; seed17000+batch_offset',
        checkpoint_selection='minimum native flow MSE on real7 dimensions and nonpadded heldout target frames; earliest tie',
        validation_native_full32_loss_also_reported=True,
        save_interval=500,log_interval=100,minimum_free_bytes=25*1024**3,maximum_training_seconds=14400,
        stopping_rule='fixed16000 updates, halt on nonfinite/resource/error; no automatic extension or test-driven tuning',
        maximum_development_smoke_episodes=2,development_smoke_episodes_used=2,
        physical_reference_episodes_per_task=40,physical_reference_indices=list(range(60,100)),
        physical_reference_role='EXPOSED_REFERENCE',physical_success='official placement AND retreat',
        inference_seed_each_episode=7,smoke_success_used_for_selection=False,reference_success_used_for_selection=False,
        discarded_probe_optimizer_updates=9,discarded_probe_anchors=204,
        probe_history='3batch4+3batch32 hardware probes;3batch32 after padding propagation correction; all discarded',
        code_files=[dict(path=str(directory/name),sha256=digest(directory/name)) for name in
            ['data.py','runtime.py','model.py','worker.py','evaluate.py','train.py','audit.py']],
        native_files=[dict(path=str(native/name),sha256=digest(native/name)) for name in
            ['src/lerobot/policies/smolvla/modeling_smolvla.py','src/lerobot/policies/smolvla/smolvlm_with_expert.py',
             'src/lerobot/policies/smolvla/configuration_smolvla.py','src/lerobot/policies/smolvla/processor_smolvla.py',
             'src/lerobot/processor/normalize_processor.py','src/lerobot/processor/converters.py',
             'src/lerobot/policies/pretrained.py','src/lerobot/optim/optimizers.py','src/lerobot/optim/schedulers.py']],
        data_files=[dict(path=str(run/'data'/name),sha256=digest(run/'data'/name)) for name in
            ['manifest.json','statistics.json','train.npy','validation.npy','state_audit_rows.json','pack_index.json']])
    write(path,protocol)
    write(directory/'freeze.json',dict(utc=now(),protocol_sha256=digest(path),resources=resource(),
        accepted_probe_sha256=digest(run/'probe/batch-32-padmask-correction/result.json'),
        queue_sha256=digest(run/'queue_audit.json'),readonly_replay_sha256=digest(run/'readonly_replay.json'),
        validation_probe_sha256=digest(run/'validation_probe.json'),smoke_source_state_and_action_verified=True))
    write(directory/'status.json',dict(status='PARTIAL',method='smolvla',phase='FROZEN_READY_TO_TRAIN',
        information_tier='RGB_PLUS_NATIVE_PROPRIOCEPTION',actual_training_updates=0,discarded_probe_optimizer_updates=9,
        physical_evaluation_episodes=0,physical_smoke_episodes=2,success_rate=None,
        ordinary_repairs_allowed=True,formal_budget_frozen=True))
    print('FROZEN_UNIQUE_SMOLVLA_PROTOCOL',digest(path),flush=True)

if __name__=='__main__':main()
