"""Freeze the only finite visual-flow port after genuine update and2 smokes."""
import inspect
import subprocess
import torchcfm.conditional_flow_matching as cfm
import torchcfm.optimal_transport as ot
import diffusers.training_utils as ema_module
import diffusers.optimization as scheduler_module
import torchvision.models.resnet as resnet_module
from .native import SOURCE,REVISION
from .data import RUN,DIRECTORY
from baselines.strong.common import PROJECT,out,read,write,digest,now,resource,TASKS

def main():
    directory=out('flow_matching_policy');path=directory/'protocol.json'
    if path.exists():print('PROTOCOL_ALREADY_FROZEN_NO_MUTATION');return
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==REVISION
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
    assert read(RUN/'probe/result.json')['status']=='PASS' and read(RUN/'probe/result.json')['actual_optimizer_calls']==1
    assert read(RUN/'rpc_audit.json')['status'].startswith('PASS')
    assert read(RUN/'validation_probe.json')['status'].startswith('PASS')
    assert read(directory/'code_checks.json')['native_data_tests']==4
    for task in TASKS:assert read(directory/('smoke_state_action_'+task+'.json'))['status']=='PASS'
    source_names=['README.md','LICENSE.md','examples/flow_pusht.py','examples/flow_mimic.py',
        'external/models/unet.py','external/models/resnet.py','external/models/pusht.py']
    native=[SOURCE/name for name in source_names]+[__import__('pathlib').Path(inspect.getfile(m)) for m in [cfm,ot,ema_module,scheduler_module,resnet_module]]
    official=dict(repository='https://github.com/HRI-EU/flow_matching',revision=REVISION,branch='master',license='BSD-3-Clause',
        visual_reference='examples/flow_pusht.py',robomimic_reference='examples/flow_mimic.py',
        actual_variant='official visual flow-component faithful port to fixed RoboCasa-X',
        native_network_imported=True,prior_local_DP_network_imported=False,pretrained_weights_used=False,
        initialization='author visual default weights=None, from scratch; seed20261113',
        public_checkpoints_not_reused='PushT/Kitchen/Robomimic observation/action tensor shapes differ; no misleading partial pretrained loading',
        original_Robomimic_lowdim_absolute_recipe_reproduced=False,full_affordance_prompt_framework_reproduced=False,
        native_files=[dict(path=str(p),sha256=digest(p)) for p in native])
    write(directory/'official_source.json',official)
    protocol=dict(method='flow_matching_policy',setting='source_target_visual',base_commit='3c9671e8ca50cfca06b8f6e46ff3e97d1461cb4c',
        official_revision=REVISION,official_license='BSD-3-Clause',TorchCFM='1.0.7',diffusers='0.35.2',
        variant=official['actual_variant'],original_Robomimic_lowdim_absolute_recipe_reproduced=False,
        initialization='from scratch weights=None; exact reset to pre-probe parameter hashes',
        information_tier='RGB96_PLUS_REAL_QPOS9_PLUS_FROZEN_CLIP768',
        dataset_source_train=540,dataset_source_validation=60,dataset_target_train=90,dataset_target_validation=10,
        target_train_per_task=45,new_demonstrations=0,generated_demos_used_for_training=0,
        RGB='one original180x320 RGB ->cv2 default LINEAR96 square, CPU float32[0,1], contiguous NCHW',
        proprioception='verified real arm qpos7 plus2 Robotiq joint qpos; UR5e seventh-arm slot zero',
        language='same frozen768D CLIP-L/14 inputs for all16 full object-specific instructions; no task-ID collapse',
        object_state_input=False,extra_real_camera=False,
        normalization='native per-dimension train-only min/max[-1,1] for qpos9 and canonical OSC7; no normalization changes at test',
        boundary_padding='native sample_sequence repeat-last; permit15 rather than7 future tail slots to preserve original anchor frames',
        sampler='byte-identical original balanced1M source/target x task anchors; consume first768000',
        original_validation_anchor_identity_and_frames_preserved=True,
        network='author GN ResNet18 and ConditionalUnet1D; no local DP substitute',
        core_layer_topology_changed=False,interface_dimensions_changed=True,
        native_UNet_down_dims=[256,512,1024],native_kernel_size=5,native_norm_groups=8,native_time_embedding=256,
        global_condition_dim=1289,action_dim=7,parameters=94874439,trainable_parameters=94874439,
        observation_horizon=1,prediction_horizon=16,execution_horizon=8,physical_execution_seconds=.4,
        training_source_distribution='Gaussian torch.randn as official source',
        flow_matcher='native TorchCFM ConditionalFlowMatcher(sigma=0)',training_loss='unchanged full16x7 mean squared velocity error, including native repeated boundary values; no masks',
        inference_source_distribution='uniform torch.rand[0,1] as official visual and Robomimic code',
        Gaussian_training_uniform_deployment_mismatch_retained=True,
        ODE_solver='official explicit Euler',integration_steps=1,integration_t0=0,integration_dt=1,
        action_interface='unchanged canonical OSC7 positive-close ->OSC12 at20Hz; no IK/controller/phase modifications',
        batch_size=32,training_updates=24000,anchors_consumed=768000,seed=20261113,
        optimizer='native-example torch.optim.AdamW',learning_rate=1e-4,weight_decay=1e-6,betas=[.9,.999],epsilon=1e-8,
        gradient_clipping=False,mixed_precision=False,
        scheduler='native diffusers get_scheduler cosine',warmup_updates=500,
        ema_constructor='native diffusers EMAModel(parameters,power=0.75)',ema_power=.75,ema_use_library_default_warmup=False,
        EMA_bookkeeping_fix='store raw before copy_to EMA for validation/checkpoint, restore raw before next optimizer call',
        validation_anchors=256,validation_interval=1000,
        validation_noise='fixed Gaussian flow-loss noise with seed17000+batch_offset; restore CPU+CUDA RNG',
        checkpoint_selection='minimum fixed target-validation EMA full16x7 velocity MSE; earliest tie; never formal success',
        save_interval=500,log_interval=100,maximum_training_seconds=14400,minimum_free_bytes=25*1024**3,
        stopping_rule='exact24000 optimizer calls, halt on nonfinite/error/resource; no automatic budget extension or tuning',
        discarded_optimizer_probe_calls=1,probe_constant_peak_LR_no_scheduler=True,
        probe_pre_contiguity_unification=True,probe_weights_reused_for_formal_training=False,
        maximum_smoke_episodes=2,smoke_episodes_used=2,smoke_successes=0,
        smoke_gate='full600/650 episodes, native-generation/normalization/OSC/qpos/video/goal acceptance; not positive task success',
        formal_episodes_per_task=40,formal_indices=list(range(60,100)),formal_role='EXPOSED_REFERENCE',
        official_success='placement AND retreat',control_frequency_Hz=20,per_episode_inference_seed=7,
        formal_success_used_for_selection=False,
        code_files=[dict(path=str(directory/name),sha256=digest(directory/name)) for name in
            ['data.py','native.py','runtime.py','model.py','worker.py','evaluate.py','audit.py','train.py']],
        native_files=official['native_files'],data_files=[dict(path=str(DIRECTORY/name),sha256=digest(DIRECTORY/name)) for name in
            ['manifest.json','statistics.json','language.json','train.npy','validation.npy']])
    write(path,protocol)
    write(directory/'freeze.json',dict(utc=now(),protocol_sha256=digest(path),resources=resource(),
        probe_sha256=digest(RUN/'probe/result.json'),RPC_gate_sha256=digest(RUN/'rpc_audit.json'),
        validation_probe_sha256=digest(RUN/'validation_probe.json'),smoke_native_state_and_action_gates='PASS_BOTH'))
    status=read(directory/'status.json');status.update(phase='FROZEN_READY_TO_TRAIN',formal_budget_frozen=True,
        discarded_probe_optimizer_calls=1,physical_smoke_episodes=2,information_tier=protocol['information_tier'])
    write(directory/'status.json',status)
    write(RUN/'formal_ready.json',dict(status='FROZEN',protocol_sha256=digest(path),next='Publish preregistration, launch one run_frozen; no more smokes'))
    print('FROZEN_ONE_NATIVE_VISUAL_FLOW_PROTOCOL',digest(path),flush=True)

if __name__=='__main__':main()
