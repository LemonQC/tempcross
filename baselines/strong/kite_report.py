"""Close the bounded KITE attempt truthfully; never launch/resume training."""
import json
import shutil
import subprocess
from pathlib import Path
from baselines.strong.common import ROOT, PROJECT, SOURCE, out, read, write, digest, now, resource, initialize


def main():
    folder=out('kite');server=ROOT/'kite';upstream=Path('/home/users/lemon/baseline-benchmark/vendor/KITE')
    git=lambda *args:subprocess.check_output(['git','-C',str(upstream),*args],text=True).strip()
    assert git('rev-parse','HEAD')=='136159544aea68f5e8b6cc64f17d221e825e3eee'
    raw=read(ROOT/'shared_geometry/ur5e_raw_summary.json');runtime=read(server/'runtime_audit.json')
    assert raw['demonstrations']==600 and raw['actions_matched_all_frames']
    assert runtime['status']=='PASS_FROZEN_PREREQUISITES_ONLY'
    assert not runtime['has_ur5e_or_panda_robotiq_only_head']
    assert 'KITE assets are ready' in (server/'download_assets.log').read_text()
    assert 'All selected assets match' in (server/'asset_verification.log').read_text()
    assert 'Installed 1 package' in (server/'pytorch3d_build_repair_01.log').read_text()
    failed=(server/'geometry_probe.log').read_text()
    assert "ModuleNotFoundError: No module named 'mimicgen'" in failed
    assert not (server/'geometry_probe').exists(), 'Unexpected simulator probe artifact: inspect before report'
    existing_module=SOURCE/'third_party/mimicgen/mimicgen/scripts/prepare_src_dataset.py'
    assert existing_module.exists()
    repair=read(server/'technical_repair_01.json')
    assert repair['id']==1 and not repair['further_technical_repairs_permitted']
    provenance=['README.md','conf/kite.yaml','asset_downloads.json','train_kinematic_decoder.py',
        'experiment/encode_contacts.py','experiment/high_level_policy/contact_encoding.py',
        'experiment/manipulation/sparse_contact_extraction.py','robot_model/robot_model.py',
        'robot_model/composite_gripper_stick_model.py',
        'experiment/high_level_policy/policy.py','experiment/pipeline_high_low_policy.py',
        'experiment/conf/train_policy/object_suite_paper2400.yaml','experiment/conf/train_policy/object_suite.yaml',
        'third_party/pytorch3d/LICENSE','third_party/Improved-3D-Diffusion-Policy/LICENSE']
    write(folder/'official_source.json',dict(repository='https://github.com/helloqxwang/KITE',commit=git('rev-parse','HEAD'),
        paper='https://arxiv.org/html/2606.22113v2',root_license='NO_TOP_LEVEL_LICENSE_FOUND',
        vendored_licenses={'pytorch3d':'BSD-3-Clause','idp3':'MIT'},upstream_source_redistributed=False,
        source=[dict(path=str(upstream/p),sha256=digest(upstream/p)) for p in provenance]))
    write(folder/'protocol.json',dict(status='NOT_FROZEN_BEFORE_BLOCKER',method='KITE',
        official_recipe_followed_to='Verified original source, released assets, native CUDA extension and frozen intent encoder',
        source_data=dict(embodiment='UR5e',identities=600,train=540,validation=60,split='Original complete-demonstration identity split'),
        target_task_demonstrations_planned=0,target_task_demonstrations_used=0,zero_shot_task_training_completed=False,
        source_robots=['ur5e'],train_target_demonstrations_per_task=0,
        observation='未部署；原生需求为object-only点云、上一intent、实测configuration及运动学模型；不同于RGB-only信息条件',
        synthetic_target_kinematic_learning='Required by method for unsupported target head; not executed',
        native_information=['object-only calibrated world-frame point cloud','previous latent intent','measured current configuration','kinematic model'],
        tier='NATIVE_3D_STATE_REQUIRED_NOT_RGB_ONLY',training_budget_frozen=False,formal_updates=None,checkpoint_rule=None,
        intended_evaluation=dict(robot='Panda',tasks=['pnp_counter_to_sink','pnp_sink_to_counter'],conditions=list(range(60,100)),
            role='EXPOSED_REFERENCE',control_hz=20,canonical_to_environment='OSC7 to existing canonical12',success='official placement AND retreat',
            actual_episodes=0),stop_rule='One documented technical repair; stop on a new integration failure',
        actual_differences=['RoboCasa-X raw HDF5 is not the native franka_robotiq unified payload',
            'Released bank lacks the bare Robotiq85/Panda target decoder; composite13 is not interchangeable with bare gripper',
            'Source intents, target kinematic decoder and OSC deployment bridge are not validated or trained']))
    write(folder/'training.json',dict(status='NOT_RUN',actual_optimizer_updates=0,actual_task_sample_exposures=0,
        actual_kinematic_training_updates=0,selected_checkpoint=None,selection_basis=None,validation_metrics=None,
        frozen_upstream_encoder_probe_only=True))
    write(folder/'evaluation.json',dict(status='NOT_RUN',episodes=0,successes=None,success_rate=None,role='EXPOSED_REFERENCE',
        cells=[dict(task=t,episodes=0,successes=None,success_rate=None) for t in ['pnp_counter_to_sink','pnp_sink_to_counter']],
        failure_trajectories=[],videos=[],failure_classification='N/A: no task trajectory exists'))
    errors=dict(technical_repairs=1,further_repairs_permitted=False,
        repaired_failure=dict(stage='NATIVE_EXTENSION_BUILD',error='cusparse.h missing from compiler-only prefix',repair_record=repair,
            result='PASS: native CUDA FPS and frozen encoder tested'),
        final_failure=dict(stage='SOURCE_GEOMETRY_PREFLIGHT_IMPORT',
            error="ModuleNotFoundError: No module named 'mimicgen'",agent_implementation_error=True,
            cause='Checker launcher omitted the already available BARX third_party/mimicgen path',
            module_exists=str(existing_module),module_sha256=digest(existing_module),
            before_simulator_construction=True,new_physics_steps=0,new_policy_episodes=0,
            no_remediation_or_relaunch_after_error=True),
        interpretation='Bounded port stopped, not evidence of KITE scientific infeasibility or policy failure',
        acquisition_logs='All index/network/Conda resolution attempts retained in absolute-path SHA index; not training updates')
    write(folder/'errors.json',errors)
    for task in ['pnp_counter_to_sink','pnp_sink_to_counter']:
        records=[r for r in read(ROOT/'shared_geometry/ur5e_raw_identity.json') if r['task']==task]
        write(folder/f'raw_identity_{task}.json',records)
    write(folder/'data_inventory.json',raw)
    write(folder/'native_runtime_audit.json',runtime)
    assets=read(upstream/'asset_downloads.json')
    write(folder/'official_assets.json',dict(release=assets['release'],verification='ALL_SELECTED_FILE_SHA256_PASS',
        bundles=[{k:v for k,v in b.items() if k!='files'}|dict(verified_files=len(b['files'])) for b in assets['bundles'] if b['name'] in ['decoder','runtime']],
        manifest_path=str(upstream/'asset_downloads.json'),manifest_sha256=digest(upstream/'asset_downloads.json'),
        target_policy_checkpoint=None))
    for name in ['installed_requirements.txt','cuda_explicit.txt']:
        shutil.copy2(server/name,folder/name)
    index=[]
    # The live report's stdout changes after this index is written. It is not
    # an experiment log; keep completed acquisition/build/probe logs instead.
    files=[p for p in server.iterdir() if p.is_file() and not (p.name.startswith('report') and p.suffix=='.log')]
    files+=[upstream/'checkpoints/decoder/release.pt',upstream/'checkpoints/decoder/dataset_meta.pt',ROOT/'shared_geometry/ur5e_raw_identity.json']
    files += [Path(r['absolute_path']) for r in raw['acquisition']['verified']]
    files += [folder/'geometry_probe.py',folder/'runtime_audit.py',PROJECT/'baselines/strong/acquire_source_states.py',PROJECT/'baselines/strong/raw_geometry_inventory.py',Path(__file__)]
    for p in sorted(set(files)):index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(folder/'sha_index.json',index)
    protected=initialize();assert all(digest(r['path'])==r['sha256'] for r in protected['files'])
    write(folder/'preservation.json',dict(status='PASS',files=protected['files'],old_baseline_reruns=0))
    write(folder/'resources.json',dict(actual_training_updates=0,task_physical_episodes=0,task_smoke_episodes=0,
        pretrained_encoder_forward_calls=1,native_cuda_fps_calls=1,
        raw_original_files_bytes=sum(int(r['bytes']) for r in raw['acquisition']['verified']),
        source_states_acquired_not_new_demonstrations=True,final_snapshot=resource(),
        native_extension_repair_build_seconds=294,large_artifacts_server_only=True))
    write(folder/'checks.json',dict(raw_identity_all600='PASS',published_asset_checksums='PASS',native_cuda_fps='PASS',
        original_frozen_contact_encoder='PASS',source_task_geometry='FAILED_BEFORE_STATE_EXTRACTION',
        target_decoder_bridge='NOT_VALIDATED',real_training_update='NOT_RUN',full_physical_smoke='NOT_RUN'))
    status=dict(status='BLOCKED',method='kite',reason_code='BOUNDED_PORT_INTEGRATION_FAILURE_AFTER_ONE_REPAIR',
        actual_training_updates=0,physical_evaluation_episodes=0,success_rate=None,technical_repairs=1,
        source_states_available=True,native_runtime_available=True,target_decoder_and_deployment_validated=False,
        full_baseline_completed=False,scientific_infeasibility_claim=False,utc=now())
    write(folder/'status.json',status);write(folder/'results.json',dict(**status,cells=[]))
    (folder/'PROTOCOL.md').write_text('''# KITE execution boundary

**NOT_FROZEN_BEFORE_BLOCKER.** Official source/assets/runtime were inspected;
no formal learning configuration, training budget, checkpoint selection or stop
criterion for model fitting was frozen because the data/deployment gate failed.
There is no trained KITE baseline and no task success rate.

The intended setting remains UR5e→Panda, original two kitchen tasks, two-finger
gripper,20Hz, canonical OSC7→12, original placement AND retreat. Conditions60–99
are EXPOSED_REFERENCE. Native point clouds/configuration/kinematic supervision
belong to a different information tier from RGB-only ACT/VQ-BeT. Panda task
demonstrations used:0. Synthetic kinematic adaptation would be allowed by KITE,
but none was executed. See protocol.json for the uncompleted gates.
''')
    (folder/'RESULTS.md').write_text('''# KITE result: BLOCKED, not a completed baseline

Official code136159544aea68f5e8b6cc64f17d221e825e3eee and both released asset
bundles were obtained and SHA256 verified. Native Torch2.9.1+cu128, CUDA12.8.93,
MuJoCo3.8.0 and the vendored PyTorch3D0.7.9 extension run in a separate prefix.
The real CUDA FPS operator and the original frozen contact encoder ran; the
encoder produced a finite256-dimensional latent from native composite-hand FK.
This is a prerequisite test, not KITE training or task evaluation.

All600 original UR5e raw demonstrations match the old action/instruction IDs
and every packed action frame;540/60 whole-demo training/validation is intact.
Full states, arm/gripper states and camera metadata exist. No raw depth field
exists. We did not establish calibrated depth/contact-label reconstruction;
missing processed RGB-only fields must not be mistaken for missing raw states.

The one technical repair supplied already installed pinned CUDA library header
paths to the extension compiler; that repair passed. A subsequent geometric
checker failed at import because my launcher omitted the existing vendored
MimicGen module path. **This is an agent integration error**, not missing public
MimicGen code, not a scientific limitation of KITE, and not a simulator failure.
No environment was constructed and no physics step occurred in that checker.
I have not repaired/relaunched it, respecting this round's one-repair ceiling.

The actual released decoder bank contains Wuji26,Allegro22,Barrett14,
Robotiq+Stick13 and MANO51 configurations, with no bare Robotiq85/Panda head.
Discarding the stick channels would not certify the target robot's kinematic
decoder. A valid future port must establish actual gripper FK/coordinate/action
correspondence, extract native source intent labels, train the required target
kinematic decoder with synthetic supervision, and pass a real update and full
smoke before freezing formal training. Existing source/Panda raw-data presence
and a successful frozen encoder do not complete those steps.

Actual new learning updates:0; task smoke/formal episodes:0; success rate:N/A;
task failure videos:N/A. No Panda task supervision, canonical diffusion-policy
substitute, manual grasp trigger or invented label was used. No conclusion
about KITE's task performance is supported. See errors.json,checks.json and
sha_index.json for reproducible evidence and all retained acquisition logs.
''')
    (folder/'README.md').write_text('''# KITE bounded attempt

**BLOCKED**; no training or task evaluation. Read RESULTS.md and errors.json.
geometry_probe.py is the preserved failed checker, not a working data converter.
Do not relaunch or fix it under this exhausted repair budget. runtime_audit.py
is a frozen prerequisite check; it does not train a baseline. Existing native
environment and original raw states are reusable if a later research instruction
authorizes a fresh port attempt. No automatic new round is started here.
''')
    from baselines.strong.matrix_details import main as matrix
    matrix();print(json.dumps(status,indent=2))


if __name__=='__main__':main()
