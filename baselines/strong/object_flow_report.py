"""Close the bounded Object-Flow label gate; no substitute network is built."""
import shutil
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,SOURCE,out,read,write,digest,now,resource,initialize


def main():
    folder=out('object_flow');server=ROOT/'object_flow'
    first_path=server/'label_probe/source-pnp_counter_to_sink/error.json'
    second_path=server/'repaired_01/label_probe/source-pnp_counter_to_sink/error.json'
    first,second=read(first_path),read(second_path);repair=read(server/'technical_repair_01.json')
    assert 'postprocess_visual_obs' in first['error']
    assert 'ur5e_agentview_left' in second['error'] and 'robot0_agentview_left' in second['error']
    assert repair['id']==1 and repair['further_technical_repairs_permitted'] is False
    assert not list((server/'repaired_01').rglob('*.png')), 'Unexpected label image; inspect before closure'
    assert not list((server/'repaired_01').rglob('acceptance.json'))
    paper=Path('/home/users/lemon/post-gate-audit-20261110/literature/cross-behavior-TRO2026.pdf');text=paper.with_suffix('.txt')
    for name in ['github_search.json','primary_availability.json','technical_repair_01.json']:
        shutil.copy2(server/name,folder/name)
    shutil.copy2(server/'attempt_00/label_probe.py',folder/'attempt_00_label_probe.py')
    write(folder/'official_source.json',dict(title='Cross-Behavior Learning with Object Flow Prediction for Robotic Manipulation',
        authors=['Longrui Chen','David Russell','Yulei Qiu','Mehmet Dogar'],doi='10.1109/TRO.2026.3701576',
        author_accepted_version='https://eprints.whiterose.ac.uk/id/eprint/242108/1/26-0166_02_MS.pdf',
        institutional_record='https://eprints.whiterose.ac.uk/id/eprint/242108/',paper_license='CC-BY-4.0',
        implementation_repository=None,implementation_sha=None,implementation_license='NOT_VERIFIED',
        author_implementation_verified=False,negative_search_is_bounded=True,
        paper_path=str(paper),paper_sha256=digest(paper),text_sha256=digest(text)))
    write(folder/'protocol.json',dict(status='NOT_FROZEN_LABEL_CORRESPONDENCE_GATE',method='Cross-Behavior Object Flow',
        intended_component_scope='METHOD_PORT_SAME_BEHAVIOR',component_completed=False,cross_behavior_claim=False,
        training_budget_frozen=False,source_robots=['ur5e'],train_target_demonstrations_per_task=0,
        intended_target_training_demonstrations_per_task=45,actual_target_demonstrations_used=0,
        observation='未部署；论文原生agent/hand RGB、EE/gripper状态、指令；流只可由学习预测器产生',
        source_pool=dict(total=600,train=540,validation=60),target_pool=dict(total=100,train=90,validation=10),
        split='Existing complete-demonstration identity split unchanged',
        label_gate=dict(only_training_ids=True,frames='0,middle,last',exact_numeric_state_roundtrip=True,
            raw_rgb_normalized_mae_max=.03,task_object_roles=['obj','container if present'],
            privileged_training_annotation='Proposed rendered visible instance masks, not certified or produced',
            actual_valid_label_frames=0),
        original_core_requirements=['Future binary object masks','Learned mask-flow predictor',
            'Frame and frame-union dice/focal supervision','Learned flow-CNN conditioning of action diffusion policy'],
        actual_core_implementation=dict(flow_predictor='NOT_IMPLEMENTED',flow_conditioned_policy='NOT_IMPLEMENTED'),
        formal_training_updates=None,checkpoint_selection=None,
        intended_evaluation=dict(robot='Panda',two_finger=True,tasks=['pnp_counter_to_sink','pnp_sink_to_counter'],
            conditions=list(range(60,100)),role='EXPOSED_REFERENCE',control_hz=20,
            action_interface='Existing canonical OSC7 to12',success='Official placement AND retreat'),
        repair_ceiling=1,repairs_used=1,stop_on_new_substantive_gate_failure=True))
    write(folder/'training.json',dict(status='NOT_RUN',actual_optimizer_updates=0,actual_sample_exposures=0,
        flow_prediction_updates=0,flow_conditioned_policy_updates=0,validation_results=None,checkpoint=None))
    write(folder/'evaluation.json',dict(status='NOT_RUN',physical_smoke_episodes=0,formal_episodes=0,successes=None,
        success_rate=None,failure_classification=None,videos=[],no_policy_trajectory_exists=True))
    write(folder/'label_gate.json',dict(status='FAILED_CAMERA_NAME_CORRESPONDENCE',first_attempt=first,repaired_attempt=second,
        numeric_state_roundtrip_passed_before_render=True,numeric_roundtrip_is_not_semantic_geometry_validation=True,
        rendered_rgb_alignment_test_completed=False,valid_segmentation_frames=0,object_flow_labels=0,
        camera_alias_mapping_validated=False,hand_view_raw_dataset_present=False,
        source_model_has_eye_in_hand_camera_in_error_listing=True,
        known_raw_states_available=True,absence_of_supervision_is_not_proved=True))
    write(folder/'errors.json',dict(technical_repairs=1,further_technical_repairs_permitted=False,
        first_failure='Agent passed a duplicate factory keyword; no environment constructed',
        repair='Removed duplicate keyword once; preserved original source/log/error and separate repaired root',
        final_failure='Raw saved camera name is absent in current reconstructed model; RGB/mask correspondence not certified',
        interpretation='Engineering/data-interface blocker, not object-flow algorithm failure or proof existing states cannot supply labels',
        no_alias_patch_or_relaunch_after_final_failure=True,
        no_bbox_or_static_mask_substitution=True,no_invented_supervision=True,
        counter_warning='The preserved probe hooks only mj_step. Kitchen initialization uses mj_step1/2; zero counter values do NOT mean zero total physics.'))
    write(folder/'resources.json',dict(actual_training_updates=0,learned_policy_inferences=0,
        formal_evaluation_episodes=0,full_physical_smoke_episodes=0,new_generated_demonstrations=0,
        original_failed_attempt_environment_constructed=False,repaired_attempt_environment_constructed=True,
        repaired_explicit_reset_calls=1,repaired_explicit_saved_state_restorations_reached=2,
        observed_mj_step_calls=0,total_mj_step1_step2_bootstrap_integration_count=None,
        bootstrap_settling_occurred=True,zero_total_physics_steps_claim=False,
        counter_limitation='Initialization executes10*int(control_timestep/model_timestep) step1/step2 pairs per reset in kitchen.py; total calls not fully instrumented.',
        final_resource_snapshot=resource()))
    files=[p for p in server.rglob('*') if p.is_file() and not p.name.startswith('report')]
    kitchen=SOURCE/'robocasa_x/robocasa/environments/kitchen/kitchen.py'
    files += [paper,text,folder/'label_probe.py',Path(__file__),kitchen,
        SOURCE/'robocasa_x/robocasa/utils/robomimic/robomimic_env_utils.py']
    write(folder/'sha_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))])
    protected=initialize();assert all(digest(r['path'])==r['sha256'] for r in protected['files'])
    write(folder/'preservation.json',dict(status='PASS',old_baseline_reruns=0,files=protected['files']))
    write(folder/'checks.json',dict(author_paper='VERIFIED_ACCEPTED_VERSION',author_source_code='NOT_VERIFIED',
        raw_source_action_identity='PASS_ALL600_FROM_PRIOR_SHARED_AUDIT',numeric_restore='PASSED_BEFORE_FIRST_RENDER',
        semantic_camera_and_mask_correspondence='FAILED_BEFORE_LABEL_CREATION',learned_flow_prediction='NOT_IMPLEMENTED',
        learned_flow_to_action_connection='NOT_IMPLEMENTED',real_training_update='NOT_RUN',full_physical_smoke='NOT_RUN'))
    status=dict(status='BLOCKED',method='object_flow',reason_code='CAMERA_LABEL_CORRESPONDENCE_AFTER_ONE_REPAIR',
        intended_scope='METHOD_PORT_SAME_BEHAVIOR',component_completed=False,full_baseline_completed=False,
        actual_training_updates=0,physical_evaluation_episodes=0,success_rate=None,technical_repairs=1,utc=now())
    write(folder/'status.json',status);write(folder/'results.json',dict(**status,cells=[]))
    (folder/'PROTOCOL.md').write_text('''# Object Flow boundary

**NOT_FROZEN**: valid label correspondence was not established, so no formal
training budget/checkpoint was frozen. Intended scope:METHOD_PORT_SAME_BEHAVIOR,
not the paper's cross-behavior experiment. Original UR5e/Panda whole-demo splits,
two tasks,two-finger,20Hz,canonical OSC7→12 and official placement AND retreat
remain unchanged. Reference60–99 is EXPOSED_REFERENCE.

The four proposed label checks use training demonstrations only and fixed
0/middle/last frames. Numeric state equality and RGB normalized MAE≤.03 were
specified before execution. A visible-instance mask from the simulator would
be privileged training supervision, unlike the paper's manually seeded Cutie
tracking. That difference must be disclosed; it was not validated or generated.
No source/target mask, learned flow predictor or action model was produced.
''')
    (folder/'RESULTS.md').write_text('''# Object Flow: BLOCKED at authentic label correspondence

The author accepted paper and institutional metadata were checked. No author
implementation was verified in the recorded bounded searches; paperCC-BY4.0
is not a source-code license. No custom network was substituted for that method.

Its representation is a future sequence of binary task-object masks, with
learned prediction and a real learned connection to the action policy. Optical
flow, future object-center coordinates or static bbox masks alone would not be
this method. Those substitutes were not built or reported as a baseline.

Existing600 UR5e identities/actions are audited, and original Panda raw states
also exist. A source+target4-demo training-only feasibility probe was specified.
The first launch failed because I supplied a duplicate wrapper keyword. The
one recorded repair removed that keyword, preserving attempt00 separately.
The repaired launch reconstructed the first UR5e training environment and
passed numeric state round-trip at frame0, then failed before RGB rendering:
saved camera `ur5e_agentview_left` is absent. Current names include
`robot0_agentview_left`,`barx_ur5e_agentview` and`robot0_eye_in_hand`.
No equivalence between those cameras was assumed or silently applied.

Numeric round-trip is not certification of joint/body/camera semantics. There
is no RGB alignment measurement, segmentation frame, flow label, model update,
validation loss, physical smoke or formal policy episode. Success rate/video
are N/A. The second failure ends this attempt under the one-repair ceiling;
it does not establish that labels cannot be recovered or that the paper fails.

Resource caveat:environment initialization settled objects through native
mj_step1/2. The probe counted only mj_step, so its zero counters do not measure
total initialization dynamics. Bootstrap happened; exact integration count is
unavailable. No learned policy trajectory was run, and initialization is not
reported as either a completed evaluation or zero total simulation work.

All old baseline evidence is unchanged. Raw failure traces, both source versions,
resource limits and absolute SHA indexes are preserved. A same-behavior component
has not been completed; a full cross-behavior reproduction is even less supported.
See MINIMUM_DATA_REQUIREMENTS.md for the separate, unexecuted data boundary.
''')
    (folder/'MINIMUM_DATA_REQUIREMENTS.md').write_text('''# Minimum data and frozen-boundary proposal — not executed

**Same-behavior component:** no new task demonstration is currently justified.
First certify legacy-camera→current-camera geometry using existing raw states
and fixed training-only RGB comparisons. Do not pick an alias by policy success.
Then verify task-relevant visible object masks, temporal identity, occlusions
and raw→packed image transforms. If simulator masks are used instead of manually
seeded tracking, disclose that privileged annotation difference. The current
probe supplies none of those labels. A source/target train-validation identity
freeze already exists:UR5e540/60,Panda90/10. Keep it intact.

Only after valid labels exist may a separately reviewed implementation preserve
learned future-mask prediction, the paper's temporal segmentation objective,
and learned flow conditioning of action diffusion. Freeze its architecture,
budget, validation-only selection and stopping rules before fitting. Success
on60–99 cannot choose any model. Original hand-view/proprioception inputs must
be retained or their omission explicitly limit the component claim. A camera
name existing in a model does not certify its correspondence to saved images.

**Full cross-behavior claim:** existing pick-and-place demonstrations do not
certify distinct source behaviors. A minimal future pilot could collect10
demonstrations per existing task with an independently specified, physically
valid different source behavior, capped20 attempts/task and reporting every
attempt. Freeze8/2 train-validation whole-trajectory groups and development
conditions before collection; never collect/select using reference60–99. Use
the same robots,two-finger gripper,tasks and official success criterion. If a
valid alternate behavior is not feasible in a direction, report that absence;
do not invent one or add another task. These numbers describe a feasibility
pilot, not a statistically adequate full-baseline sample-size claim.

No new cross-behavior data, task, generator or training round is authorized or
automatically started by this document. Researcher review must precede it.
''')
    (folder/'README.md').write_text('''# Object Flow bounded attempt

**BLOCKED**; intended METHOD_PORT_SAME_BEHAVIOR is not completed.
0 learning updates,0 policy task episodes,success N/A. The actual label-probe
code and one failed repair are preserved. No learned flow predictor or policy
is claimed. Bootstrap initialization physics occurred; its full substep count
was not measured. Do not relaunch/repair under this exhausted round budget.
''')
    from baselines.strong.matrix_details import main as matrix
    matrix();print(status)


if __name__=='__main__':main()
