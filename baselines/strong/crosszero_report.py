"""Record the CrossZero source/interface gate without inventing a policy."""
import shutil
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,out,read,write,digest,now,resource,initialize


def main():
    folder=out('crosszero');server=ROOT/'crosszero'
    search=read(server/'github_search.json');availability=read(server/'primary_availability.json')
    contract=read(server/'controller_contract.json')
    assert contract['status']=='PASS_INTERFACE_INSPECTION_ONLY'
    assert contract['direct_joint_target8_rejected']
    assert not contract['goal_setter_assigns_joint_nullspace_target']
    assert search['official_repository_verified'] is False
    assert all(not x.get('returned_model_ids') for x in availability['checks'] if x['kind']=='huggingface_models' and x.get('http_status')==200)
    paper=Path('/home/users/lemon/post-gate-audit-20261110/literature/crosszero.pdf')
    text=paper.with_suffix('.txt');assert paper.exists() and text.exists()
    for source,target in [(server/'github_search.json','official_search.json'),(server/'primary_availability.json','primary_availability.json'),
        (server/'controller_contract.json','controller_contract.json'),(server/'target_raw_field_inventory.json','target_raw_field_inventory.json')]:
        shutil.copy2(source,folder/target)
    write(folder/'official_source.json',dict(publisher_url='https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ge_CrossZero_Zero-Shot_of_Unseen_Embodiment_Manipulation_via_Object-Centric_Hierarchical_Diffusion_CVPRW_2026_paper.html',
        source_commit=None,source_license='NOT_VERIFIED',official_implementation_verified=False,official_weights_verified=False,
        search_conclusion='No author-verified implementation/weights found in the recorded bounded queries; not proof none exists',
        paper_path=str(paper),paper_sha256=digest(paper),text_path=str(text),text_sha256=digest(text)))
    protocol=dict(status='NOT_FROZEN_SOURCE_AND_INTERFACE_GATE',method='CrossZero',training_budget_frozen=False,
        source_robots=['ur5e'],observation='未部署；原生robot-masked XYZRGB点云、EE状态、关节状态；训练另需物体功能位姿监督',
        train_target_demonstrations_per_task=0,actual_target_demonstrations_used=0,
        zero_shot_scope='Paper level1 EE policy only; unseen level2 IK diffusion uses50 policy rollout episodes',
        source_demonstrations=dict(original=600,train=540,validation=60,raw_state_identity_audit='Reused verified original pool, no new demonstrations'),
        intended_evaluation=dict(tasks=['pnp_counter_to_sink','pnp_sink_to_counter'],robot='Panda',two_finger_gripper=True,
            conditions=list(range(60,100)),role='EXPOSED_REFERENCE',control_hz=20,action_interface='Canonical OSC7 to existing12',
            success='Official placement AND retreat'),
        formal_training_budget=None,checkpoint_rule=None,model_selection_from_reference_success=False,
        new_target_distillation_rollouts=0,technical_repair_budget=1,technical_repairs_used=0,
        forbidden_shortcuts=['Drop point cloud or robot masking','Drop functional-pose objective','Drop learned IK diffusion',
            'Use old BARX/BC rollouts as if produced by a new CrossZero EE policy','Forward joint angles as OSC deltas',
            'Set controller.initial_joint from policy output through an added control channel'])
    write(folder/'protocol.json',protocol)
    write(folder/'training.json',dict(status='NOT_RUN',actual_optimizer_updates=0,actual_sample_exposures=0,
        ee_diffusion_updates=0,ik_diffusion_updates=0,checkpoint=None,validation_results=None))
    write(folder/'evaluation.json',dict(status='NOT_RUN',actual_episodes=0,successes=None,success_rate=None,
        failure_classification=None,videos=[],role='EXPOSED_REFERENCE'))
    write(folder/'data_inventory.json',dict(source_original_raw_identity_summary=read(ROOT/'shared_geometry/ur5e_raw_summary.json'),
        target_original_demos=100,target_joint_fields_present=True,target_demos_used_for_learning=0,
        source_joint_fields_present=True,depth_present_in_raw_files=False,
        calibrated_robot_masked_cloud_reconstruction_validated=False,functional_pose_supervision_certified=False,
        out_of_domain_crosszero_policy_teacher_rollouts_available=0,
        inference='States and geometry may support reconstruction; no fabricated contact, pose or IK label has been generated'))
    errors=dict(technical_repairs=0,reason_code='REFERENCE_IMPLEMENTATION_AND_FAITHFUL_JOINT_DEPLOYMENT_UNVERIFIED',
        source_gate='No author-verified source/checkpoint/reference output established',
        reconstruction_gate='Paper core and many hyperparameters are available; native functional-pose/16D action conventions and checkpoint-merging details remain uncertified',
        interface_gate='Native learned qpos sequences have no direct channel in fixed Cartesian OSC; a FK projection loses redundant posture intent',
        limits='Neither negative search nor channel counting proves scientific impossibility. A faithful implementation/bridge has not been established in this setting.',
        acquisition_errors=['One direct Hugging Face request timed out; subsequent primary-site request succeeded',
            'Web open of publisher returned403; recorded requests retrieval succeeded200'],
        method_failures_observed=0,full_baseline_completed=False)
    write(folder/'errors.json',errors)
    checks=dict(official_source_gate='NOT_VERIFIED',native_module_definitions='REVIEWED_IN_PAPER',
        raw_joint_state_availability='PASS_FIELD_AND_SOURCE_IDENTITY_AUDIT',fixed_action_boundary='PASS_AUDIT_NOT_A_PORT',
        functional_pose_labels='NOT_CERTIFIED',faithful_target_ik_deployment='NOT_ESTABLISHED',
        real_training_update='NOT_RUN',physical_smoke='NOT_RUN')
    write(folder/'checks.json',checks)
    protected=initialize();assert all(digest(r['path'])==r['sha256'] for r in protected['files'])
    write(folder/'preservation.json',dict(status='PASS',old_baseline_reruns=0,files=protected['files']))
    files=[p for p in server.iterdir() if p.is_file() and not p.name.startswith('report')]
    files += [paper,text,folder/'availability_audit.py',folder/'controller_contract.py',Path(contract['source_path']),PROJECT/'baselines/common/action_adapter.py',Path(__file__)]
    write(folder/'sha_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))])
    write(folder/'resources.json',dict(actual_training_updates=0,policy_inferences=0,new_physics_steps=0,formal_episodes=0,
        teacher_rollout_episodes=0,full_physical_smoke_episodes=0,resource_snapshot=resource(),large_artifacts='No new weights/videos; cached paper remains server-side'))
    status=dict(status='BLOCKED',method='crosszero',actual_training_updates=0,physical_evaluation_episodes=0,
        success_rate=None,technical_repairs=0,reason_code=errors['reason_code'],full_baseline_completed=False,utc=now())
    write(folder/'status.json',status);write(folder/'results.json',dict(**status,cells=[]))
    (folder/'PROTOCOL.md').write_text('''# CrossZero source and port boundary

**NOT_FROZEN**: source/faithful deployment gates were not passed. No training
budget or checkpoint is selected. Same RoboCasa-X/UR5e/Panda/two-finger/two-task
setting; official placement AND retreat,20Hz and canonical OSC7→12 stay fixed.
Formal60–99 would be EXPOSED_REFERENCE. Native XYZRGB point cloud/EE/joint-state
information and pose supervision are distinct from RGB-only baselines.

No target-task learning or target distillation rollout was performed. In the
paper, zero-shot describes level1; unseen level2 is distilled using50 rollouts
from that EE policy. Those must not be training on reference60–99, called unseen
test, or replaced silently by historical rollouts from another model.
''')
    (folder/'RESULTS.md').write_text('''# CrossZero: BLOCKED before training

The publisher page was actually retrieved with HTTP200 and contained no GitHub
link. Recorded GitHub title/module queries and Hugging Face model search did not
establish an author-verified implementation or weights. This bounded negative
result does not establish non-publication. Earlier403/connection errors are
recorded separately; they are not evidence that a repository does not exist.

The paper specifies the essential robot-masked1024×6 XYZRGB observation,
PointNet encoder, object functional-pose auxiliary supervision, EE diffusion,
learned IK diffusion, and several optimizer/sampling settings. A reimplementation
is conceivable; lack of one minor hyperparameter alone is not declared fatal.
However, the functional-pose schema, native action convention and merging
procedure have not been certified, and the fixed deployment contract is open.
We did not build a different DP3 policy and call it CrossZero.

The actual canonical adapter rejects a joint7+gripper1 vector; its7 channels
mean Cartesian6+gripper1. The installed OSC goal setter assigns position and
orientation but not a learned joint posture. It uses controller.initial_joint
for nullspace torques. Mapping predicted joint positions through FK loses that
redundant posture; changing the nullspace target would add a control pathway
outside the fixed interface. The audit therefore establishes that a simple
direct/FK bridge is not faithful. It does **not** prove all history-dependent
Cartesian control bridges impossible, or predict physical failure rates.

Data absence is not asserted:600 source trajectories exactly match the old
identity/action pool and retain full/arm/gripper states;100 original Panda
trajectories have7-joint observations and full states. Raw depth is absent but
may be recoverable. Calibrated robot-masked point clouds and native functional
pose labels have not been validated. No counterfactual labels were fabricated.

Actual new training updates0, policy inference0, target distillation rollout0,
task smoke/formal evaluation0; success rate N/A, no failure video exists. Source
license and implementation SHA are N/A because no author implementation was
verified. Reusable work is the source search, native requirements/data inventory
and executed fixed-controller contract audit; it is not a full baseline.

Required before a faithful attempt: verified native schemas/reference outputs,
certified geometry/pose supervision, and a reviewed bridge preserving the IK
module under the same benchmark interface. No extra tasks, controller changes,
or new round were automatically started. Details:protocol.json,checks.json,
controller_contract.json,errors.json and absolute SHA index.
''')
    (folder/'REIMPLEMENTATION_ASSESSMENT.md').write_text('''# Reimplementation assessment

Primary paper: [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ge_CrossZero_Zero-Shot_of_Unseen_Embodiment_Manipulation_via_Object-Centric_Hierarchical_Diffusion_CVPRW_2026_paper.html).
The cached PDF/text and current publisher snapshot are indexed by SHA256.

| Core requirement | Available specification | Current platform gate |
|---|---|---|
| Robot-masked point cloud | XYZRGB1024; PointNet64/128/256/512 | Original RGB/states exist; calibrated depth/mask conversion not validated |
| Object functional pose | Auxiliary L2 and joint objective,Sec3.2 | Native annotation schema/frame/loss coefficient not certified |
| EE diffusion | FiLM1D U-Net,horizon8,10DDIM; several optimizer settings | Single-arm mapping of published16D schema not certified |
| IK diffusion | EE action and current q→q sequence,Sec3.3 | Actual joint fields exist; no valid joint-target deployment bridge |
| Unseen-robot adaptation |50 EE-policy rollouts then teacher-IK distillation,Sec3.3.3–4 | No new CrossZero EE policy or teacher rollouts; old models cannot be relabelled |
| Checkpoint merging | Mentioned inSec3.3.5 | Exact operation unspecified in inspected material |

These are uncertified implementation differences, not proof that the algorithm
cannot be implemented. A paper-only candidate would need explicit fixed choices
and verification. The unresolved joint-controller bridge independently prevents
a faithful full-method result in this attempt. No core module was dropped.
''')
    (folder/'README.md').write_text('''# CrossZero bounded source/interface gate

**BLOCKED**,0 training updates,0 physical evaluations, success N/A.
RESULTS.md states the evidence and limits; no trained CrossZero code is claimed.
availability_audit.py and controller_contract.py are actual executed audits,
not a substitute policy. Full paper/failed retrieval logs remain at SHA-indexed
server paths. No automatic further training or interface modification.
''')
    from baselines.strong.matrix_details import main as matrix
    matrix();print(status)


if __name__=='__main__':main()
