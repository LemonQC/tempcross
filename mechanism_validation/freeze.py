"""Create the final fixed protocol before any intervention; no simulator steps."""
import json
import subprocess
from mechanism_validation.common import *
from baselines.barx.registry import LEGACY


def main():
    if (ROOT/'physical_launches.jsonl').exists():raise RuntimeError('Already started; protocol is immutable')
    control=dict(gain_position=.4,gain_rotation=.4,max_position_step_m=.012,max_rotation_step_rad=.12,
        position_tolerance_m=.015,rotation_tolerance_degrees=10,pose_dwell_states=5,contact_dwell_states=5,
        stable_rise_m=.03,stable_dwell_states=10,clearance_up_m=.08,pregrasp_up_m=.08,lift_m=.08,
        close_min_steps=16,preplace_up_m=.12,release_steps=20,retreat_up_m=.15,
        phase_caps=dict(clearance=60,pregrasp=160,descend=100,close=60,lift=100,transport=130,preplace=130,place=140,retreat=100))
    paths=[*sorted((PROJECT/'mechanism_validation').glob('*.py')),OUT/'source_contact_bank.json',OUT/'scene_audit.json',OUT/'bundle_integrity.json']
    config=dict(version='MECHANISM_202611101108_V1',baseline_commit='fe149302b6f9f00c96fca6236e22df4e4a72cb06',directive_commit='789774a8a7c75e6e37d007c9903159172b201b94',
        source_commit='0f435089f48b1f16d225f868dbef855a5e75d1e0',task=TASK,source_robot='ur5e',target_robot='panda',conditions=CASES,methods=['M0','M1','M2'],
        role='OUTCOME_SELECTED_HARD_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_TEST',maximum_physical_episodes=24,control_freq=20,max_episode_steps=600,seed=7,
        checkpoint=str(LEGACY['A_barx_joint']),checkpoint_sha256=digest(LEGACY['A_barx_joint']),control=control,
        M0='PRIVILEGED_REFERENCE: common corrected grasp; source-derived fixed object trajectory with handoff attachment held fixed; continuous TCP OSC tracking',
        M1='PRIVILEGED_PREFIX_THEN_ORIGINAL_RGB_LANGUAGE_BARX: no object state passed to policy; reset queue at stable lift; act_horizon8',
        M2='PRIVILEGED_REFERENCE: identical prefix and object goals; actual object error via continuously re-observed object-to-TCP attachment; same OSC gains',
        prefix='Anchor source object-relative grasp to exact target initial object. Open clearance, pregrasp, descend; pose-gated close; confirmed stable lift. Advance on pose arrival plus dwell, never phase time alone. Phase caps abort without alternative search.',
        postgrasp='Source transport midpoint and release expressed in target sink frame. Transport and preplace have a common vertical clearance floor max(handoff object z, source release object z+.12). Then exact source release object pose. Same targets for M0/M2. Hold TCP and open20, then retreat+.15 world z. No object chasing after release.',
        physical_contact='At least one raw object contact on each left/right finger; each side accepts its existing fingerpad_collision OR fingertip_collision. Old pad-only _check_grasp is separately retained. Contact distance must be <=0.',
        stable_grasp='Physical bilateral contact AND object COM rise >=.03m for >=10 consecutive observed states; prefix handoff additionally requires lift TCP within .015m/10deg for5 states',
        source_extraction='First10-state physical bilateral run preceding first10-state contact+3cm lift. First open after confirmed lift; otherwise final bilateral contact loss (case94, not a commanded release). No synthetic source success.',
        phase_failures='Timeout, lost grasp, or global600-step budget are outcomes; no fallback, repeated trial, threshold search or additional development rollout. Official success alone authorizes early success termination.',
        prefix_equivalence='Independently execute all three prefixes from identical official state, without restoring a handoff snapshot. Compare every executed action, flattened simulator state and handoff numerical controller fields, tolerance1e-8. Save full mjSTATE_INTEGRATION plus numeric MjData/controller/gripper/RNG/policy queue state. A failed prefix has no handoff and is reported as such.',
        gates=dict(A='PASS iff common-prefix stable-grasp count>=6/8 AND max(M0 official successes,M2 official successes)>=4/8. Max of individual method counts, not union of successful cases.',
            B='If A passes and M1>=4/8, select PREGRASP visual/proprio contact acquisition',
            C='If A passes and M1<=2/8 and M2>=4/8, select POSTGRASP visual/proprio task-effect feedback',
            D='Acknowledge ordinary geometry/feedback effectiveness; require a distinct decision mechanism against MimicGen,KITE,CrossZero,object-flow transfer,visual servo/MPC before new network',
            E='Otherwise AMBIGUOUS or NO_GO; no second round. Gates are prototype decisions, not paper statistics.'),
        policy_information=['RGB','verbatim instruction'],no_retraining=True,no_dynamics_changes=True,no_new_expert_demonstrations=True,
        integrity={str(p.relative_to(PROJECT)):digest(p) for p in paths},external_source_trace_integrity={str(p):digest(p) for p in sorted((ROOT/'source_traces').glob('*.json'))})
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==config['source_commit']
    write_json(CONFIG,config);print(json.dumps(dict(protocol=str(CONFIG),sha256=digest(CONFIG),episodes=24,physics_steps=0)))


if __name__=='__main__':main()
