"""Zero simulation source, data identity, registry and action-frame audit."""
import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from common import OUT, ROOT, SOURCE, TASKS, digest, install_paths, read, write


def frame_check():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mimicgen.env_interfaces.robosuite import RobosuiteInterface
    from common import install_annotation_frame_repair
    controller = SimpleNamespace(output_max=np.array([.05] * 3 + [.5] * 3),
                                 input_ref_frame='base')
    composite = SimpleNamespace(get_controller=lambda arm: controller)
    fake = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(composite_controller=composite)]))
    pose = np.eye(4)
    pose[:3, 3] = [.2, -.1, .8]
    fake.get_robot_eef_pose = lambda: pose.copy()
    action = np.array([.2, -.3, .1, .05, -.08, .03, -1.])
    rows = []
    decoder = RobosuiteInterface.action_to_target_pose
    for yaw in [0., .7, 1.57, -2.1]:
        rotation = Rotation.from_euler('z', yaw).as_matrix()
        fake.get_controller_base_pose = lambda: (np.zeros(3), rotation)
        target = decoder(fake, action)
        recovered = RobosuiteInterface.target_pose_to_action(fake, target)
        expected_position = pose[:3, 3] + rotation @ (.05 * action[:3])
        rows.append(dict(base_yaw=yaw, original_roundtrip_max_error=float(abs(recovered-action[:6]).max()),
                         original_translation_error_m=float(np.linalg.norm(target[:3,3]-expected_position))))
    install_annotation_frame_repair()
    repaired = RobosuiteInterface.action_to_target_pose
    for row in rows:
        rotation = Rotation.from_euler('z', row['base_yaw']).as_matrix()
        fake.get_controller_base_pose = lambda: (np.zeros(3), rotation)
        target = repaired(fake, action)
        recovered = RobosuiteInterface.target_pose_to_action(fake, target)
        row['repaired_roundtrip_max_error'] = float(abs(recovered-action[:6]).max())
        assert row['repaired_roundtrip_max_error'] < 1e-5
    assert max(r['original_roundtrip_max_error'] for r in rows) > .1
    return rows


def main():
    if (OUT / 'audit.json').exists():
        print('Audit already recorded; no simulation or experiment repeated.')
        return
    install_paths()
    import mimicgen
    import h5py
    import numpy as np
    from mimicgen.configs import config_factory
    from p3_validation.prepare_data import fingerprint
    manifest = read(PROJECT / 'configs/datasets/baseline_target_manifest.json')
    selections = []
    for task in TASKS:
        train = sorted([r for r in manifest if r['task'] == task and r['split'] == 'train'],
                       key=lambda r: r['episode_id'])
        assert len(train) == 45
        for r in train[:5]:
            with h5py.File(r['original_file'], 'r') as f:
                demo = f['data'][r['original_demo']]
                actions = demo['actions'][:, :7].astype(np.float32)
                actions[-1] = actions[-2]
                actions[-1, :6] = 0
                instruction = json.loads(demo.attrs['ep_meta'])['lang']
                assert fingerprint(actions, len(actions), instruction) == r['episode_id']
            selections.append(r)
    write(OUT / 'source_seed_manifest.json', selections)
    registry = {}
    for name in ['PnPCounterToSink', 'PnPSinkToCounter']:
        cfg = config_factory(name=name, config_type='robosuite')
        registry[name] = json.loads(cfg.dump())
    files = [SOURCE / name for name in [
        'scripts/prepare_mimicgen_source.py', 'scripts/generate_mimicgen.py',
        'barx/mimicgen_release.py', 'configs/mimicgen.json',
        'third_party/mimicgen/mimicgen/scripts/prepare_src_dataset.py',
        'third_party/mimicgen/mimicgen/scripts/generate_dataset.py',
        'third_party/mimicgen/mimicgen/datagen/data_generator.py',
        'third_party/mimicgen/mimicgen/datagen/selection_strategy.py',
        'third_party/mimicgen/mimicgen/datagen/waypoint.py',
        'third_party/mimicgen/mimicgen/env_interfaces/robosuite.py',
        'third_party/mimicgen/mimicgen/env_interfaces/robocasa/single_stage/mg_pnp.py',
        'third_party/mimicgen/mimicgen/configs/robocasa/single_stage/config_pnp.py']]
    documents = [PROJECT / 'docs' / n for n in ['BASELINE_PROTOCOL.md',
        'BASELINE_REPRODUCIBILITY_MATRIX.md', 'BASELINE_RESULTS.md', 'BARX_TARGET_ADAPTATION_RESULTS.md']]
    result = dict(status='ZERO_SIMULATION_AUDIT_COMPLETE_REPAIR_REQUIRED',
                  official_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
                  official_worktree_clean=not bool(subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()),
                  module=mimicgen.__file__, simulator_environments_created=0, physics_steps=0,
                  source_seed_demonstrations=len(selections), target_train_pool=90, target_validation_pool=10,
                  versions={p:importlib.metadata.version(p) for p in ['torch','mujoco','robosuite','robomimic','numpy']},
                  source_files=[dict(path=p,bytes=Path(p).stat().st_size,sha256=digest(p)) for p in sorted({r['original_file'] for r in selections})],
                  reviewed_inputs=[dict(path=str(p),sha256=digest(p)) for p in files+documents],
                  registered_task_configs=registry, action_frame_check=frame_check(),
                  compatibility_notes=['Task-local PYTHONPATH exposes bundled mimicgen; no package installation or dependency upgrades.',
                    'Original action_to_target_pose omits base-to-world conversion while its inverse encodes world-to-base; one isolated annotation repair required.',
                    'Initial inspection used unsupported config_factory(base_type=...); corrected diagnostic call to the actual config_type signature before creating any environment.'])
    write(OUT / 'audit.json', result)
    state = read(OUT / 'status.json')
    state['completed_stages'].append('zero_simulation_source_and_interface_audit')
    state['phase'] = 'PRE_PHYSICS_PROTOCOL_FREEZE'
    state['input_sha256'].update({r['path']:r['sha256'] for r in result['source_files']})
    state['repair_attempts'] = [dict(id=1,stage='PRE_PHYSICS_INTEGRATION',
       scope='task-local vendored import path and base-to-world source annotation inverse',
       evidence=str(OUT/'audit.json'), controller_unchanged=True, repair_code=str(OUT/'common.py'),
       sha256=digest(OUT/'common.py'), physical_episodes_repeated=0,
       further_repairs_permitted=False)]
    write(OUT / 'status.json', state)
    print(json.dumps({k:v for k,v in result.items() if k in ['status','source_seed_demonstrations','action_frame_check']},indent=2))


from common import PROJECT
if __name__ == '__main__':
    main()
