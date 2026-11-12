"""Task-local paths, atomic progress, and a single frozen integration repair."""
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path('/home/users/lemon/CodexProject/tempcross')
SOURCE = Path('/home/users/lemon/coft-p2/barx')
ROOT = Path('/home/users/lemon/mimicgen-baseline-20261112')
OUT = PROJECT / 'results/strong_baselines/mimicgen'
PYTHON = SOURCE / '.venv/bin/python'
TASKS = ['pnp_counter_to_sink', 'pnp_sink_to_counter']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install_paths():
    for path in [SOURCE / 'third_party/mimicgen', SOURCE, SOURCE / 'policy', PROJECT]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4')


def install_annotation_frame_repair():
    """Decode saved base-frame OSC deltas into world-frame annotation targets.

    Does not change the controller, generated-action encoder, gains, physics,
    waypoint algorithm, noise, source selection, or official success function.
    """
    from mimicgen.env_interfaces.robosuite import RobosuiteInterface
    import mimicgen.utils.pose_utils as poses
    import robosuite.utils.transform_utils as transforms
    import numpy as np

    original = RobosuiteInterface.action_to_target_pose

    def decode(self, action, relative=True):
        if not relative:
            return original(self, action, relative=False)
        controller = self.env.robots[0].composite_controller.get_controller('right')
        if controller.input_ref_frame != 'base':
            return original(self, action, relative=True)
        position, rotation = poses.unmake_pose(self.get_robot_eef_pose())
        _, base_rotation = self.get_controller_base_pose()
        delta_position = np.asarray(action[:3]) * controller.output_max[:3]
        delta_rotation = np.asarray(action[3:6]) * controller.output_max[3:6]
        world_delta_rotation = base_rotation @ transforms.quat2mat(
            transforms.axisangle2quat(delta_rotation)) @ base_rotation.T
        return poses.make_pose(position + base_rotation @ delta_position,
                               world_delta_rotation @ rotation)

    RobosuiteInterface.action_to_target_pose = decode
