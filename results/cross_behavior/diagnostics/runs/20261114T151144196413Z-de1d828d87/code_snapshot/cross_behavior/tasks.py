"""Task-only wrappers around the pinned RoboCasa Kitchen and native Stack assets.

No friction, actuator, robot, gripper or controller parameter is changed here.
Object placements below are native initial-condition sampling, never policy steps.
"""
import xml.etree.ElementTree as ET
import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen, FixtureType
from robocasa.environments.kitchen.barx_tasks import BARXTaskMixin
from robocasa.environments.kitchen.single_stage.kitchen_drawer import OpenDrawer
from robosuite.models.objects import BoxObject
from robosuite.environments.manipulation.stack import Stack as NativeStack
from robocasa.models.objects.objects import MJCFObject


class SamplingBox(MJCFObject):
    """Bounding-box interface only; compiled physics remains the native BoxObject.

    RoboCasa's sampler otherwise falls back to a missing fixture radius site
    when a robosuite generated object meets a Kitchen fixture. No global patch,
    collision-check removal or altered physical parameter is needed.
    """
    def __init__(self, native):
        self.native = native

    def __getattr__(self, key):
        return getattr(self.native, key)

    @property
    def bottom_offset(self):
        return self.native.bottom_offset

    @property
    def top_offset(self):
        return self.native.top_offset

    @property
    def horizontal_radius(self):
        return self.native.horizontal_radius

    def get_bbox_points(self, trans=None, rot=None):
        from robosuite.utils.transform_utils import quat2mat
        signs = np.array([[-1,-1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,1],
                          [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]])
        R = np.eye(3) if rot is None else quat2mat(rot)
        return (signs * self.native.size) @ R.T + (np.zeros(3) if trans is None else trans)


class CounterBlocks(BARXTaskMixin, Kitchen):
    behavior = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref('sink', dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref('counter', dict(id=FixtureType.COUNTER, ref=self.sink))
        self.init_robot_base_pos = self.counter

    def _get_obj_cfgs(self):
        names = ['cubeA'] if self.behavior == 'push' else ['cubeA', 'cubeB']
        if self.behavior == 'stack':
            return [dict(name=name, placement=dict(fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.counter, loc='nn'),
                    size=(.06,.06), pos=(-1.,-1.),
                    offset=(.05 if name == 'cubeA' else .20, .05), rotation=0.)) for name in names]
        return [dict(name=name, placement=dict(
            fixture=self.counter, sample_region_kwargs=dict(ref=self.counter, loc='nn'),
            size=(.06, .06), pos=(.25, -1.),
            offset=((-.06 if name == 'cubeA' else .09), .05), rotation=0.)) for name in names]

    def _create_obj(self, cfg):
        size = [.02] * 3 if cfg['name'] == 'cubeA' else [.025] * 3
        color = [1, 0, 0, 1] if cfg['name'] == 'cubeA' else [0, 1, 0, 1]
        obj = BoxObject(name=cfg['name'], size_min=size, size_max=size, rgba=color)
        return obj, dict(cat='block', groups_containing_sampled_obj=['block'],
                         mjcf_path='native:robosuite.BoxObject')

    def _get_placement_initializer(self, cfg_list, z_offset=.01):
        initializer = super()._get_placement_initializer(cfg_list, z_offset)
        for sampler in initializer.samplers.values():
            sampler.mujoco_objects = [SamplingBox(obj) if isinstance(obj, BoxObject) else obj
                                     for obj in sampler.mujoco_objects]
        return initializer

    def _load_model(self):
        super()._load_model()
        self.surface_z = float(self.counter.pos[2] + self.counter.sample_reset_region(env=self)['offset'][2])
        self.cubeA = self.objects['cubeA']
        if self.behavior == 'stack':
            self.cubeB = self.objects['cubeB']
            self.table_offset = np.array([0., 0., self.surface_z])
        else:
            start = np.asarray(self.object_placements['cubeA'][0])
            direction = np.array([np.cos(self.counter.rot), np.sin(self.counter.rot), 0.])
            self.goal = start + .14 * direction
            self.goal[2] = self.surface_z + .001
            # Thin visible outline, non-colliding; no physical support is added.
            for i, (x, y, sx, sy) in enumerate(((0, -.06, .06, .002), (0, .06, .06, .002),
                                               (-.06, 0, .002, .06), (.06, 0, .002, .06))):
                delta = np.array([x*np.cos(self.counter.rot)-y*np.sin(self.counter.rot),
                                  x*np.sin(self.counter.rot)+y*np.cos(self.counter.rot), 0.])
                ET.SubElement(self.model.worldbody, 'geom', dict(name=f'push_goal_outline_{i}',
                    type='box', pos=' '.join(map(str, self.goal + delta)), size=f'{sx} {sy} .0005',
                    euler=f'0 0 {self.counter.rot}', rgba='0 0 1 1', contype='0', conaffinity='0', group='1'))

    def _setup_references(self):
        super()._setup_references()
        self.cubeA_body_id = self.obj_body_id['cubeA']
        if self.behavior == 'stack':
            self.cubeB_body_id = self.obj_body_id['cubeB']

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        from cross_behavior.common import PROJECT, read
        meta['lang'] = read(PROJECT / 'configs/tasks/cross_behavior_contracts.json')[self.behavior]['instruction']
        return meta

    def _check_success(self):
        # Push uses the new frozen contract in the independent evaluator.
        if self.behavior == 'push':
            return False
        return bool(NativeStack._check_success(self))

    staged_rewards = NativeStack.staged_rewards


class CBPush(CounterBlocks):
    behavior = 'push'


class CBStack(CounterBlocks):
    behavior = 'stack'


class CBDrawer(BARXTaskMixin, OpenDrawer):
    """Native OpenDrawer assets, initialization, instruction and opening predicate."""


def create(task, robot, seed):
    from barx.benchmark import EMBODIMENTS
    from robocasa.utils.controller_utils import load_robocasa_controller_config
    spec = EMBODIMENTS[robot]
    cls = dict(push=CBPush, stack=CBStack, drawer=CBDrawer)[task]
    env = cls(robots=[spec.robot], gripper_types=spec.gripper,
              controller_configs=load_robocasa_controller_config(robot=spec.robot),
              has_renderer=False, has_offscreen_renderer=True,
              camera_names=[spec.camera], camera_widths=320, camera_heights=180,
              camera_depths=True, control_freq=20, horizon=600, ignore_done=True,
              hard_reset=False, layout_and_style_ids=[(0, 0)],
              generative_textures='100p', translucent_robot=False,
              rng=np.random.default_rng(seed), seed=seed)
    env.reset()
    assert env.control_freq == 20 and env.action_dim == 12
    return env, spec.camera
