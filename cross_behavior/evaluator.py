"""Privileged diagnostic logger and immutable task success evaluator.

Nothing returned here is supplied to Expert.predict().
"""
from dataclasses import dataclass, field
import numpy as np


def contacts(env):
    return [(env.sim.model.geom_id2name(c.geom1), env.sim.model.geom_id2name(c.geom2))
            for c in env.sim.data.contact[:env.sim.data.ncon]]


@dataclass
class Evaluator:
    task: str
    contract: dict
    initial: dict = field(default_factory=dict)
    stable: int = 0
    contact_steps: int = 0
    first_contact: int = -1
    max_continuous_contact: int = 0
    continuous_contact: int = 0
    violation: bool = False
    events: dict = field(default_factory=dict)

    def observe(self, env, action, step):
        pairs = contacts(env)
        fingers = {name for g in env.robots[0].gripper.values() for name in g.contact_geoms}
        robot_names = {name for pair in pairs for name in pair if name and name.startswith('robot0_')}
        robot_collision_pairs = [p for p in pairs if any(n in robot_names for n in p) and
                                  not all(n and (n.startswith('robot0_') or n in fingers) for n in p)]
        c = self.contract[self.task]
        site = env.robots[0].eef_site_id['right']
        tcp = env.sim.data.site_xpos[site]
        record = dict(step=step, contacts=pairs, robot_collision_pairs=robot_collision_pairs,
                      tcp=tcp.tolist(), native_success=False)
        if self.task == 'drawer':
            states = env.drawer.get_door_state(env=env)
            handle_geoms = {n for n in [env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom)]
                            if n and env.drawer.name in n and 'handle' in n}
            touch = any((a in fingers and b in handle_geoms) or (b in fingers and a in handle_geoms) for a,b in pairs)
            handle_ids = [env.sim.model.geom_name2id(n) for n in handle_geoms]
            handle_center = np.mean(env.sim.data.geom_xpos[handle_ids], axis=0) if handle_ids else None
            retreat = bool(handle_center is not None and np.linalg.norm(tcp-handle_center) >= c['retreat_distance_m'])
            prev = self.initial.get('previous_joint', states)
            joint_speed = max([abs(states[n]-prev[n])*20 for n in states] or [float('inf')])
            self.initial['previous_joint'] = states
            native = bool(env._check_success())
            released = action[6] < 0 and not touch
            valid = native and released and retreat and self.first_contact >= 0 and joint_speed < c['maximum_joint_speed_per_second']
            record.update(drawer_normalized_joint=states, maximum_joint_speed=joint_speed,
                          handle_geoms=sorted(handle_geoms), handle_contact=touch, released=bool(released),
                          retreat=retreat, native_success=native)
        else:
            aid = env.obj_body_id['cubeA']
            pos = env.sim.data.body_xpos[aid].copy()
            aid_geom = set(env.objects['cubeA'].contact_geoms)
            touch = any((a in fingers and b in aid_geom) or (b in fingers and a in aid_geom) for a,b in pairs)
            v = env.sim.data.get_body_xvelp(env.objects['cubeA'].root_body)
            w = env.sim.data.get_body_xvelr(env.objects['cubeA'].root_body)
            speed, angular_speed = float(np.linalg.norm(v)), float(np.linalg.norm(w))
            self.initial.setdefault('position', pos.tolist())
            initial = np.asarray(self.initial['position'])
            displacement = float(np.linalg.norm(pos[:2]-initial[:2]))
            supported = any(((a in aid_geom and b not in aid_geom and b not in fingers) or
                             (b in aid_geom and a not in aid_geom and a not in fingers)) for a,b in pairs)
            fall = bool(pos[2] < env.surface_z-.02)
            if self.task == 'push':
                local = np.array([[np.cos(env.counter.rot), np.sin(env.counter.rot)],
                                  [-np.sin(env.counter.rot), np.cos(env.counter.rot)]]) @ (pos[:2]-env.goal[:2])
                # Full rotated cube footprint must fit, not just its center.
                object_rot = env.sim.data.body_xmat[aid].reshape(3, 3)
                extent = np.abs(np.array([[np.cos(env.counter.rot), np.sin(env.counter.rot), 0],
                                          [-np.sin(env.counter.rot), np.cos(env.counter.rot), 0]]) @ object_rot) @ np.full(3, .02)
                in_goal = bool(np.all(np.abs(local)+extent <= np.asarray(c['goal_half_size_m'])))
                lift = float(pos[2]-initial[2])
                self.violation |= bool(fall or lift > c['maximum_lift_m'] or action[6] > 0)
                valid = (in_goal and supported and displacement >= c['minimum_displacement_m'] and
                         not self.violation and speed < c['maximum_linear_speed_m_s'] and angular_speed < c['maximum_angular_speed_rad_s'])
                record.update(goal_center=env.goal.tolist(), in_goal=in_goal, lift_m=lift,
                              grasp_or_lift_or_fall_violation=self.violation,
                              lateral_error_m=float(local[1]), goal_error_xy_m=local.tolist())
            else:
                bid = env.obj_body_id['cubeB']
                bpos = env.sim.data.body_xpos[bid].copy()
                bgeom = set(env.objects['cubeB'].contact_geoms)
                ab_contact = any((a in aid_geom and b in bgeom) or (b in aid_geom and a in bgeom) for a,b in pairs)
                b_supported = any((a in bgeom and b not in bgeom and b not in aid_geom and b not in fingers) or
                                  (b in bgeom and a not in bgeom and a not in aid_geom and a not in fingers) for a,b in pairs)
                native = bool(env._check_success())
                xy = float(np.linalg.norm(pos[:2]-bpos[:2]))
                height_error = float(abs(pos[2]-bpos[2]-c['expected_center_height_difference_m']))
                released = bool(action[6] < 0 and not touch)
                retreat = bool(np.linalg.norm(tcp-pos) > c['retreat_distance_m'])
                b_speed = float(np.linalg.norm(env.sim.data.get_body_xvelp(env.objects['cubeB'].root_body)))
                b_angular = float(np.linalg.norm(env.sim.data.get_body_xvelr(env.objects['cubeB'].root_body)))
                valid = (native and ab_contact and b_supported and xy < c['maximum_xy_error_m'] and
                         height_error < c['maximum_height_error_m'] and released and retreat and not fall and
                         max(speed, b_speed) < c['maximum_linear_speed_m_s'] and max(angular_speed, b_angular) < c['maximum_angular_speed_rad_s'])
                record.update(native_success=native, object_B_position=bpos.tolist(), ab_contact=ab_contact,
                              support_B_contact=b_supported, xy_error_m=xy, height_error_m=height_error,
                              released=released, retreat=retreat, toppled=bool(pos[2] < bpos[2] or xy > .05))
                if pos[2]-initial[2] > .05:
                    self.events.setdefault('lift_step', step)
                if ab_contact:
                    self.events.setdefault('support_contact_step', step)
            record.update(object_position=pos.tolist(), linear_speed_m_s=speed,
                          angular_speed_rad_s=angular_speed, displacement_m=displacement,
                          supported=supported, fall=fall, gripper_object_contact=touch)
        if touch:
            if self.first_contact < 0:
                self.first_contact = step
            self.contact_steps += 1
            self.continuous_contact += 1
            self.max_continuous_contact = max(self.max_continuous_contact, self.continuous_contact)
        else:
            self.continuous_contact = 0
        self.stable = self.stable+1 if valid else 0
        record.update(stable_control_steps=self.stable, success=self.stable >= c['stable_control_steps'],
                      first_contact_step=self.first_contact, contact_steps=self.contact_steps,
                      max_continuous_contact_steps=self.max_continuous_contact, events=dict(self.events))
        return record
