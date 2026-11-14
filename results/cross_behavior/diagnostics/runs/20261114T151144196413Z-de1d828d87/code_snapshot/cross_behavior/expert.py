"""Non-learning RGB-D demonstration expert, separate from ground-truth evaluation."""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from cross_behavior.sensing import colored_position, drawer_handle


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(np.linalg.norm(vector), 1e-12)


def orientation(s, approach, separation):
    a, b = unit(s.approach_local), unit(s.sep_local)
    x = unit(approach)
    y = unit(np.asarray(separation) - x*np.dot(separation, x))
    return np.column_stack((x, y, np.cross(x, y))) @ np.column_stack((a, b, np.cross(a, b))).T


@dataclass
class Expert:
    task: str
    phase: int = 0
    elapsed: int = 0
    memory: dict = field(default_factory=dict)
    error: str = ''

    def predict(self, s):
        """Only Sensors enters this API. No environment, evaluator or contact flags."""
        self.elapsed += 1
        if not self.memory:
            if self.task in ('push', 'stack'):
                red = colored_position(s, 'red', counter_block=self.task == 'stack')
                target = colored_position(s, 'blue' if self.task == 'push' else 'green',
                                          counter_block=self.task == 'stack')
                if red is None or target is None:
                    self.error = 'RGBD_OBJECT_OR_GOAL_NOT_DETECTED'
                    return np.array([0., 0., 0., 0., 0., 0., -1.]), {'error': self.error}
                direction = unit(np.r_[target[:2]-red[:2], 0.])
                self.memory = dict(start=red, target=target, direction=direction,
                                   initial_tcp=s.tcp.copy(), initial_rotation=s.rotation.copy())
            else:
                handle = drawer_handle(s)
                if handle is None:
                    self.error = 'RGBD_HANDLE_NOT_IDENTIFIABLE'
                    return np.array([0., 0., 0., 0., 0., 0., -1.]), {'error': self.error}
                point, normal = handle
                self.memory = dict(handle=point, normal=normal, initial_tcp=s.tcp.copy(),
                                   initial_rotation=s.rotation.copy())
            self.error = ''
            self.elapsed = 0
        m = self.memory
        if self.task == 'push':
            d = m['direction']
            R = orientation(s, [0, 0, -1], np.cross([0, 0, 1], d))
            behind = m['start'] - .055*d
            finish = m['target'].copy()
            finish[2] = m['start'][2]
            points = [behind+[0, 0, .14], behind, finish, finish+[0, 0, .16]]
            local_point = s.left_pad_local
            grips = [-1., -1., -1., -1.]
            holds = [15, 12, 8, 30]
        elif self.task == 'stack':
            R = orientation(s, [0, 0, -1], np.cross([0, 0, 1], m['direction']))
            a, b = m['start'], m['target']
            grasp = a.copy()
            above_a = a+[0, 0, .16]
            above_b = np.r_[b[:2], a[2]+.16]
            place = b+[0, 0, .045]
            points = [above_a, grasp, grasp, above_a, above_b, place, place, above_b]
            local_point = s.pad_local
            grips = [-1., -1., 1., 1., 1., 1., -1., -1.]
            holds = [8, 12, 35, 8, 8, 10, 35, 30]
        else:
            normal, handle = m['normal'], m['handle']
            R = orientation(s, -normal, [0, 0, 1])
            points = [handle+.12*normal, handle, handle, handle+.40*normal,
                      handle+.40*normal, handle+.55*normal+[0, 0, .08]]
            local_point = s.pad_local
            grips = [-1., -1., 1., 1., -1., -1.]
            holds = [10, 10, 35, 15, 35, 30]
        phase = min(self.phase, len(points)-1)
        target = np.asarray(points[phase]) - R @ local_point
        translation = target - s.tcp
        rotation = Rotation.from_matrix(R @ s.rotation.T).as_rotvec()
        at_target = np.linalg.norm(translation) < .012 and np.linalg.norm(rotation) < .12
        # A bounded pose servo, unchanged underlying OSC gains/action scaling.
        # Drawer-only documented expert repair. Native OSC gains/scales stay fixed.
        max_delta = .02 if self.task == 'drawer' else .006
        delta = translation * min(1., max_delta/max(np.linalg.norm(translation), 1e-9))
        aa = rotation * min(1., .10/max(np.linalg.norm(rotation), 1e-9))
        action = np.r_[s.origin_rotation.T @ delta/.05, s.origin_rotation.T @ aa/.5, grips[phase]]
        action = np.clip(action, -1., 1.)
        if at_target:
            m['hold'] = m.get('hold', 0) + 1
        else:
            m['hold'] = 0
        if m['hold'] >= holds[phase] and self.phase < len(points)-1:
            self.phase += 1
            self.elapsed = 0
            m['hold'] = 0
        return action, dict(phase=phase, target_tcp=target.tolist(),
                            translation_error_m=float(np.linalg.norm(translation)),
                            orientation_error_rad=float(np.linalg.norm(rotation)),
                            at_target=bool(at_target), phase_steps=self.elapsed,
                            perceived={k:v.tolist() for k,v in m.items() if isinstance(v, np.ndarray)})
