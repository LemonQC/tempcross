"""Declared gripper interventions. B1 has no simulator/oracle input."""
from dataclasses import dataclass


@dataclass
class GripperCorrection:
    threshold: float
    hysteresis: float = .03
    minimum_hold: int = 8

    def __post_init__(self):
        if not 0 < self.threshold < 1 or self.hysteresis < 0 or self.minimum_hold < 1:
            raise ValueError('Invalid gripper control parameters')
        self.reset()

    def reset(self):
        self.state = -1.
        self.remaining = 0

    def command(self, raw_gripper):
        if self.remaining:
            self.remaining -= 1
            return self.state
        previous = self.state
        if self.state < 0 and raw_gripper < self.threshold:
            self.state = 1.
        elif self.state > 0 and raw_gripper >= self.threshold+self.hysteresis:
            self.state = -1.
        if self.state != previous:
            self.remaining = self.minimum_hold-1
        return self.state


def select_conditions(payload, states, models, indices):
    """Select only after the caller's official loader validated the full bundle."""
    from copy import deepcopy
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(states) for i in indices):
        raise ValueError('Invalid/duplicate condition indices')
    if not len(payload['episodes']) == len(states) == len(models):
        raise ValueError('Full bundle arrays do not align')
    selected = deepcopy(payload)
    selected['episodes'] = [deepcopy(payload['episodes'][i]) for i in indices]
    selected['episode_count'] = len(indices)
    # Original entry episode fields and IDs must remain untouched.
    return selected, [states[i] for i in indices], [models[i] for i in indices]
