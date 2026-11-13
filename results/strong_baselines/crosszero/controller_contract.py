"""Read-only check of the fixed action boundary, not a learned CrossZero port."""
import ast
import importlib.util
from pathlib import Path
from baselines.strong.common import ROOT,write,digest


def main():
    import numpy as np
    from baselines.common.action_adapter import canonical_action
    source=Path(importlib.util.find_spec('robosuite').origin).parent/'controllers/parts/arm/osc.py'
    text=source.read_text();tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OperationalSpaceController')
    goal=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='set_goal')
    assigned={n.attr for a in ast.walk(goal) if isinstance(a,ast.Assign) for target in a.targets for n in ast.walk(target) if isinstance(n,ast.Attribute)}
    assert 'goal_pos' in assigned and 'goal_ori' in assigned and 'initial_joint' not in assigned
    assert 'self.control_dim = 6 if self.use_ori else 3' in text
    assert 'self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel' in text
    rejected=False
    try:canonical_action(np.zeros(8))
    except ValueError:rejected=True
    assert rejected
    action,_=canonical_action(np.zeros(7));assert action.shape==(12,)
    result=dict(status='PASS_INTERFACE_INSPECTION_ONLY',source_path=str(source),source_sha256=digest(source),
        arm_command_semantics='6 Cartesian OSC dimensions',gripper_command_dimensions=1,environment_dimensions=12,
        panda_joint_position_plus_gripper_dimensions=8,direct_joint_target8_rejected=True,
        goal_setter_assigns_joint_nullspace_target=False,nullspace_posture_reference='controller.initial_joint',
        conclusion='Direct native joint-target deployment is absent. FK-to-OSC would discard redundant joint posture; no faithful bridge established.',
        limitation='This does not prove arbitrary history-dependent control under OSC is impossible. It does not test physical task success.',
        test_vectors_are_policy_predictions=False,new_training_updates=0,new_policy_inferences=0,new_physics_steps=0)
    write(ROOT/'crosszero/controller_contract.json',result)
    print(result)


if __name__=='__main__':main()
