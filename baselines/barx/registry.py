from pathlib import Path
from baselines.common.paths import ROOT
P2=Path('/home/users/lemon/coft-p2/p2/physical-20261108')
ARTIFACTS=Path('/home/users/lemon/coft-p2/artifacts/runs')
LEGACY={
 'A_barx_no_reps':ARTIFACTS/'no_reps/checkpoints/step-050000-epoch-15-loss=0.8424.pt',
 'A_barx_joint':ARTIFACTS/'joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt',
 'C_barx_no_reps_6000':P2/'adapted-checkpoints/no_reps/checkpoints/step-003000-epoch-00-loss=restricted.pt',
 'C_barx_joint_96000':Path('/home/users/lemon/coft-p3/p3-20261109/adapted/checkpoints/step-012000-epoch-01-loss=restricted96000.pt'),
}
