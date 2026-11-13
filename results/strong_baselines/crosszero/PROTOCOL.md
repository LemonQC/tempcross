# CrossZero source and port boundary

**NOT_FROZEN**: source/faithful deployment gates were not passed. No training
budget or checkpoint is selected. Same RoboCasa-X/UR5e/Panda/two-finger/two-task
setting; official placement AND retreat,20Hz and canonical OSC7→12 stay fixed.
Formal60–99 would be EXPOSED_REFERENCE. Native XYZRGB point cloud/EE/joint-state
information and pose supervision are distinct from RGB-only baselines.

No target-task learning or target distillation rollout was performed. In the
paper, zero-shot describes level1; unseen level2 is distilled using50 rollouts
from that EE policy. Those must not be training on reference60–99, called unseen
test, or replaced silently by historical rollouts from another model.
