# Minimum data and frozen-boundary proposal — not executed

**Same-behavior component:** no new task demonstration is currently justified.
First certify legacy-camera→current-camera geometry using existing raw states
and fixed training-only RGB comparisons. Do not pick an alias by policy success.
Then verify task-relevant visible object masks, temporal identity, occlusions
and raw→packed image transforms. If simulator masks are used instead of manually
seeded tracking, disclose that privileged annotation difference. The current
probe supplies none of those labels. A source/target train-validation identity
freeze already exists:UR5e540/60,Panda90/10. Keep it intact.

Only after valid labels exist may a separately reviewed implementation preserve
learned future-mask prediction, the paper's temporal segmentation objective,
and learned flow conditioning of action diffusion. Freeze its architecture,
budget, validation-only selection and stopping rules before fitting. Success
on60–99 cannot choose any model. Original hand-view/proprioception inputs must
be retained or their omission explicitly limit the component claim. A camera
name existing in a model does not certify its correspondence to saved images.

**Full cross-behavior claim:** existing pick-and-place demonstrations do not
certify distinct source behaviors. A minimal future pilot could collect10
demonstrations per existing task with an independently specified, physically
valid different source behavior, capped20 attempts/task and reporting every
attempt. Freeze8/2 train-validation whole-trajectory groups and development
conditions before collection; never collect/select using reference60–99. Use
the same robots,two-finger gripper,tasks and official success criterion. If a
valid alternate behavior is not feasible in a direction, report that absence;
do not invent one or add another task. These numbers describe a feasibility
pilot, not a statistically adequate full-baseline sample-size claim.

No new cross-behavior data, task, generator or training round is authorized or
automatically started by this document. Researcher review must precede it.
