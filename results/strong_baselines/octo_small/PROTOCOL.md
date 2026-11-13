# Octo-Small: frozen official-runtime finetuning protocol

Status at freeze: **PARTIAL / FROZEN_WAITING_LOSSLESS_CACHE**. Real updates,
native-data checks and both complete task smoke runs passed before the formal
budget was frozen in [protocol.json](protocol.json) and [freeze.json](freeze.json).
Formal training and evaluation must be distinguished from these discarded probes.

Use [official Octo](https://github.com/octo-models/octo) commit
`241fb3514b7c40957a86d869fecb7c7fc353f540` (MIT), native JAX/Flax transformer,
frozen T5-base, and native 7-dimensional, 4-step DiffusionActionHead. Initialize
from [author Octo-Small1.5 weights](https://huggingface.co/rail-berkeley/octo-small-1.5)
revision `dc9aa3019f764726c770814b27e4ab0fc6e32a58`. All published checkpoint
hashes must pass. An isolated Python3.10 environment preserves existing baselines.

Initial main experiment: **UR5e+Panda / source_target**. Reuse the original
whole-demonstration split: UR5e540train/60validation, Panda90train/10validation
(45target train per task). Source and target, then the two tasks, receive equal
sampling weight. Use the original256 target-validation anchors. No generated
MimicGen data, exposed reference images, or evaluation outcomes enter training.

Inputs are saved primary RGB180×320 resized by Octo's official transform to256²,
two consecutive observations and the original task language (native T5 max16
tokens). Missing wrist and goal images have zero padding with official false
modality masks. No proprioception, simulator object state, depth or second real
camera is added. Report the larger resolution and Octo/T5 pretraining separately
from older RGB128 comparisons. Do not claim matched pretraining.

Canonical OSC7 is mapped reversibly to native base-frame metre/radian deltas
using unchanged controller limits(0.05m,0.5rad), plus gripper openness0..1
(`(1-canonical_close)/2`). Train-only mean/std normalize the first6 dimensions;
gripper remains unnormalized. Decode to original canonical OSC7 then apply the
existing OSC7→12 adapter at20Hz. No new threshold, action controller or state rule.

Native `chunk_act_obs`, image resize/augmentation, model, diffusion loss and
`create_optimizer` are reused directly. Use full finetuning with official frozen
T5 keys; do not replace the action head or language encoder. The small diagnostic
update uses constant3e-5 to verify changed parameters; it is discarded before
formal training, whose optimizer recipe and budget are now frozen below. Ordinary
engineering errors are recorded and repaired; they do not by themselves prove
RTX3090 infeasibility.

Full-task smoke uses original development condition0 of each Panda task through
the established evaluator and its600/650-step limits. Interface execution and
official placement AND retreat checks are the gate, not smoke success. Neither
smoke outcome nor conditions60–99 may select a checkpoint.

Formal evaluation, if gates pass: original Panda conditions60–99,40 per task,
**EXPOSED_REFERENCE**, same initial states,20Hz, action interface and complete
official success criterion. Preserve every episode/video/telemetry and SHA on
the server; publish code, actual update counts, validation, metrics and indexes.
No old baseline is rerun. No other large model trains while Octo is active.

## Frozen budget and model selection

One joint source_target model: batch32,24000 updates,768000 sampled anchors
(384000source/384000target; each task half). Reset to original pretrained weights;
discard the6 diagnostic optimizer updates (batch8 and32,3each). Native AdamW,
peak3e-4,2000 warmup updates, cosine to zero at24000, decay0.01, clip1.0,
bfloat16 first moment and frozen `*hf_model*` keys. No validation-driven extension.

Validate at1000..24000 on the original256 held-out target anchors, using fixed
diffusion keys `17000+batch_offset`. Select minimum native DDPM noise-prediction
loss, earliest tie. Save complete optimizer/model/RNG state every500 updates;
retain one best native checkpoint plus the resume state. Fixed update budget;
halt on nonfinite/error, free disk below25GiB, or14400 training seconds. Never
choose a checkpoint using smoke or EXPOSED_REFERENCE success.

Differences from official generic Bridge finetuning example: batch32 vs256,
24000 vs50000 updates, history2 (released checkpoint's native history) vs example1,
custom existing HDF5 identities and balanced UR5e/Panda sampling, fixed validation
anchors, unchanged RoboCasa-X action interface. Model/loss/augmentation/optimizer
implementations and all pretrained core weights remain official. This is a
declared faithful task port with a bounded GPU budget, not the default full budget.

Actual gates: both batch8 and32 performed3 updates;252 tensors changed, T5 unchanged.
136670604 total parameters =109628544 frozen T5 +27042060 remaining parameters.
Batch32 hot updates averaged0.4201s; final snapshot19198MiB (~18.7GiB, including
desktop), not a measured exact peak. Both smoke episodes finished600/650steps,
official success false, and passed state/action/queue/video/goal acceptance.
These2 smoke trials are development data, not80 formal trials or a success estimate.

Native T5 max16 tokens truncates the17-token Sink→Counter strings by one token;
the16 original instructions still have zero cross-task token collisions.
Keep this native preprocessing limitation explicit; no task string is rewritten.
