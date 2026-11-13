# Octo-Small: official-runtime preflight

Status: **PARTIAL / PREFLIGHT**. Formal budget, checkpoint selection and stopping
rules will be frozen only after an actual parameter-changing update and complete
task smoke. No training or performance result is implied by this document.

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
formal training, whose optimizer recipe and budget are frozen later. Ordinary
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
