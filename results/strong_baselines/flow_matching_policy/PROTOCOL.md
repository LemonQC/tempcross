# One frozen Flow Matching Policy visual port

This round handles only the last baseline. Source and core-difference audit:
OFFICIAL_AUDIT.md and official_source.json. No old baseline is trained/evaluated.

Use HRI-EU/flow_matching@516e8e18875b27741bdbdb8a252e904032c90723,
BSD-3-Clause, native GN ResNet18 and ConditionalUnet1D with unchanged default
widths/time embedding. Train with TorchCFM1.0.7's ConditionalFlowMatcher(sigma0),
Gaussian starting noise and full16x7 velocity MSE. Deployment retains the public
uniform-noise, single-Euler behavior and predicts16/executes8. This inconsistency
with training is disclosed, not silently repaired. The original Robomimic
lowdim/absolute-action and full affordance-prompt recipes are not reproduced.

## Data and information

One joint source_target_visual model. Original UR5e540/60 and Panda90/10
complete-demonstration split:45 target training demonstrations per task. Reuse
the exact existing balanced1M train-anchor sequence and256 target-validation
anchors, preserving IDs and frame positions. No reference rollout, new demo or
generated success-reference data enters training. Original timing anomalies in
the released demonstrations are retained, not resampled.

Report a separate **RGB96+true qpos9+frozen CLIP768 instruction** information tier.
Keep every object-specific instruction, not just a direction ID. The original
joint_states field is velocity and is not used as position. Robot positions
come from the already certified saved-state cache; deployment reads named arm
and Robotiq joints and checks every policy input against pre-action qpos. No
object pose, depth, extra camera, heatmap or new kinematic label is supplied.

Apply native train-only min/max normalization to canonical OSC7 and qpos9.
RGB uses the original camera, default cv2 LINEAR resize96 square and CPU
float32[0,1] normalization, contiguous NCHW in both data and deployment. Language
vectors are byte-identical to the existing frozen CLIP-L/14 cache. Native
repeat-last sequence sampling permits15 rather than7 tail slots, retaining
the original anchor frames; no loss mask or invented future physical label.

## Gates before formal training

One actual diagnostic AdamW update at batch32/peakLR1e-4, no warmup scheduler;
native optimizer counter1 and208 changed tensors,94,874,439 trainable parameters.
Peak allocated memory2.344762880GB and update0.4132s on RTX3090. Diagnostic weights
are discarded and formal initialization must match the pre-probe hashes exactly.
This probe preceded the NCHW preprocessing unification; no second optimizer
probe was silently performed. Ordinary fixes and original failed RPC evidence
are retained in errors.json/server logs.

Four native data tests, readonly four raw trajectories and two previously
certified successful references, direct-versus-RPC17 calls/3 native queries,
and native EMA/fixed-validation/RNG restoration all passed. Exactly two full
development smokes used condition0,600/650steps, task successfalse for both.
State, native velocity/Euler generation,8-action queues, OSC12 mapping, video
and official-goal acceptance passed. No additional smoke is permitted.

## Unique finite training and evaluation

Freeze **batch32 ×24000 optimizer calls =768000 anchors**, seed20261113.
Each source/target × task group receives192000 anchors. Source example
batch64/thousands-of-epochs training is not claimed; this is a bounded faithful
port. All network weights are trainable from scratch, native weights=None.

Use native AdamW1e-4,weight decay1e-6, default betas(.9,.999),eps1e-8; no
gradient clipping or mixed precision. Native cosine scheduler,500 warmup calls.
Keep native diffusers0.35.2 EMAModel(power=.75) including its default
use_ema_warmup=False. Store raw before EMA copy; restore before the next update.

Every1000 updates compute EMA velocity MSE over the unchanged256 heldout target
anchors, using seed17000+batch_offset and restoring CPU/CUDA RNG. Select the
lowest value, earliest tie. Save complete raw model/Adam/scheduler/EMA/RNG every
500; retain only one validation-best EMA and the latest full resume checkpoint.
Report full-run optimizer/sample exposure separately from the selected EMA's
update count and effective seen anchors. Never use exposed success to select.

Stop at24000 calls, nonfinite/error, free disk below25GiB or14400 cumulative
training seconds. No automatic extension, parameter search or second model.
After completed training and optimizer/EMA counter audit, run Panda40/task on
unchanged60–99 conditions, labelled **EXPOSED_REFERENCE**. Same initial states,
RoboCasa-X, two tasks, double-finger gripper,20Hz, OSC7→12 and official
**placement AND retreat**. Keep videos, telemetry, native noise/velocity/actions,
raw individual official results and absolute-path/SHA indexes. Then publish,
verify GitHub, and stop without another baseline or new Method.
