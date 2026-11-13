# SmolVLA: one frozen state-conditioned baseline

This round handles **SmolVLA only**. Octo-Small and every earlier baseline are
frozen. No Flow Matching Policy candidate or new Method starts automatically.

Use official Hugging Face LeRobot **v0.4.4**, commit
`8fff0fde7c79f23a93d845d1a50e985de01f8b8a` (Apache-2.0), public
`lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` (Apache-2.0),
model SHA256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
Backbone metadata is pinned to `4cde9d208e19519c36917aa170fffc7e381aa0ad`.
Official code/architecture/flow matching are unchanged. Native strict loading of
the complete policy checkpoint uses native `load_vlm_weights=False` to skip
redundantly downloading the base VLM weights; no used parameter is left random.

## Information and data

One **source_target_state** setting: original540/60UR5e and90/10Panda complete
demonstration split (45target train per task). No new demonstration or reference
rollout enters training. Use the original balanced sampler and256 target-validation
anchors. State and action statistics use only630 training demos.

Report separately from RGB-only models: primary RGB180×320, original language,
7 arm slots and2 Robotiq joint coordinates. UR5e has6 arm joints; its7th slot is
zero padding. Native preprocessing pads9state dimensions to32, aspect-preserving
resizes/pads one real image to512² and tokenizes48 positions. No depth, object
state, target pose or extra real camera is supplied. Default optional dataset
image augmentation is disabled, matching the official default.

The raw HDF5 field `joint_states` is **velocity**, not position. All700 demos were
checked against saved flat state, qpos time derivatives and exact gripper qpos.
Robot qpos comes from the actual old saved state layout; live evaluation reads
the same named joint coordinates from the current simulator, whose flat joint
ordering differs. Every recorded policy state is checked against pre-action qpos.
Existing100 non20Hz saved intervals remain unchanged: no interpolation or new
labels are introduced. The physical benchmark remains20Hz.

Readonly control checks use four preselected raw training trajectories and two
already accepted generated-success references (conditions1/7, original train
parents). The latter are **not SmolVLA training data**. Raw rewards are all zero
and are not treated as success certification. No physics step or new generation
is used for these checks.

## Native action and training mechanisms

Train on original canonical OSC7 with positive-close gripper. Native mean/std
normalization,32-dimensional flow expert,10 integration steps and50-step chunks
are preserved. Native `select_action` executes all50 predictions (2.5s at20Hz).
The crop32→7, unnormalization and unchanged OSC7→12 interface are independently
audited. No IK, controller tuning, action trigger or manual phase rule is added.

Native processor transition conversion drops the dataset-only boundary flag.
The adapter therefore carries the genuine `actions_id_pad` mask across that
conversion under the exact key the unchanged native forward expects. Corrected
real updates were checked before the formal protocol freeze.

Keep native expert/state-projection finetuning:450046176 total parameters,
99880992 trainable, frozen VLM unchanged. Training uses the unchanged native
full32-dimension masked flow loss. Validation reports this loss and also the
real7-dimension valid-frame flow MSE, avoiding selection dominated by25 padded
output dimensions. This is a declared validation metric difference, not a new
training objective. Fixed noise seeds and CPU/CUDA RNG restoration were checked.

## Frozen finite budget

After real-update, queue/RPC, readonly replay and both complete smoke gates:
**batch32,16000updates,512000anchors**, source and target half, tasks half.
RTX3090 batch32 measured about0.718s/hot update and6.16GB peak allocated.
This budget fits a14400s training cap; it differs from the official example
batch64/20000steps. It is a declared bounded faithful port, not the full default
recipe. Nine actual diagnostic updates are discarded (including six before the
padding propagation correction). Formal training resets to official weights.

Use native AdamW1e-4,betas(0.9,0.95),eps1e-8,decay1e-10,clip10. Native scheduler
auto-scales its1000/30000 warmup/decay preset to533/16000 for this budget, with
final LR2.5e-6. No AMP override. Save full model/optimizer/scheduler/RNG every500
updates; retain one validation-best model. Validate every1000 on the unchanged
256 anchors; select minimum real7 valid-frame flow MSE, earliest tie. No extension
on plateau or failed physical success. Halt on nonfinite/error, free disk below
25GiB or14400training seconds.

Development smoke budget is exhausted: exactly2 episodes,600/650steps, success
false for both. Their interface/video/state/official-goal checks passed. They are
not formal evaluation or a task success estimate, and do not select a checkpoint.

After training completes, evaluate original Panda conditions60–99,40/task,
**EXPOSED_REFERENCE**, original initial states/controllers/camera20Hz and official
**placement AND retreat**. Keep complete videos, telemetry, individual official
results, raw32/7 action logs and SHA indexes. No exposed success selects weights.
Finish or report a substantive limitation, publish, then stop this round.
