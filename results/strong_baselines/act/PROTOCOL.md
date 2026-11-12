# ACT: RGB/language/canonical-OSC faithful port

This is an adaptation of the [official ACT implementation](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488), MIT license. The candidate configuration is `candidate_protocol.json`; it becomes the immutable `protocol.json` only after both real-update probes and all four full-horizon physical smoke runs pass acceptance. The native ALOHA setup is not claimed reproduced.

## Preserved core and declared differences

The official ResNet18/DETR CVAE, posterior action encoder, 32-dimensional latent, masked L1 plus 10×KL, zero-latent inference and exponential temporal aggregation are retained. `prepare.py` copies the pinned source without modifying the original checkout and writes `official_io.patch`.

The native 14-dimensional joint-action and joint-state interfaces become 7-dimensional canonical OSC actions and a frozen 768-dimensional CLIP instruction vector. The instruction vector occupies the original conditioning slot; **measured robot state is not a hidden input**. This is a language-conditioned port and is not an exact original observation setup. Images use the existing 128×128 packed RGB, one camera. A recorded forecast-validity mask replaces upstream's nonzero-value sentinel so a valid zero action is not discarded. No phase rules or privileged object/contact state generate actions.

## Data and budget

Whole-demonstration identity splits are unchanged. Only UR5e source rows are used: 270 train / 30 validation demonstrations per task; Panda has 45 train / 5 validation per task. There are 50 available Panda human demonstrations per direction, of which five remain held out. IIWA/Kinova3 are excluded in this UR5e→Panda comparison rather than silently changing the source population.

Target-only and source+target each have 24,000 updates with 16 target anchors per update, hence **384,000 target anchor exposures each**. Joint additionally has 16 UR5e anchors per update, giving 384,000 source exposures and batch size 32. Task sampling is balanced, then uniform episode/frame. Action statistics use only the training pool of that setting. Future chunks contain up to 100 actions; terminal padding is masked using the official loss convention. Input is the current RGB/instruction, never future RGB.

These are anchor exposures, not unique demonstrations, epochs, or independent samples. The consumed-anchors CSV records every actual identity/frame. The fixed update budget is an implementation difference from upstream epoch-based examples; the paper's task success rates are not promised by this budget.

## Selection, stopping and deployment

Validate every 1,000 updates on the same 256 historical target validation anchors, with fixed posterior RNG. Choose minimum official CVAE validation loss, tie to earlier update. No target reference success affects hyperparameters, checkpoint, budget or stopping. Probe weights are discarded and formal training starts from seed 7. Stop at the exact budget, a nonfinite error, 12-hour group wall cap, or 25 GiB disk floor. Keep only best validation, latest resumable checkpoint and probe provenance.

Single GPU, sequential groups. Both groups require one actual update and a full original-horizon condition-0 smoke on each task. Smoke passes by correct finite execution/recording, not task success after one update. A technique failure permits one documented repair across this method; further substantive obstruction is reported before proceeding to VQ-BeT.

## Physical evaluation and acceptance

Each selected model receives 40 Panda trials per task on the exact existing 60–99 frozen states: **160 formal episodes total for ACT**. All are `EXPOSED_REFERENCE`, not unseen test data. Control is 20 Hz, canonical clipping/12-dimensional environment action is unchanged, horizons are 600/650, and official success remains placement AND retreat. Neither a phase completion nor a contact proxy is success.

Every completed episode saves unmodified policy-view video, actions, all forecasts, actual environment actions, state/contact telemetry and official predicates. Acceptance independently reconstructs ACT's temporal aggregation, action conversion, exact frozen initial states, official outcome and every video frame. Failed episodes remain in the denominator. Technical partial episodes are preserved separately and never reported as policy failures. Per-condition completion files allow already completed episodes to be reused without replay.

Final classification can be `COMPLETE_FAITHFUL_PORT` only after both trained checkpoints and all 160 accepted formal episodes exist. Otherwise it remains `PARTIAL` or `BLOCKED`, with success rate N/A for unrun cells. The old 1280+160 evaluation records and TOCAT branch are untouched.
