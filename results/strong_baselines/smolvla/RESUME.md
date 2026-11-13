# SmolVLA round only: frozen formal pipeline active

Repository base39ecc90acf671bff9fefbf9fe4fe50575c8311e0. Octo and all earlier
baseline code/results/weights remain frozen. This round stops after SmolVLA
completion or substantive limitation; never start Flow Matching Policy.

Read server session_resume.json and smolvla/pipeline_process.json before acting.
If a pipeline is live, do not launch another trainer. run_frozen.py uses an
exclusive lock and runs native train -> optimizer audit ->80 formal episodes ->
acceptance, then waits for GitHub publication. It never starts another method.

All gates PASS:700 original identities and real-qpos extraction;640 pixel-exact
raw-RGB caches;4 native data tests; strict full official weights;9 actual discarded
probe updates (6before padding-flag fix,3corrected),122 tensors changed with frozen
VLM unchanged; native queue50 and byte/list RPC; two certified successful-demo
read-only references; fixed validation/RNG restoration. Exactly2 development
smokes ran600/650steps, task successfalse; full state/action/video/goal acceptance
passed. No additional smoke is permitted. Formal pipeline launched after GitHub readback at commit89968fd877e89223c77fff8f41bb3a10742ec9cc; PID2325348. Inspect live logs/state before acting. No formal evaluation at launch.

Official LeRobot v0.4.4@8fff0fde7c79f23a93d845d1a50e985de01f8b8a, Apache2.0.
Weights lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6,
SHA2567cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb.
Native maxstate32/maxaction32, flow10steps, chunk50/execute50 are unchanged.
Information tier RGB+real arm/gripper qpos9; not RGB-only. Old joint_states are
qvel, never substituted for qpos. Old raw flat ordering differs from current
models; deployment reads named joints and verifies each pre-action input row.
No object pose, new camera, new demonstration or generated training data.

Frozen unique run:batch32×16000,512000 fixed balanced anchors; native expert-only
AdamW/scheduler (auto warm533); full32 training loss unchanged, selection uses
real7 valid-frame fixed-target-validation flow MSE at1000..16000, earliest tie.
Checkpoints/full optimizer+CPU/CUDA RNG every500. Cap14400training seconds/free25GiB.
All16k updates must finish before80 physical trials. Formal60–99 per task is
EXPOSED_REFERENCE; no success-driven tuning. Training resets to official weights.

Launch from scientific repo with old simulation Python:
`/home/users/lemon/coft-p2/barx/.venv/bin/python -m results.strong_baselines.smolvla.run_frozen`
Native model runs in isolated smolvla/env, while fixed RoboCasa-X uses the old
simulation environment. Full logs/large weights/media stay under
`/home/users/lemon/strong-baselines-20261112/smolvla/`.
After accepted results, verify hashes, commit/push/read GitHub back and stop.
