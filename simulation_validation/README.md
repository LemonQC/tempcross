# COFT-P2: official BARX gate

This directory does not introduce a simulator, task, camera, planner or learned policy.
It uses upstream BARX commit `0f435089f48b1f16d225f868dbef855a5e75d1e0` and the
unmodified `policy/experiments/robot/robocasa_x/evaluate.py` through a read-only telemetry wrapper.

External execution root: `/home/users/lemon/coft-p2`. Existing environments, including
`cbl4090_clean` if available on another executor, must not be modified.

## Required preparation

```bash
# Use an independent Python 3.10 environment and upstream uv.lock.
cd /home/users/lemon/coft-p2/barx
uv sync --locked --no-dev
uv run --locked --no-dev python -m unittest discover -s tests -v
# Policy inference also needs the upstream pinned FlashAttention 2.5.5.
# Upstream installation.md requires the train extra, but no training command is run.
uv sync --locked --extra train --no-dev
uv run --locked --no-dev python robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes
```

CUDA development toolkit must match the pinned Torch build if compiling FlashAttention;
the detected host toolkit is 11.3, while Torch is pinned to CUDA 12.1. The official
Dao-AILab v2.5.5 release also supplies a prebuilt cp310/torch2.2/CUDA12/ABI-false wheel.
Neither installation route has been validated by this task unless explicitly shown in
the execution report. Do not silently switch to a different attention implementation.

Download inference artifacts only, from this repository root with the BARX environment:

```bash
export HF_HOME=/home/users/lemon/coft-p2/artifacts/hf
export BARX_VQ_ROOT=/home/users/lemon/coft-p2/artifacts/vq
/home/users/lemon/coft-p2/barx/.venv/bin/python -m simulation_validation.download_inference \
  --source /home/users/lemon/coft-p2/barx --root /home/users/lemon/coft-p2/artifacts
```

This excludes RLDS and the unused base VLM checkpoint. DINO, SigLIP and Qwen runtime
backbones remain necessary for the official constructor. No Reps is fetched after
P0 passes, or explicitly with `--include-no-reps`. Revisions are fixed, and the
upstream VQ and Joint Reps SHA256 values are verified.

## Gate and evaluation

```bash
scripts/run_barx_gate.sh
# Optional P0 only:
BARX_RUN_ROOT=/home/users/lemon/coft-p2/rollouts/p0-only scripts/run_barx_gate.sh --p0-only
```

The gate runs official tests, then one complete Joint Reps episode on each robot.
It requires complete official summaries, episode records and decodable videos.
It never counts a technical crash as an unsuccessful task episode. Only after P0
passes does it run the fixed four cells of 20 episodes, using the first official
conditions, max_steps=600, act_horizon=8, seed=7 and prior normalization mg_pnp_lite.
Both methods infer actions without representation prompts. Runs do not overwrite
earlier directories. Use a new external run root when resolving a documented gate
error; never replace failed episodes with favorable conditions.

Official logs and videos stay external. Telemetry stores qpos/qvel, body poses,
non-image observations, target pose when available, contact geometry names,
gripper contact, predicted chunks and padded simulator actions. Official policy
images are saved by the upstream video writer without changing camera input.
The old COFT pinhole function is used only for posthoc target-center frustum
projection with the actual simulator camera; it is not an occlusion estimator,
not Eye-in-Hand evidence and never a task-success label.

The telemetry wrapper and full gate require real integration validation. Compile
checks and gate unit tests alone do not establish simulator or policy execution.
See `docs/SIM_VALIDATION_REPORT.md` for the actual reached stage and errors.
