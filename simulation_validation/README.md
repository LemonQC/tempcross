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
The official prebuilt route has now been validated on the RTX 3090, including
actual policy rollouts. Do not silently switch to a different attention implementation.

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
BARX_RUN_ROOT=/home/users/lemon/coft-p2/rollouts/new-evaluation scripts/run_barx_gate.sh
# Optional P0 only:
BARX_RUN_ROOT=/home/users/lemon/coft-p2/rollouts/p0-only scripts/run_barx_gate.sh --p0-only
```

The gate runs official tests, then one complete Joint Reps episode on each robot.
It requires complete official summaries, episode records and decodable videos.
It never counts a technical crash as an unsuccessful task episode. Only after P0
passes does it run the fixed four cells of 20 episodes, using the first official
conditions, max_steps=600, act_horizon=8, seed=7 and prior normalization mg_pnp_lite.
Both methods infer actions without representation prompts. The gate refuses any nonempty output directory before writing logs or its ledger. Runs do not overwrite
earlier directories. Use a new external run root when resolving a documented gate
error; never replace failed episodes with favorable conditions.

Official logs and videos stay external. Telemetry stores qpos/qvel, body poses,
non-image observations, target pose when available, contact geometry names,
gripper contact, predicted chunks and padded simulator actions. Official policy
images are saved by the upstream video writer without changing camera input.
The old COFT pinhole function is used only for posthoc target-center frustum
projection with the actual simulator camera; it is not an occlusion estimator,
not Eye-in-Hand evidence and never a task-success label.

The telemetry wrapper and full gate have passed integration validation: P0 plus
80/80 P1 physical rollouts. Compile checks and unit tests alone are not that evidence.
See `docs/SIM_VALIDATION_REPORT.md` for actual results and limitations.

## Verified installation recovery (user-authorized continuation)

The continued installation completed in the external `.venv`: CUDA matmul,
FlashAttention 2.5.5 CUDA kernel, NVIDIA MuJoCo EGL and both official robot
registrations passed. All 76 upstream tests passed with no skip. This is runtime
verification, **not** a completed P0 episode. Evidence:
`results/simulation/install_resume/`.

`mirror_install.py` exports the upstream lock with the train extra, retains artifact
SHA256 values, and changes only registry download transport to Tsinghua's mirror.
It requires bootstrap Python 3.10 with `tomli`, `packaging` and `uv`, the fixed Git
checkouts in the private uv cache, and the downloaded official FlashAttention wheel.
It deliberately fails rather than resolving alternative versions or modifying the
upstream lock. Review `mirror-manifest.json` for every URL/hash and Git commit.
The Flash wheel hash is our recorded artifact hash, not an upstream published checksum.

```bash
cd /home/users/lemon/CodexProject/tempcross
/home/users/lemon/coft-p2/bootstrap/bin/python -m simulation_validation.mirror_install \
  --source /home/users/lemon/coft-p2/barx \
  --uv /home/users/lemon/coft-p2/bootstrap/bin/uv \
  --cache /home/users/lemon/coft-p2/uv-cache \
  --wheels /home/users/lemon/coft-p2/install-resume/wheels \
  --out /home/users/lemon/coft-p2/install-resume/mirror

# Standalone verification; does not invoke uv sync or change dependencies:
/home/users/lemon/coft-p2/barx/.venv/bin/python -m simulation_validation.verify_runtime \
  --out /home/users/lemon/coft-p2/install-resume/runtime-recheck.json

# Official kitchen archive contents/checksums, resumable transport:
NO_PROXY=public.boxcloud.com,.public.boxcloud.com \
no_proxy=public.boxcloud.com,.public.boxcloud.com \
/home/users/lemon/coft-p2/barx/.venv/bin/python -m simulation_validation.install_kitchen_assets \
  --archive-root /home/users/lemon/coft-p2/artifacts/kitchen-archives
```

The upstream pins have four metadata conflicts (see `dependency_check.txt`):
TensorFlow-metadata and tianshou disagree with upstream's explicit protobuf
3.20.3 override; the pinned robosuite declares MuJoCo >=3.2.3 and optional whole-body
`mink`, whereas BARX pins MuJoCo 3.1.1 and does not install mink. We retain those
upstream choices. Registered UR5e/Panda and EGL pass; that does not establish every
controller or complete policy rollout. Do not interpret `uv pip check` as clean.

## Physical-result analysis

After all four cells complete:

```bash
scripts/analyze_barx_run.sh \
  /home/users/lemon/coft-p2/rollouts/user-authorized-resume \
  results/simulation/closed_loop \
  /home/users/lemon/coft-p2/rollouts/user-authorized-resume/analysis/video_reviews.json
```

The optional third argument contains explicit video/state review records. Without
it failed episodes remain `UNKNOWN / VIDEO_REVIEW_REQUIRED`; the script never
invents causal annotations. `review_sheets.py` generates labeled sampled frames
for inspection alongside complete HDF5/JSONL histories. Samples are not dense
frame-by-frame human annotations. Success labels always come from upstream
`episodes.jsonl`; raw telemetry serializes some NumPy booleans as strings, which
must never be counted by Python truthiness. The analyzer rejects non-boolean
official labels and mismatched summary counts.

Stage diagnostics include EEF-to-target-center distance, finger-contact events,
object height/displacement, commanded gripper closure, exact policy-to-environment
action mapping, simulator joint limits and center-frustum projection. They do not
provide force closure, collision causality, area visibility or task success labels.
The report distinguishes a brief contact/rise from confirmed acquisition and
transport. Complete decoding checks each video's frame count against official
executed steps. All weights, raw state/action histories and videos remain external.
