#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BARX_SOURCE=${BARX_SOURCE:-/home/users/lemon/coft-p2/barx}
export BARX_ARTIFACT_ROOT=${BARX_ARTIFACT_ROOT:-/home/users/lemon/coft-p2/artifacts}
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
export HF_HOME="$BARX_ARTIFACT_ROOT/hf"
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$BARX_SOURCE/.venv/bin/python" -m simulation_validation.gate \
  --source "$BARX_SOURCE" --artifacts "$BARX_ARTIFACT_ROOT" \
  --out "${BARX_RUN_ROOT:-/home/users/lemon/coft-p2/rollouts/formal}" "$@"
