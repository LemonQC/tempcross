#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: scripts/analyze_barx_run.sh RUN_ROOT OUT_DIR [VIDEO_REVIEWS_JSON]' >&2
  exit 2
fi
run_root=$1
analysis_out=$2
barx_python=${BARX_PYTHON:-/home/users/lemon/coft-p2/barx/.venv/bin/python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$analysis_out"
review_args=()
if [[ $# -eq 3 ]]; then review_args=(--reviews "$3"); fi
"$barx_python" -m simulation_validation.analyze_rollouts --root "$run_root" --out "$analysis_out" "${review_args[@]}"
"$barx_python" -m simulation_validation.plot_results --analysis "$analysis_out/analysis.json" --out "$analysis_out"
"$barx_python" -m simulation_validation.audit_pairing --root "$run_root" --out "$analysis_out/physical_pairing_audit.json"
"$barx_python" -m simulation_validation.validate_videos --root "$run_root" --out "$analysis_out/video_validation.json"
"$barx_python" -m simulation_validation.video_index --root "$run_root" --out "$analysis_out/video_index.json" "${review_args[@]}"
"$barx_python" -m simulation_validation.external_manifest --root "$run_root" --out "$analysis_out/external_files.csv"
