#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON="${BARX_SOURCE:-/home/users/lemon/coft-p2/barx}/.venv/bin/python"
# Prepared data and the remotely verified freeze marker must already exist.
# Every stage reuses an exact completed job and refuses to overwrite a failure.
"$BENCH_PYTHON" -m baselines.common.run_probes
"$BENCH_PYTHON" -m baselines.common.smoke_suite
"$BENCH_PYTHON" -m baselines.common.barx_smoke
"$BENCH_PYTHON" -m baselines.common.train_suite
"$BENCH_PYTHON" -m baselines.common.audit_matching
"$BENCH_PYTHON" -m baselines.common.evaluate_suite
"$BENCH_PYTHON" -m baselines.common.collect --final
"$BENCH_PYTHON" scripts/summarize_baselines.py
