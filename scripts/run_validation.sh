#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" scripts/build_models.py
"$PYTHON" -m pytest -q tests
"$PYTHON" -m coftprobe.experiment --mode demo --scenes 12 --yaws 8 --seed 20261107 --out evidence/v0
"$PYTHON" -m coftvalidation.experiment --out evidence/v1
"$PYTHON" scripts/audit_results.py evidence/v1
