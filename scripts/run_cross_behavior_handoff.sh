#!/usr/bin/env bash
set -euo pipefail
HANDOFF_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$HANDOFF_REPO_DIR"
if (( $# == 0 )); then
  set -- run
fi
exec python scripts/cross_behavior_handoff.py "$@"
