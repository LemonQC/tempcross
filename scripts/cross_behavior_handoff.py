#!/usr/bin/env python3
"""Offline evidence handoff; works without Codex or any connected application."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_behavior.handoff import main

if __name__ == '__main__':
    sys.exit(main())
