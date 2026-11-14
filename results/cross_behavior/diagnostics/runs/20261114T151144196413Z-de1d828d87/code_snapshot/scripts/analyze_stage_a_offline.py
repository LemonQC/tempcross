#!/usr/bin/env python3
"""Read saved JSONL/RGB-D only. Never imports a simulator or a training stack."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_behavior.offline_diagnostic import main

if __name__ == '__main__':
    sys.exit(main())
