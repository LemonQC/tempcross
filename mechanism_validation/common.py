import os
from pathlib import Path
from baselines.common.paths import PROJECT, SOURCE, ROOT as BASELINE_ROOT, digest, write_json, environment

ROOT = Path(os.environ.get('MECHANISM_ROOT', '/home/users/lemon/mechanism-20261110'))
OUT = PROJECT / 'results/mechanism'
CASES = [64, 68, 71, 72, 76, 88, 93, 94]
TASK = 'pnp_counter_to_sink'
CONFIG = PROJECT / 'configs/mechanism_protocol.json'


def load_config():
    import json
    return json.loads(CONFIG.read_text())
