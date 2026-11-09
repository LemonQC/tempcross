"""See `python -m baselines.common.evaluator --help` for the shared evaluator."""
import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runpy
if __name__=='__main__':runpy.run_module('baselines.common.evaluator',run_name='__main__')
