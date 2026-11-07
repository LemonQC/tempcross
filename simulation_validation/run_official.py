"""Run the official BARX evaluator with read-only state logging."""
import argparse
import os
from pathlib import Path
import runpy
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--embodiment',choices=['ur5e','panda'],required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/'policy')]
    from barx.benchmark import EMBODIMENTS
    from simulation_validation.observe import Recorder
    rec=Recorder(out/'telemetry');rec.install();arm=EMBODIMENTS[a.embodiment]
    sys.argv=[str(source/'policy/experiments/robot/robocasa_x/evaluate.py'),
              '--pretrained_checkpoint',str(Path(a.checkpoint).resolve()),'--embodiment',a.embodiment,
              '--robot',arm.robot,'--gripper_types',arm.gripper,'--camera',arm.camera,
              '--task','XPnPCounterToSink','--unnorm_key','mg_pnp_lite','--num_trials_per_task',str(a.episodes),
              '--start_seed','1000','--seed','7','--max_steps','600','--act_horizon','8',
              '--conditions_dir',str(source/'evaluation/conditions'),'--rollout_dir',str(out/'official')]
    os.chdir(source/'policy')
    try:runpy.run_path(sys.argv[0],run_name='__main__')
    finally:rec.close()
if __name__=='__main__':main()
