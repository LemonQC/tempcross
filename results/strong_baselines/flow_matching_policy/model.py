"""Pinned simulator -> isolated official HRI-EU Flow Matching Policy runtime; no policy substitute."""
import os
import subprocess
import time
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
from baselines.strong.common import ROOT,PROJECT

RUN=ROOT/'flow_matching_policy'

class Policy:
    def __init__(self,checkpoint,role='DEVELOPMENT_OR_REFERENCE'):
        self.socket=f'/tmp/tempcross-flow_matching_policy-{os.getpid()}.sock';stamp=f'{time.time_ns()}-{os.getpid()}'
        directory=RUN/'inference';directory.mkdir(parents=True,exist_ok=True)
        self.log=(directory/f'{stamp}.log').open('w');ledger=directory/f'{stamp}.jsonl'
        environment=dict(os.environ)
        for name in ['PYTHONPATH','MUJOCO_GL','USE_TF']:environment.pop(name,None)
        self.process=subprocess.Popen([str(RUN/'env/bin/python'),'-u','-m','results.strong_baselines.flow_matching_policy.worker',
            '--socket',self.socket,'--checkpoint',str(checkpoint),'--ledger',str(ledger),'--role',role],
            cwd=PROJECT,env=environment,stdout=self.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+300
        while not Path(self.socket).exists():
            if self.process.poll() is not None:raise RuntimeError('Native HRI-EU Flow Matching Policy worker failed: '+str(self.log.name))
            if time.monotonic()>deadline:
                self.process.terminate();raise TimeoutError('Native worker load timeout: '+str(self.log.name))
            time.sleep(.2)
        self.connection=Client(self.socket,family='AF_UNIX');self.runtime_metadata=self.connection.recv()
        self.episode=-1;self.reset()
    def reset(self):
        self.episode+=1;self.step=0;self.connection.send({'command':'reset'})
        assert self.connection.recv()['status']=='RESET'
    def predict_action(self,observation):
        rgb=np.asarray(observation['rgb'],np.uint8);state=np.asarray(observation['state'],np.float32)
        assert rgb.shape==(180,320,3) and state.shape==(9,) and np.isfinite(state).all()
        self.connection.send(dict(command='predict',rgb=rgb.tobytes(),shape=list(rgb.shape),state=state.tolist(),
            instruction=observation['instruction'],episode=self.episode,step=self.step))
        result=self.connection.recv();self.step+=1
        prediction=dict(action=np.asarray(result['action'],np.float32),representation='canonical_osc',new_query=result['new_query'])
        if result['new_query']:prediction['native_query']=np.asarray(result['native_query'],np.float32)
        return prediction
    def close(self):
        if hasattr(self,'connection'):
            try:self.connection.send({'command':'close'});self.connection.close()
            except (BrokenPipeError,EOFError,OSError):pass
            del self.connection
        if hasattr(self,'process'):
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.terminate()
        Path(self.socket).unlink(missing_ok=True)
        if hasattr(self,'log'):self.log.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
