"""Keep the pinned simulator stack separate from official Octo's JAX runtime."""
import collections
import os
from pathlib import Path
import subprocess
import time
from multiprocessing.connection import Client
import numpy as np
from baselines.strong.common import ROOT,PROJECT

RUN=ROOT/'octo_small'

class Policy:
    def __init__(self,checkpoint):
        self.socket=f'/tmp/tempcross-octo-{os.getpid()}.sock'
        stamp=f'{time.time_ns()}-{os.getpid()}'
        logdir=RUN/'inference';logdir.mkdir(parents=True,exist_ok=True)
        self.log=(logdir/f'{stamp}.log').open('w');ledger=logdir/f'{stamp}.jsonl'
        environment=dict(os.environ)
        for key in ('PYTHONPATH','USE_TF','MUJOCO_GL','CUDA_VISIBLE_DEVICES'):
            environment.pop(key,None)
        environment.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',TOKENIZERS_PARALLELISM='false',
            HF_HOME=str(RUN/'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        self.process=subprocess.Popen([str(RUN/'env/bin/python'),'-u','-m',
            'results.strong_baselines.octo_small.runtime','--serve',self.socket,
            '--checkpoint',str(checkpoint),'--ledger',str(ledger)],cwd=PROJECT,env=environment,
            stdout=self.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+300
        while not Path(self.socket).exists():
            if self.process.poll() is not None:raise RuntimeError('Octo worker failed; see '+str(self.log.name))
            if time.monotonic()>deadline:
                self.process.terminate();raise TimeoutError('Octo model load timeout; see '+str(self.log.name))
            time.sleep(.2)
        self.connection=Client(self.socket,family='AF_UNIX');self.runtime_metadata=self.connection.recv()
        self.episode=-1;self.reset()
    def reset(self):
        self.history=collections.deque(maxlen=2);self.queue=collections.deque();self.step=0;self.episode+=1
        self.connection.send({'command':'reset'});assert self.connection.recv()['status']=='RESET'
    def predict_action(self,observation):
        self.history.append(np.asarray(observation['rgb'],np.uint8).copy())
        new=not self.queue
        if new:
            history=list(self.history)
            valid=[True,True] if len(history)==2 else [False,True]
            if len(history)==1:history=[history[0],history[0]]
            self.connection.send(dict(command='predict',images=history,valid=valid,
                instruction=observation['instruction'],episode=self.episode,step=self.step))
            result=self.connection.recv();chunk=np.asarray(result['canonical'],np.float32)
            self.queue.extend(chunk)
        action=self.queue.popleft();self.step+=1
        prediction=dict(action=action,representation='canonical_osc',new_query=new)
        if new:prediction['native_query']=chunk
        return prediction
    def close(self):
        if hasattr(self,'connection'):
            try:self.connection.send({'command':'close'});self.connection.close()
            except (EOFError,BrokenPipeError,OSError):pass
            del self.connection
        if hasattr(self,'process'):
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.terminate()
        Path(self.socket).unlink(missing_ok=True)
        if hasattr(self,'log'):self.log.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
