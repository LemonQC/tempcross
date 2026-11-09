"""Capture the executing stack without changing packages or physics."""
import datetime,importlib.metadata,json,platform,subprocess,sys
from baselines.common.paths import ROOT,SOURCE,VENDOR,DEPS,PROJECT,write_json

def main():
 versions={}
 for name in ['torch','torchvision','transformers','diffusers','huggingface-hub','robomimic','mujoco','mujoco-py','numpy','scipy','h5py','opencv-python','tensorflow','tensorflow-datasets','dlimp','safetensors']:
  try:versions[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:versions[name]='NOT_DISTRIBUTION_REGISTERED'
 # External PYTHONPATH dependency versions are read without replacing the old environment.
 for name in ['diffusers','huggingface-hub']:
  matches=list(DEPS.glob(name.replace('-','_')+'-*.dist-info'))
  for p in matches:versions['external_'+name]=p.name.removesuffix('.dist-info')
 sources={}
 for name,path in [('barx',SOURCE),('robomimic',VENDOR),('benchmark',PROJECT)]:
  sources[name]=dict(path=str(path),commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).splitlines())
 result=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,executable=sys.executable,platform=platform.platform(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),packages=versions,source=sources,simulation_and_training_seed=7,anchor_sampling_seed=20261109,formal_workers=4,formal_concurrent_latency_not_isolated=True,isolated_latency_in_separate_measurements=True,precision='float32 new BC/DP; unchanged released BARX inference',environment_mutation='No package replaced in the fixed BARX .venv; robomimic and two pinned pure-Python dependencies supplied via external sys.path')
 write_json(ROOT/'environment.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
