"""Two full old-development rollouts per new probe, never on final conditions."""
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,time
from baselines.common.paths import ROOT,PROJECT,PYTHON,TASKS,environment,write_json


def run_spec(spec):
 group,method=spec
 for _ in range(120):
  choices=[]
  for b in [16,8,4]:
   p=ROOT/f'probes/{group}_{method}_b{b}/status.json'
   if p.exists():
    s=json.loads(p.read_text())
    if s['status']=='COMPLETE':choices.append(s)
  if choices:break
  time.sleep(5)
 else:return dict(label=f'{group}_{method}',status='PROBE_NOT_AVAILABLE')
 probe=choices[0];label=f'probe_{group}_{method}';results=[]
 for task in TASKS:
  job=ROOT/'smoke'/label/task/'panda/job.json'
  if job.exists() and json.loads(job.read_text())['status']=='COMPLETE':results.append(json.loads(job.read_text()));continue
  log=ROOT/'smoke'/f'{label}-{task}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:
   code=subprocess.run([str(PYTHON),'-m','baselines.common.evaluator','--label',label,'--kind','robomimic','--checkpoint',probe['checkpoint'],'--task',task,'--smoke'],cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT).returncode
  if code:return dict(label=f'{group}_{method}',status='TECHNICAL_FAILURE',log=str(log))
  results.append(json.loads(job.read_text()))
 return dict(label=f'{group}_{method}',status='COMPLETE',rollouts=2,results=results)


def main():
 specs=[('B','bc_gmm'),('B','bc_rnn'),('B','bc_transformer'),('B','diffusion_policy'),('C','bc_transformer'),('C','diffusion_policy'),('A','bc_transformer'),('A','diffusion_policy')]
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(run_spec,specs):
   rows.append(result);write_json(ROOT/'smoke_suite.json',dict(status='RUNNING',results=rows));print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
 write_json(ROOT/'smoke_suite.json',dict(status='COMPLETE',results=rows))
if __name__=='__main__':main()
