from concurrent.futures import ThreadPoolExecutor
import json,subprocess
from baselines.common.paths import ROOT,PROJECT,PYTHON,TASKS,environment,write_json
from baselines.barx.registry import LEGACY


def run(spec):
 label,checkpoint,task=spec;directory=ROOT/'smoke'/label/task/'panda'
 job=directory/'job.json'
 if job.exists():return json.loads(job.read_text())
 log=ROOT/'smoke'/f'{label}-{task}.log';log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('w') as f:
  code=subprocess.run([str(PYTHON),'-m','baselines.common.evaluator','--label',label,'--kind','barx','--checkpoint',str(checkpoint),'--task',task,'--smoke'],cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT).returncode
 return json.loads(job.read_text()) if job.exists() else dict(label=label,task=task,status='TECHNICAL_FAILURE',log=str(log))


def main():
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for d in pool.map(run,[(label,p,t) for label,p in LEGACY.items() for t in TASKS]):
   rows.append(d);write_json(ROOT/'barx_smoke_suite.json',dict(status='RUNNING',results=rows));print(json.dumps(dict(label=d['label'],task=d['task'],status=d['status'])),flush=True)
 write_json(ROOT/'barx_smoke_suite.json',dict(status='COMPLETE',results=rows))
if __name__=='__main__':main()
