"""Run every eligible existing method on exactly the same untouched test cells."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,time
from pathlib import Path
from baselines.common.paths import ROOT,PROJECT,PYTHON,TASKS,environment,write_json,digest
from baselines.barx.registry import LEGACY


def wait_for_training():
 while True:
  path=ROOT/'training_suite.json'
  if path.exists():
   d=json.loads(path.read_text())
   if d['status'].startswith('COMPLETE'):return d
  time.sleep(15)


def main():
 state=wait_for_training();methods=[];blocked=[]
 for label in state['models']:
  path=ROOT/'training'/label/'selected.json'
  if not path.exists():blocked.append(dict(label=label,reason=state['models'][label]));continue
  s=json.loads(path.read_text());methods.append(dict(label=label,kind='robomimic',checkpoint=s['checkpoint'],group=label[0]))
 for label,path in LEGACY.items():
  jobs=[ROOT/'smoke'/label/t/'panda/job.json' for t in TASKS]
  if not all(p.exists() and json.loads(p.read_text())['status']=='COMPLETE' for p in jobs):blocked.append(dict(label=label,reason='COMMON_EVALUATOR_SMOKE_TECHNICAL_FAILURE'));continue
  methods.append(dict(label=label,kind='barx',checkpoint=str(path),group=label[0],legacy=True))
 suite=dict(status='SELECTED_CHECKPOINT_DEVELOPMENT_SMOKE',methods=methods,blocked=blocked,final_test_model_selection_closed=True,formal_workers=4,cells=[])
 ledger=ROOT/'formal_suite.json';write_json(ledger,suite)
 reference_scope=PROJECT/'configs/evaluation/baseline_source_reference_scope.json';references=json.loads(reference_scope.read_text())['labels']
 suite['source_reference_scope_sha256']=digest(reference_scope);write_json(ledger,suite)
 # Probe rollouts validate the algorithm interface; these additionally load the
 # actual selected weights, still only on old development condition 0.
 smoke_specs=[(m,t,'panda') for m in methods if m['kind']=='robomimic' for t in TASKS]+[(m,t,'ur5e') for m in methods if m['label'] in references for t in TASKS]
 def selected_smoke(spec):
  m,task,robot=spec;label='selected_'+m['label'];job=ROOT/'smoke'/label/task/robot/'job.json'
  if job.exists():return json.loads(job.read_text())
  log=ROOT/'smoke'/f'{label}-{task}-{robot}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:
   subprocess.run([str(PYTHON),'-m','baselines.common.evaluator','--label',label,'--kind',m['kind'],'--checkpoint',m['checkpoint'],'--task',task,'--robot',robot,'--smoke'],cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT)
  return json.loads(job.read_text()) if job.exists() else dict(label=label,status='TECHNICAL_FAILURE',log=str(log))
 with ThreadPoolExecutor(max_workers=4) as pool:
  suite['selected_checkpoint_smokes']=list(pool.map(selected_smoke,smoke_specs))
 failed={r['label'].removeprefix('selected_') for r in suite['selected_checkpoint_smokes'] if r['status']!='COMPLETE'}
 blocked.extend(dict(label=label,reason='SELECTED_CHECKPOINT_SMOKE_TECHNICAL_ERROR') for label in sorted(failed))
 methods=[m for m in methods if m['label'] not in failed];suite['methods']=methods
 suite['status']='ISOLATED_LATENCY';write_json(ledger,suite)
 for m in methods:
  log=ROOT/'latency'/f"{m['label']}.log";log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:
   code=subprocess.run([str(PYTHON),'-m','baselines.common.latency','--label',m['label'],'--kind',m['kind'],'--checkpoint',m['checkpoint']],cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT).returncode
  if code:blocked.append(dict(label=m['label'],reason='ISOLATED_LATENCY_TECHNICAL_ERROR',log=str(log)))
 suite['status']='FROZEN_FINAL_EVALUATION';write_json(ledger,suite)
 specs=[(m,t,'panda') for m in methods for t in TASKS]+[(m,t,'ur5e') for m in methods if m['label'] in references for t in TASKS]
 def run(spec):
  m,task,robot=spec;job=ROOT/'formal'/m['label']/task/robot/'job.json'
  if job.exists():return json.loads(job.read_text())
  log=ROOT/'formal'/f"{m['label']}-{task}-{robot}.log";log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:
   code=subprocess.run([str(PYTHON),'-m','baselines.common.evaluator','--label',m['label'],'--kind',m['kind'],'--checkpoint',m['checkpoint'],'--task',task,'--robot',robot],cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT).returncode
  return json.loads(job.read_text()) if job.exists() else dict(status='TECHNICAL_FAILURE',label=m['label'],task=task,robot=robot,log=str(log))
 with ThreadPoolExecutor(max_workers=suite['formal_workers']) as pool:
  for f in as_completed([pool.submit(run,x) for x in specs]):
   row=f.result();suite['cells'].append(row);write_json(ledger,suite);print(json.dumps({k:row.get(k) for k in ['label','task','robot','status','episodes','successes']}),flush=True)
 suite['status']='COMPLETE_WITH_TECHNICAL_BLOCKERS' if any(c['status']!='COMPLETE' for c in suite['cells']) else 'COMPLETE';write_json(ledger,suite)
if __name__=='__main__':main()
