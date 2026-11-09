"""Run existing models; extend budgets only by frozen held-out-data rules."""
import json,subprocess,time
from baselines.common.paths import ROOT,PROJECT,PYTHON,TASKS,environment,write_json

SPECS=[('B','bc_gmm'),('B','bc_rnn'),('B','bc_transformer'),('B','diffusion_policy'),('C','bc_transformer'),('C','diffusion_policy'),('A','bc_transformer'),('A','diffusion_policy')]

def plateau(s):
 v=[x['value'] for x in s['validation_history']]
 if len(v)<6:return False,None
 before=sum(v[-6:-3])/3;after=sum(v[-3:])/3;gain=(before-after)/max(abs(before),1e-12)
 return gain<=.02,gain


def train_to(group,method,end):
 label=f'{group}_{method}';directory=ROOT/'training'/label
 previous=directory/'status.json'
 if previous.exists():
  d=json.loads(previous.read_text())
  if d['status']=='TIER_COMPLETE' and d['anchor_positions']>=end:return d
  if d['status']=='TECHNICAL_FAILURE':return d
 log=ROOT/'training'/f'{label}-to-{end:06d}.log';log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('w') as stream:
  code=subprocess.run([str(PYTHON),'-m','baselines.common.train','--group',group,'--method',method,'--end-positions',str(end),'--target-batch','16'],cwd=PROJECT,env=environment(),stdout=stream,stderr=subprocess.STDOUT).returncode
 if not previous.exists():return dict(status='TECHNICAL_FAILURE',label=label,error='No state, inspect '+str(log))
 return json.loads(previous.read_text())


def wait_smoke(group,method):
 for _ in range(240):
  complete=True
  for task in TASKS:
   path=ROOT/'smoke'/f'probe_{group}_{method}'/task/'panda/job.json'
   if not path.exists():complete=False;break
   d=json.loads(path.read_text())
   if d['status']=='TECHNICAL_FAILURE':return False
   if d['status']!='COMPLETE':complete=False;break
  if complete:return True
  time.sleep(5)
 return False


def main():
 ledger=ROOT/'training_suite.json';state=dict(status='FIRST_TIERS',models={},predeclared_budget_policy=True,final_test_inspected=False)
 def save():write_json(ledger,state)
 save()
 for group,method in SPECS:
  label=f'{group}_{method}'
  if not wait_smoke(group,method):state['models'][label]=dict(status='BLOCKED_SMOKE_TECHNICAL');save();continue
  d=train_to(group,method,96000);state['models'][label]=d;save();print(json.dumps(dict(label=label,status=d['status'],positions=d.get('anchor_positions'))),flush=True)
 state['status']='VALIDATION_ONLY_BUDGET_EXTENSIONS';save()
 families=[['B_bc_gmm'],['B_bc_rnn'],['B_bc_transformer','C_bc_transformer'],['B_diffusion_policy','C_diffusion_policy'],['A_bc_transformer'],['A_diffusion_policy']]
 for family in families:
  family=[label for label in family if state['models'][label].get('status')=='TIER_COMPLETE']
  if not family:continue
  while True:
   reports=[plateau(state['models'][label]) for label in family];current=max(state['models'][label]['anchor_positions'] for label in family)
   if all(x[0] for x in reports) or current>=768000:
    for label,(stopped,gain) in zip(family,reports):
     d=state['models'][label];d.update(selection_status='SELECTED_FINAL_CHECKPOINT',budget_stop_reason='VALIDATION_NO_FURTHER_GAIN' if stopped else 'BUDGET_LIMITED_NOT_CONVERGED',recent_validation_relative_improvement=gain)
     write_json(ROOT/'training'/label/'selected.json',d)
    save();break
   end=current*2
   for label in family:
    group,method=label.split('_',1);d=train_to(group,method,end);state['models'][label]=d;save();print(json.dumps(dict(label=label,status=d['status'],positions=d.get('anchor_positions'))),flush=True)
   if any(state['models'][label]['status']!='TIER_COMPLETE' for label in family):break
 state['status']='COMPLETE_WITH_ACCURATE_BLOCKERS' if any(s.get('status')!='TIER_COMPLETE' for s in state['models'].values()) else 'COMPLETE'
 save()
if __name__=='__main__':main()
