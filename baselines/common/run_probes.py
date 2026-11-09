import json,subprocess
from baselines.common.paths import ROOT,PROJECT,PYTHON,environment,write_json

def main():
 specs=[('B','bc_gmm'),('B','bc_rnn'),('B','bc_transformer'),('B','diffusion_policy'),('C','bc_transformer'),('C','diffusion_policy'),('A','bc_transformer'),('A','diffusion_policy')]
 rows=[]
 for group,method in specs:
  for batch in [16,8,4]:
   label=f'{group}_{method}_b{batch}';path=ROOT/'probes'/label/'status.json'
   if path.exists():status=json.loads(path.read_text())
   else:
    log=ROOT/'probes'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:code=subprocess.run([str(PYTHON),'-m','baselines.common.probe','--method',method,'--group',group,'--target-batch',str(batch)],cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT).returncode
    status=json.loads(path.read_text()) if path.exists() else dict(status='TECHNICAL_FAILURE',error='No status; inspect '+str(log))
   rows.append(status);write_json(ROOT/'probe_suite.json',dict(status='RUNNING',results=rows));print(json.dumps(dict(group=group,method=method,batch=batch,status=status['status'])),flush=True)
   if status['status']=='COMPLETE' or not status.get('is_cuda_oom'):break
 write_json(ROOT/'probe_suite.json',dict(status='COMPLETE',results=rows))
if __name__=='__main__':main()
