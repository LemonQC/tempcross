"""Finish read-only delivery generation once the actual supervised runs close."""
import datetime,json,subprocess,time,traceback
from baselines.common.paths import ROOT,PROJECT,PYTHON,environment,write_json

def main():
 ledger=ROOT/'delivery_ready.json'
 write_json(ledger,dict(status='WAITING_FOR_REAL_FORMAL_RESULTS',no_experiments_repeated=True))
 while True:
  p=ROOT/'formal_suite.json';a=ROOT/'closed_cell_audit.json'
  if p.exists() and json.loads(p.read_text())['status'].startswith('COMPLETE'):
   suite=json.loads(p.read_text());needed=sum(c['status']=='COMPLETE' for c in suite['cells'])
   audit=json.loads(a.read_text()) if a.exists() else {'cells':[]}
   if len(audit['cells'])>=needed:
    assert all(c['status']=='VERIFIED' for c in audit['cells']),audit
    break
  time.sleep(15)
 try:
  commands=[[str(PYTHON),'-m','baselines.common.validate'],[str(PYTHON),'-m','baselines.common.collect','--final'],[str(PYTHON),'scripts/summarize_baselines.py'],[str(PROJECT/'.venv/bin/python'),'-m','pytest','tests','-q'],['git','diff','--check']]
  for i,cmd in enumerate(commands):
   log=ROOT/f'delivery-step-{i:02d}.log'
   with log.open('w') as f:subprocess.run(cmd,cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT,check=True)
  status=json.loads((PROJECT/'results/baselines/status.json').read_text())
  write_json(ledger,dict(status='RESULTS_AND_REPORTS_READY_FOR_GITHUB_PUBLICATION',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_episodes=status['formal_episodes'],formal_cells=status['formal_complete_cells'],no_experiments_repeated=True,github_publication_still_required=True))
 except Exception as exc:
  write_json(ledger,dict(status='DELIVERY_TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
