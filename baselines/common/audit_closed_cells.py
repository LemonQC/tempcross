"""Validate closed formal cells while other cells continue, without simulator steps."""
import json,time,traceback
from baselines.common.paths import ROOT,write_json
from baselines.common.validate import validate_job

def main():
 checked={}
 while True:
  for path in sorted((ROOT/'formal').rglob('job.json')):
   if str(path) in checked:continue
   job=json.loads(path.read_text())
   if job['status']!='COMPLETE':continue
   try:
    result=validate_job(path);checked[str(path)]=dict(job=str(path),status='VERIFIED',episodes=result['episodes'])
   except Exception as exc:
    checked[str(path)]=dict(job=str(path),status='VALIDATION_TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc())
   write_json(ROOT/'closed_cell_audit.json',dict(cells=list(checked.values()),verified_episodes=sum(r.get('episodes',0) for r in checked.values()),read_only_physics=True));print(json.dumps(checked[str(path)]),flush=True)
  suite=ROOT/'formal_suite.json'
  if suite.exists() and json.loads(suite.read_text())['status'].startswith('COMPLETE'):break
  time.sleep(15)
if __name__=='__main__':main()
