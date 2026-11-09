"""Read actual ledgers without training, inference, or simulator steps."""
import json
from baselines.common.paths import ROOT

def main():
 models=[]
 for p in sorted((ROOT/'training').glob('*/status.json')):
  d=json.loads(p.read_text());models.append(dict(label=p.parent.name,status=d['status'],positions=d['anchor_positions'],selected=(p.parent/'selected.json').exists(),tier=d.get('prescribed_tier_positions')))
 f=ROOT/'formal_suite.json';suite=json.loads(f.read_text()) if f.exists() else {};jobs=[json.loads(p.read_text()) for p in (ROOT/'formal').rglob('job.json')]
 smoke=[json.loads(p.read_text()) for p in (ROOT/'smoke').glob('selected_*/*/*/job.json')]
 audit=ROOT/'closed_cell_audit.json';delivery=ROOT/'delivery_ready.json'
 print(json.dumps(dict(selected=sum(d['selected'] for d in models),active=[d for d in models if d['status']=='TRAINING'],training_failures=[d for d in models if d['status']=='TECHNICAL_FAILURE'],formal_phase=suite.get('status','WAITING_FOR_TRAINING'),formal_episodes=sum(j.get('completed_episodes',0) for j in jobs),formal_complete_cells=sum(j['status']=='COMPLETE' for j in jobs),formal_failures=[dict(label=j['label'],task=j['task'],robot=j['robot'],error=j.get('error')) for j in jobs if j['status']=='TECHNICAL_FAILURE'],selected_smoke_complete=sum(j['status']=='COMPLETE' for j in smoke),smoke_failures=[dict(label=j['label'],task=j['task'],robot=j['robot'],error=j.get('error')) for j in smoke if j['status']=='TECHNICAL_FAILURE'],audit=json.loads(audit.read_text()) if audit.exists() else None,delivery=json.loads(delivery.read_text()) if delivery.exists() else None)))
if __name__=='__main__':main()
