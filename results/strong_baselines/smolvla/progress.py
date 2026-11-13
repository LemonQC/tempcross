"""Persist only observed native update/episode progress, not assumed results."""
import json
from baselines.strong.common import ROOT,out,read,write,now,TASKS

def main():
    run=ROOT/'smolvla';directory=out('smolvla');current=read(directory/'status.json')
    if current['status']=='COMPLETE_FAITHFUL_PORT':print(json.dumps(current));return
    base=run/'training/source_target_state';state=read(base/'state.json') if (base/'state.json').exists() else {}
    rows=[]
    if (base/'training.jsonl').exists():
        for line in (base/'training.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    last=next((r for r in reversed(rows) if 'metrics' in r),None)
    episodes=[]
    for task in TASKS:
        for p in sorted((run/'physical/formal/source_target_state'/task).glob('condition-*/episode.json')):
            r=read(p);episodes.append(dict(task=task,index=r['official_condition_index'],success=r['success'],steps=r['steps']))
    count=max(state.get('actual_training_updates',0),last['update'] if last else 0)
    report=dict(status='PARTIAL',phase=current['phase'],as_of_utc=now(),verified_persisted_training_updates=count,
        resumable_saved_update=state.get('saved_step',0),frozen_training_budget=16000,
        completed_official_episodes=len(episodes),episodes=episodes,role='EXPOSED_REFERENCE',
        success_rate=sum(r['success'] for r in episodes)/len(episodes) if episodes else None,
        latest_training_record=last,validation_history=[r for r in rows if 'validation' in r],formal_budget_changed=False)
    write(run/'live_progress.json',report);write(directory/'progress.json',report)
    write(directory/'status.json',dict(current,actual_training_updates=count,physical_evaluation_episodes=len(episodes),success_rate=report['success_rate']))
    from baselines.strong.matrix_current import main as matrix
    matrix();print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','validation_history','latest_training_record']}))

if __name__=='__main__':main()
