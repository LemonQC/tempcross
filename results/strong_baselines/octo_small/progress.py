"""Expose persisted real progress without inventing unfinished evaluation rates."""
import json
from baselines.strong.common import ROOT,out,read,write,now,TASKS

def main():
    run=ROOT/'octo_small';directory=out('octo_small')
    current=read(directory/'status.json')
    if current['status']=='COMPLETE_FAITHFUL_PORT':print(json.dumps(current));return
    training=run/'training/source_target';last=None;validations=[]
    log=training/'training.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'metrics' in r:last=r
            if 'validation_ddpm_loss' in r:validations.append(r)
    state=read(training/'state.json') if (training/'state.json').exists() else {}
    episodes=[]
    for task in TASKS:
        path=run/'physical/formal/source_target'/task
        for p in sorted(path.glob('condition-*/episode.json')):
            r=read(p);episodes.append(dict(task=task,index=r['official_condition_index'],success=r['success'],steps=r['steps']))
    updates=max(state.get('actual_training_updates',0),last['update'] if last else 0)
    result=dict(as_of_utc=now(),status='PARTIAL',phase=current['phase'],
        verified_persisted_training_updates=updates,resumable_saved_update=state.get('saved_step',0),
        frozen_training_budget=24000,completed_official_episodes=len(episodes),episodes=episodes,
        success_rate=None if not episodes else sum(r['success'] for r in episodes)/len(episodes),
        formal_role='EXPOSED_REFERENCE',last_training_log=last,validation_history=validations,
        pipeline_process=read(run/'pipeline_process.json'),formal_budget_changed=False)
    write(run/'live_progress.json',result);write(directory/'progress.json',result)
    write(directory/'status.json',dict(current,actual_training_updates=updates,
        physical_evaluation_episodes=len(episodes),success_rate=result['success_rate']))
    from baselines.strong.matrix_current import main as matrix
    matrix();print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','validation_history','last_training_log']}))

if __name__=='__main__':main()
