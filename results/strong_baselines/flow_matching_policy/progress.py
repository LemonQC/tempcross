"""Persist observed counts, never convert unrun reference episodes to0%."""
import json
from .data import RUN
from baselines.strong.common import out,read,write,now

def main():
    directory=out('flow_matching_policy');status=read(directory/'status.json')
    if status['status']=='COMPLETE_FAITHFUL_PORT':print(json.dumps(status));return
    base=RUN/'training/source_target_visual';state=read(base/'state.json') if (base/'state.json').exists() else {}
    rows=[]
    if (base/'training.jsonl').exists():
        for line in (base/'training.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    last=next((r for r in reversed(rows) if 'loss' in r),None)
    count=max(state.get('actual_training_updates',0),last['update'] if last else 0)
    episodes=[read(p) for p in sorted((RUN/'physical/formal/source_target_visual').glob('*/condition-*/episode.json'))]
    report=dict(status='PARTIAL',phase=status['phase'],as_of_utc=now(),actual_logged_optimizer_calls=count,
        resumable_saved_step=state.get('saved_step',0),frozen_optimizer_budget=24000,
        selected_step=state.get('best_step'),selected_anchors=(state['best_step']*32 if state.get('best_step') else None),
        latest_training_record=last,validation_history=[r for r in rows if 'validation' in r],
        completed_formal_episodes=len(episodes),role='EXPOSED_REFERENCE',
        partial_success_rate=sum(r['success'] for r in episodes)/len(episodes) if episodes else None,
        formal_budget_changed=False)
    write(RUN/'live_progress.json',report);write(directory/'progress.json',report)
    write(directory/'status.json',dict(status,actual_training_updates=count,physical_evaluation_episodes=len(episodes)))
    from baselines.strong.matrix_current import main as matrix
    matrix();print(json.dumps({k:v for k,v in report.items() if k not in ['latest_training_record','validation_history']}))

if __name__=='__main__':main()
