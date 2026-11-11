"""Distinguish total run cost from the selected checkpoint's actual exposure."""
from collections import Counter,defaultdict
import csv
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from common import OUT,ROOT,digest,manifests,read,write


def main():
    selected=read(OUT/'selection.json');state=read(ROOT/'training/status.json');protocol=read(OUT/'protocol.json')
    assert state['status']=='COMPLETE' and not selected['used_physical_success']
    n=selected['step']*protocol['effective_batch']
    rows={r['episode_id']:r for r in manifests()}
    counts=Counter();transforms=Counter();frames=defaultdict(set);prefix=hashlib.sha256();size=0
    path=ROOT/'training/consumed_samples.csv'
    with path.open('rb') as f:
        line=f.readline();keys=next(csv.reader([line.decode()]));prefix.update(line);size+=len(line)
        for i in range(n):
            line=f.readline();assert line
            row=dict(zip(keys,next(csv.reader([line.decode()]))))
            assert int(row['index'])==i and rows[row['episode_id']]['split']=='train'
            assert 0<=int(row['frame'])<rows[row['episode_id']]['frames']
            counts[row['dataset'],row['task']]+=1;transforms[int(row['transform_index'])]+=1
            frames[row['dataset'],row['episode_id']].add(int(row['frame']))
            prefix.update(line);size+=len(line)
    write(OUT/'selected_checkpoint_exposure.json',dict(
        selected_optimizer_updates=selected['step'],selected_sample_positions=n,
        total_run_optimizer_updates=state['completed_optimizer_updates'],total_run_sample_positions=state['sample_positions'],
        selected_effective_batch=protocol['effective_batch'],selected_checkpoint_adapter=selected['adapter'],
        selection_used_physical_success=False,no_validation_demonstrations_in_selected_prefix=True,
        dataset_task_positions=[dict(dataset=k[0],task=k[1],positions=v) for k,v in sorted(counts.items())],
        unique_demonstrations={d:sum(k[0]==d for k in frames) for d in ['mg_pnp_lite','panda_pnp']},
        unique_frame_positions=sum(len(s) for s in frames.values()),joint_transform_index_counts=dict(transforms),
        full_run_trace_path=str(path),full_run_trace_sha256=digest(path),
        prefix_definition='SHA256 of original CSV header plus exactly the first selected_sample_positions complete original CSV records',
        selected_trace_prefix_bytes=size,selected_trace_prefix_sha256=prefix.hexdigest(),
        table_label_768000_denotes_total_run_budget_not_deployed_model_exposure=True))


if __name__=='__main__':main()
