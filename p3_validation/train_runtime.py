"""Measured upstream LoRA execution with exact consumed-input provenance."""
import atexit
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time
from p3_validation.common import ROOT,SOURCE
from p2_validation.offline_audit import sha256

STREAMS=[]
ACTIVE_RUN_LABEL=None


class AuditedStream:
    def __init__(self,root):
        self.root=Path(root);self.n=0;self.counts=Counter();self.frames=defaultdict(set);self.meta={};self.closed=False
        label=ACTIVE_RUN_LABEL or Path(sys.argv[sys.argv.index('--out')+1]).name
        self.directory=ROOT/'training_inputs'/label
        self.directory.mkdir(parents=True,exist_ok=False)
        self.stream=(self.directory/'consumed_samples.csv').open('w',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=['index','dataset','episode_id','original_frame_index','raw_sample_sha256'],lineterminator='\n');self.writer.writeheader()
        STREAMS.append(self);atexit.register(self.close)

    def as_numpy_iterator(self):
        metadata=json.loads((self.root/'complete.json').read_text())
        with (self.root/'sample_trace.csv').open() as f:
            trace=csv.DictReader(f)
            for i in range(metadata['samples']):
                row=next(trace);assert int(row['index'])==i
                with gzip.open(self.root/f'sample-{i:06d}.pkl.gz','rb') as stream:data=stream.read()
                assert hashlib.sha256(data).hexdigest()==row['raw_sample_sha256']
                sample=pickle.loads(data);name=sample['dataset_name'].decode();eid=sample['p3_episode_id'].decode();frame=int(sample['p3_frame_index'])
                assert (row['dataset'],row['episode_id'],int(row['original_frame_index']))==(name,eid,frame)
                self.writer.writerow(row);self.n+=1;self.counts[name]+=1;self.frames[(name,eid)].add(frame)
                self.meta[(name,eid)]=(sample['p3_file_path'].decode(),int(sample['p3_episode_length']))
                if self.n%128==0:self.progress()
                yield sample

    def progress(self):
        self.stream.flush()
        result=dict(samples_consumed=self.n,dataset_counts=dict(self.counts),unique_target_demonstrations=sum(k[0]=='panda_pnp' for k in self.frames),unique_source_demonstrations=sum(k[0]=='mg_pnp_lite' for k in self.frames),exact_order_and_raw_hashes_match_frozen_cache=True)
        p=self.directory/'input_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)

    def close(self):
        if self.closed:return
        self.progress();self.stream.close();self.closed=True
        rows=[]
        for (name,eid),frames in sorted(self.frames.items()):
            original_file,length=self.meta[(name,eid)]
            rows.append(dict(dataset=name,episode_id=eid,original_file=original_file,unique_consumed_frames=len(frames),original_frames=length))
        from p3_validation.prepare_data import write_csv
        write_csv(self.directory/'consumed_demonstration_coverage.csv',rows)
        p=self.directory/'input_status.json';d=json.loads(p.read_text());d.update(consumed_trace_sha256=sha256(self.directory/'consumed_samples.csv'),input_stream_closed=True);p.write_text(json.dumps(d,indent=2)+'\n')


def main():
    global ACTIVE_RUN_LABEL
    sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    import torch
    from safetensors.torch import save_file
    from peft import get_peft_model_state_dict
    import p2_validation.prepare_training as transport
    transport.CachedStream=AuditedStream
    out=Path(sys.argv[sys.argv.index('--out')+1]);telemetry=ROOT/'training_execution'/out.name
    ACTIVE_RUN_LABEL=out.name
    telemetry.mkdir(parents=True,exist_ok=False)
    state=dict(completed_backwards=0,finite_loss_checks=0,nonfinite_losses=0,actual_training_strategy='upstream_ddp_lora',runtime_source_sha256=sha256(Path(__file__)))
    ledger=telemetry/'execution.json';start=time.monotonic()
    def save():
        state['elapsed_seconds']=time.monotonic()-start
        p=ledger.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ledger)
    loss_log=(telemetry/'backward_losses.csv').open('w',newline='');writer=csv.writer(loss_log,lineterminator='\n');writer.writerow(['completed_backward','raw_loss'])
    backward=torch.Tensor.backward
    def checked_backward(tensor,*args,**kwargs):
        value=float(tensor.detach().float().item());state['last_loss']=value
        if not math.isfinite(value):
            state['nonfinite_losses']+=1;save();raise FloatingPointError('Nonfinite policy loss before backward')
        state['finite_loss_checks']+=1
        result=backward(tensor,*args,**kwargs)
        state['completed_backwards']+=1;writer.writerow([state['completed_backwards'],value])
        if state['completed_backwards']%100==0:loss_log.flush();save()
        return result
    torch.Tensor.backward=checked_backward
    from prismatic.training.strategies.lora_ddp import LoraDDPStrategy
    setup=LoraDDPStrategy.run_setup
    def measured_setup(strategy,*args,**kwargs):
        result=setup(strategy,*args,**kwargs)
        initial={k:v.detach().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(strategy.vlm.module).items()}
        destination=telemetry/'initial_adapter.safetensors';save_file(initial,str(destination))
        records=[]
        for key,tensor in initial.items():
            payload=tensor.view(torch.uint8).numpy().tobytes()
            records.append(dict(key=key,shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(payload).hexdigest()))
        (telemetry/'initial_parameter_hashes.json').write_text(json.dumps(records,indent=2)+'\n')
        state['initial_adapter_sha256']=sha256(destination);state['trainable_adapter_tensors']=len(initial);save()
        return result
    LoraDDPStrategy.run_setup=measured_setup
    try:
        from p2_validation.train_adapter import main as upstream_main
        upstream_main();state['launcher_returned_successfully']=True
    except BaseException as exc:
        state['launcher_returned_successfully']=False;state['error']=repr(exc);raise
    finally:
        for stream in STREAMS:stream.close()
        loss_log.close();save()


if __name__=='__main__':main()
