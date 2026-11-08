"""Execute official BARX adaptation with measured memory and matched inputs.

Only runtime/data transport hooks are external: single-rank FSDP NO_SHARD,
the fixed official-mixture cache, optimizer-step telemetry, and optional upstream
ddp-lora strategy. Original BARX source and learning objectives remain intact.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--method',choices=['no_reps','joint_reps'],required=True)
    p.add_argument('--route',choices=['full','lora'],required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--batch-size',type=int,required=True);a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.out).resolve();cache=Path(a.cache).resolve()
    if out.exists() and any(out.iterdir()):
        # A supervisor may reach a job already completed by an independent
        # worker. Reuse only an exact, fully saved prescribed training run;
        # never restart updates or silently accept an incomplete directory.
        status_path=out/'training_status.json'
        previous=json.loads(status_path.read_text()) if status_path.exists() else {}
        metadata=json.loads((cache/'complete.json').read_text())
        expected=dict(status='COMPLETE',method=a.method,route=a.route,prescribed_steps=a.steps,
                      batch_size=a.batch_size,completed_optimizer_steps=a.steps,sample_positions=a.steps*a.batch_size,
                      original_checkpoint=a.checkpoint,sample_trace_sha256=metadata['sample_trace_sha256'])
        if all(previous.get(key)==value for key,value in expected.items()):
            saved=list((out/'runs').rglob(f'step-{a.steps:06d}-*'))
            if a.route=='lora':saved=[q for q in saved if q.is_dir() and (q/'adapter_config.json').exists()
                                     and (q/'adapter_model.safetensors').exists() and (q/'optimizer.pt').exists()]
            else:saved=[q for q in saved if q.suffix=='.pt']
            if len(saved)==1:
                print('Reusing exact completed training job:',out,flush=True);return
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);sys.path[:0]=[str(source),str(source/'policy')]
    import torch
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.set_visible_devices([],'GPU');tf.random.set_seed(7)
    from prismatic.vla.datasets.datasets import RLDSDataset
    from p2_validation.prepare_training import CachedStream
    metadata=json.loads((cache/'complete.json').read_text());assert a.steps*a.batch_size<=metadata['samples']
    statistics=json.loads((cache/'dataset_statistics.json').read_text())
    def fixed_data(self,config):
        assert self.data_mix=='panda_xp_900_pnp'
        assert config['traj_transform_kwargs']['future_action_window_size']==7
        return CachedStream(cache),metadata['dataset_length'],statistics
    RLDSDataset.make_dataset=fixed_data
    from prismatic.training.strategies.fsdp import FSDPStrategy
    original_init=FSDPStrategy.__init__
    def single_rank(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.fsdp_sharding_strategy=torch.distributed.fsdp.ShardingStrategy.NO_SHARD
    FSDPStrategy.__init__=single_rank
    status=dict(status='RUNNING',method=a.method,route=a.route,prescribed_steps=a.steps,batch_size=a.batch_size,
       completed_optimizer_steps=0,sample_positions=0,sample_trace_sha256=metadata['sample_trace_sha256'],
       dataset_counts_full_sequence=metadata['dataset_counts'],gpu=torch.cuda.get_device_name(0),
       original_checkpoint=a.checkpoint,official_source_unmodified=True,
       equivalence='RESTRICTED_SINGLE_GPU_NOT_OFFICIAL_GLOBAL_BATCH_256',single_rank_fsdp='NO_SHARD')
    ledger=out/'training_status.json';start=time.monotonic()
    def save():
        status.update(elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        ledger.write_text(json.dumps(status,indent=2)+'\n')
    torch.cuda.reset_peak_memory_stats();save()
    original_step=torch.optim.AdamW.step
    def measured_step(optimizer,*args,**kwargs):
        result=original_step(optimizer,*args,**kwargs);torch.cuda.synchronize()
        status['completed_optimizer_steps']+=1;status['sample_positions']+=a.batch_size;save()
        return result
    torch.optim.AdamW.step=measured_step
    sys.argv=[str(source/'policy/vla-scripts/train.py'),'--vla.type','prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full',
       '--vla.data_mix','panda_xp_900_pnp','--vla.action_tokenizer','xp_900_pnp_action_tokenizer',
       '--vla.expected_world_size','1','--vla.global_batch_size',str(a.batch_size),'--vla.per_device_batch_size',str(a.batch_size),
       '--vla.learning_rate','2e-5','--vla.lr_scheduler_type','constant','--vla.max_steps',str(a.steps),
       '--vla.use_wrist_image','False','--vla.image_sequence_len','1','--vla.transform_types',a.method,
       '--vla.train_strategy','ddp-lora' if a.route=='lora' else 'fsdp-full-shard',
       '--vla.shuffle_buffer_size','10000','--data_root_dir',str(cache.parent/'data'),
       '--run_root_dir',str(out/'runs'),'--run_id',a.method,'--run_id_note',a.route,
       '--pretrained_checkpoint',str(Path(a.checkpoint).resolve()),'--is_resume','False',
       '--dataset_statistics_map','{"panda_pnp":"mg_pnp_lite"}','--image_aug','False','--save_interval',str(a.steps)]
    os.chdir(source/'policy')
    try:
        runpy.run_path(sys.argv[0],run_name='__main__');status['status']='COMPLETE'
    except BaseException as error:
        status.update(status='FAILED',error=repr(error),is_cuda_oom=isinstance(error,torch.cuda.OutOfMemoryError));raise
    finally:save()


if __name__=='__main__':main()
