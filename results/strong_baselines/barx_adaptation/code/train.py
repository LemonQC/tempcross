"""Official BARX LoRA with measured accumulation and offline validation.

This is an engineering replacement for the upstream VLA loop, which asserts
gradient_accumulation_steps==1. Model, transforms, LoRA factory, AdamW, clipping,
LR and token losses are upstream. Accumulation represents eight virtual ranks
of 32 samples, each rank's token mean averaged over ranks, using microbatch 8.
No validation or physical-success feedback changes the prescribed run budget.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import csv
import gzip
import json
import math
import os
import pickle
import random
import time
import traceback
import numpy as np
from common import CHECKPOINT, DATA, OLD_STATS, OUT, ROOT, digest, read, setup_source, write


def forward(model, batch):
    import torch
    device=next(model.parameters()).device
    def move(value):
        return {k:move(v) for k,v in value.items()} if isinstance(value,dict) else value.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(input_ids=move(batch['input_ids']), attention_mask=move(batch['attention_mask']),
                     pixel_values=move(batch['pixel_values']), labels=move(batch['labels']))


def tokens(batch):
    # Multimodal insertion adds only IGNORE labels; upstream shifts by one.
    return int((batch['labels'][:, 1:] != -100).sum())


def validate(model, dataset, collator, step, tag, output_prefix=''):
    import torch
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    action = next(t for t, _ in dataset.batch_transforms if type(t) is RLDSBatchTransform)
    idx = read(OUT / f'{tag}_index.json')['frames']
    sums = Counter(); counts = Counter(); correct = Counter(); rows = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(idx), 8):
            entries = idx[start:start + 8]
            items = []
            for entry in entries:
                assert digest(entry['path']) == entry['sha256']
                items.append(action(pickle.loads(__import__('pathlib').Path(entry['path']).read_bytes())))
            batch = collator(items)
            output = forward(model, batch)
            patches = model.base_model.model.vision_backbone.num_patches if hasattr(model, 'base_model') else model.vision_backbone.num_patches
            logits = output.logits[:, patches:-1].float()
            labels = batch['labels'][:, 1:].to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits.transpose(1,2), labels, reduction='none', ignore_index=-100)
            for j, entry in enumerate(entries):
                mask = labels[j] != -100
                n = int(mask.sum()); ce = float(loss[j][mask].sum())
                hit = int(((logits[j].argmax(-1)==labels[j]) & mask).sum())
                task = entry['task']; sums[task] += ce; counts[task] += n; correct[task] += hit
                rows.append(dict(episode_id=entry['episode_id'], frame=entry['frame'], task=task,
                                 supervised_tokens=n, response_nll_sum=ce, correct_tokens=hit))
            del logits, loss, output
    if was_training:
        model.train()
    per_task = {t:dict(response_nll=sums[t]/counts[t], response_token_accuracy=correct[t]/counts[t], supervised_tokens=counts[t]) for t in sorted(counts)}
    result = dict(step=step, tag=tag, samples=len(idx), per_task=per_task,
                  macro_task_response_nll=sum(v['response_nll'] for v in per_task.values())/len(per_task),
                  metric='TEACHER_FORCED_ACTION_RESPONSE_INCLUDING_STOP_TOKENS_NOT_PHYSICAL_SUCCESS',
                  episode_disjoint_from_new_training=True, per_frame=rows)
    assert math.isfinite(result['macro_task_response_nll'])
    write(ROOT / 'validation' / f'{output_prefix}{tag}-step-{step:06d}.json', result)
    return {k:v for k,v in result.items() if k!='per_frame'}


def setup(config, probe=False):
    setup_source()
    import torch
    import torch.distributed as dist
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    dist.init_process_group('nccl')
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.random.set_seed(config['seed'])
    from prismatic.models import load_vla
    from prismatic.training import get_train_strategy
    from prismatic.vla.materialize import get_vla_dataset_and_collator
    if probe:
        from prismatic.vla.datasets.datasets import RLDSDataset
        from p2_validation.prepare_training import CachedStream
        cache = __import__('pathlib').Path('/home/users/lemon/coft-p3/matched-samples-96000')
        metadata = read(cache/'complete.json')
        RLDSDataset.make_dataset = lambda self,cfg:(CachedStream(cache),metadata['dataset_length'],read(OLD_STATS))
    else:
        from data import install
        install('train')
    model = load_vla(CHECKPOINT, load_for_training=True, image_sequence_len=1)
    model.freeze_backbones('vla-full-train')
    dataset, tokenizer, collator = get_vla_dataset_and_collator(
        DATA, 'panda_xp_900_pnp', image_transform=model.vision_backbone.get_image_transform(),
        tokenizer=model.llm_backbone.get_tokenizer(), prompt_builder_fn=model.llm_backbone.prompt_builder_fn,
        default_image_resolution=model.vision_backbone.default_image_resolution, shuffle_buffer_size=10000,
        transform_types='joint_reps', image_aug=False, action_tokenizer='xp_900_pnp_action_tokenizer',
        future_obj_pose_window_size=40, future_2D_trace_window_size=40, obj_pose_stride=8,
        ee_pose_2D_stride=8, image_window_size=1, use_wrist_image=False,
        dataset_statistics_map={'panda_pnp':'mg_pnp_lite'})
    strategy = get_train_strategy(
        train_strategy='ddp-lora', vlm=model, device_id=0, stage='vla-full-train', epochs=1000,
        max_steps=config['optimizer_updates'], global_batch_size=256, per_device_batch_size=8,
        learning_rate=2e-5, weight_decay=0., max_grad_norm=1., lr_scheduler_type='constant', warmup_ratio=0.,
        enable_gradient_checkpointing=False, enable_mixed_precision_training=True)
    directory = ROOT / ('accumulation-probe' if probe else 'training')
    directory.mkdir(parents=True, exist_ok=False)
    strategy.run_setup(directory, len(dataset))
    return strategy, dataset, collator, directory


def optimizer_update(strategy, items, collator):
    """One 256-sample update, matching mean of eight 32-sample token means."""
    import torch
    assert len(items)==256
    totals = []; micro = 0
    strategy.optimizer.zero_grad(set_to_none=True)
    for rank_start in range(0,256,32):
        # Count after the SAME collator truncation used for forward.
        batches = [collator(items[i:i+8]) for i in range(rank_start,rank_start+32,8)]
        denominator = sum(tokens(b) for b in batches)
        assert denominator>0
        for batch in batches:
            n = tokens(batch)
            # world_size=1: avoid repeated unused-parameter DDP bookkeeping.
            with strategy.vlm.no_sync():
                output = forward(strategy.vlm,batch)
                value = float(output.loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f'Nonfinite loss before backward: {value}')
                weight = n/denominator/8
                (output.loss*weight).backward()
                totals.append(value*weight);micro+=1
            del output
    norm = strategy.clip_grad_norm()
    finite=[torch.isfinite(p.grad).all() for p in strategy.vlm.parameters() if p.grad is not None]
    if not bool(torch.stack(finite).all()):
        raise FloatingPointError('Nonfinite gradient before optimizer')
    strategy.optimizer.step();strategy.lr_scheduler.step()
    strategy.optimizer.zero_grad(set_to_none=True)
    return sum(totals),micro


def probe(config):
    """Real full-batch/microbatch gradient check and a discarded 256-sample update."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    strategy, dataset, collator, directory = setup(config,True)
    raw = iter(dataset.dataset.as_numpy_iterator())
    rng = np.random.default_rng(config['seed'])
    transforms,weights=zip(*dataset.batch_transforms)
    items=[transforms[int(rng.choice(len(transforms),p=weights))](next(raw)) for _ in range(256)]
    model=strategy.vlm.module;model.eval()
    initial={k:v.detach().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(model).items()}
    save_file(initial,str(directory/'initial_adapter.safetensors'))
    strategy.optimizer.zero_grad(set_to_none=True)
    batch=collator(items[:8]);n=tokens(batch)
    output=forward(model,batch);full_loss=float(output.loss.detach());output.loss.backward();del output
    full={k:p.grad.detach().float().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
    strategy.optimizer.zero_grad(set_to_none=True)
    weighted_loss=0.
    for start in range(0,8,2):
        b=collator(items[start:start+2]);out=forward(model,b);weight=tokens(b)/n
        weighted_loss+=float(out.loss.detach())*weight;(out.loss*weight).backward();del out
    delta=reference=0.
    for k,p in model.named_parameters():
        if k in full:
            d=p.grad.detach().float().cpu()-full[k]
            delta+=float(d.square().sum());reference+=float(full[k].square().sum())
    relative=math.sqrt(delta/max(reference,1e-30))
    # BF16 kernels are not bitwise batch-size invariant; fixed a priori tolerance.
    assert relative<.10 and abs(full_loss-weighted_loss)/max(abs(full_loss),1e-8)<.02,(relative,full_loss,weighted_loss)
    strategy.optimizer.zero_grad(set_to_none=True);model.train();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();loss,micro=optimizer_update(strategy,items,collator);torch.cuda.synchronize()
    seconds=time.monotonic()-start
    final=get_peft_model_state_dict(model)
    changed=sum(not torch.equal(initial[k],v.detach().cpu()) for k,v in final.items())
    assert changed>0
    result=dict(status='COMPLETE',role='DISCARDED_RESOURCE_AND_ACCUMULATION_PROBE_NOT_FORMAL_TRAINING',
                full_batch8_loss=full_loss,weighted_four_batch2_loss=weighted_loss,relative_gradient_l2_error=relative,
                gradient_relative_tolerance=.10,loss_relative_tolerance=.02,gradient_check_mode='eval_to_disable_stochasticity_gradients_enabled',
                actual_accumulated_optimizer_updates=1,actual_microbatch_backwards=micro,sample_positions=256,
                changed_adapter_tensors=changed,update_seconds=seconds,estimated_3000_update_hours=seconds*3000/3600,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                loss=loss,finite_gradients=True)
    write(directory/'status.json',result);write(OUT/'accumulation_probe.json',result)
    print(json.dumps(result,indent=2),flush=True)
    torch.distributed.destroy_process_group()


def train(config):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    marker=read(ROOT/'frozen_publication.json')
    assert marker['github_readback_verified'] and marker['protocol_sha256']==digest(OUT/'protocol.json')
    for path,h in {**marker['code_sha256'],**marker['data_sha256']}.items():
        assert digest(path)==h,'Frozen code changed: '+path
    strategy,dataset,collator,directory=setup(config)
    model=strategy.vlm.module
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(model).items()},str(directory/'initial_adapter.safetensors'))
    write(directory/'dataset_statistics.json',dataset.dataset_statistics)
    # JSON cannot encode NumPy arrays: write a safe serializable copy explicitly.
    state=dict(status='RUNNING',completed_optimizer_updates=0,completed_backwards=0,sample_positions=0,
               prescribed_updates=config['optimizer_updates'],effective_batch=256,microbatch=8,
               reference_loss_groups='8 virtual ranks x 32 examples',seed=config['seed'],
               original_checkpoint=str(CHECKPOINT),original_checkpoint_sha256=digest(CHECKPOINT),
               frozen_commit=marker['remote_commit'],best_validation=None,technical_failures_not_task_failures=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def status():
        state.update(elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        write(directory/'status.json',state)
    stream=(directory/'consumed_samples.csv').open('x',newline='')
    from data import trace_iterator
    inputs=iter(trace_iterator(dataset,np.random.default_rng(config['seed']),stream))
    curve=[]
    try:
        # Reference only; step zero cannot win trained-checkpoint selection.
        for tag in ['validation_target','validation_source_ur5e']:
            curve.append(validate(model,dataset,collator,0,tag))
        write(OUT/'validation_curve.json',curve);status();model.train()
        with (directory/'updates.csv').open('x',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['update','sample_positions','microbatch_backwards','loss','lr','elapsed_seconds'])
            for step in range(1,config['optimizer_updates']+1):
                if time.monotonic()-start>config['maximum_training_wall_hours']*3600:
                    raise TimeoutError('Frozen maximum training wall time reached; do not silently reduce budget')
                if __import__('shutil').disk_usage(ROOT).free<config['minimum_free_disk_gib']*1024**3:
                    raise OSError('Frozen minimum free disk floor reached')
                items=[next(inputs) for _ in range(256)]
                loss,micro=optimizer_update(strategy,items,collator)
                state.update(completed_optimizer_updates=step,completed_backwards=step*micro,sample_positions=step*256,last_loss=loss)
                writer.writerow([step,step*256,micro,loss,strategy.optimizer.param_groups[0]['lr'],time.monotonic()-start])
                f.flush();status()
                if step%10==0:
                    print(json.dumps({k:state[k] for k in ['completed_optimizer_updates','sample_positions','last_loss','elapsed_seconds']}),flush=True)
                if step in config['validation_steps']:
                    checkpoint=directory/'checkpoints'/f'step-{step:06d}'
                    model.save_pretrained(checkpoint)
                    torch.save(dict(optimizer=strategy.optimizer.state_dict(),scheduler=strategy.lr_scheduler.state_dict(),
                                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=step),checkpoint/'optimizer.pt')
                    for tag in ['validation_target','validation_source_ur5e']:
                        result=validate(model,dataset,collator,step,tag);curve.append(result)
                        if tag=='validation_target':
                            candidate=dict(step=step,macro_task_response_nll=result['macro_task_response_nll'],adapter=str(checkpoint))
                            if state['best_validation'] is None or (candidate['macro_task_response_nll'],step)<(state['best_validation']['macro_task_response_nll'],state['best_validation']['step']):
                                state['best_validation']=candidate
                    write(OUT/'validation_curve.json',curve);status()
        state['status']='COMPLETE';status()
        write(OUT/'selection.json',dict(**state['best_validation'],selection_rule=config['checkpoint_selection'],
                                       used_physical_success=False,training_completed_before_test=True,
                                       actual_optimizer_updates=state['completed_optimizer_updates'],actual_sample_positions=state['sample_positions']))
    except BaseException as exc:
        state.update(status='TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc(),is_cuda_oom=isinstance(exc,torch.cuda.OutOfMemoryError));status();raise
    finally:
        stream.close()
        write(OUT/'training_status.json',state)
        torch.distributed.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');args=p.parse_args()
    config=read(OUT/'protocol.json')
    if args.probe:probe(config)
    else:train(config)


if __name__=='__main__':
    main()
