"""Unmodified official Octo module/loss/optimizer in an isolated JAX runtime."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import sys
import time
import argparse
import hashlib
from pathlib import Path
sys.path.insert(0,'/home/users/lemon/baseline-benchmark/vendor/octo')
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import jax
import jax.numpy as jnp
import flax
import optax
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import TrainState,create_optimizer
from scripts.configs.finetune_config import get_config
from baselines.strong.common import ROOT,read,write,resource,digest,now
from .data import Dataset,DIRECTORY,decode_action

RUN=ROOT/'octo_small'
os.environ.setdefault('HF_HOME',str(RUN/'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

def load_model(checkpoint=None):
    assert jax.default_backend()=='gpu', 'CPU fallback is not an RTX3090 resource probe'
    print('JAX_DEVICES',jax.devices(),flush=True)
    if checkpoint is None:
        model=OctoModel.load_pretrained(str(RUN/'pretrained'),step=300000)
    else:
        p=Path(checkpoint)
        model=OctoModel.load_pretrained(str(p.parents[2]),step=int(p.parent.parent.name))
    stats={k:np.array(v) for k,v in read(DIRECTORY/'action_statistics.json').items()}
    return model.replace(dataset_statistics={'robocasa_x_train':{'action':stats}})

def optimizer(model,num_steps,probe=False):
    kwargs=get_config('full,language_conditioned').optimizer.to_dict()
    kwargs['frozen_keys']=model.config['optimizer']['frozen_keys']
    if probe:
        # Keep the official schedule-dict code path. Its constant-scalar branch
        # closes over a subsequently overwritten kwargs entry in this revision.
        kwargs['learning_rate']=dict(name='cosine',init_value=3e-5,peak_value=3e-5,
            warmup_steps=1,decay_steps=100,end_value=3e-5)
    else:kwargs['learning_rate']['decay_steps']=num_steps
    tx,lr,_=create_optimizer(model.params,**kwargs)
    return tx,lr,kwargs

def functions(model,lr):
    # Exact official scripts/finetune.py learning path. No auxiliary head,
    # language feature replacement, low-rank layer or controller heuristic.
    def loss_fn(params,batch,rng,train=True):
        bound=model.module.bind({'params':params},rngs={'dropout':rng})
        embeddings=bound.octo_transformer(batch['observation'],batch['task'],
            batch['observation']['timestep_pad_mask'],train=train)
        return bound.heads['action'].loss(embeddings,batch['action'],
            batch['observation']['timestep_pad_mask'],batch['action_pad_mask'],train=train)
    @jax.jit
    def update(state,batch):
        rng,dropout_rng=jax.random.split(state.rng)
        (loss,info),grads=jax.value_and_grad(loss_fn,has_aux=True)(state.model.params,batch,dropout_rng,True)
        state=state.apply_gradients(grads=grads,rng=rng)
        return state,dict(info,loss=loss,grad_norm=optax.global_norm(grads),learning_rate=lr(state.step-1))
    @jax.jit
    def validate(params,batch,rng):return loss_fn(params,batch,rng,False)
    return update,validate

def flat_hashes(params):
    flat=flax.traverse_util.flatten_dict(params)
    return {'.'.join(k):dict(sha256=hashlib.sha256(np.asarray(v).tobytes()).hexdigest(),
        parameters=int(np.size(v))) for k,v in flat.items()}

def host_metrics(metrics):return {k:float(np.asarray(v)) for k,v in jax.device_get(metrics).items()}

def serve(socket,checkpoint,ledger):
    from multiprocessing.connection import Listener
    from .data import resize_image
    model=load_model(checkpoint)
    stats=read(DIRECTORY/'action_statistics.json');rng=jax.random.PRNGKey(7)
    metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),devices=[str(x) for x in jax.devices()],
        input_fields=['RGB256','instruction'],history=2,horizon=4,jax=jax.__version__,
        native_source='241fb3514b7c40957a86d869fecb7c7fc353f540',raw_action_ledger=str(ledger))
    Path(socket).unlink(missing_ok=True)
    with Listener(socket,family='AF_UNIX') as listener,Path(ledger).open('a') as stream:
        with listener.accept() as connection:
            connection.send(metadata)
            while True:
                try:request=connection.recv()
                except EOFError:break  # evaluator process exited after its completed job
                if request['command']=='close':break
                if request['command']=='reset':
                    # Original evaluator seeds every episode alike. Reset both
                    # observation history and diffusion RNG, with no cross-episode state.
                    rng=jax.random.PRNGKey(7);connection.send({'status':'RESET'});continue
                images=np.stack([resize_image(x) for x in request['images']])[None]
                observation=dict(image_primary=images,image_wrist=np.zeros((1,2,128,128,3),np.uint8),
                    timestep_pad_mask=np.array([request['valid']],bool),
                    pad_mask_dict=dict(image_primary=np.ones((1,2),bool),image_wrist=np.zeros((1,2),bool)))
                task=model.create_tasks(texts=[request['instruction']]);rng,key=jax.random.split(rng)
                normalized=np.asarray(model.sample_actions(observation,task,rng=key))[0]
                assert normalized.shape==(4,7) and np.isfinite(normalized).all()
                physical=np.where(np.array(stats['mask'],bool),normalized*np.array(stats['std'])+np.array(stats['mean']),normalized).astype(np.float32)
                canonical=decode_action(physical)
                record=dict(utc=now(),episode=request['episode'],step=request['step'],instruction=request['instruction'],
                    normalized=normalized.tolist(),native_physical=physical.tolist(),canonical_osc=canonical.tolist(),
                    diffusion_key=np.asarray(key).tolist(),history_valid=request['valid'])
                stream.write(__import__('json').dumps(record)+'\n');stream.flush()
                connection.send(dict(canonical=canonical))

def probe(batch_size):
    directory=RUN/'probe'/f'batch-{batch_size}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists() and read(directory/'result.json')['status']=='PASS':
        print('EXISTING_PASS_NO_REPEAT',flush=True);return
    report=dict(status='RUNNING',batch_size=batch_size,actual_training_updates=0,start=resource(),utc=now())
    write(directory/'result.json',report)
    data=None
    try:
        model=load_model();before=flat_hashes(model.params)
        data=Dataset();t=time.monotonic();batch,traces=data.batch(0,batch_size,model,augment=True)
        assert batch['action'].shape==(batch_size,2,4,7)
        print('BATCH_READY',time.monotonic()-t,traces,flush=True)
        tx,lr,kwargs=optimizer(model,100,probe=True)
        state=TrainState.create(rng=jax.random.PRNGKey(20261113),model=model,tx=tx)
        update,_=functions(model,lr);durations=[];history=[]
        for i in range(3):
            t=time.monotonic();state,metrics=update(state,batch);metrics=host_metrics(metrics)
            durations.append(time.monotonic()-t);assert all(np.isfinite(v) for v in metrics.values())
            history.append(metrics);report['actual_training_updates']=int(state.step)
            write(directory/'result.json',report)
            print('REAL_UPDATE',i+1,metrics,'seconds',durations[-1],flush=True)
        after=flat_hashes(state.model.params)
        changes=[k for k in before if before[k]['sha256']!=after[k]['sha256']]
        frozen=[k for k in before if 'hf_model' in k]
        assert changes and frozen and not set(changes).intersection(frozen)
        assert any('transformer' in k for k in changes) and any('heads_action' in k for k in changes)
        state.model.save_pretrained(int(state.step),str(directory/'native'))
        checkpoint=directory/'native'/str(int(state.step))/'default/checkpoint'
        assert checkpoint.exists()
        write(directory/'parameter_audit.json',dict(before=before,after=after,changed=changes,frozen=frozen))
        report.update(status='PASS',actual_training_updates=int(state.step),durations_seconds=durations,metrics=history,
            warm_update_seconds=float(np.mean(durations[1:])),parameters_total=sum(v['parameters'] for v in before.values()),
            parameters_language_frozen=sum(before[k]['parameters'] for k in frozen),changed_tensors=len(changes),
            frozen_language_changed_tensors=0,optimizer=kwargs,training_traces=traces,
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),end=resource())
    except BaseException as error:
        import traceback
        report.update(status='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc(),end=resource())
        raise
    finally:
        write(directory/'result.json',report)
        if data is not None:data.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe-batch',type=int)
    parser.add_argument('--serve');parser.add_argument('--checkpoint');parser.add_argument('--ledger')
    args=parser.parse_args()
    if args.serve:serve(args.serve,args.checkpoint,args.ledger)
    else:
        assert args.probe_batch is not None;probe(args.probe_batch)
