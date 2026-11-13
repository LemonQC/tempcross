"""Native 50-step queue parity and processor roundtrip; no physics/updates."""
import numpy as np
import torch
from .runtime import RUN,load,prepare_batch
from .data import Dataset
from baselines.strong.common import out,read,write,digest

def main(checkpoint):
    torch.set_num_threads(4);policy,(pre,post)=load(checkpoint)
    data=Dataset();j=next(i for i,r in enumerate(data.rows) if r['domain']=='target' and r['split']=='train')
    data.anchors=np.array([[j,0]],np.int32);raw,traces=data.batch(0,1)
    originals=raw['action'].clone();batch=prepare_batch(pre,raw)
    torch.testing.assert_close(post(batch['action']),originals,rtol=1e-6,atol=1e-6)
    batch.pop('action');batch.pop('actions_id_pad')
    calls=[];original=policy.model.sample_actions
    def traced(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(result.detach().cpu());return result
    policy.model.sample_actions=traced  # observational wrapper; outputs unchanged
    noise=torch.randn((1,50,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(7))
    policy.reset();expected=policy.predict_action_chunk(batch,noise=noise)
    policy.reset();actions=[]
    for i in range(50):actions.append(policy.select_action(batch,noise=noise))
    actual=torch.stack(actions,dim=1)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    assert len(calls)==2 and len(policy._queues['action'])==0
    assert all(x.shape==(1,50,32) for x in calls)
    torch.testing.assert_close(expected.cpu(),calls[0][...,:7],rtol=0,atol=0)
    result=dict(status='PASS_NATIVE_QUEUE_AND_NORMALIZATION',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        native_generation_calls=2,selected_actions=50,native_chunk_size=50,denoising_steps=10,
        raw_action_dimensions=32,used_action_dimensions=7,native_queue_exactly_matches_full_chunk=True,
        action_normalization_roundtrip=True,native_padding_and_crop_preserved=True,
        original_training_anchor=traces[0],physics_step_calls=0,new_policy_episodes=0,new_training_updates=0)
    write(RUN/'queue_audit.json',result);write(out('smolvla')/'queue_audit.json',result)
    data.close();print('NATIVE_QUEUE50_RAW32_TO7_NORMALIZATION_PASS',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);args=p.parse_args();main(args.checkpoint)
