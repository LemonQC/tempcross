"""Native SmolVLA select_action queue over a byte/list-only local RPC."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np
import torch
from .runtime import RUN,load
from baselines.strong.common import digest,now

def main(socket,checkpoint,ledger,role):
    torch.set_num_threads(4);policy,(pre,post)=load(checkpoint);policy.eval()
    capture={};native=policy.model.sample_actions
    def traced(*args,**kwargs):
        actions=native(*args,**kwargs);capture['raw']=actions.detach().cpu().numpy();return actions
    policy.model.sample_actions=traced  # observation-only, unchanged native return value
    meta=dict(model='official SmolVLAPolicy.select_action',official_source='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),information=['RGB180x320->native512','instruction','arm_qpos7','Robotiq_qpos2'],
        action_generation='native flow matching,10 steps,chunk50,execute50',raw_action_ledger=str(ledger),role=role,
        torch=torch.__version__,device=torch.cuda.get_device_name(),state_fields=9,extra_real_cameras=0)
    Path(socket).unlink(missing_ok=True)
    with Listener(socket,family='AF_UNIX') as listener,Path(ledger).open('a') as stream:
        with listener.accept() as conn:
            conn.send(meta)
            while True:
                try:r=conn.recv()
                except EOFError:break
                if r['command']=='close':break
                if r['command']=='reset':
                    policy.reset();torch.manual_seed(7);conn.send({'status':'RESET'});continue
                rgb=np.frombuffer(r['rgb'],np.uint8).reshape(r['shape'])
                raw={'observation.images.camera1':torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255,
                    'observation.state':torch.tensor([r['state']],dtype=torch.float32),'task':[r['instruction']]}
                batch=pre(raw);new=len(policy._queues['action'])==0
                with torch.inference_mode():normalized=policy.select_action(batch)
                canonical=post(normalized).numpy()[0]
                assert canonical.shape==(7,) and np.isfinite(canonical).all()
                result=dict(action=canonical.tolist(),new_query=new)
                if new:
                    native32=capture.pop('raw');assert native32.shape==(1,50,32)
                    used=torch.from_numpy(native32[...,:7].copy()).to('cuda')
                    chunk=post(used).numpy()[0]
                    np.testing.assert_array_equal(chunk[0],canonical)
                    record=dict(utc=now(),episode=r['episode'],step=r['step'],role=role,
                        instruction=r['instruction'],robot_state=r['state'],normalized32=native32[0].tolist(),
                        normalized7=native32[0,:,:7].tolist(),canonical_chunk=chunk.tolist())
                    stream.write(json.dumps(record)+'\n');stream.flush();result['native_query']=chunk.tolist()
                conn.send(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--ledger',required=True);p.add_argument('--role',required=True);a=p.parse_args();main(a.socket,a.checkpoint,a.ledger,a.role)
