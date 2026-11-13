"""Verify native saved optimizer counters and frozen VLM without a GPU call."""
import hashlib
import torch
from baselines.strong.common import ROOT,out,read,write,digest,now

def main():
    run=ROOT/'smolvla';path=run/'training/source_target_state/resume.pt'
    state=torch.load(path,map_location='cpu',weights_only=False);step=state['step']
    counts=sorted(set(int(s['step'].item()) for s in state['optimizer']['state'].values() if 'step' in s))
    assert counts==[step] and state['scheduler']['last_epoch']==step
    before=read(run/'probe/batch-32-padmask-correction/parameter_audit.json')['before'];changed=[];frozen=[]
    for name,record in before.items():
        value=state['model'][name].detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        different=hashlib.sha256(value).hexdigest()!=record['sha256']
        if record['trainable']:
            if different:changed.append(name)
        else:
            frozen.append(name);assert not different,name
    assert changed and frozen
    result=dict(status='PASS',saved_update=step,optimizer_counters=counts,scheduler_last_epoch=state['scheduler']['last_epoch'],
        changed_trainable_tensors=len(changed),frozen_tensors_verified=len(frozen),frozen_tensors_changed=0,
        resume_path=str(path),resume_sha256=digest(path),cpu_only=True,new_training_updates=0,new_physics_steps=0,utc=now())
    write(run/'optimizer_audit.json',result);write(out('smolvla')/'optimizer_audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
