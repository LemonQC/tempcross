"""Verify native saved Adam/scheduler/EMA counters and real learned tensors."""
import hashlib
import torch
from .data import RUN
from baselines.strong.common import out,read,write,digest,now

def main():
    path=RUN/'training/source_target_visual/resume.pt';s=torch.load(path,map_location='cpu',weights_only=False)
    counts=sorted(set(int(v['step'].item()) for v in s['optimizer']['state'].values() if 'step' in v))
    assert counts==[s['update']] and s['scheduler']['last_epoch']==s['update'] and s['ema']['optimization_step']==s['update']
    initial=read(RUN/'training/source_target_visual/initial_parameter_audit.json')['parameters'];changed=[]
    for name,r in initial.items():
        value=s['model'][name].detach().contiguous().numpy().tobytes()
        if hashlib.sha256(value).hexdigest()!=r['sha256']:changed.append(name)
    assert any(n.startswith('vision_encoder.') for n in changed) and any(n.startswith('noise_pred_net.') for n in changed)
    report=dict(status='PASS',saved_update=s['update'],optimizer_step_counters=counts,
        scheduler_last_epoch=s['scheduler']['last_epoch'],EMA_optimization_step=s['ema']['optimization_step'],
        changed_trainable_tensors=len(changed),initialization_exactly_pre_probe=True,
        resume_path=str(path),resume_sha256=digest(path),new_optimizer_updates=0,new_physics_steps=0,cpu_only=True,utc=now())
    write(RUN/'optimizer_audit.json',report);write(out('flow_matching_policy')/'optimizer_audit.json',report);print(report,flush=True)

if __name__=='__main__':main()
