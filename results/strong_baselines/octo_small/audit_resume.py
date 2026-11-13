"""Read saved native optimizer counters/parameter hashes without GPU inference."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import hashlib
import numpy as np
import flax.serialization
import flax.traverse_util
from baselines.strong.common import ROOT,out,read,write,digest,now

def counters(tree,path=()):
    if isinstance(tree,dict):
        for key,value in tree.items():
            if key=='count':yield dict(path='.'.join((*path,key)),value=int(value))
            else:yield from counters(value,(*path,str(key)))
    elif isinstance(tree,(list,tuple)):
        for i,value in enumerate(tree):yield from counters(value,(*path,str(i)))

def main():
    directory=ROOT/'octo_small/training/source_target';snapshot=directory/'audit_snapshot.msgpack'
    snapshot.unlink(missing_ok=True);os.link(directory/'resume.msgpack',snapshot)
    try:
        sha=digest(snapshot);raw=flax.serialization.msgpack_restore(snapshot.read_bytes());step=int(raw['step'])
        counts=list(counters(raw['opt_state']));assert counts and all(c['value']==step for c in counts)
        baseline=read(ROOT/'octo_small/probe/batch-8/parameter_audit.json')['before']
        params=flax.traverse_util.flatten_dict(raw['model']['params'],sep='.')
        current={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in params.items()}
        assert set(current)==set(baseline)
        frozen=[k for k in current if 'hf_model' in k]
        changed=[k for k in current if current[k]!=baseline[k]['sha256']]
        assert frozen and changed and not set(frozen)&set(changed)
        result=dict(status='PASS',saved_update=step,optimizer_counters=counts,changed_parameter_tensors=len(changed),
            frozen_t5_changed_tensors=0,resume_state_sha256=sha,rng=np.asarray(raw['rng']).tolist(),
            whole_pretrained_parameter_tree_retained=True,no_gpu_forward_or_training=True,utc=now())
        write(ROOT/'octo_small/optimizer_resume_audit.json',result);write(out('octo_small')/'optimizer_resume_audit.json',result)
        print(result,flush=True)
    finally:snapshot.unlink(missing_ok=True)

if __name__=='__main__':main()
