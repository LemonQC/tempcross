"""Merge only the prescribed final adapter; verify real parameter changes."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from p3_validation.common import ROOT,SOURCE,CHECKPOINT,CACHE
from p2_validation.offline_audit import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--steps',type=int,required=True);a=p.parse_args()
    adapter=Path(a.adapter);directory=ROOT/'adapted'
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(directory)
    sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    import torch
    torch.set_num_threads(4)
    from safetensors import safe_open
    from peft import PeftModel
    from prismatic.models import load_vla
    training_dir=adapter.parents[3];status=json.loads((training_dir/'training_status.json').read_text())
    assert status['status']=='COMPLETE' and status['completed_optimizer_steps']==a.steps and status['sample_positions']==96000
    execution=ROOT/'training_execution'/training_dir.name
    initial=execution/'initial_adapter.safetensors';final=adapter/'adapter_model.safetensors';changes=[]
    with safe_open(initial,framework='pt',device='cpu') as left,safe_open(final,framework='pt',device='cpu') as right:
        assert set(left.keys())==set(right.keys())
        for key in left.keys():
            x=left.get_tensor(key).float();y=right.get_tensor(key).float();assert torch.isfinite(y).all()
            delta=y-x;changes.append(dict(key=key,changed=not torch.equal(x,y),delta_l2=float(delta.norm()),max_abs_delta=float(delta.abs().max())))
    assert sum(x['changed'] for x in changes)>0
    directory.mkdir(parents=True);(directory/'parameter_changes.json').write_text(json.dumps(dict(total_adapter_tensors=len(changes),changed_tensors=sum(x['changed'] for x in changes),all_final_adapter_values_finite=True,initial_sha256=sha256(initial),final_sha256=sha256(final),tensors=changes),indent=2)+'\n')
    model=load_vla(CHECKPOINT,load_for_training=False);model=PeftModel.from_pretrained(model,str(adapter)).merge_and_unload()
    epoch=int(re.search(r'epoch-(\d+)',adapter.name).group(1));checkpoints=directory/'checkpoints';checkpoints.mkdir()
    destination=checkpoints/f'step-{a.steps:06d}-epoch-{epoch:02d}-loss=restricted96000.pt'
    torch.save({'model':{key:getattr(model,key).state_dict() for key in model.all_module_keys}},destination)
    shutil.copyfile(CHECKPOINT.parents[1]/'config.json',directory/'config.json');shutil.copyfile(CACHE/'dataset_statistics.json',directory/'dataset_statistics.json')
    provenance=dict(source_checkpoint=str(CHECKPOINT),source_checkpoint_sha256=sha256(CHECKPOINT),adapter=str(adapter),adapter_sha256=sha256(final),merged_checkpoint=str(destination),merged_checkpoint_sha256=sha256(destination),actual_updates=a.steps,sample_positions=96000,role='ENHANCED_BUDGET_RESTRICTED_LORA_NOT_FULL_STANDARD_FINETUNING')
    (directory/'adapter_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
