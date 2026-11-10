"""Merge only the validation-selected trained adapter into its original prior."""
import shutil
from common import CHECKPOINT, OUT, ROOT, digest, read, setup_source, write


def main():
    setup_source()
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from prismatic.models import load_vla
    torch.set_num_threads(4)
    selected=read(OUT/'selection.json');state=read(ROOT/'training/status.json')
    assert state['status']=='COMPLETE' and state['completed_optimizer_updates']==3000 and state['sample_positions']==768000
    assert selected['used_physical_success'] is False
    adapter=__import__('pathlib').Path(selected['adapter']);initial=ROOT/'training/initial_adapter.safetensors'
    final=adapter/'adapter_model.safetensors';changes=[]
    with safe_open(initial,framework='pt',device='cpu') as a,safe_open(final,framework='pt',device='cpu') as b:
        assert set(a.keys())==set(b.keys())
        for k in a.keys():
            x=a.get_tensor(k).float();y=b.get_tensor(k).float();assert torch.isfinite(y).all()
            changes.append(dict(key=k,changed=not torch.equal(x,y),delta_l2=float((y-x).norm())))
    assert any(r['changed'] for r in changes)
    write(OUT/'parameter_changes.json',dict(selected_step=selected['step'],adapter_tensors=len(changes),
          changed_tensors=sum(r['changed'] for r in changes),all_final_values_finite=True,tensors=changes))
    model=load_vla(CHECKPOINT,load_for_training=False)
    model=PeftModel.from_pretrained(model,str(adapter)).merge_and_unload()
    dest=ROOT/'adapted';dest.mkdir(exist_ok=False);(dest/'checkpoints').mkdir()
    checkpoint=dest/'checkpoints'/f"step-{selected['step']:06d}-epoch-00-loss=validationselected.pt"
    torch.save({'model':{k:getattr(model,k).state_dict() for k in model.all_module_keys}},checkpoint)
    # The original configuration stays on the server; do not publish credential fields.
    shutil.copyfile(CHECKPOINT.parents[1]/'config.json',dest/'config.json')
    shutil.copyfile(ROOT/'training/dataset_statistics.json',dest/'dataset_statistics.json')
    write(OUT/'model_index.json',dict(source_checkpoint=str(CHECKPOINT),source_sha256=digest(CHECKPOINT),
         adapter=str(adapter),adapter_sha256=digest(final),merged_checkpoint=str(checkpoint),
         merged_sha256=digest(checkpoint),merged_bytes=checkpoint.stat().st_size,selected_step=selected['step'],
         actual_training_optimizer_updates=3000,sample_positions=768000,selection_used_physical_success=False))


if __name__=='__main__':main()
