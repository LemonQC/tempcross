"""Offline descriptive comparisons; never retrain or rerun old physical tests."""
from types import SimpleNamespace
import gc
from common import CHECKPOINT, OUT, ROOT, digest, read, setup_source, write


def main():
    setup_source()
    import torch
    torch.set_num_threads(4)
    from prismatic.models import load_vla
    from prismatic.vla.datasets.datasets import RLDSBatchTransform
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from train import validate
    paths={
        'original_joint': CHECKPOINT,
        'restricted_joint_6000': '/home/users/lemon/coft-p2/p2/physical-20261108/adapted-checkpoints/joint_reps/checkpoints/step-003000-epoch-00-loss=restricted.pt',
        'restricted_joint_96000': '/home/users/lemon/coft-p3/p3-20261109/adapted/checkpoints/step-012000-epoch-01-loss=restricted96000.pt'}
    if (OUT/'baseline_validation.json').exists():
        existing=read(OUT/'baseline_validation.json')
        if len(existing)==6 and all(r['checkpoint_sha256']==digest(paths[r['model_label']]) and
                r['validation_index_sha256']==digest(OUT/f"{r['tag']}_index.json") for r in existing):
            print('Verified existing completed offline baseline validation; no repeated inference',flush=True)
            return
        raise ValueError('Preserve incomplete or mismatched baseline validation')
    results=[]
    for label,path in paths.items():
        model=load_vla(__import__('pathlib').Path(path),load_for_training=False).to(device='cuda',dtype=torch.bfloat16).eval()
        tokenizer=model.llm_backbone.get_tokenizer()
        transform=RLDSBatchTransform(tokenizer=tokenizer,image_transform=model.vision_backbone.get_image_transform(),
                    prompt_builder_fn=model.llm_backbone.prompt_builder_fn,action_tokenizer=model.action_tokenizer,
                    image_window_size=1,use_wrist_image=False,predict_stop_token=True)
        data=SimpleNamespace(batch_transforms=[(transform,1.)])
        collator=PaddedCollatorForActionPrediction(tokenizer.model_max_length,tokenizer.pad_token_id,padding_side='right')
        for tag in ['validation_target','validation_source_ur5e']:
            r=validate(model,data,collator,0,tag,output_prefix=label+'-')
            r.update(model_label=label,checkpoint=str(path),checkpoint_sha256=digest(path),
                     validation_index_sha256=digest(OUT/f'{tag}_index.json'),
                     validation_exposure='SEE_RECIPE_AUDIT_OLD_ADAPTERS_MAY_HAVE_SEEN_THESE_EPISODES')
            results.append(r)
        write(OUT/'baseline_validation.json',results)
        del model,transform,data,collator;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
