"""Merge an upstream PEFT adapter into its exact BARX source checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ['source','checkpoint','adapter','out','statistics']:p.add_argument('--'+name,required=True)
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.out).resolve()
    sys.path[:0]=[str(source),str(source/'policy')]
    import torch
    from peft import PeftModel
    from prismatic.models import load_vla
    model=load_vla(Path(a.checkpoint),load_for_training=False)
    model=PeftModel.from_pretrained(model,a.adapter).merge_and_unload()
    out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    state={key:getattr(model,key).state_dict() for key in model.all_module_keys}
    destination=out/'checkpoints/step-003000-epoch-00-loss=restricted.pt'
    torch.save({'model':state},destination)
    shutil.copyfile(Path(a.checkpoint).parents[1]/'config.json',out/'config.json')
    shutil.copyfile(a.statistics,out/'dataset_statistics.json')
    (out/'adapter_provenance.json').write_text(json.dumps(dict(source_checkpoint=a.checkpoint,adapter=a.adapter,
           merged_checkpoint=str(destination),role='RESTRICTED_LORA_ADAPTATION'),indent=2))


if __name__=='__main__':main()
