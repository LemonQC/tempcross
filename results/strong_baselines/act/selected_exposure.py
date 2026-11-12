"""Distinguish full training cost from the selected checkpoint's own exposures."""
import csv
from collections import Counter
from baselines.strong.common import ROOT,out,read,write,digest

def main():
    import torch
    result=[]
    for group in ['target_only','source_target']:
        selection=read(out('act')/(group+'_selection.json'))
        if selection.get('checkpoint') is None:continue
        checkpoint=selection['checkpoint'];assert digest(checkpoint)==selection['sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        step=selection['selected']['step'];assert saved['step']==step and saved['group']==group
        counts=Counter()
        with (ROOT/'act/training'/group/'consumed_anchors.csv').open() as stream:
            for row in csv.DictReader(stream):
                if int(row['update'])<=step:counts[row['domain']]+=1
        assert counts['target']==16*step
        assert counts['source']==(16*step if group=='source_target' else 0)
        result.append(dict(group=group,checkpoint=checkpoint,sha256=selection['sha256'],selected_checkpoint_updates=step,
            selected_checkpoint_target_anchor_exposures=counts['target'],selected_checkpoint_source_anchor_exposures=counts['source'],
            complete_training_run_updates=selection['actual_optimizer_updates'],
            complete_training_run_target_anchor_exposures=selection['target_anchor_exposures'],
            complete_training_run_source_anchor_exposures=selection['source_anchor_exposures'],
            verified_against_actual_checkpoint_metadata_and_consumption_log=True,
            physical_success_used_for_selection=False))
        del saved
    write(out('act')/'selected_checkpoint_exposures.json',result)
    print(__import__('json').dumps(result,indent=2))

if __name__=='__main__':main()
