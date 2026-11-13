"""Check actual native window labels, image cache and tokenizer information."""
import os
import sys
sys.path.insert(0,'/home/users/lemon/baseline-benchmark/vendor/octo')
from baselines.strong.common import ROOT,write,read
os.environ['HF_HOME']=str(ROOT/'octo_small/huggingface')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import h5py
from octo.utils.spec import ModuleSpec
from .data import DIRECTORY,Dataset,encode_action,normalized_action

def main():
    config=read(ROOT/'octo_small/pretrained/config.json')
    processor=ModuleSpec.instantiate(config['text_processor'])()
    rows=read(DIRECTORY/'manifest.json');group={}
    for i,row in enumerate(rows):
        if row['split']=='train':group.setdefault((row['domain'],row['task']),i)
    data=Dataset();data.anchors=np.array([(j,t) for j in group.values() for t in [0,1,rows[j]['frames']-2,rows[j]['frames']-1]],np.int32)
    traces=[]
    for i,(j,t) in enumerate(data.anchors):
        j,t=int(j),int(t);row=rows[j];item=data.item(i)
        assert item['image'].shape==(2,256,256,3) and item['action'].shape==(2,4,7)
        assert item['action_pad_mask'].shape==(2,4,7)
        np.testing.assert_array_equal(item['timestep_pad_mask'],[t>0,True])
        with h5py.File(row['raw_file'],'r') as f:
            actions=f['data'][row['raw_demo']]['actions'][:,:7].astype(np.float32)
        actions[-1]=actions[-2];actions[-1,:6]=0
        history=np.maximum([t-1,t],0)
        expected=actions[np.minimum(history[:,None]+np.arange(4),len(actions)-1)]
        np.testing.assert_allclose(item['action'],normalized_action(encode_action(expected),data.stats),atol=0,rtol=0)
        traces.append(item['trace'])
    languages=sorted(set(r['instruction'] for r in rows));encoded=processor.encode(languages)['input_ids']
    full=processor.tokenizer(languages,add_special_tokens=True,truncation=False)['input_ids']
    information=[];ids_to_tasks={}
    for text,tokens,untruncated in zip(languages,encoded,full):
        tasks=sorted(set(r['task'] for r in rows if r['instruction']==text));key=tuple(map(int,tokens))
        ids_to_tasks.setdefault(key,set()).update(tasks)
        information.append(dict(instruction=text,tasks=tasks,tokens=tokens.tolist(),untruncated_length=len(untruncated),
            truncated=len(untruncated)>len(tokens)))
    result=dict(status='PASS',boundary_windows_checked=len(traces),traces=traces,
        native_chunk_labels_equal=True,history_padding_checked=True,source_and_target_both_tasks_checked=True,
        tokenizer_native_max_length=16,tokenizer_distinct_instructions=len(languages),
        cross_task_token_collisions=sum(len(v)>1 for v in ids_to_tasks.values()),languages=information,
        new_simulator_steps=0)
    write(ROOT/'octo_small/native_data_audit.json',result);data.close()
    print('NATIVE_DATA_AUDIT_PASS',result['boundary_windows_checked'],'windows',result['cross_task_token_collisions'],'cross-task token collisions',flush=True)

if __name__=='__main__':main()
