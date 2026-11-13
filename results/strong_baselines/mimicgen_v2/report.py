"""Accept the bounded component; keep full MimicGen status PARTIAL honestly."""
from collections import Counter
import csv
from pathlib import Path
from baselines.strong.common import PROJECT,ROOT,TASKS,out,read,write,digest,resource,now

def main():
    import torch,numpy as np
    folder=out('mimicgen_v2');cfg=read(folder/'protocol.json');freeze=read(folder/'freeze.json')
    assert digest(folder/'protocol.json')==freeze['protocol_sha256']
    for p,sha in freeze['code_sha256'].items():assert digest(PROJECT/p)==sha,p
    for p,sha in read(ROOT/'mimicgen_v2/data/complete.json')['sha256'].items():assert digest(p)==sha,p
    for p,sha in read(folder/'official_source.json')['sha256'].items():assert digest(p)==sha,p
    state=read(ROOT/'mimicgen_v2/training/status.json');selected=read(folder/'selection.json')
    assert state['status']=='COMPLETE' and state['actual_optimizer_updates']==48000 and selected['selected_step']==48000
    assert not selected['used_physical_success'] and digest(selected['checkpoint'])==selected['sha256']
    saved=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    counters=sorted({int(v['step']) for v in saved['model']['optimizers']['policy']['state'].values() if 'step' in v});assert counters==[48000]
    control=read(folder/'reused_control.json');assert saved['config']==read(control['config'])
    assert digest(control['checkpoint'])==control['checkpoint_sha256'] and digest(control['initial_weights'])==control['initial_weights_sha256']
    initial=torch.load(control['initial_weights'],map_location='cpu',weights_only=False);nets=saved['model']['nets'];changed=[]
    for name,value in initial.items():
        assert name in nets and torch.isfinite(nets[name]).all()
        if not torch.equal(value,nets[name]):changed.append(name)
    assert changed
    write(folder/'optimizer_acceptance.json',dict(status='PASS',actual_native_optimizer_step_counters=counters,changed_parameter_tensors=len(changed),
        exact_control_architecture=True,random_initial_weights_sha256=control['initial_weights_sha256'],trained_checkpoint_sha256=selected['sha256']))
    rows=read(ROOT/'mimicgen_v2/data/manifest.json');anchors=np.load(ROOT/'mimicgen_v2/data/train.npy',mmap_mode='r');counts=Counter();ids=set();n=0
    with (ROOT/'mimicgen_v2/training/consumed_anchors.csv').open() as stream:
        for k,r in enumerate(csv.DictReader(stream)):
            j,t=map(int,anchors[k]);expected=rows[j]
            assert int(r['update'])==k//16+1 and r['origin']==expected['origin'] and r['task']==expected['task'] and r['episode_id']==expected['episode_id'] and int(r['frame'])==t
            assert expected['split']=='train';counts[(r['origin'],r['task'])]+=1;ids.add(r['episode_id']);n+=1
    assert n==768000 and len(ids)==153
    assert sum(v for (origin,_),v in counts.items() if origin=='human')==state['human_anchor_exposures']
    assert sum(v for (origin,_),v in counts.items() if origin=='generated')==state['generated_anchor_exposures']
    write(folder/'actual_exposures.json',dict(status='PASS',actual_updates=48000,total_anchors=n,supervised_action_labels=n,
        input_history_positions=10*n,unique_training_demo_ids=len(ids),by_origin_and_task=[dict(origin=k[0],task=k[1],anchors=v) for k,v in sorted(counts.items())],
        trace=str(ROOT/'mimicgen_v2/training/consumed_anchors.csv'),trace_sha256=digest(ROOT/'mimicgen_v2/training/consumed_anchors.csv')))
    cells=[];episodes=[];jobs=[]
    for task in TASKS:
        accepted=read(folder/f'evaluation_human_plus_generated_{task}.json');assert accepted['status']=='PASS' and accepted['episodes']==40
        rr=[r['episode']|dict(failure_stage=r['failure_stage']) for r in accepted['rows']]
        assert {r['official_condition_index'] for r in rr}==set(range(60,100)) and all(r['checkpoint_sha256']==selected['sha256'] for r in rr)
        cells.append(dict(group='human_plus_generated',task=task,episodes=40,successes=sum(r['success'] for r in rr),success_rate=sum(r['success'] for r in rr)/40,
            role='EXPOSED_REFERENCE',failure_counts=dict(Counter(r['failure_stage'] for r in rr)),checkpoint=selected['checkpoint'],checkpoint_sha256=selected['sha256']))
        episodes+=rr;jobs.append(read(ROOT/'mimicgen_v2/physical/formal/human_plus_generated'/task/'job.json'))
    assert len(episodes)==80
    with (folder/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episodes[0]),lineterminator='\n');writer.writeheader();writer.writerows(episodes)
    old_status=read(out('mimicgen')/'status.json');assert old_status['status']=='BLOCKED'
    assert digest(out('mimicgen')/'status.json')==freeze['original_mimicgen_status_sha256']
    preserved=read(ROOT/'preservation_freeze.json')
    for r in preserved['files']:assert digest(r['path'])==r['sha256']
    write(folder/'preservation.json',dict(status='PASS',files=preserved['files'],original_mimicgen_gate='BLOCKED_UNCHANGED',old_baseline_reruns=0,new_generation_attempts=0))
    write(folder/'results.json',dict(status='PARTIAL',component_status='BOUNDED_DOWNSTREAM_CONTROL_COMPLETE',scope=cfg['method'],
        cells=cells,formal_episodes=80,new_policy_successes=sum(r['success'] for r in episodes),reused_control_cells=control['cells'],
        reused_control_episodes=80,new_control_episodes=0,new_control_training_updates=0,generated_train_counts=cfg['existing_generated_train_counts'],
        full_original_mimicgen_reproduced=False,cross_embodiment_generation=False,independent_unseen_test=False,role='EXPOSED_REFERENCE',old_baseline_reruns=0))
    index=[];seen=set();allocated=0
    for p in sorted((ROOT/'mimicgen_v2').rglob('*')):
        if p.is_file():
            s=p.stat();index.append(dict(path=str(p),bytes=s.st_size,sha256=digest(p)))
            if (s.st_dev,s.st_ino) not in seen:seen.add((s.st_dev,s.st_ino));allocated+=s.st_blocks*512
    for p in sorted(ROOT.glob('mimicgen-v2-*.log')):index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(folder/'sha_index.json',index)
    write(folder/'model_index.json',dict(new_augmented_policy=selected,reused_human_control=control,probe=dict(path=str(ROOT/'mimicgen_v2/probe/latest.pt'),sha256=digest(ROOT/'mimicgen_v2/probe/latest.pt'))))
    write(folder/'resources.json',dict(training_seconds=state['elapsed_seconds'],peak_training_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'],
        peak_training_gpu_reserved_bytes=state['peak_gpu_reserved_bytes'],formal_job_wall_seconds=sum(j['duration_seconds'] for j in jobs),
        formal_control_steps=sum(r['steps'] for r in episodes),dedicated_allocated_bytes_counting_hardlinks_once=allocated,
        formal_optimizer_updates=48000,probe_optimizer_updates=1,total_new_optimizer_updates=48001,new_formal_episodes=80,new_smoke_episodes=2,new_generation_attempts=0,final_snapshot=resource()))
    write(folder/'errors.json',dict(new_technical_repairs=0,source_review_findings=['AST identity rejected because variable names/unused None branch differ;120 numerical cases match, no generator or algorithm modified'],
        legacy_generation_repairs=old_status['repair_attempts'],full_original_generation_failure=read(out('mimicgen')/'blocked_results.json')['reason']))
    write(folder/'status.json',dict(status='PARTIAL',component_status='BOUNDED_DOWNSTREAM_CONTROL_COMPLETE',method='mimicgen_v2',
        actual_training_updates=48000,probe_optimizer_updates=1,physical_smoke_episodes=2,physical_evaluation_episodes=80,
        success_rate=sum(r['success'] for r in episodes)/80,technical_repairs=0,new_generation_attempts=0,
        full_original_mimicgen_status='BLOCKED_PRESERVED',cross_embodiment_generation=False,role='EXPOSED_REFERENCE',all_new_trajectories_accepted=True,old_baseline_reruns=0))
    write(folder/'delivery_acceptance.json',dict(status='PASS_BOUNDED_COMPONENT_ONLY',utc=now(),actual_new_training_updates=48001,new_formal_episodes=80,new_smoke_episodes=2,
        optimizer_counters_verified=True,all_consumed_anchors_verified=True,old_baseline_hashes_unchanged=True,original_full_gate_preserved=True,
        full_baseline_claim=False,new_generation_attempts=0,never_test_selected=True))
    lines=['# MimicGen v2结果','','**PARTIAL；有界下游组件完成**。不等量50/13生成数据和Panda源身份使本实验不能代替完整MimicGen跨具身基线；原50+50门槛BLOCKED保持不变。','',
        '| 任务 | 旧人类BC对照（复用） | 新human+generated策略 | 新评测次数 |','|---|---:|---:|---:|']
    for c in cells:
        old=next(r for r in control['cells'] if r['task']==c['task']);lines.append(f"| {c['task']} | {old['successes']}/40 | {c['successes']}/40 | 40 |")
    lines+=['', '全部均标EXPOSED_REFERENCE，不是独立未见测试。新策略仅一次48000更新、固定最终checkpoint，与复用对照的架构、随机初始参数、总batch/更新预算匹配；不把旧自适应停止程序与新固定预算称严格相同的因果对照。',
        '', f"实际768000训练anchor中，人类{state['human_anchor_exposures']}，生成{state['generated_anchor_exposures']}；每anchor监督最后一步动作，加载10帧因果历史。原始消耗逐行核对，见actual_exposures.json。",
        '', '177条既有生成记录全部核查并解码视频，63条成功示范全部使用；没有新增生成、重跑旧BC物理评测或增加目标人类示范。新策略80条正式评测与两条smoke分开，失败阶段为描述性代理，完整目标仍为placement AND retreat。',
        '', '权重、原始轨迹、视频和训练日志的服务器绝对路径与SHA见model_index.json、sha_index.json；既有生成大文件见existing_generation_index.json。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from baselines.strong.matrix_details import main as matrix
    matrix();print('Accepted bounded MimicGen v2 component; overall MimicGen remains PARTIAL')

if __name__=='__main__':main()
