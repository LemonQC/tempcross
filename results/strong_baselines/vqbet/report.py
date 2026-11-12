"""Accept only trained native components plus 160 audited physical episodes."""
from collections import Counter
import csv
from baselines.strong.common import ROOT,TASKS,out,read,write,digest,resource

def audit_trace(directory,state,batch):
    counts=Counter();ids={'target':set(),'source':set()};previous=0
    with (directory/'consumed_anchors.csv').open() as f:
        for r in csv.DictReader(f):
            step=int(r['stage_update']);assert previous<=step<=state['actual_stage_updates'];previous=step
            assert int(r['frame'])>=9;counts[r['domain']]+=1;ids[r['domain']].add(r['episode_id'])
    assert counts['target']==state['target_anchor_exposures'] and counts['source']==state['source_anchor_exposures']
    with (directory/'updates.csv').open() as f:
        updates=list(csv.DictReader(f));assert [int(r['stage_update']) for r in updates]==list(range(1,state['actual_stage_updates']+1))
    return dict(status='PASS',actual_updates=len(updates),anchor_exposures=dict(counts),unique_training_demo_ids={k:len(v) for k,v in ids.items()},
        anchor_trace=str(directory/'consumed_anchors.csv'),trace_sha256=digest(directory/'consumed_anchors.csv'))

def main():
    folder=out('vqbet');cfg=read(folder/'protocol.json');cells=[];episodes=[];states=[];exposures=[]
    for group in cfg['groups']:
        for stage in ['vqvae','bet']:
            directory=ROOT/'vqbet/training'/group/stage;state=read(directory/'status.json');assert state['status']=='COMPLETE'
            assert state['actual_stage_updates']==cfg[stage+'_updates']
            expected=cfg['vqvae_updates'] if stage=='vqvae' else cfg['bet_full_optimizer_updates']*2+cfg['bet_offset_only_updates']
            assert state['actual_optimizer_calls']==expected
            selected=read(folder/(group+'_'+stage+'_selection.json'));assert not selected['used_physical_success']
            assert digest(selected['checkpoint'])==selected['sha256'];states.append(state)
            exposures.append(dict(group=group,stage=stage,**audit_trace(directory,state,cfg[stage+'_target_batch'])))
        selected=read(folder/(group+'_bet_selection.json'))
        for task in TASKS:
            accepted=read(folder/f'evaluation_{group}_{task}.json');assert accepted['status']=='PASS' and accepted['episodes']==40
            rows=[r['episode']|dict(failure_stage=r['failure_stage']) for r in accepted['rows']]
            assert {r['official_condition_index'] for r in rows}==set(range(60,100))
            assert all(r['checkpoint_sha256']==selected['sha256'] for r in rows)
            cells.append(dict(group=group,task=task,episodes=40,successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/40,
                checkpoint=selected['checkpoint'],checkpoint_sha256=selected['sha256'],selected_step=selected['selected']['step'],role='EXPOSED_REFERENCE',failure_counts=dict(Counter(r['failure_stage'] for r in rows))))
            episodes.extend(rows)
    assert len(episodes)==160
    with (folder/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episodes[0]),lineterminator='\n');writer.writeheader();writer.writerows(episodes)
    write(folder/'results.json',dict(status='COMPLETE_FAITHFUL_PORT',cells=cells,formal_episodes=160,successes=sum(r['success'] for r in episodes),
        independent_unseen_test=False,role='EXPOSED_REFERENCE',old_baseline_reruns=0,probe_episodes_excluded=4))
    write(folder/'actual_exposure_reconciliation.json',dict(status='PASS',stages=exposures,
        bet_action_labels_per_anchor=10,vqvae_action_labels_per_anchor=1,raw_canonical_actions_not_standardized=True))
    preserved=read(ROOT/'preservation_freeze.json')
    for row in preserved['files']:assert digest(row['path'])==row['sha256']
    write(folder/'preservation.json',dict(status='PASS',files=preserved['files'],old_baseline_reruns=0,tocat_used=False))
    index=[]
    for p in sorted((ROOT/'vqbet').rglob('*')):
        if p.is_file():index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    for p in sorted(ROOT.glob('vqbet-*.log')):index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(folder/'sha_index.json',index)
    write(folder/'errors.json',dict(technical_repairs=1,repair=read(folder/'technical_repair.json'),
        preflight_coding_corrections=[dict(stage='before_data_preparation_or_optimizer',issue='Patch guard matched cached and raw visual goal branches',fix='Restrict match to detached three-dimensional cached goal branch',training_updates_at_issue=0,physical_steps_at_issue=0)],
        core_algorithm_repairs=[],task_failure_scope='Descriptive stages only; no causal claim'))
    write(folder/'resources.json',dict(training_seconds=sum(s['elapsed_seconds'] for s in states),component_training_states=states,final_snapshot=resource(),
        formal_stage_updates=sum(s['actual_stage_updates'] for s in states),formal_optimizer_calls=sum(s['actual_optimizer_calls'] for s in states),probe_stage_updates=4,probe_optimizer_calls=6,
        prior_probe_stage_updates=4,prior_probe_optimizer_calls=6,total_executed_stage_updates=sum(s['actual_stage_updates'] for s in states)+8,
        total_executed_optimizer_calls=sum(s['actual_optimizer_calls'] for s in states)+12))
    write(folder/'status.json',dict(status='COMPLETE_FAITHFUL_PORT',method='vqbet',actual_training_updates=sum(s['actual_stage_updates'] for s in states),
        actual_optimizer_calls=sum(s['actual_optimizer_calls'] for s in states),probe_stage_updates=4,probe_optimizer_calls=6,physical_smoke_episodes=4,
        physical_evaluation_episodes=160,success_rate=sum(r['success'] for r in episodes)/160,technical_repairs=1,role='EXPOSED_REFERENCE',all_trajectories_and_videos_accepted=True,old_baseline_reruns=0,
        prior_probe_stage_updates=4,prior_probe_optimizer_calls=6,prior_physical_smoke_episodes=4,total_physical_smoke_episodes=8))
    lines=['# VQ-BeT结果','','COMPLETE_FAITHFUL_PORT：官方残差VQ、GPT码预测、连续偏移和两阶段更新保留；语言输入、canonical动作及有限计算预算差异见PROTOCOL.md。','',
        '| 设置 | 任务 | 官方成功/实际评测 | 成功率 | BeT checkpoint step |','|---|---|---:|---:|---:|']
    for c in cells:lines.append(f"| {c['group']} | {c['task']} | {c['successes']}/40 | {c['success_rate']*100:.1f}% | {c['selected_step']} |")
    lines+=['','条件60–99标记为EXPOSED_REFERENCE，不是独立未见测试。只按原身份留出目标示范的验证损失选择RVQ和BeT checkpoint，未用物理成功率选模。',
        '', '每设置RVQ实际6000次更新，BeT实际24000次更新；BeT前12000次同时更新码预测和偏移，后12000次只更新偏移。每个组件阶段384000个目标anchor；joint另有等量UR5e anchor。BeT每anchor含10个因果观测及对应动作标签，RVQ每anchor一个动作。',
        '', '使用官方发布支持的冻结ResNet18特征模式；不声称复现视觉编码器微调或论文全部计算预算。动作窗口为官方UR3配方的1，每物理控制步重新预测。',
        '', '唯一一次技术修复统一缓存和部署的FP32计算（关闭cuDNN及matmul TF32），在正式训练前完成。原四次组件探针更新/六次优化器调用及四条smoke另行归档，修复后四次组件探针更新/六次优化器调用及四条smoke单列；八条smoke都不计正式160条。全部视频、原始状态/动作/接触及权重留在服务器，绝对路径及哈希见sha_index.json及attempts/initial_tf32/sha_index.json；失败分类是描述性代理。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from baselines.strong.matrix_details import main as matrix
    matrix()

if __name__=='__main__':main()
