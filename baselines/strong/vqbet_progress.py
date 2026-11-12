"""Read-only snapshot of actual completed VQ-BeT stages; never launches jobs."""
from collections import Counter
from baselines.strong.common import ROOT,TASKS,out,read,write,digest,now,resource

def main():
    folder=out('vqbet');stages=[];models=[];cells=[];indices=[];cfg=read(folder/'protocol.json')
    for group in ['target_only','source_target']:
        for stage in ['vqvae','bet']:
            p=ROOT/'vqbet/training'/group/stage/'status.json'
            if not p.exists():continue
            state=read(p);stages.append(state)
            if state['status']=='COMPLETE':
                selected=read(folder/(group+'_'+stage+'_selection.json'))
                assert digest(selected['checkpoint'])==selected['sha256'] and not selected['used_physical_success']
                selected_count=selected['selected']['step']*cfg[stage+'_target_batch']
                models.append(dict(group=group,stage=stage,selected_checkpoint_target_anchor_exposures=selected_count,
                    selected_checkpoint_source_anchor_exposures=selected_count if group=='source_target' else 0,
                    action_labels_per_anchor=10 if stage=='bet' else 1,**selected))
                for name in ['status.json','updates.csv','consumed_anchors.csv','best.pt','latest.pt','resource_start.json']:
                    q=p.parent/name
                    if q.exists():indices.append(dict(path=str(q),bytes=q.stat().st_size,sha256=digest(q)))
        for task in TASKS:
            p=folder/f'evaluation_{group}_{task}.json'
            if not p.exists():continue
            accepted=read(p);assert accepted['status']=='PASS' and accepted['episodes']==40
            rows=accepted['rows'];cells.append(dict(group=group,task=task,episodes=40,successes=accepted['successes'],success_rate=accepted['successes']/40,
                role='EXPOSED_REFERENCE',failure_counts=dict(Counter(r['failure_stage'] for r in rows))))
    closed=list((ROOT/'vqbet/physical/formal').glob('*/*/condition-*/episode.json'))
    completed=[s for s in stages if s['status']=='COMPLETE'];stage_updates=sum(s['actual_stage_updates'] for s in completed)
    write(folder/'model_index.json',models);write(folder/'training_artifact_index.json',indices)
    write(folder/'selected_checkpoint_exposures.json',dict(components=models,note='Separate codebook pretraining and behavior training exposures; full executed budgets remain distinct from selected checkpoint exposure'))
    if read(folder/'status.json')['status']=='COMPLETE_FAITHFUL_PORT':
        assert len(models)==4 and len(cells)==4
        print('Final model/selected-exposure indices refreshed; historical partial snapshot retained.');return
    result=dict(status='PARTIAL',snapshot_utc=now(),server_root=str(ROOT),cells=cells,completed_training_components=completed,
        completed_component_stage_updates=stage_updates,completed_component_optimizer_calls=sum(s['actual_optimizer_calls'] for s in completed),
        formally_accepted_episodes=sum(c['episodes'] for c in cells),closed_physical_episodes_at_snapshot=len(closed),
        in_progress_components=[s for s in stages if s['status']!='COMPLETE'],independent_unseen_test=False,role='EXPOSED_REFERENCE',
        no_unfinished_success_rates=True,old_baseline_reruns=0)
    write(folder/'partial_results.json',result)
    write(folder/'progress_resources.json',resource())
    lines=['# VQ-BeT阶段快照','','这是断点快照，不是完整Baseline结果。最终状态以status.json和RESULTS.md为准。','',
        f"已完成组件更新：{stage_updates}；已经闭合的物理episode：{len(closed)}；通过完整40条单元验收的评测：{result['formally_accepted_episodes']}条。",'',
        '| 设置 | 组件 | 实际更新 | 验证选择step | 验证损失 |','|---|---|---:|---:|---:|']
    for m in models:lines.append(f"| {m['group']} | {m['stage']} | {m['actual_stage_updates']} | {m['selected']['step']} | {m['selected']['loss']:.8f} |")
    lines+=['','未完成或未验收的单元不填写0%成功率。checkpoint和原始训练日志的绝对路径及SHA见model_index.json及training_artifact_index.json。',
        '', '原TF32精度尝试和修复版探针分别归档，不计正式训练更新或正式成功率。已完成阶段复用，不重训。']
    (folder/'PARTIAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    from baselines.strong.matrix_details import main as matrix
    matrix()
    print(result['snapshot_utc'],stage_updates,len(closed),result['formally_accepted_episodes'])

if __name__=='__main__':main()
