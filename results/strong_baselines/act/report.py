"""Accept a full baseline only with real trained checkpoints and all 160 trials."""
from collections import Counter
import csv
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,TASKS,out,read,write,digest,resource

def main():
    folder=out('act');cells=[];episodes=[];updates=[];training_seconds=0
    for group in ['target_only','source_target']:
        state=read(ROOT/'act/training'/group/'status.json');cfg=read(folder/'protocol.json')
        assert state['status']=='COMPLETE' and state['actual_optimizer_updates']==cfg['optimizer_updates_per_group']
        selection=read(folder/(group+'_selection.json'));assert digest(selection['checkpoint'])==selection['sha256'] and not selection['used_physical_success']
        updates.append(state['actual_optimizer_updates']);training_seconds+=state['elapsed_seconds']
        for task in TASKS:
            accepted=read(folder/f'evaluation_{group}_{task}.json');assert accepted['status']=='PASS' and accepted['episodes']==40
            rr=[r['episode']|dict(failure_stage=r['failure_stage']) for r in accepted['rows']]
            assert {r['official_condition_index'] for r in rr}==set(range(60,100))
            assert all(r['checkpoint_sha256']==selection['sha256'] for r in rr)
            cells.append(dict(group=group,task=task,episodes=40,successes=sum(r['success'] for r in rr),success_rate=sum(r['success'] for r in rr)/40,
                              checkpoint=selection['checkpoint'],checkpoint_sha256=selection['sha256'],selected_step=selection['selected']['step'],
                              role='EXPOSED_REFERENCE',failure_counts=dict(Counter(r['failure_stage'] for r in rr))))
            episodes.extend(rr)
    assert len(episodes)==160
    with (folder/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(episodes[0]),lineterminator='\n');w.writeheader();w.writerows(episodes)
    write(folder/'results.json',dict(status='COMPLETE_FAITHFUL_PORT',cells=cells,formal_episodes=160,successes=sum(r['success'] for r in episodes),independent_unseen_test=False,
                                    role='EXPOSED_REFERENCE',new_target_demonstrations=0,old_baseline_reruns=0,probe_episodes_excluded=4))
    inputs=read(ROOT/'preservation_freeze.json')
    for row in inputs['files']:assert digest(row['path'])==row['sha256']
    write(folder/'preservation.json',dict(status='PASS',files=inputs['files'],old_baseline_reruns=0,tocat_used=False))
    index=[]
    for p in sorted((ROOT/'act').rglob('*')):
        if p.is_file():index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    for p in sorted(ROOT.glob('act-*.log')):index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(folder/'sha_index.json',index)
    write(folder/'errors.json',dict(technical_repairs=1,errors=[read(folder/'technical_repair.json')],scope='The upstream decoder repair is declared; task failures are in episodes.csv'))
    prior=read(folder/'attempts/upstream_first_decoder/summary.json')
    write(folder/'resources.json',dict(corrected_training_seconds=training_seconds,
        prior_technical_attempt_training_seconds=prior['state']['elapsed_seconds'],
        total_executed_training_seconds=training_seconds+prior['state']['elapsed_seconds'],
        formal_optimizer_updates=48000,prior_attempt_optimizer_updates=24000,total_probe_optimizer_updates=4,
        final_snapshot=resource(),training_groups=[read(ROOT/'act/training'/g/'status.json') for g in ['target_only','source_target']]))
    write(folder/'status.json',dict(status='COMPLETE_FAITHFUL_PORT',method='act',actual_training_updates=sum(updates),updates_per_group=updates,probe_optimizer_updates=2,
        physical_smoke_episodes=4,physical_evaluation_episodes=160,success_rate=sum(r['success'] for r in episodes)/160,
        technical_repairs=1,prior_technical_attempt_updates=24000,total_executed_training_updates=72000,
        prior_probe_optimizer_updates=2,prior_physical_smoke_episodes=4,
        role='EXPOSED_REFERENCE',all_trajectories_and_videos_accepted=True,old_baseline_reruns=0))
    lines=['# ACT结果','', '**COMPLETE_FAITHFUL_PORT**：官方CVAE/动作分块/时序集成核心保留；RGB＋语言到canonical OSC接口适配差异见PROTOCOL.md。', '',
           '| 设置 | 任务 | 官方成功/实际评测 | 成功率 | checkpoint step |', '|---|---|---:|---:|---:|']
    for c in cells:lines.append(f"| {c['group']} | {c['task']} | {c['successes']}/{c['episodes']} | {100*c['success_rate']:.1f}% | {c['selected_step']} |")
    lines+=['','唯一一次技术修复把官方第一层decoder中间输出改为最后层输出，实际七层均更新。原版target-only的24000次训练、2次探针更新和4条smoke完整保留并单列，不进入修复版正式成功率。',
            '', '条件60–99全部为EXPOSED_REFERENCE，不是独立未见测试。模型只由留出示范验证损失选择；没有根据这里的成功率调参。',
            '',f'每设置实际24000次更新，384000个目标anchor暴露；joint另有384000个UR5e anchor。训练合计{training_seconds:.1f}秒。探针2次更新和4条物理smoke单列，不计正式160条。',
            '', '全部失败轨迹保留在episodes.csv。原始视频、状态、动作、接触及checkpoint的服务器绝对路径和SHA256见sha_index.json。失败阶段为描述性代理，不推断严格物理根因。',
            '', '原1280＋160条结果及其启动账本哈希不变；TOCAT分支未使用。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from baselines.strong.matrix import main as matrix
    matrix()

if __name__=='__main__':main()
