"""Describe saved generation failures and lineage; never changes a trajectory."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from common import PROJECT, OUT, ROOT, TASKS, read, write
from report import table


def main():
    import h5py
    import numpy as np
    state=read(OUT/'status.json');assert state['status'] in ['COMPLETE','BLOCKED']
    spec=importlib.util.spec_from_file_location('saved_interaction_audit',PROJECT/'evidence/interaction_initiation/audit.py')
    contact_tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(contact_tools)
    rows=[];groups=defaultdict(list);contexts=defaultdict(list)
    for p in sorted((ROOT/'augmentation/probe').glob('*/episode.json')):
        r=read(p)
        with h5py.File(r['telemetry']) as f:
            closed=np.flatnonzero(f['actions/environment'][:,6]>0)
            first=int(closed[0]) if len(closed) else None
            bodies=json.loads(f.attrs['body_names']);names=json.loads(f.attrs['geom_names'])
            geoms={g for g,b in zip(names,f['model/geom_bodyid'][:]) if bodies[b].startswith('obj_')}
            logs=[json.loads(x) for x in Path(r['contact_log']).read_text().splitlines()]
            contact=contact_tools.physical_contact(logs,geoms)
            if first is None:contact[:]=False
            else:contact[:first+1]=False
            rise=f['target/position'][:,2]-f['target/position'][0,2]
            lift=contact_tools.longest(contact & (rise>=.03))>=10
            placement=bool(f['mimicgen/post_action_placement'][:].any())
            success=bool(np.any(f['mimicgen/post_action_placement'][:]&f['mimicgen/post_action_retreat'][:]))
            assert success==r['success']
        stage=('SUCCESS' if success else 'NO_CLOSE_COMMAND' if first is None else
            'CLOSE_WITHOUT_BILATERAL_CONTACT' if not contact.any() else
            'CONTACT_WITHOUT_STABLE_LIFT' if not lift else
            'PLACEMENT_WITHOUT_RETREAT' if placement else 'STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT')
        lineage=read(ROOT/'augmentation/lineage'/(r['id']+'.json'))
        row=dict(id=r['id'],task=r['task'],condition=r['condition'],condition_id=r['condition_id'],
            success=success,steps=r['steps'],issued_close=first is not None,first_close=first,
            bilateral_contact=bool(contact.any()),stable_lift_proxy=lift,ever_placement=placement,
            failure_stage=stage,source_episode_ids=[s['episode_id'] for s in lineage['selected_sources']],
            source_subtasks=[s['subtask'] for s in lineage['selected_sources']],
            trajectory=r['trajectory'],telemetry=r['telemetry'],video=r['video'])
        rows.append(row);groups[r['task']].append(row);contexts[r['task'],r['condition']].append(row)
    table(OUT/'generation_failure_episodes.csv',rows)
    write(OUT/'all_generation_lineage.json',
          [read(p) for p in sorted((ROOT/'augmentation/lineage').glob('*.json'))])
    summary=[]
    for task,rr in sorted(groups.items()):
        summary.append(dict(task=task,attempts=len(rr),successes=sum(r['success'] for r in rr),
            physical_steps=sum(r['steps'] for r in rr),failure_counts=dict(Counter(r['failure_stage'] for r in rr)),
            distinct_frozen_conditions=len({r['condition_id'] for r in rr}),
            unique_source_parent_ids=sorted({s for r in rr for s in r['source_episode_ids']})))
    table(OUT/'generation_condition_cells.csv',[dict(task=k[0],condition=k[1],attempts=len(rr),successes=sum(r['success'] for r in rr)) for k,rr in sorted(contexts.items())])
    write(OUT/'generation_failure_summary.json',dict(cells=summary,causal_mechanism_identified=False,
        interpretation='Stage proxies on repeated development conditions; not policy success rates or independent samples.',
        new_physics_steps=0,new_model_calls=0))
    if (state['status']=='BLOCKED' and state['actual_optimizer_updates']==0
            and any(c['task']==TASKS[1] and c['successes']<50 for c in summary)):
        p=PROJECT/'docs/MIMICGEN_BASELINE_RESULTS.md'
        cfg=read(OUT/'downstream_protocol.json')
        probe=read(OUT/'probe_validation.json')
        total=sum(c['attempts'] for c in summary)
        success=sum(c['successes'] for c in summary)
        blocked=dict(status='BLOCKED',reason=state.get('reason'),
            probe_generation_successes=probe['generation_successes'],probe_generation_attempts=5,
            probe_replay_successes=probe['replay_successes'],probe_replay_attempts=5,
            augmentation_completed_attempts=total,augmentation_official_successes=success,
            per_task=summary,actual_optimizer_updates=state['actual_optimizer_updates'],
            selected_checkpoint=None,formal_policy_evaluation_episodes=len(state.get('evaluated_episode_ids',[])),
            formal_policy_success_rate=None,old_1280_plus_160_not_rerun=True,
            no_claim_of_algorithm_infeasibility=True)
        write(OUT/'blocked_results.json',blocked)
        if state['actual_optimizer_updates']==0 and not state.get('evaluated_episode_ids'):
            write(OUT/'selection_not_run.json',dict(status='NOT_RUN',checkpoint=None,
                reason='Augmentation data gate not met; no downstream model trained or selected.'))
        lines=['# MimicGen有界Baseline：实际结果','',
            '状态：**BLOCKED**。完整MimicGen学习Baseline尚未形成；未把几何生成器或单条成功轨迹当成部署视觉策略。','',
            f"阻碍发生在额外训练数据生成阶段：冻结目标为两任务各{cfg['augmentation']['successful_demos_per_task']}条成功示范，每任务最多{cfg['augmentation']['max_attempts_per_task']}次尝试。当前固定源池/配方在Sink→Counter未达到门槛，未降低数据目标、扩大生成量或改训练配置。原始阶段错误与精确退出信息见`phase2_execution.json`。",'',
            '| 任务 | 官方成功 / 实际额外生成 | 目标成功数 |','| --- | ---: | ---: |']
        for c in summary:lines.append(f"| {c['task']} | {c['successes']}/{c['attempts']} ({100*c['successes']/c['attempts']:.1f}%) | 50 |")
        lines+=['',f"额外生成共{total}条物理轨迹、{success}条官方成功，合计{sum(c['physical_steps'] for c in summary)}个控制步。全部成功与失败都保留，所用为既有开发条件0–19的变体，不是独立测试样本。",'',
            '## 已完成与未完成','',
            f"- 首轮生成{probe['generation_successes']}/5、对应动作重放{probe['replay_successes']}/5，共8/10；五对相关轨迹仅用于可行性门槛。逐条视频、动作、接触、物理状态及重放初态/动作前缀通过核查。",
            '- 已审查两项官方生成接口，准备每方向固定5条Panda human train的原始状态注释；没有使用人类validation示范生成数据。源身份与全部父片段选择分别见`source_seed_manifest.json`、每次lineage原件。',
            '- 已冻结唯一BC-Transformer配置、seed7、48000更新/768000 anchors、固定验证和checkpoint选择规则。它们只是已准备的执行代码/配置，不能写成已训练。',
            f"- 实际下游optimizer更新：**{state['actual_optimizer_updates']}**；checkpoint：**未产生**；选模：**未执行**；正式视觉策略评测：**未执行，成功率N/A**。",
            '- 原1280＋160条Baseline物理实验零重跑，原输入和原HDF5保持SHA256不变。没有启动其他Baseline或新Method。','',
            '## 实现修复及复现边界','',
            '预检发现原源动作反解缺少base→world旋转，完成一次隔离注释修复；纯数学往返测试不当成物理成功率。探针第三条的NumPy int64 JSON索引记录错误使用物理探针阶段单独允许的一次修复，已有403步失败轨迹只读恢复，未重新执行。首次协议将两阶段修复合并的勘误在剩余探针前公开提交。详见`probe_logging_repair.json`和协议追加说明。',
            '崩溃时NumPy RNG状态未保存，剩余探针按预注册seed7重启，未声称与无中断执行逐位一致；额外生成则从开始就按每attempt固定种子记录。没有根据成功率换seed、源池、噪声或控制器。','',
            '源示范是Panda目标域human train每方向5条，而非零目标示范的UR5e迁移。生成器使用训练时物体/目标/EE位姿和接触阶段；原定部署模型仅RGB＋原文指令。额外生成成本应与BC/DP/BARX分列，不能声称等数据公平。','',
            '## 已有生成轨迹的只读失败诊断','',
            '| 任务 | 成功/已完成生成 | 物理步数 | 失败阶段计数 |','| --- | ---: | ---: | --- |']
        for c in summary:lines.append(f"| {c['task']} | {c['successes']}/{c['attempts']} | {c['physical_steps']} | {json.dumps(c['failure_counts'])} |")
        lines+=['','阶段代理是已保存实际动作、双侧接触、物体body原点上升3cm持续10状态及官方放置/退离的描述，不识别严格物理根因。生成率低不能归因于视觉策略，因为根本未进入学习阶段；也不能据此断言MimicGen一般无效或证明需要新算法。',
            '本轮科学结论限定为：当前固定源池、原任务配置及有界数据预算未形成预注册的两任务强对照。保留这些真实结果并在此停止，不用替代规则控制器、少量成功示范或未运行评测填补完整Baseline。','',
            f'原始目录：`{ROOT}`。实际物理账本：`physical_launches.jsonl`；逐条结果：`generation_and_replay_episodes.csv`、`generation_failure_episodes.csv`；分条件结果：`generation_condition_cells.csv`。视频、动作、接触和状态的绝对路径、字节数/SHA256见`external_file_index.json`，大文件留服务器。',
            '协议：[MIMICGEN_BASELINE_PROTOCOL.md](MIMICGEN_BASELINE_PROTOCOL.md)。本次汇总没有新物理step或模型调用。']
        p.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
