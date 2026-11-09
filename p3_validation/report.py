"""Write completion reports only after all frozen physical evaluations validate."""
from collections import Counter
import csv
import datetime
import json
from pathlib import Path
import statistics
from p3_validation.common import ROOT,PROJECT


def read(path):return json.loads(Path(path).read_text())


def main():
    out=PROJECT/'evidence/p3/execution';data=read(out/'analysis.json');suite=read(out/'training_suite.json');validation=read(out/'validation/summary.json')
    assert suite['status']=='COMPLETE' and suite['actual_sample_positions']==96000
    assert validation['complete_independent_suite'] and validation['videos_verified']==124
    assert len(data['episodes'])==100
    cells={r['cell']:r for r in data['cells']};name='p3-96000-joint_reps-panda';new=cells[name]
    assert all(r['status']=='COMPLETE' and r['episodes']==20 for r in cells.values())
    batch=suite['selected_batch'];training=read(out/f'training/joint-reps-96000-b{batch}/training_status.json')
    execution=read(out/f'training/joint-reps-96000-b{batch}/execution.json');inputs=read(out/f'training/joint-reps-96000-b{batch}/input_status.json')
    assert training['status']=='COMPLETE' and training['sample_positions']==inputs['samples_consumed']==96000
    updates=training['completed_optimizer_steps'];assert updates*batch==96000 and execution['completed_backwards']==updates and execution['nonfinite_losses']==0
    manifest=read(out/'data/complete.json');provenance=read(out/'adapter_provenance.json');parameters=read(out/'parameter_changes.json')
    assert inputs['consumed_trace_sha256']==manifest['sample_trace_sha256'] and parameters['changed_tensors']>0
    with (out/'data/demonstration_coverage.csv').open() as f:coverage=list(csv.DictReader(f))
    direction_positions=Counter();direction_demos=Counter()
    for r in coverage:
        if r['dataset']=='panda_pnp':direction_positions[r['direction']]+=int(r['sampled_positions']);direction_demos[r['direction']]+=1
    assert sum(direction_positions.values())==48412 and sorted(direction_demos.values())==[50,50]
    groups={cell:{r['official_condition_index']:r for r in data['episodes'] if r['cell']==cell} for cell in cells}
    def changed(candidate,baseline):
        a=groups[candidate];b=groups[baseline]
        return ([i for i in b if not b[i]['success'] and a[i]['success']],[i for i in b if b[i]['success'] and not a[i]['success']])
    rescue,regression=changed(name,'original-joint_reps-panda')
    comparisons=read(out/'paired_analysis.json');diag=read(out/'diagnostic_analysis.json')
    source=cells['original-joint_reps-ur5e']['successes'];original=cells['original-joint_reps-panda']['successes'];k=new['successes']
    successful=[i for i,r in groups[name].items() if r['success']];stages=dict(Counter(r['observed_stage'] for r in groups[name].values()))
    source_successful=[i for i,r in groups['original-joint_reps-ur5e'].items() if r['success']]
    panda_cells=[n for n in cells if n!='original-joint_reps-ur5e']
    persistent=[i for i in source_successful if not any(groups[n][i]['success'] for n in panda_cells)]
    stage_description=f"{stages.get('NO_CLOSE_COMMAND',0)}条整段未发闭爪命令，{stages.get('NO_BILATERAL_CONTACT',0)}条闭爪但无双指接触，{stages.get('CONTACT_WITHOUT_SUSTAINED_LIFT',0)}条有接触但无稳定抬升，{stages.get('POST_ACQUISITION_TASK_FAILURE',0)}条抓持后失败，{k}条官方成功"
    stable=sum(r['sustained_contact_lift_proxy'] for r in groups[name].values());ci=new['wilson95']
    if k>=source:
        assessment=f'增强普通适配达到或超过这批源机器人的观测成功次数（{k}/20 对 {source}/20），支持此前训练预算/覆盖不足可能解释主要均值差距。小样本不能证明等效或普遍充分，不能据此宣称问题已全部解决。'
    elif k>original:
        assessment=f'增强普通适配从 {original}/20 改善到 {k}/20，但观测次数仍低于源机器人 {source}/20。结果与普通适配可以缩小差距相容；20个条件尚不足以确定稳定收益、剩余差距大小或唯一原因。'
    else:
        assessment=f'增强普通适配为 {k}/20，未超过原始Panda的 {original}/20，低于源机器人 {source}/20。本次受限配方没有解决完整任务差距；它仍只有官方暴露量的1/8，不能据此否定充分标准适配或所有标准方法。'
    labels={'original-joint_reps-ur5e':'Original Joint / UR5e','original-joint_reps-panda':'Original Joint / Panda','b1-joint_reps-panda':'B1 fixed 0.85 / Panda','p2-6000-joint_reps-panda':'Old 6000-position Joint LoRA / Panda',name:'New 96000-position Joint LoRA / Panda'}
    table='| 方法 | 官方完整任务 | 稳定接触抬升代理 |\n| --- | --- | --- |\n'+'\n'.join(f"| {labels[n]} | {cells[n]['successes']}/20 ({cells[n]['success_rate']:.0%}) | {sum(r['sustained_contact_lift_proxy'] for r in groups[n].values())}/20 |" for n in labels)
    with (out/f'training/joint-reps-96000-b{batch}/backward_losses.csv').open() as f:loss=[float(r['raw_loss']) for r in csv.DictReader(f)]
    assert len(loss)==updates
    peak=training['peak_gpu_allocated_bytes']/2**30;reserved=training['peak_gpu_reserved_bytes']/2**30
    strong=f'''# P3-B：96000样本位置的增强受限Joint LoRA实际完成

已完成 **{updates}次真实optimizer更新 × batch {batch} = 96000样本位置**，随后仅用预指定最终checkpoint完成独立40–59的20次物理评测，官方成功为 **{k}/20（{k/20:.0%}）**，Wilson 95%区间为{ci[0]:.1%}–{ci[1]:.1%}。成功条件为{successful}；相对原始Panda救回{rescue}、退化{regression}，不以平均数掩盖失败集合变化。

{assessment}

## 实际训练、来源与资源

batch=8的独占GPU单步forward/backward/optimizer探针真实通过，探针权重丢弃；正式模型从原始XP-900 Joint Reps 50k重新开始。实际路线是单GPU上游DDP LoRA，rank32、alpha16、dropout0、AdamW LR2e-5常数、weight decay0、BF16，没有新架构。固定最终更新数{updates}，没有根据40–59挑checkpoint。用时{training['elapsed_seconds']:.1f}秒，主训练峰值分配{training['peak_gpu_allocated_bytes']} bytes（{peak:.2f} GiB）、预留{training['peak_gpu_reserved_bytes']} bytes（{reserved:.2f} GiB）。没有资源OOM，因此未继续尝试batch4/2。

全部{updates}次backward和optimizer记录完整、损失有限、nonfinite计数0。前500步原始loss均值{statistics.mean(loss[:500]):.6f}，末500步{statistics.mean(loss[-500:]):.6f}；训练loss不能代替闭环成功率。初始/最终adapter逐张量比较：{parameters['total_adapter_tensors']}个保存张量中 **{parameters['changed_tensors']}个真实改变**，最终值均有限。探针的一步参数核验与正式模型分开保存，不将非零初始化误当成更新。

源/目标实际输入为47588/48412位置；源池1800、实际涉及115条；目标两个方向各50条、全部100条原始示范均被访问。目标方向实际位置数：`{json.dumps(dict(direction_positions))}`。88261个唯一原始帧位置、7739次重复帧、138次预处理内容完全重复；目标40673/41065原始帧被采到，59/100目标示范帧覆盖完整。它不是机械重复旧6000缓存。完整实际消费CSV与准备序列SHA256逐字节相同：`{inputs['consumed_trace_sha256']}`；每个输入的原始payload hash和episode/frame身份在读取时均核对。

沿用官方RLDS源/目标transition混合权重0.05:1、源归一化与VQ；归一化统计与P2逐值相同。本轮显式使用全train split（100目标/1800源），P2默认95%，同时batch和更新数均增加，因此不能把效果单独归因为训练步数。这仍是 **增强预算的受限LoRA，只有官方参考768000样本位置的1/8**，并非完整标准微调。仅训练Joint Reps。

## 最终checkpoint和真实加载评测

- 原始checkpoint：`{provenance['source_checkpoint']}`，SHA256 `{provenance['source_checkpoint_sha256']}`。
- 最终adapter权重：`{provenance['adapter']}/adapter_model.safetensors`，SHA256 `{provenance['adapter_sha256']}`。
- 实际评测的合并checkpoint：`{provenance['merged_checkpoint']}`，SHA256 `{provenance['merged_checkpoint_sha256']}`。

评测job中的路径/文件hash、合并provenance和training suite已强制核对一致；它不是仅写出了脚本。最终模型20次视频/状态全部校验，稳定接触抬升为{stable}/20，官方成功为{k}/20，两者分开报告。阶段观测为：**{stage_description}**。未发闭爪命令是实际执行层观测，不能未经进一步验证就断言神经网络没有抓取意图或表达能力。

[每步原始损失与优化器统计](../evidence/p3/execution/training/) · [实际参数变化](../evidence/p3/execution/parameter_changes.json) · [输入及示范覆盖](../evidence/p3/execution/data/) · [同条件救回/退化](../evidence/p3/execution/paired_cases.csv) · [完整权重路径/大小/SHA256](../evidence/p3/execution/model_artifacts.csv) · [训练曲线](../evidence/p3/execution/plots/)。

## 技术恢复与可解释性限制

已保留两次技术恢复证据：96000缓存写完后覆盖CSV字段不一致，仅恢复统计而未重采样；追踪器首次接入受上游argv重写影响，0更新/0样本即失败，修复固定run label后batch8探针及正式训练成功。这些不是资源OOM，也不是任务0%成功。所有原始日志保留服务器。

目标数据与40–59检查未发现可检测的精确物理状态重叠，但缺少逐示范XML/关节排序，不能穷尽证明轨迹独立。单训练seed、20个新条件的结果也不能证明充分标准适配已经穷尽。
'''
    (PROJECT/'docs/P3_STRONGER_ADAPTATION.md').write_text(strong)
    decision=f'''# P3科研判断：停止当前COFT新网络分支

本轮已实际完成24次冻结开发干预、96000样本位置Joint LoRA训练及100次独立物理评测。结论是 **不进入新的T-RO算法设计**：已有物理恢复能力与普通适配结果尚未建立标准方法无法解决的具体缺口。

## 独立40–59的全部结果

{table}

新96000模型相对原始Panda救回{rescue}、退化{regression}。完整的原始condition ID、真实seed、模型/状态hash及所有Panda方法配对记录见[冻结身份](../evidence/p3/frozen/independent_conditions_40_59.json)、[配对结果](../evidence/p3/execution/paired_cases.csv)。不以文件trial000–019替代原官方索引40–59。

新模型阶段观测：**{stage_description}**。同批UR5e成功条件为{source_successful}，其中{persistent}在全部四个Panda方法中仍未完成；这是可复核的持续失败集合，尚不是标准方法不可解证明。

{assessment}

每格仅20次、一次训练seed，成功率区间较宽；未见显著差异不代表方法等效。UR5e与Panda按官方索引及真实seed对应，同Panda方法的初始状态逐元素相同。跨机器人对象资产相同、设施位置相同，物体存在小量settling差异；原始元数据中两条类别成员顺序差异单独记录，原始条件及验证hash没有修改。[场景核查](../evidence/p3/execution/cross_embodiment_scene_audit.json)

## 六个必须回答的问题

1. **空间还是闭爪主因？** 现有设计不能给所有失败唯一排序。A2/A3同位姿目标而夹爪调度不同，A3确实恢复抓持；但A1在22/24/37/39已连续双指接触211–256状态而不抬升，原臂后续动作/反馈也是混淆。29/34无接触、26仅瞬态接触，空间对齐是有效恢复手段，不等于已证明唯一必要因素。
2. **A3真实恢复多少？** 稳定接触抬升8/8、官方完整任务5/8；相对已知B0救回七次稳定抓持、四次完整任务并保留阳性35。A1为1/8抓持、1/8任务；A2为0/8、0/8。八例是已选开发案例，特权物体位姿加标准OSC不是公平RGB新算法。
3. **增强Joint LoRA成功率？** 新40–59为{k}/20（{k/20:.0%}），真实{updates}次更新、96000输入；只使用冻结末步checkpoint，无测试集选模。
4. **普通控制/适配能否解释差距？** 标准几何迁移、接近和闭合已经证明这八例的物理抓取可恢复；{assessment} 不能把开发集特权5/8与独立RGB评测直接排名，也不能把1/8曝光的LoRA称为完整微调。
5. **有什么标准方法不能解决的可复现问题？** 本轮没有证明这样的缺口。可复核的剩余现象是A3的22/29/39抓持后任务失败，以及新模型失败阶段`{json.dumps(stages,ensure_ascii=False)}`。A3固定路径存在速度/步幅上限导致的跟踪滞后，没有近奇异或限位证据，不能据此排除标准轨迹控制。RGB物体配准、视觉伺服和接触意图解码强基线也尚未被实测排除。
6. **是否进入T-RO新算法？** **否，停止当前具体COFT网络分支。** 不训练新网络、不在相同20条件继续调阈值追数字。保留已跑通的BARX/RoboCasa-X环境、原P0/P1/P2历史和本轮全部原始证据。

## 既有方法与创新边界

物体中心操作片段的空间迁移、拼接和连续执行已有[MimicGen原论文](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf)先例；不能将本轮A3包装成独立创新。[RoboTAP官方项目](https://robotap.github.io/)的点跟踪视觉伺服构成RGB标准方法先例，其公开4D设置也不能未经实测就等同于本任务全部6D要求。[ReKep原论文](https://arxiv.org/abs/2409.01652)的3D关键点时空约束与SE(3)优化使用RGB-D，不能直接冒称同RGB信息的公平对照。“空间＋时间”概念本身不足以支持新算法立项。

## 完整性与限制

初始协议在f6bf766公开；A1保持时长的公开修订4cc1129发生在A类运行前、尚未查看新40–59结果时，当时两个公平基线已启动。五格正式评测和训练规则不变，未在看结果后调参。可选特权新20次已在协议中选择不做，本轮实际预算就是24开发＋100正式，没有冒算或遗漏。

{validation['videos_verified']}个视频的全部{validation['decoded_frames']}帧已解码核对，状态长度、有限性和冻结条件核验通过。视频/权重/HDF5保留服务器，见[124次轨迹索引](../evidence/p3/execution/all_rollouts.csv)、[完整文件大小/SHA256](../evidence/p3/execution/external_files.csv)与[checkpoint索引](../evidence/p3/execution/model_artifacts.csv)。训练与示范重叠审计的元数据限制仍保留，不能宣称已做穷尽独立性证明。
'''
    (PROJECT/'docs/P3_SCIENTIFIC_DECISION.md').write_text(decision)
    status=dict(status='P3_COMPLETE',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_directive='202611091028.md',
        initial_protocol_commit='f6bf766440b4606ee697ef209ffd902d6b4dc1b7',amended_diagnostic_protocol_commit='4cc1129a2a9ed0732b1320f17fba280da271306c',
        p3_a=dict(status='COMPLETE',physical_rollouts=24,summary=diag['summary'],A1_post_contact_arm_behavior_is_a_confound=True),
        p3_b=dict(status='COMPLETE',actual_optimizer_updates=updates,batch=batch,actual_sample_positions=96000,completed_backwards=execution['completed_backwards'],nonfinite_losses=0,consumed_trace_sha256=inputs['consumed_trace_sha256'],changed_adapter_tensors=parameters['changed_tensors'],total_adapter_tensors=parameters['total_adapter_tensors'],checkpoint=provenance,exposure_fraction_of_official=0.125,data_manifest=manifest,target_positions_by_direction=dict(direction_positions)),
        p3_c=dict(status='COMPLETE',official_indices=list(range(40,60)),formal_episodes=100,cells=list(cells.values()),stronger_model_successes=k,stronger_model_episodes=20,stronger_model_success_rate=k/20,comparisons=comparisons),
        scientific_decision='NO_GO_CURRENT_COFT_NETWORK_NO_PROVEN_GAP_BEYOND_STANDARD_METHODS',execution_root=str(ROOT),validation=validation)
    (PROJECT/'evidence/p3/status.json').write_text(json.dumps(status,indent=2)+'\n')
    readme=f'''# P3实际完成证据

24次开发干预＋96000样本位置训练＋100次独立40–59物理评测全部完成。新Joint LoRA为{k}/20；A3稳定抓持8/8、官方完整任务5/8。当前决定：停止当前COFT新网络分支，尚未证明标准方法不能解决的缺口。

- [五格结果](cells.csv)、[100条逐次统计](episodes.csv)、[全配对救回/退化](paired_comparisons.csv)、[逐条件阶段变化](paired_cases.csv)、[阶段测量](formal_stage_measurements.csv)、[阶段定义与限制](stage_measurement_definitions.json)。
- [24条诊断](diagnostic_results.csv)、[分支/世界目标](diagnostic_cases/)、[A2/A3完全相同目标核查](A2_A3_matched_world_goals.json)。A类是特权开发诊断，不能混入公平排名。
- [124次完整轨迹路径/大小/SHA256](all_rollouts.csv)、[全部服务器文件索引](external_files.csv)、[权重及优化器状态索引](model_artifacts.csv)、[全部视频解码与状态有限性核验](validation/summary.json)、[同Panda初态完全相同核验](validation/panda_matched_initial_states.json)。
- [训练尝试](training_attempts.csv)、[每步真实损失/更新/输入记录](training/)、[初始与最终参数变化](parameter_changes.json)、[实际评测checkpoint来源](adapter_provenance.json)。
- [96k采样来源/重复/覆盖](data/complete.json)、[全部原始示范覆盖](data/demonstration_coverage.csv)、[100条目标原始demo映射](data/original_target_demonstrations.json)、[完整96k序列索引](data/sample_trace_index.json)。
- [冻结源接触模板](../frozen/source_contact_bank.json)、[官方40–59身份及真实seed/hash](../frozen/independent_conditions_40_59.json)、[全bundle验证](../frozen/official_condition_bundle_integrity.json)、[跨机器人场景核查](cross_embodiment_scene_audit.json)。
- [诊断/逐条件/训练曲线](plots/)、[雅可比核验](jacobian_validation.json)、[局部条件数](recorded_kinematic_diagnostics.csv)、[OSC跟踪速度分析](osc_tracking_budget.csv)。后两项只读已有状态，没有增加仿真或调参。

服务器根目录：`{ROOT}`。新独立评测原始JSONL：`formal/<cell>/official/**/episodes.jsonl`；状态：`formal/<cell>/telemetry/episode-000.h5`至`episode-019.h5`，对应原官方索引40–59。诊断：`diagnostics/condition-XXX/A1|A2|A3/`。完整缓存：`/home/users/lemon/coft-p3/matched-samples-96000/`。训练初始/最终adapter、完整消费CSV和原始JSONL日志均在服务器，精确位置见索引。

官方success与接触/抬升/运输/开爪/退离测量分开。单模板不是合法抓取全集；A1原臂缺少有效抬升，不能将所有差异唯一归为空间。目标示范与测试状态未发现可检测重叠，缺少逐demo XML的审计限制仍然保留。

[因果报告](../../../docs/P3_SPATIOTEMPORAL_CAUSAL_REPORT.md) · [实际训练报告](../../../docs/P3_STRONGER_ADAPTATION.md) · [六问科研判断](../../../docs/P3_SCIENTIFIC_DECISION.md) · [协议修订](../../../docs/P3_PROTOCOL_AMENDMENT_01.md)。
'''
    (out/'README.md').write_text(readme)
    path=PROJECT/'README.md';text=path.read_text();lines=text.splitlines()
    replacement=f'> **P3真实完成。** 24次开发物理干预，A3稳定抓持8/8、完整任务5/8；Joint LoRA完成{updates}次更新、96000样本位置；独立40–59五格100次评测全部完成，新模型{k}/20。未证明标准方法无法解决的缺口，停止当前COFT新网络分支。见 [因果报告](docs/P3_SPATIOTEMPORAL_CAUSAL_REPORT.md)、[实际适配](docs/P3_STRONGER_ADAPTATION.md)、[六问科研判断](docs/P3_SCIENTIFIC_DECISION.md)、[完整证据与大文件索引](evidence/p3/execution/README.md)。'
    matches=[i for i,line in enumerate(lines) if line.startswith('> **P3')];assert len(matches)==1
    lines[matches[0]]=replacement;path.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='REPORTS_COMPLETE',new_successes=k,new_episodes=20,actual_updates=updates,actual_sample_positions=96000,decision=status['scientific_decision']),indent=2))


if __name__=='__main__':main()
