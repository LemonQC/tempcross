"""Reporting only: cannot train, select a checkpoint, or run a simulation."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from common import OUT,PROJECT,ROOT,digest,read,write


def main():
    state_path=ROOT/'training/status.json'
    state=read(state_path) if state_path.exists() else dict(status='NOT_STARTED',completed_optimizer_updates=0,sample_positions=0,completed_backwards=0)
    execution=read(ROOT/'execution.json')
    frozen=read(ROOT/'frozen_publication.json')
    terminal=execution['status'] in ['COMPLETE','TECHNICAL_FAILURE']
    lines=['# BARX目标适配：实际执行结果','',
           f"执行状态：**{execution['status']}**，阶段 `{execution.get('active_stage','unknown')}`。",
           f"实际完成 **{state['completed_optimizer_updates']}/3000 次正式optimizer更新、{state['completed_backwards']} 次正式backward、{state['sample_positions']}/768000 样本位置**。",
           f"冻结提交：[{frozen['remote_commit']}](https://github.com/LemonQC/tempcross/commit/{frozen['remote_commit']})。",
           '协议与全部训练/评测控制代码在首个正式更新前已远程读回；本报告脚本只汇总实际文件，不改变训练、选模或物理评测。','',
           '训练配置为发布版Joint Reps LoRA，rank32/alpha16、AdamW 2e-5常数、BF16、batch8累积32；达到参考样本位置预算时仍不等价于论文全参数适配。单训练seed；没有新Method。','']
    if (OUT/'implementation_details.json').exists():
        detail=read(OUT/'implementation_details.json')
        lines.extend(['补充实现核查：shuffle buffer为**10000**，与两轮旧受限LoRA相同，低于上游所选配置类默认的256000。10000在训练前冻结代码中已固定，初版协议JSON未单列，此处补录，未改变执行配置。',
                      '实际采用BOUNDS_Q99；生效的q01、q99、mask及用于零维处理的min/max，与发布先验逐值一致。未采用的mean/std存在约3.28e-7/6.56e-6的数值差异，没有据此修改既有统计。',
                      '验证包在正式训练前保存并逐次核对SHA256；准备阶段的TensorFlow RNG状态未单独保存，精确重放以保留的验证pickle及其索引为准，不承诺仅靠seed7重新生成相同验证包。详见 `implementation_details.json`。',''])
    old=read(OUT/'baseline_validation.json')
    lines.extend(['旧模型固定离线验证如下。Panda的320帧来自10条示范；旧6000适配实际见过其中1条，旧96000见过全部10条，因此旧adapter分数不当成干净未见验证。UR5e验证也可能在源pretraining中被见过，只诊断新增适配的遗忘。','',
                  '| 旧模型 | Panda两任务平均action-response NLL | UR5e两任务平均NLL |',
                  '| --- | ---: | ---: |'])
    for label in ['original_joint','restricted_joint_6000','restricted_joint_96000']:
        target=next(r for r in old if r['model_label']==label and r['tag']=='validation_target')
        source=next(r for r in old if r['model_label']==label and r['tag']=='validation_source_ur5e')
        lines.append(f"| {label} | {target['macro_task_response_nll']:.6f} | {source['macro_task_response_nll']:.6f} |")
    lines.extend(['','NLL为teacher-forced action-response（含stop token），不是闭环成功率。不得据离线loss宣称任务成功。',''])
    if (OUT/'selection.json').exists():
        selection=read(OUT/'selection.json')
        lines.extend([f"验证选中 **step {selection['step']}**，Panda两任务平均NLL **{selection['macro_task_response_nll']:.6f}**。实际总训练更新数仍按上文报告，不把选中的较早step当成训练只进行了该步数。",
                      f"**被物理评测的模型自身经历 {selection['step']} 次更新、{selection['step']*256} 个样本位置**；768000是完整训练运行的预算，不能写成该较早checkpoint已见过768000位置。候选仅为500/1000/1500/2000/2500/3000，按预定验证指标选模，未使用物理成功率。",''])
        if (OUT/'selected_checkpoint_exposure.json').exists():
            exposure=read(OUT/'selected_checkpoint_exposure.json')
            lines.extend([f"选中模型输入前缀的实际域/任务计数：`{json.dumps(exposure['dataset_task_positions'])}`；唯一示范数：`{json.dumps(exposure['unique_demonstrations'])}`。完整运行与部署checkpoint的暴露分别见 `consumption.json` 和 `selected_checkpoint_exposure.json`。",''])
    if execution['status']=='COMPLETE':
        result=read(OUT/'results.json');assert result['status']=='COMPLETE' and result['new_evaluation_episodes']==160
        assert state['completed_optimizer_updates']==3000 and state['sample_positions']==768000
        cells=result['cells']
        lines.extend(['已完成 **160条新增物理rollout**。均为已暴露60–99参考条件，不能宣称独立未见测试；旧1280次实验零重跑。RoboCasa-X环境、两任务、动作接口、20Hz、600/650步上限和官方放置＋退离成功定义均未修改。','',
                      '| 模型 / 已暴露60–99 | C→S Panda | S→C Panda | C→S UR5e | S→C UR5e |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for label in ['A_barx_joint','C_barx_joint_96000','new_joint_adaptation_768000']:
            values=[]
            for task,robot in [('pnp_counter_to_sink','panda'),('pnp_sink_to_counter','panda'),('pnp_counter_to_sink','ur5e'),('pnp_sink_to_counter','ur5e')]:
                rr=[r for r in cells if r['label']==label and r['task']==task and r['robot']==robot]
                values.append(f"{rr[0]['successes']}/{rr[0]['episodes']} ({100*rr[0]['success_rate']:.1f}%)" if rr else '未评测')
            shown=(f"新适配（运行768k；选中checkpoint {selection['step']*256//1000}k）" if label=='new_joint_adaptation_768000' else label)
            lines.append('| '+shown+' | '+' | '.join(values)+' |')
        lines.extend(['','旧Joint6000在不同20–39条件为Counter→Sink Panda 3/20；不与本表当成同条件收益。数据划分、暴露预算及有效batch同时改变，当前对照不能单独归因某一个因素。','',
                      '同条件逐条变化如下；“救回”为旧失败/新成功，“退化”为旧成功/新失败。它们是固定已暴露条件下的描述，不是严格因果配对或未见测试。','',
                      '| 历史对照 | 任务 / 机器人 | 救回 | 退化 | 均成功 | 均失败 |',
                      '| --- | --- | ---: | ---: | ---: | ---: |'])
        pair_groups=defaultdict(list)
        for pair in csv.DictReader((OUT/'paired_cases.csv').open()):
            pair_groups[pair['baseline'],pair['task'],pair['robot']].append(pair)
        for (baseline,task,robot),rr in sorted(pair_groups.items()):
            both_success=sum(r['baseline_success']=='True' and r['candidate_success']=='True' for r in rr)
            both_failure=sum(r['baseline_success']=='False' and r['candidate_success']=='False' for r in rr)
            rescued=sum(r['rescued']=='True' for r in rr);regressed=sum(r['regressed']=='True' for r in rr)
            assert both_success+both_failure+rescued+regressed==len(rr)
            lines.append(f'| {baseline} | {task} / {robot} | {rescued} | {regressed} | {both_success} | {both_failure} |')
        target=[r for r in cells if r['new_physical_run'] and r['robot']=='panda']
        target_success=sum(r['successes'] for r in target);target_n=sum(r['episodes'] for r in target)
        lines.extend(['',f'本配方在Panda两方向合计官方成功 **{target_success}/{target_n}**。逐任务结果应分别判断，不能用验证NLL下降替代闭环收益，也不能把增加暴露预算直接表述为跨具身问题已解决。','',
                      '| 新模型 | 官方成功 | 发出闭爪 | 稳定抬升代理 | 失败阶段计数 |',
                      '| --- | ---: | ---: | ---: | --- |'])
        for r in cells:
            if not r['new_physical_run']:continue
            failures={k:v for k,v in r['failure_counts'].items() if k!='SUCCESS'}
            lines.append(f"| {r['task']} / {r['robot']} | {r['successes']}/{r['episodes']} | {r['issued_close']} | {r['stable_lift']} | {json.dumps(failures,ensure_ascii=False)} |")
        counter=next(r for r in target if r['task']=='pnp_counter_to_sink')
        sink=next(r for r in target if r['task']=='pnp_sink_to_counter')
        lines.extend(['',f"关键观察：新Panda Counter→Sink仍有{counter['failure_counts'].get('NO_CLOSE_COMMAND',0)}/40条全程未发闭爪；Sink→Counter则有{sink['failure_counts'].get('CLOSE_WITHOUT_BILATERAL_CONTACT',0)}/40条发闭爪但未形成双侧接触。此次适配未消除方向相关的启动/接触阶段缺口；阶段统计不能单独识别视觉、运动历史或动作预测的根因。",
                      '本轮唯一建议是保留此已完成的参考预算LoRA作为Baseline，如实报告收益与退化，在此停止本阶段。不能把这些结果包装成新算法必要性的证明，也不能据此否定尚未实现的充分全参数适配。'])
        consumption=read(OUT/'consumption.json')
        lines.extend(['','失败阶段统一重读新旧原始H5/接触JSONL，含指尖及指垫。稳定抬升仅为接触＋body原点抬升3cm持续10状态的代理；不能充作官方成功或物理根因。完整逐条结果和救回/退化集合在 `episodes.csv`、`paired_cases.csv`。','',
                      f"实际输入域/任务计数：`{json.dumps(consumption['counts'],ensure_ascii=False)}`；唯一示范数：`{json.dumps(consumption['unique_demonstrations'])}`；唯一帧位置{consumption['unique_frame_positions']}。实际读取trace逐条检查均来自本轮train episode，无验证示范输入。",
                      f"正式训练用时{state.get('elapsed_seconds',0)/3600:.3f}小时；GPU分配/预留峰值{state.get('peak_gpu_allocated_bytes',0)}/{state.get('peak_gpu_reserved_bytes',0)} bytes。训练和模型选择均已完成，原物理账本哈希保持不变。",''])
        model=read(OUT/'model_index.json')
        lines.extend(['选中模型及大文件：','',f"- 原权重：`{model['source_checkpoint']}`，SHA256 `{model['source_sha256']}`。",
                      f"- 选中adapter：`{model['adapter']}`，SHA256 `{model['adapter_sha256']}`。",
                      f"- 实际物理评测合并权重：`{model['merged_checkpoint']}`，SHA256 `{model['merged_sha256']}`。",
                      '- 原始H5、视频、完整训练trace及验证结果的绝对路径与SHA256见 `artifact_index.json`、`physical_jobs.json`、`consumption.json`；大型文件留在服务器。','',
                      '这些结果只回答此固定训练配方的适配表现；不能据单seed、已暴露条件或LoRA路线排除充分全参数适配。本阶段仅交付BARX适配Baseline，不据此提出新算法或启动其他方法。'])
    elif execution['status']=='TECHNICAL_FAILURE':
        failed=execution['stages'][-1] if execution['stages'] else {}
        lines.extend([f"执行因技术限制停止：`{json.dumps(failed,ensure_ascii=False)}`。",
                      f"训练状态中的错误：`{state.get('error','见相应阶段原始日志')}`。",
                      '没有把未完成训练写成3000更新，没有把未运行评测写成0%成功；不自动改batch、预算或算法。尚未产生的交付项明确保留为未完成。'])
    else:
        lines.extend(['**本轮正式训练/后续流水线仍在执行，尚不能给出新模型的官方完整任务成功率。**',
                      '完成预定训练后，将按冻结验证规则选一个checkpoint，顺序合并、运行160条新增参考条件、验证原始H5和视频，再汇总失败类别。当前不把计划预算或历史成功数写成新结果。'])
    lines.extend(['','完整协议：[BARX_TARGET_ADAPTATION_PROTOCOL.md](BARX_TARGET_ADAPTATION_PROTOCOL.md)。原始运行目录：`/home/users/lemon/barx-target-adaptation-20261110/`。'])
    if (OUT/'figures/training_and_validation.svg').exists():
        lines.extend(['','[训练与验证曲线](../results/strong_baselines/barx_adaptation/figures/training_and_validation.svg)。'])
    if (OUT/'figures/exposed_reference_success.svg').exists():
        lines.extend(['','[官方成功率对照图](../results/strong_baselines/barx_adaptation/figures/exposed_reference_success.svg)；误差棒为各40条条件的描述性Wilson区间，不作为独立未见泛化或因果效应检验。'])
    if (OUT/'delivery_checks.json').exists():
        checks=read(OUT/'delivery_checks.json')
        lines.extend(['',f"交付核查：`{checks['status']}`；从160条新H5重算官方放置＋退离，核对640条新旧逐条记录与各cell/失败计数。`raw_rollout_index.json`保留全部新H5、接触JSONL和视频的绝对路径、字节数及SHA256。该核查没有策略推理或物理step。"])
        if checks['after_close_flags_on_episodes_without_close']:
            lines.extend(['','派生字段限制：部分未发闭爪的轨迹存在开放夹爪双侧接触，冻结collector将其计入`bilateral_contact_after_close`。这些条目详见`delivery_checks.json`，该字段在这些条目上只能读作“曾双侧接触”；失败阶段仍按`NO_CLOSE_COMMAND`优先，官方成功次数不受影响。'])
    (PROJECT/'docs/BARX_TARGET_ADAPTATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'live_execution.json',dict(execution=execution,training=state,terminal=terminal))
    if terminal:
        write(OUT/'frozen_publication.json',frozen)


if __name__=='__main__':main()
