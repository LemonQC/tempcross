"""Current official-candidate scope; historical method artifacts stay immutable."""
from baselines.strong.common import PROJECT,out,read


def main():
    scope=read(PROJECT/'results/strong_baselines/research_scope.json')
    methods=['act','vqbet','mimicgen_v2','kite','crosszero','object_flow']+scope['ordered_candidates']
    roles={'act':'完成保留，不重跑','vqbet':'完成保留，不重跑','mimicgen_v2':'保留有界组件，不再生成',
        'kite':'候选暂缓；旧报告保留','crosszero':'归档，无新增复现预算','object_flow':'归档，无新增复现预算',
        'octo_small':'当前优先候选','smolvla':'后续候选','flow_matching_policy':'后续候选'}
    lines=['# T-RO Strong Baseline Master Matrix','',
        f"当前轮仅执行{scope.get('current_round_only',scope['active_method'])}，完成或确认实质阻碍后停止；不自动启动后续候选。Octo-Small及已完成结果保持冻结。",
        'CrossZero与Cross-Behavior Object Flow移出必须复现名单；历史BLOCKED代码/报告不变。KITE保留候选，暂缓其运动学解码器。',
        '允许必要的普通工程修复；一次Python错误不再触发永久BLOCKED。不得更换核心算法、放宽成功判据或无限扩大GPU预算。','',
        '既有BC/DP/BARX1280、BARX适配160、ACT/VQ-BeT及MimicGen v2均不重跑。MimicGen v2保留48000更新和80条评测。',
        '正式条件60–99一律 **EXPOSED_REFERENCE**。未执行不填0%成功率。原任务、双指夹爪、20Hz、canonical OSC7→12及官方placement AND retreat不变。','',
        '| 方法 | 当前研究角色 | 实际状态 | 正式训练更新 | 已验收正式评测 | 报告 |',
        '|---|---|---|---:|---:|---|']
    for m in methods:
        s=read(out(m)/'status.json');link='RESULTS.md' if (out(m)/'RESULTS.md').exists() else 'status.json'
        if m in ['octo_small','smolvla'] and s['status']=='COMPLETE_FAITHFUL_PORT':roles[m]='联合设置已验收，禁止重跑'
        if m==scope['active_method'] and s['status'] not in ['COMPLETE_OFFICIAL','COMPLETE_FAITHFUL_PORT','COMPLETE_REIMPLEMENTATION']:roles[m]='本轮唯一执行候选'
        if m==scope['active_method'] and s['status'] in ['COMPLETE_OFFICIAL','COMPLETE_FAITHFUL_PORT','COMPLETE_REIMPLEMENTATION']:roles[m]='本轮已验收，禁止重跑'
        lines.append(f"| {m} | {roles[m]} | {s['status']} | {s.get('actual_training_updates',0)} | {s.get('physical_evaluation_episodes',0)} | [证据](../results/strong_baselines/{m}/{link}) |")
    lines += ['', 'COMPLETE_OFFICIAL、COMPLETE_FAITHFUL_PORT、COMPLETE_REIMPLEMENTATION只用于完成训练和正式完整任务评测的对应范围。PARTIAL/BLOCKED/NOT_RUN不是完整Baseline。',
        'KITE旧BLOCKED是历史尝试状态，不代表当前候选已被永久否定。归档方法不计入新的必须凑齐数量。', '',
        '## 真实正式评测','', '| 方法/设置 | 信息条件 | 任务 | 官方成功/总数 | 范围 |','|---|---|---|---:|---|']
    for m in methods:
        p=out(m)/'results.json'
        if not p.exists():continue
        r=read(p)
        for c in r.get('cells',[]):
            assert c['episodes']>0 and c.get('role','EXPOSED_REFERENCE')=='EXPOSED_REFERENCE'
            information={'octo_small':'RGB256+语言','smolvla':'RGB512+语言+qpos9','flow_matching_policy':'RGB96+CLIP768+qpos9'}.get(m,'见已冻结协议')
            lines.append(f"| {m}/{c['group']} | {information} | {c['task']} | {c['successes']}/{c['episodes']} | {r['status']}/EXPOSED_REFERENCE |")
    lines += ['', '## 信息和计算边界','',
        '每方法保留官方版本/许可证、预训练、目标示范数量、额外观测监督、训练样本暴露与checkpoint选择记录。不得把多相机/机器人状态输入称为RGB-only等输入比较。',
        'Octo-Small当前主设置为UR5e+Panda联合微调，使用原540/60源、90/10目标完整示范划分。主RGB256和语言是原生支持的输入子集；缺失腕部相机按官方padding mask处理，未引入额外机器人状态。其输入分辨率与T5/Octo预训练区别于旧RGB128模型。',
        '模型真实更新和完整任务smoke通过后才冻结正式训练预算与验证选模规则。当前阶段不得把安装成功或仅forward称为完整Baseline。',
        'Octo-Small联合设置已完成24000正式更新、80正式评测：两任务均0/40。另有6次丢弃探针更新和2条smoke；target-only未运行。零成功率来自实际完整任务评测。',
        'SmolVLA已验收的联合设置使用RGB+关节/夹爪状态信息条件，保持原canonical OSC7动作。原HDF5的joint_states经核对是速度，不能误作位置；状态适配核对真实保存qpos和部署命名关节读数。该候选不得称为RGB-only等输入比较。',
        'Flow Matching Policy采用HRI-EU官方视觉流组件：RGB96+真实qpos9+冻结CLIP768完整指令，原版GN ResNet18/条件U-Net、TorchCFM速度目标及16预测/8执行。单列信息条件；不称为原Robomimic低维/绝对动作配方或完整affordance框架复现。',
        'MimicGen v2仍为PARTIAL有界组件：两任务新结果2/40与0/40，复用旧人类BC各1/40，总数均2/80。',
        'VQ-BeT的60000阶段迭代对应84000次正式optimizer调用，探针/旧技术尝试成本另列。', '',
        '[上一轮完整性审计（历史）](https://github.com/LemonQC/tempcross/blob/8b274c9a014dccab2f8ddc364b64f10667cc8e2d/docs/STRONG_BASELINE_MASTER_MATRIX.md)；[当前范围](../results/strong_baselines/research_scope.json)。']
    (PROJECT/'docs/STRONG_BASELINE_MASTER_MATRIX.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
