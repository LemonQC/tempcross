# T-RO Strong Baseline Master Matrix

当前安排：Octo-Small → SmolVLA → Flow Matching Policy，单卡顺序执行。只对当前方法推进训练。
CrossZero与Cross-Behavior Object Flow移出必须复现名单；历史BLOCKED代码/报告不变。KITE保留候选，暂缓其运动学解码器。
允许必要的普通工程修复；一次Python错误不再触发永久BLOCKED。不得更换核心算法、放宽成功判据或无限扩大GPU预算。

既有BC/DP/BARX1280、BARX适配160、ACT/VQ-BeT及MimicGen v2均不重跑。MimicGen v2保留48000更新和80条评测。
正式条件60–99一律 **EXPOSED_REFERENCE**。未执行不填0%成功率。原任务、双指夹爪、20Hz、canonical OSC7→12及官方placement AND retreat不变。

| 方法 | 当前研究角色 | 实际状态 | 正式训练更新 | 已验收正式评测 | 报告 |
|---|---|---|---:|---:|---|
| act | 完成保留，不重跑 | COMPLETE_FAITHFUL_PORT | 48000 | 160 | [证据](../results/strong_baselines/act/RESULTS.md) |
| vqbet | 完成保留，不重跑 | COMPLETE_FAITHFUL_PORT | 60000 | 160 | [证据](../results/strong_baselines/vqbet/RESULTS.md) |
| mimicgen_v2 | 保留有界组件，不再生成 | PARTIAL | 48000 | 80 | [证据](../results/strong_baselines/mimicgen_v2/RESULTS.md) |
| kite | 候选暂缓；旧报告保留 | BLOCKED | 0 | 0 | [证据](../results/strong_baselines/kite/RESULTS.md) |
| crosszero | 归档，无新增复现预算 | BLOCKED | 0 | 0 | [证据](../results/strong_baselines/crosszero/RESULTS.md) |
| object_flow | 归档，无新增复现预算 | BLOCKED | 0 | 0 | [证据](../results/strong_baselines/object_flow/RESULTS.md) |
| octo_small | 联合设置已验收，禁止重跑 | COMPLETE_FAITHFUL_PORT | 24000 | 80 | [证据](../results/strong_baselines/octo_small/RESULTS.md) |
| smolvla | 本轮唯一执行候选 | PARTIAL | 12000 | 0 | [证据](../results/strong_baselines/smolvla/status.json) |
| flow_matching_policy | 后续候选 | NOT_RUN | 0 | 0 | [证据](../results/strong_baselines/flow_matching_policy/status.json) |

COMPLETE_OFFICIAL、COMPLETE_FAITHFUL_PORT、COMPLETE_REIMPLEMENTATION只用于完成训练和正式完整任务评测的对应范围。PARTIAL/BLOCKED/NOT_RUN不是完整Baseline。
KITE旧BLOCKED是历史尝试状态，不代表当前候选已被永久否定。归档方法不计入新的必须凑齐数量。

## 真实正式评测

| 方法/设置 | 任务 | 官方成功/总数 | 范围 |
|---|---|---:|---|
| act/target_only | pnp_counter_to_sink | 5/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| act/target_only | pnp_sink_to_counter | 4/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| act/source_target | pnp_counter_to_sink | 8/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| act/source_target | pnp_sink_to_counter | 2/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| vqbet/target_only | pnp_counter_to_sink | 0/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| vqbet/target_only | pnp_sink_to_counter | 0/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| vqbet/source_target | pnp_counter_to_sink | 0/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| vqbet/source_target | pnp_sink_to_counter | 0/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| mimicgen_v2/human_plus_generated | pnp_counter_to_sink | 2/40 | PARTIAL/EXPOSED_REFERENCE |
| mimicgen_v2/human_plus_generated | pnp_sink_to_counter | 0/40 | PARTIAL/EXPOSED_REFERENCE |
| octo_small/source_target | pnp_counter_to_sink | 0/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |
| octo_small/source_target | pnp_sink_to_counter | 0/40 | COMPLETE_FAITHFUL_PORT/EXPOSED_REFERENCE |

## 信息和计算边界

每方法保留官方版本/许可证、预训练、目标示范数量、额外观测监督、训练样本暴露与checkpoint选择记录。不得把多相机/机器人状态输入称为RGB-only等输入比较。
Octo-Small当前主设置为UR5e+Panda联合微调，使用原540/60源、90/10目标完整示范划分。主RGB256和语言是原生支持的输入子集；缺失腕部相机按官方padding mask处理，未引入额外机器人状态。其输入分辨率与T5/Octo预训练区别于旧RGB128模型。
模型真实更新和完整任务smoke通过后才冻结正式训练预算与验证选模规则。当前阶段不得把安装成功或仅forward称为完整Baseline。
Octo-Small联合设置已完成24000正式更新、80正式评测：两任务均0/40。另有6次丢弃探针更新和2条smoke；target-only未运行。零成功率来自实际完整任务评测。
SmolVLA本轮单独使用RGB+关节/夹爪状态信息条件，保持原canonical OSC7动作。原HDF5的joint_states经核对是速度，不能误作位置；状态适配核对真实保存qpos和部署命名关节读数。该候选不得称为RGB-only等输入比较。
MimicGen v2仍为PARTIAL有界组件：两任务新结果2/40与0/40，复用旧人类BC各1/40，总数均2/80。
VQ-BeT的60000阶段迭代对应84000次正式optimizer调用，探针/旧技术尝试成本另列。

[上一轮完整性审计（历史）](https://github.com/LemonQC/tempcross/blob/8b274c9a014dccab2f8ddc364b64f10667cc8e2d/docs/STRONG_BASELINE_MASTER_MATRIX.md)；[当前范围](../results/strong_baselines/research_scope.json)。
