# Strong Baseline Master Matrix

研究基准：main。TOCAT 独立分支保留，不用于训练。原 BC/DP/BARX 1280 条及 BARX 适配 160 条评测不重跑。

正式条件 60–99 均标为 **EXPOSED_REFERENCE**，不是独立未见测试。未评测写 N/A，不以 0% 代替。

| 方法 | 状态 | 已验证训练更新（不含探针） | 已验收正式物理评测 | 成功率 | 当前证据 |
|---|---|---:|---:|---|---|
| act | COMPLETE_FAITHFUL_PORT | 48000 | 160 | 分设置见下表 | [状态](../results/strong_baselines/act/status.json) |
| vqbet | COMPLETE_FAITHFUL_PORT | 60000 | 160 | 分设置见下表 | [状态](../results/strong_baselines/vqbet/status.json) |
| mimicgen_v2 | PARTIAL | 48000 | 80 | 0.025 | [状态](../results/strong_baselines/mimicgen_v2/status.json) |
| kite | BLOCKED | 0 | 0 | N/A | [状态](../results/strong_baselines/kite/status.json) |
| crosszero | BLOCKED | 0 | 0 | N/A | [状态](../results/strong_baselines/crosszero/status.json) |
| object_flow | BLOCKED | 0 | 0 | N/A | [状态](../results/strong_baselines/object_flow/status.json) |

状态含义：COMPLETE_OFFICIAL 为核心与原生设置完整复现；COMPLETE_FAITHFUL_PORT 为核心保留、已声明平台/接口适配并完成训练及官方完整任务评测；COMPLETE_REIMPLEMENTATION 为已声明的忠实重实现；PARTIAL 为尚未闭环；BLOCKED 为具体阻碍已记录；NOT_RUN 为尚未执行。

执行顺序：ACT → VQ-BeT → MimicGen v2 → KITE → CrossZero → Object Flow。单卡每次只运行一个大型训练；每种算法完成或阻碍报告后单独提交。

原始50+50生成示范门槛失败的MimicGen结论保持不变；mimicgen_v2为单独协议。不同观测、目标示范量及额外监督在各方法协议中分列，不作等信息或等数据暗示。

## 按设置与任务分列的真实评测

下表只读取已经存在的结果单元；未执行单元不填0%，不同设置不合并伪装成一个模型。

| 方法 | 设置 | 任务 | 官方成功/总次数 | 成功率 | 结果作用域 |
|---|---|---|---:|---:|---|
| act | target_only | pnp_counter_to_sink | 5/40 | 12.5% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| act | target_only | pnp_sink_to_counter | 4/40 | 10.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| act | source_target | pnp_counter_to_sink | 8/40 | 20.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| act | source_target | pnp_sink_to_counter | 2/40 | 5.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| vqbet | target_only | pnp_counter_to_sink | 0/40 | 0.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| vqbet | target_only | pnp_sink_to_counter | 0/40 | 0.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| vqbet | source_target | pnp_counter_to_sink | 0/40 | 0.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| vqbet | source_target | pnp_sink_to_counter | 0/40 | 0.0% | COMPLETE_FAITHFUL_PORT / EXPOSED_REFERENCE |
| mimicgen_v2 | human_plus_generated | pnp_counter_to_sink | 2/40 | 5.0% | PARTIAL / EXPOSED_REFERENCE |
| mimicgen_v2 | human_plus_generated | pnp_sink_to_counter | 0/40 | 0.0% | PARTIAL / EXPOSED_REFERENCE |

## 数据和信息条件

数据量、预训练、目标示范、原生传感器及额外监督必须随各方法分列；不默认等数据或等信息。

| 方法 | 本轮源具身 | 部署观测（未完成者仅列原生需求） | Panda训练示范/任务 | 协议 |
|---|---|---|---|---|
| act | ["ur5e"] | ["128x128 RGB", "frozen CLIP instruction embedding"] | 45 | [冻结协议](../results/strong_baselines/act/protocol.json) |
| vqbet | ["ur5e"] | ["128x128 RGB, resized nearest to224 for native frozen ResNet18", "frozen CLIP-L instruction768, learned prefix projection"] | 45 | [冻结协议](../results/strong_baselines/vqbet/protocol.json) |
| mimicgen_v2 | ["panda_seed_demonstrations_for_generation"] | ["128x128 RGB", "frozen CLIP-L instruction"] | 45 | [冻结协议](../results/strong_baselines/mimicgen_v2/protocol.json) |
| kite | ["ur5e"] | 未部署；原生需求为object-only点云、上一intent、实测configuration及运动学模型；不同于RGB-only信息条件 | 0 | [未冻结边界](../results/strong_baselines/kite/protocol.json) |
| crosszero | ["ur5e"] | 未部署；原生robot-masked XYZRGB点云、EE状态、关节状态；训练另需物体功能位姿监督 | 0 | [未冻结边界](../results/strong_baselines/crosszero/protocol.json) |
| object_flow | ["ur5e"] | 未部署；论文原生agent/hand RGB、EE/gripper状态、指令；流只可由学习预测器产生 | 0 | [未冻结边界](../results/strong_baselines/object_flow/protocol.json) |

旧BC/DP源池包含IIWA、Kinova3和UR5e。新增ACT及VQ-BeT依本轮UR5e/Panda设置仅取UR5e的600条源示范，保留原有身份划分；旧结果不重训、不覆盖，也不把跨源池比较称为等数据预算比较。BARX预训练/适配及MimicGen生成监督的来源仍以各自原协议为准。

ACT唯一一次decoder输出修复及原版24000更新尝试另列归档。VQ-BeT唯一一次TF32精度修复发生在正式训练前，原四次组件探针更新和四条smoke保留，修复版独立目录重新验收。不能用原技术尝试替代修复版训练完成量；技术尝试、探针和正式评测成本分别记账。

## 完整性边界

六项中完整Baseline为2/6；PARTIAL为1项；BLOCKED为3项。只要未达到6/6，就不能称六项Baseline已补齐。

MimicGen v2的有界下游组件完成仍保持PARTIAL；其新策略总成功数2/80，与复用旧人类BC的2/80相同，不能用单一任务的计数变化声称总体增益。

VQ-BeT表中的60000为两阶段训练迭代数，对应84000次正式optimizer调用；探针和废弃技术尝试在各自资源账本另列，不混成相同计算预算。

| 未完成方法 | 已记录阻碍 | 结论边界 |
|---|---|---|
| kite | 原生扩展已可用；一次修复后，数据检查器的导入路径错误使目标几何/解码部署未闭环 | [BLOCKED报告](../results/strong_baselines/kite/RESULTS.md)；N/A成功率，不是算法失败 |
| crosszero | 未核实作者实现/权重；固定OSC下的原生joint-IK部署桥接未建立 | [BLOCKED报告](../results/strong_baselines/crosszero/RESULTS.md)；N/A成功率，不是算法失败 |
| object_flow | 一次启动修复后遇到保存相机名与重建模型不一致；尚无认证流标签、学习预测器或动作连接 | [BLOCKED报告](../results/strong_baselines/object_flow/RESULTS.md)；N/A成功率，不是算法失败 |
