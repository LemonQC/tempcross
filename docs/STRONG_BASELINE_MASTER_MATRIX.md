# Strong Baseline Master Matrix

研究基准：main。TOCAT 独立分支保留，不用于训练。原 BC/DP/BARX 1280 条及 BARX 适配 160 条评测不重跑。

正式条件 60–99 均标为 **EXPOSED_REFERENCE**，不是独立未见测试。未评测写 N/A，不以 0% 代替。

| 方法 | 状态 | 已验证训练更新（不含探针） | 已验收正式物理评测 | 成功率 | 当前证据 |
|---|---|---:|---:|---|---|
| act | COMPLETE_FAITHFUL_PORT | 48000 | 160 | 分设置见下表 | [状态](../results/strong_baselines/act/status.json) |
| vqbet | PARTIAL | 30000 | 0 | N/A | [状态](../results/strong_baselines/vqbet/status.json) |
| mimicgen_v2 | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/mimicgen_v2/status.json) |
| kite | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/kite/status.json) |
| crosszero | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/crosszero/status.json) |
| object_flow | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/object_flow/status.json) |

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

## 数据和信息条件

数据量、预训练、目标示范、原生传感器及额外监督必须随各方法分列；不默认等数据或等信息。

| 方法 | 本轮源具身 | 部署观测（冻结协议） | Panda训练示范/任务 | 协议 |
|---|---|---|---|---|
| act | ["ur5e"] | ["128x128 RGB", "frozen CLIP instruction embedding"] | 45 | [冻结协议](../results/strong_baselines/act/protocol.json) |
| vqbet | ["ur5e"] | ["128x128 RGB, resized nearest to224 for native frozen ResNet18", "frozen CLIP-L instruction768, learned prefix projection"] | 45 | [冻结协议](../results/strong_baselines/vqbet/protocol.json) |
| mimicgen_v2 | 尚未核对 | 尚未核对 | N/A | NOT_RUN |
| kite | 尚未核对 | 尚未核对 | N/A | NOT_RUN |
| crosszero | 尚未核对 | 尚未核对 | N/A | NOT_RUN |
| object_flow | 尚未核对 | 尚未核对 | N/A | NOT_RUN |

旧BC/DP源池包含IIWA、Kinova3和UR5e。新增ACT及VQ-BeT依本轮UR5e/Panda设置仅取UR5e的600条源示范，保留原有身份划分；旧结果不重训、不覆盖，也不把跨源池比较称为等数据预算比较。BARX预训练/适配及MimicGen生成监督的来源仍以各自原协议为准。

ACT唯一一次decoder输出修复及原版24000更新尝试另列归档。VQ-BeT唯一一次TF32精度修复发生在正式训练前，原四次组件探针更新和四条smoke保留，修复版独立目录重新验收。不能用原技术尝试替代修复版训练完成量；技术尝试、探针和正式评测成本分别记账。
