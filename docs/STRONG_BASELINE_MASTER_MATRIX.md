# Strong Baseline Master Matrix

研究基准：main。TOCAT 独立分支保留，不用于训练。原 BC/DP/BARX 1280 条及 BARX 适配 160 条评测不重跑。

正式条件 60–99 均标为 **EXPOSED_REFERENCE**，不是独立未见测试。未评测写 N/A，不以 0% 代替。

| 方法 | 状态 | 已验证训练更新（不含探针） | 已验收正式物理评测 | 成功率 | 当前证据 |
|---|---|---:|---:|---|---|
| act | PARTIAL | 0 | 0 | N/A | [状态](../results/strong_baselines/act/status.json) |
| vqbet | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/vqbet/status.json) |
| mimicgen_v2 | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/mimicgen_v2/status.json) |
| kite | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/kite/status.json) |
| crosszero | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/crosszero/status.json) |
| object_flow | NOT_RUN | 0 | 0 | N/A | [状态](../results/strong_baselines/object_flow/status.json) |

状态含义：COMPLETE_OFFICIAL 为核心与原生设置完整复现；COMPLETE_FAITHFUL_PORT 为核心保留、已声明平台/接口适配并完成训练及官方完整任务评测；COMPLETE_REIMPLEMENTATION 为已声明的忠实重实现；PARTIAL 为尚未闭环；BLOCKED 为具体阻碍已记录；NOT_RUN 为尚未执行。

执行顺序：ACT → VQ-BeT → MimicGen v2 → KITE → CrossZero → Object Flow。单卡每次只运行一个大型训练；每种算法完成或阻碍报告后单独提交。

原始50+50生成示范门槛失败的MimicGen结论保持不变；mimicgen_v2为单独协议。不同观测、目标示范量及额外监督在各方法协议中分列，不作等信息或等数据暗示。
