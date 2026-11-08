# P2-B：未执行真实干预

状态：`NOT_RUN`。本地工作区、分支/reflog、外部运行目录及 GitHub 审计未找到已完成的 P2-B 实验。此前已有分析建议，没有完成 B1/B2 物理实现验证。本次也未运行训练或仿真。

| 方法 | 机器人 | 新测试已执行 episodes | 成功次数/成功率 |
|---|---|---:|---|
| B0 Original Joint Reps | UR5e | 0 | N/A |
| B0 Original Joint Reps | Panda | 0 | N/A |
| Original No Reps | Panda | 0 | N/A |
| B1 Simple Gripper Correction | Panda | 0 | N/A |
| B2 Privileged Grasp Oracle | Panda | 0 | N/A |

新条件 20–39 的原始结果和视频路径均为 null。没有已测的 rescue/regression 配对统计。旧 P1 的 9/20、0/20 等结果属于条件 0–19，不能移入这张表。

本次源码 `p2_validation/offline_audit.py` 是读取旧数据的诊断工具，不是 B1/B2 控制器。指垫中心棱柱对成功例的覆盖只有 6/15，因此没有把它冒充已验证 B2。未经实际干预，无法回答 B1/B2 能救回多少任务。

机器可读状态见 [status.json](../evidence/p2/status.json)，真实已完成的离线证据见 [P2_GRASP_CAUSAL_AUDIT.md](P2_GRASP_CAUSAL_AUDIT.md)。
