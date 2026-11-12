# ACT阶段结果：target-only已闭环，整体仍为PARTIAL

已完成修复版target-only的真实训练与80条正式物理评测；source+target尚在训练，不能将ACT整体称为完整Baseline。

| 设置 | 任务 | 官方完整任务成功/实际次数 | 成功率 | 角色 |
|---|---|---:|---:|---|
| target-only | Counter→Sink | 5/40 | 12.5% | EXPOSED_REFERENCE |
| target-only | Sink→Counter | 4/40 | 10.0% | EXPOSED_REFERENCE |
| source+target | 两任务 | 未评测 | N/A | 尚未完成 |

target-only实际训练24000次更新，消费384000个目标anchors；实际使用90条原始训练示范，10条验证示范保持留出。选中的第17000次checkpoint由固定验证损失0.5206316确定，实际见过272000个目标anchors，没有根据物理成功率选模。

权重留在服务器：`/home/users/lemon/strong-baselines-20261112/act_decoder_repair_01/act/training/target_only/best.pt`。

SHA256：`b1cff0e2705a419a0fc90a8ddb3960c09d36208081b53edcef214e12313fe5e3`。

80条轨迹均已逐条核对原冻结初态、完整官方placement AND retreat判据、实际动作转换、100步预测的时序集成和视频全部帧。逐条结果及失败分类见`target_only_episodes.csv`，完整视频/动作/状态/接触及训练日志的绝对路径和SHA见`target_only_artifact_index.json`。大文件未上传GitHub。

Counter→Sink的失败描述为30条闭爪但无双侧接触、5条接触但未达到稳定抬升代理。Sink→Counter为34条闭爪但无双侧接触、1条接触但未达到稳定抬升代理、1条稳定抬升代理后无官方放置。这些是描述性阶段分类，不宣称严格因果根因。

官方源码仅有已声明的维度适配及唯一一次decoder输出修复。原版只用首层输出的24000次训练尝试、2次更新探针和4条smoke均单独保留，不进入本表。修复版另有2次更新探针和4条smoke，不计正式80条。

条件60–99已暴露，本表不是独立未见测试。旧BC/DP/BARX的1280条及BARX适配160条均未重跑。源池/训练预算差异以总矩阵和协议为准，不声称与全部历史方法等数据或等计算量。
