# BARX目标适配：实际执行结果

当前状态：**正式训练前冻结阶段**。已完成官方路径和两轮受限配置核查、batch8真实forward/backward/optimizer检查、256样本累积与梯度检查、整条示范划分、560个固定验证帧准备、256条实时RLDS输入检查。**正式新模型更新数目前为0，尚无新物理成功率。** 本文件将在实际训练、验证选模及独立记账的新增评测完成后更新；不把预定预算填写成已完成。

固定协议见 [BARX_TARGET_ADAPTATION_PROTOCOL.md](BARX_TARGET_ADAPTATION_PROTOCOL.md)，代码及证据见 [results/strong_baselines/barx_adaptation/](../results/strong_baselines/barx_adaptation/)。

当前可确认的历史同条件参考（只读原1280条，未重跑）：

| 旧模型 / 60–99 | Counter→Sink Panda | Sink→Counter Panda | Counter→Sink UR5e | Sink→Counter UR5e |
| --- | ---: | ---: | ---: | ---: |
| 原始Joint | 1/40 | 7/40 | 16/40 | 14/40 |
| 旧Joint 96000位置LoRA | 2/40 | 1/40 | 未评测 | 未评测 |
| 本轮验证选模的新适配 | 尚未评测 | 尚未评测 | 尚未评测 | 尚未评测 |

旧Joint6000位置LoRA在不同的20–39条件为Counter→Sink Panda **3/20**，不与上表当成同条件比较。旧受限LoRA不称为充分标准适配；本轮使用官方LoRA路由、参考暴露预算与累积执行，也不冒称论文全参数复现。
