# VQ-BeT断点

当前修复版根目录：`/home/users/lemon/strong-baselines-20261112/vqbet_input_repair_01`。

已发布并远程读取核验：冻结协议/探针 `1f8c7023e7d74450361b6a18e7252ecc8d036016`；target-only完整两阶段训练 `285397bea7f988e23dbb605a0572b6466a99ab11`。两组件实际更新6000/24000；验证选择4000/1000。模型和日志路径/哈希见model_index.json、training_artifact_index.json。阶段快照不是完整Baseline。

控制进程由独立session启动，PID记录在上述根目录的vqbet-run-process.json，主日志为vqbet-run.log。恢复会话时先检查该PID和子进程；进程存活时继续读取状态，不再次启动。额度中断不需要重训。该控制器只执行VQ-BeT，不自动启动后续算法。

执行顺序已冻结：target-only RVQ/BeT训练 → target-only两任务正式评测 → source+target RVQ/BeT训练 → source+target两任务正式评测 → report验收。每个正式单元40条，60–99均标EXPOSED_REFERENCE。已经完成的训练status为COMPLETE时复用；完成的物理episode保持原文件。发现不完整episode目录时拒绝自动重跑。

确认原控制器已退出、审查退出原因后，才可使用 `STRONG_BASELINE_RUN_ROOT=/home/users/lemon/strong-baselines-20261112/vqbet_input_repair_01 python -m results.strong_baselines.vqbet.run formal` 恢复可恢复阶段。唯一技术修复已用完；新的实质性阻碍先记录BLOCKED，不自动调整算法或接口。

完成report后，用同一根目录运行 `python -m baselines.strong.vqbet_finalize`，统一刷新全部四个组件权重、日志、选中checkpoint样本暴露、验证曲线及真实优化器计数/RVQ冻结证据，并核对160条正式评测、八条smoke和原1280+160条保护哈希。它保留历史PARTIAL快照。提交最终VQ-BeT结果、从GitHub接口重新读取文件后，再处理MimicGen v2。
