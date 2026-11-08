# P2 实际执行证据（指令 202611082145）

120 次正式新条件评测完成，另有 24 次 B1 开发及 30 次 B2 物理重放（包含重复遥测和初次技术故障，不是 30 个独立案例）。历史 0–19 证据仍在原目录。

- [汇总与区间](cells.csv)、[逐次结果](episodes.csv)、[102 条正式失败](formal_failures.csv)。阶段和持续接触抬升是代理，官方 success 未改，原因未明仍为 UNKNOWN。
- [救回/退化](paired_comparisons.csv)、[逐条件阶段变化](paired_stage_changes.csv)、[阶段计数](stage_counts.csv)。
- [B2 七组可信探针](b2_development.json)、[过程统计](b2_mechanism.csv)、[v2 历史](b2_development_v2.json)、[v3 遥测核查](b2_telemetry_integrity.json)。六个负探针来自三条轨迹，救回 0；阳性对照通过。正式 Oracle 未验证，无 B2 新测试成功率。
- [训练完整性](training_integrity.json)、[No Reps 3000 步](training/no_reps-lora/optimizer_metrics.csv)、[Joint 3000 步](training/joint_reps-lora/optimizer_metrics.csv)、[全参数 OOM 日志](training/full-step-probe/console.log)。样本和更新数匹配，不是充分官方训练或严格 FLOPs 匹配。
- [Checkpoint 与哈希](checkpoint_index.json)、[完整大文件路径与 SHA256](external_files.csv)、[策略视频](video_index.json)、[174 视频全覆盖](video_completeness.json)。模型、HDF5 状态动作、B2 完整逐步 JSON、快照和原始训练 JSONL 保留服务器。
- [条件身份](condition_integrity.json)、[144 条策略轨迹核验](state_integrity.json)、[官方源码及 bundle](official_source_integrity.json)、[版本审计](execution_versions.json)。控制规则未按正式结果调整。
- [数据清单](data_manifest.json)、[重叠审计及限制](target_data_audit.json)、[相同样本序列](matched_sample_trace.csv)、[目标方向计数](matched_target_directions.json)。未发现可检测重复，但缺少逐轨迹 XML，不能穷尽证明轨迹独立。
- [轨迹图](plots/)、[B1 配对](plots/b1-rescued-pairs.svg)、[B2 配对](plots/b2-causal-probes.svg)、[训练曲线](plots/training-curves.svg)。

主机根目录：`/home/users/lemon/coft-p2/p2/physical-20261108/`。
新测试原始文件：`formal/<cell>/official/**/episodes.jsonl`，`formal/<cell>/telemetry/episode-000.h5` 至 `episode-019.h5`，精确路径见索引。
缓存：`/home/users/lemon/coft-p2/p2/matched-samples-6000/`；固定数据版本见清单。

只读复核使用官方环境：`/home/users/lemon/coft-p2/barx/.venv/bin/python -m p2_validation.finalize_evidence --root /home/users/lemon/coft-p2/p2/physical-20261108 --out <新的证据目录>`。该命令不启动训练或物理仿真。真实执行入口为 `p2_validation.execute`、`p2_validation.replay_intervention`、`p2_validation.train_suite`。

[科研决定](../../../docs/P2_SCIENTIFIC_DECISION.md)：停止当前具体 COFT 算法分支，充分标准适配无效和可物理救回残余问题未获所需联合证据。40–59 未使用。

原始文件名按本轮试次 000–019 编号，对应官方条件 20–39；以 `episodes.csv` 的 `trial_index`、`official_condition_index`、`condition_id` 联合映射为准。原 bundle entry 的 episode/seed/哈希保留在 job.json。官方 `episodes.jsonl` 的 success 是 JSON 布尔值，并经类型核验；辅助遥测 JSONL 中 task 值可能由 NumPy 布尔字符串化，不用于成功率统计。

仓库 console.log 是保留全部文本内容、转换进度条换行并清除行尾空格的可读副本；原始 stdout/JSONL 的精确字节和 SHA256 保留服务器索引。
