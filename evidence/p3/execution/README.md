# P3 实际证据（增强适配训练中）

已完成24次A类开发干预和40–59的80次旧模型公平评测；96000样本Joint LoRA真实训练进行中，新模型20次评测尚未执行，不能当作0%。当前状态见 [机器状态](../status.json)。

- [A类逐次结果](diagnostic_results.csv)、[汇总](diagnostic_summary.csv)、[原始分支与目标](diagnostic_cases/)、[A2/A3目标完全相同核验](A2_A3_matched_world_goals.json)。
- [源接触模板](../frozen/source_contact_bank.json)、[40–59冻结身份及哈希](../frozen/independent_conditions_40_59.json)、[官方bundle验证](../frozen/official_condition_bundle_integrity.json)。
- [公平单元结果](cells.csv)、[逐次原始统计](episodes.csv)、[同条件救回/退化](paired_comparisons.csv)、[逐条件阶段比较](paired_cases.csv)。文件中的trial_index000–019对应原官方索引40–59，绝不以文件编号替代原条件索引。
- [已闭合物理文件绝对路径/大小/SHA256](closed_physical_artifacts_before_strong_evaluation.csv)、[104个视频及状态完整性核验](validation/summary.json)。后续新模型20次将追加最终完整索引。
- [实际训练尝试](training_attempts.csv)、[每步损失、配置与执行记录](training/)、[真实单步参数变化](probe_parameter_verification.json)。原始控制台/JSONL及大权重保留服务器，仓库控制台副本移除进度条并清理行尾空格，不改科学数值。
- [新96k数据元信息](data/complete.json)、[不同原始示范覆盖](data/demonstration_coverage.csv)、[目标原始demo映射](data/original_target_demonstrations.json)、[完整96k序列的主机索引](data/sample_trace_index.json)。总曝光量、唯一帧、重复帧分别报告。
- [轨迹图](plots/diagnostic_trajectories.svg)、[解析雅可比核验](jacobian_validation.json)、[局部运动学与误差](recorded_kinematic_diagnostics.csv)、[轨迹速度与OSC响应](osc_tracking_budget.csv)。这些只读分析不增加任何物理step。

主机根目录：`/home/users/lemon/coft-p3/p3-20261109/`。
A类原始文件：`diagnostics/condition-XXX/A1|A2|A3/`。
新独立评测：`formal/<cell>/official/**/episodes.jsonl`，`formal/<cell>/telemetry/episode-000.h5`至`episode-019.h5`。
新缓存：`/home/users/lemon/coft-p3/matched-samples-96000/`，完整序列原始SHA见索引。独立数据重叠只做可检测物理状态检查，缺少逐示范XML，不能穷尽证明无轨迹重叠。

A类不是独立测试或公平视觉策略排名。A1保留原臂路径而没有标准抬升，A3增加抬升/运输，因此结果不能唯一归于静态位姿。A2/A3相同路径上的夹爪调度差异可直接比较。单一源接触模板不是合法抓取全集Oracle。

[因果报告](../../../docs/P3_SPATIOTEMPORAL_CAUSAL_REPORT.md) · [增强适配](../../../docs/P3_STRONGER_ADAPTATION.md) · [当前科研判断](../../../docs/P3_SCIENTIFIC_DECISION.md) · [公开协议修订](../../../docs/P3_PROTOCOL_AMENDMENT_01.md)。
