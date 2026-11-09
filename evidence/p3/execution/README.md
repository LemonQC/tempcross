# P3实际完成证据

24次开发干预＋96000样本位置训练＋100次独立40–59物理评测全部完成。新Joint LoRA为2/20；A3稳定抓持8/8、官方完整任务5/8。当前决定：停止当前COFT新网络分支，尚未证明标准方法不能解决的缺口。

- [五格结果](cells.csv)、[100条逐次统计](episodes.csv)、[全配对救回/退化](paired_comparisons.csv)、[逐条件阶段变化](paired_cases.csv)、[阶段测量](formal_stage_measurements.csv)、[阶段定义与限制](stage_measurement_definitions.json)。
- [24条诊断](diagnostic_results.csv)、[分支/世界目标](diagnostic_cases/)、[A2/A3完全相同目标核查](A2_A3_matched_world_goals.json)。A类是特权开发诊断，不能混入公平排名。
- [124次完整轨迹路径/大小/SHA256](all_rollouts.csv)、[全部服务器文件索引](external_files.csv)、[权重及优化器状态索引](model_artifacts.csv)、[全部视频解码与状态有限性核验](validation/summary.json)、[同Panda初态完全相同核验](validation/panda_matched_initial_states.json)。
- [训练尝试](training_attempts.csv)、[每步真实损失/更新/输入记录](training/)、[初始与最终参数变化](parameter_changes.json)、[实际评测checkpoint来源](adapter_provenance.json)。
- [96k采样来源/重复/覆盖](data/complete.json)、[全部原始示范覆盖](data/demonstration_coverage.csv)、[100条目标原始demo映射](data/original_target_demonstrations.json)、[完整96k序列索引](data/sample_trace_index.json)。
- [冻结源接触模板](../frozen/source_contact_bank.json)、[官方40–59身份及真实seed/hash](../frozen/independent_conditions_40_59.json)、[全bundle验证](../frozen/official_condition_bundle_integrity.json)、[跨机器人场景核查](cross_embodiment_scene_audit.json)。
- [诊断/逐条件/训练曲线](plots/)、[雅可比核验](jacobian_validation.json)、[局部条件数](recorded_kinematic_diagnostics.csv)、[OSC跟踪速度分析](osc_tracking_budget.csv)。后两项只读已有状态，没有增加仿真或调参。

服务器根目录：`/home/users/lemon/coft-p3/p3-20261109`。新独立评测原始JSONL：`formal/<cell>/official/**/episodes.jsonl`；状态：`formal/<cell>/telemetry/episode-000.h5`至`episode-019.h5`，对应原官方索引40–59。诊断：`diagnostics/condition-XXX/A1|A2|A3/`。完整缓存：`/home/users/lemon/coft-p3/matched-samples-96000/`。训练初始/最终adapter、完整消费CSV和原始JSONL日志均在服务器，精确位置见索引。

官方success与接触/抬升/运输/开爪/退离测量分开。单模板不是合法抓取全集；A1原臂缺少有效抬升，不能将所有差异唯一归为空间。目标示范与测试状态未发现可检测重叠，缺少逐demo XML的审计限制仍然保留。

[因果报告](../../../docs/P3_SPATIOTEMPORAL_CAUSAL_REPORT.md) · [实际训练报告](../../../docs/P3_STRONGER_ADAPTATION.md) · [六问科研判断](../../../docs/P3_SCIENTIFIC_DECISION.md) · [协议修订](../../../docs/P3_PROTOCOL_AMENDMENT_01.md)。
