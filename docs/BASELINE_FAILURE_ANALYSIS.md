# Baseline阶段诊断与技术异常

官方任务success是唯一主成功判据。到达8cm、接触后抬升3cm持续10个状态、接触搬运10cm只是预先声明的诊断代理。阶段代理不构成严格递进链，物体中心距离也不等于抓取可达性；不以代理失败否定官方成功。放置及撤离子判据直接调用官方定义，不送入policy。

| 设置 | 任务 | 机器人 | 到达代理 | 闭爪命令 | 实测闭爪5mm | 双指接触 | 稳定抬升 | 搬运代理 | 官方放置 | 官方success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_bc_gmm | pnp_counter_to_sink | panda | 14/40 | 40/40 | 40/40 | 3/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| B_bc_gmm | pnp_sink_to_counter | panda | 0/40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| B_bc_rnn | pnp_counter_to_sink | panda | 7/40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| B_bc_rnn | pnp_sink_to_counter | panda | 0/40 | 40/40 | 40/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |

Panda失败阶段计数：`{"CLOSE_WITHOUT_BILATERAL_CONTACT": 157, "CONTACT_WITHOUT_STABLE_LIFT": 3}`。这是观察到的断点分类，不能从单次轨迹直接认定视觉、时间建模、动作表征或迁移是根因。

每episode的最大抬升、接触持续、阶段标志及视频路径见[失败索引](../results/baselines/failure_index.csv)，原始物理轨迹支持复查。闭爪命令与实测指垫中心距离减少≥5mm分别记录，后者只说明确有闭爪运动，不声称完全闭合或抓住物体。观察到的策略失败只有在完整合法rollout结束后才进入分母。

技术分类记录checkpoint loading、frozen environment/condition restore、network inference、action interface、simulation step、telemetry/success audit等实际出错阶段。详见technical_status.csv及原job traceback；技术异常与未运行方法均不填0%。训练不足在训练预算表单列，不能从失败episode倒推。

首次DP探针的三个失败属于记录器对zero-LR warmup首步的参数变化断言错误，日志原样保留；第二步核验确有参数变化，探针权重丢弃，正式算法和warmup规则不变。目标HDF初次打包的慢迭代技术尝试也保留，训练只使用完整hash验证后的包。

本轮不重新执行P0–P3。B1、标准OSC/IK、物体相对位姿迁移和A3仍是历史PRIVILEGED_GEOMETRY_REFERENCE/DIAGNOSTIC_ONLY，未混入RGB主表。

KITE有公开官方代码，阻碍是与固定旧栈的依赖及BARX接触意图/decoder数据接口；CrossZero尚未核实可执行官方仓库与本任务资产，且原点云输入超出RGB共同边界。BARX Single Rep/ECoT未核实匹配预训练权重，单卡从头源VLA训练未授权预算；它们是BLOCKED而非0%。RoboCasa新版DP不是本固定任务上的第二个独立算法。
