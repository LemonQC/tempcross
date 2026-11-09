# 统一Baseline冻结协议

本轮只运行现有方法，不开发COFT或新的网络。机器规则见[protocol.json](../configs/baselines/protocol.json)，两任务Panda/UR5e原官方60–99身份、状态值及哈希见[冻结条件](../configs/evaluation/baseline_conditions_60_99.json)。完整100条件先按官方验证器验证，再取40条，不改变真实seed、XML、物体或成功判据。Counter-to-Sink上限600步，Sink-to-Counter按官方上限650步，均20Hz。

新BC/DP模型使用相同100条目标示范池，按完整示范固定划分每任务45训练/5验证（总90/10），ID和源1620/180划分均提交。最终60–99绝不用于训练、选阈值、选checkpoint或延长预算。开发smoke只用原条件0，每任务一次完整rollout。每核心模型先单步真实forward/backward/optimizer，丢弃probe权重，再固定种子重新训练。

第一档为96000个anchor样本位置。若固定验证指标仍改善，按预定规则延长至192k/384k/768k：每12000个目标anchor检查同一256条验证anchor，比较最后三次均值与前三次，改善≤2%视为验证平台；否则升下一档。Transformer/DP的B/C成对模型共同延长直到两者平台或上限，以保留相同目标暴露量。到上限仍改善的模型标为BUDGET_LIMITED_NOT_CONVERGED，不将明显欠训练DP当作算法本质失败。

B每次更新一批target，C每次更新同一批target再加一批source；target batch按真实显存探针16→8→4选择，C总batch为两倍。故目标暴露量和更新次数匹配，但C总样本及计算更多，须分列报告，不能同时声称等总预算。source在IIWA/Kinova3/UR5e和两任务六组均衡抽样；target两任务均衡、示范均衡，frame均匀。所有方法复用固定anchor序列。完整消费trace和每demo覆盖保留服务器并给出hash。

anchor不是等价于图像/动作标签总数：GMM为单帧单动作；RNN为10步因果序列；Transformer为10帧历史监督当前动作；DP观测2帧、预测16步、执行8步。分别记录展开后的观测/标签暴露与padding，不把历史重复隐藏在anchor数字里。

新模型只见RGB、任务原文。相同OpenCV INTER_AREA将RGB缩至128×128，官方CropRandomizer训练116×116、测试中心crop；CLIP-L/14冻结语言投影，无额外仿真真值。官方规范的归一化7维OSC动作直接复用，原TFDS夹爪已经是[-1,+1]，不二次翻转；拼12维时base/torso四维为0、arm-mode为-1。BARX保留其224px原预处理、source统计/VQ及gripper01→实际控制转换。

Group A（零target）、B（target-only）、C（source+target）分表。已有BARX适配见过full100，且预算/初始化/分辨率不同，单列LEGACY_FULL100_RESTRICTED_LORA，不能与90/10新模型作严格数据匹配因果排名。旧P0–P3只作为历史，不重跑旧0–59结果。

最终checkpoint为按验证延长规则决定的末步，不由最终测试挑best；保留96k第一档权重和每档训练/验证曲线。GMM/RNN/Transformer的验证同时报告固定seed期望动作MSE和NLL，避免low-noise eval方差使NLL不可比；DP用固定noise/timestep去噪MSE。不同算法验证loss不直接排名。

所有方法使用同一BasePolicy接口、动作适配和评估循环。状态/接触等特权信息只写telemetry，永不交给RGB policy。正式每格40次，两任务分开；UR5e source reference单列。episode保存native输出、实际动作、TCP、夹爪、对象、接触、视频、condition ID和真实seed。技术异常单列，不填成策略失败或0%。阶段代理只诊断，官方success是主指标。

几何/真值方法留在PRIVILEGED_GEOMETRY_REFERENCE或DIAGNOSTIC_ONLY，不混入RGB公平主表。主结果同时给Wilson95%、同条件救回/退化、推理耗时、训练预算和显存，区分资源受限结果与算法能力。
