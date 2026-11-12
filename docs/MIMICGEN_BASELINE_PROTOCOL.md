# MimicGen 有界强Baseline：首次物理运行前冻结协议

指令：`202611121050.md`；起点：`24f1658ccc12cd0b31a5f5184c025bd27981bfc4`，保留已验证结果提交`506d929d156d02ba95773fefad65c638b8eac5a4`。本阶段只研究MimicGen能否形成“有效示范生成→视觉策略训练→同任务物理评测”的完整对照。旧1280＋160条物理实验不重复、不改写。没有新Method、KITE、CrossZero或Object Flow。

## 零新仿真的可行性核查

已读取统一Baseline协议、复现矩阵、完整结果及BARX适配结果；逐文件审查现有BARX生成包装器、源状态注释器、MimicGen DataGenerator、分段选择/插值、动作编解码和两项任务配置。证据：[audit.json](../results/strong_baselines/mimicgen/audit.json)。

官方环境源码固定`0f435089f48b1f16d225f868dbef855a5e75d1e0`，含BARX所带MimicGen快照，registry标注源revision `62aca464ee14edeac48769c6be82464a7337a5d3`。实际依赖为Python3.10、MuJoCo3.1.1、robosuite1.5.1、robomimic0.3.0、Torch2.2.0+cu121。一张RTX3090，启动前无训练或仿真进程。下游如获准使用既有robomimic `ae5799f0fae05c4559ee1f9645b0f77eb5251929` 官方BC-Transformer，不修改网络。

原始Panda HDF5两任务各50条，均有12维原动作、低维物理状态和`ep_meta`，可用官方接口提取物体位姿与阶段信号；不需要臆造这些监督。未在本轮读取到可用的原始UR5e人类源HDF5，故本轮生成种子明确使用**Panda目标域人类示范**，不能称为零目标示范跨具身生成。XP-900源RLDS的1800条不自动等价于可重建环境的MimicGen原始HDF5。

## 数据身份与信息边界

保持原整示范划分：每方向45训练/5验证，共90训练/10验证。生成种子从每任务45条train按`episode_id`排序固定取前5条，总10条；无成功率筛选。已核对动作/长度/完整指令指纹，清单在`source_seed_manifest.json`。注释只写服务器新副本，绝不改原数据。Panda全池100条、训练可用90条、生成种子10条与实际消费量分别记录。

生成器在训练数据构造时使用特权物体/目标位姿、末端位姿、接触阶段及源示范动作；这是MimicGen的输入，不冒充RGB-only运行。假如下游通过门槛，其部署输入只能是原外部RGB及完整任务原文；不能输入物体真值、EE、接触、深度或阶段标签。保留原128px/116px crop、冻结CLIP-L/14语言投影及canonical7D动作的数据接口。图像方向不额外翻转。

额外生成示范及其所有失败尝试、物理步数、存储和生成时间单列成本，不能与BC/DP/BARX宣称等数据公平。本轮尚未产生生成数据，不预填示范数或下游成功率。

## 唯一一次集成修复及忠实性限制

默认虚拟环境未安装独立`mimicgen`包；使用任务进程局部PYTHONPATH读取已固定的`third_party/mimicgen`，不安装/升级依赖。纯离线检查还证实：`action_to_target_pose`将base-frame动作直接当world-frame增量，而反向编码确实将world转base；在yaw=-2.1的合成数值检查中动作往返最大误差0.6241，平移目标误差3.13cm。该检查不是物理成功率，也不证明历史失败由此导致。

唯一有界修复在`common.py`：仅源动作注释的反向解码补充base→world旋转；修复后四个预定yaw的往返误差均<3e-6。原DataGenerator、目标位姿→动作编码、阶段规则、noise0.05、nearest_neighbor_interpolation（k=5）、插值、控制器和源checkout不变。阶段1结束偏移保持C→S的10–20、S→C的20–30；阶段2沿用sink/container参考。不是另造规则控制器，也不宣称逐字节官方复现。**修复预算1/1已使用；后续技术阻碍不再自动修复或重试。** 初始检查脚本曾误传`config_factory(base_type=...)`，在零环境创建时依实际签名改为`config_type`，不计作物理轨迹或算法结果。

## 首轮10条物理轨迹的冻结门槛

为遵守“不超过10条生成与重放探针”，采用**5条生成＋对应5条动作重放，总计10条**，而非10条生成后另加10条重放。Counter→Sink固定开发条件0、1、2；Sink→Counter固定0、1。每条生成后在同一原保存初态重放其实际动作，不用state forcing代替真实动作执行。随机种子7；官方DataGenerator逐阶段选择源片段，`select_src_per_subtask=True`、`transform_first_robot_pose=True`、`interpolate_from_last_target_pose=True`，其余任务参数原样。

使用原RoboCasa-X `XPnPCounterToSink`/`XPnPSinkToCounter`、PandaOmron＋Robotiq85、20Hz、12维控制布局（arm6/gripper1/base3/torso1/mode1；base/torso为0、mode=-1）。不关闭碰撞、不调整增益。最长600/650步，达到官方完整任务成功或步数上限即停止。成功来自原`info.is_success.task`，另与保存的placement AND retreat逐步核对。生成器只返回普通任务失败时可继续预定剩余ID；权限、数据、引擎或代码异常则BLOCKED并终止。

每条保存原RGB视频、实际动作、完整物理状态、物体/夹爪及接触日志，独立记账；失败轨迹也保留。官方源注释器通过恢复已有状态计算datagen_info，不执行新任务动作；其环境初始化和状态重建与动态生成/重放分别记录。只有总成功≥6/10、两任务的生成和重放均至少一次成功、数据/接触/视频/动作和成功定义检查通过，才允许进入下一门槛。不得扩展这10条或用历史成功填补。

## 通过探针后才冻结的下游阶段

本提交**不启动下游训练或正式评测**。仅探针通过，才在新提交中固定额外生成数据规模/尝试上限、完整数据谱系、训练/验证包、唯一BC-Transformer配置、训练seed7、固定更新预算和checkpoint选择/停止规则。使用预定验证示范或固定验证损失选权重，不使用物理测试分数。若生成分支共享人类种子、物体/条件或轨迹谱系，划分按整条谱系处理，不能随机分帧制造独立性。

正式物理评测最多两方向×40＝80条Panda，复用原60–99，仅标`EXPOSED_REFERENCE`。原有条件均已暴露；当前没有可声称独立未见的测试集。不得借此生成新条件/任务。历史BC-Transformer target-only/source+target和BARX结果只读引用；目标示范、生成额外数据、源混合、训练预算及监督信息差异公开，不重新训练对照。

## 状态、断点及产物

`results/strong_baselines/mimicgen/status.json`使用IN_PROGRESS/BLOCKED/COMPLETE，记录阶段、输入SHA、生成/重放/评测ID、checkpoint和实际更新量；服务器根`/home/users/lemon/mimicgen-baseline-20261112`。协议、源ID、代码在第一次环境创建前冻结并记录哈希。启动使用排他文件锁；已完成ID读已有结果，未完成轨迹目录保留且拒绝覆盖，终止状态拒绝再运行。每阶段原子落盘。

不声称Codex会自动恢复额度。若会话额度耗尽，读取状态、PID和日志后由用户或外部Linux续跑器以`codex exec resume`接续；本任务不操作计费、安全配置或付费API。没有运行中进程时也只能执行未完成部分，不能重新生成已有轨迹或重训。

正式交付为本协议、`MIMICGEN_BASELINE_RESULTS.md`、状态、必要代码/配置、真实物理账本与SHA索引。权重/HDF5/视频留服务器。若任一必要门槛失败，交付BLOCKED及证据，不把未运行训练/评测写成0%结果；之后不进入其他Baseline或Method。

## 探针记录器修复勘误（仅在剩余ID执行前追加）

初版把预检和物理探针两阶段的修复预算错误合并成全局一次。原指令分别写明预检“一次有界的修复尝试”，以及物理探针“失败可最多作一次有文档记录的实现修复”。依原用户指令，此处分别计数：预检集成修复1/1已使用；现在只使用物理探针实现修复1/1，之后不再修复。原文和首次冻结提交保留，不隐去该解释变化。

第3条生成执行403步后正常返回失败，记录器在JSON保存`src_demo_inds`中的NumPy int64时报错。前两条生成与其重放已成功；第三条的完整动作/状态/RGB、接触和视频也已保存。本次只将索引显式转Python int；不改DataGenerator、控制器、噪声、源选择、条件、种子、10条总上限或6/10成功门槛。

`recover_record.py`只读现有第三条数据，核对403步动作/404个状态/403帧视频、官方成功为False、文件SHA不变后恢复缺失的episode JSON。该轨迹不重新执行；其具体源片段索引因异常未写出，标为不可用，不反推造值。所有探针数据排除在后续训练之外。

原进程未保存崩溃时的NumPy RNG状态。续跑沿用预注册seed7，跳过已有ID而不重演随机draw，不承诺未执行ID与无中断运行逐位相同。该差异在剩余轨迹前公开，不依据成功分数选择新seed或参数。恢复原已执行5条，剩余只执行第三条重放和Sink→Counter的两条生成/两条重放，总轨迹仍不超过10。
