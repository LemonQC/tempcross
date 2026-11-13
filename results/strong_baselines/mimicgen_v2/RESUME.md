# MimicGen v2断点

已于2026-11-13T04:25Z完成有界组件并验收：48000次正式更新、80条正式评测、两个smoke。Counter→Sink2/40，Sink→Counter0/40；复用旧人类BC各1/40。`delivery_acceptance.json`为`PASS_BOUNDED_COMPONENT_ONLY`，全部768000个消耗anchor逐行匹配冻结身份/时刻，原1280+160条证据及旧MimicGen门槛的保护哈希未改变。控制进程已正常结束。**不要重训、重评或重新生成示范**；总体PARTIAL是有意保留的完整方法边界，不是未跑完这个组件。

最终固定权重`/home/users/lemon/strong-baselines-20261112/mimicgen_v2/training/final.pt`，SHA256`2e7d97bcc436d890ce2af9067984f3af93d868b118dbca78815dbf8d15055c09`。完整结果见RESULTS.md/results.json，80条逐条记录见episodes.csv和sha_index.json。最终发布并从GitHub回读后，按本轮六项Baseline顺序处理KITE；MimicGen后台脚本本身不会启动下一方法。

固定协议已发布并核验31个远程文件：`641794f9c519b2b1170569891c3f85c27222b360`。VQ-BeT最终结果已完成发布：`a817d714a312f7392de4762722ebc798ebe97a26`，四单元真实0/40。

当前MimicGen v2根目录 `/home/users/lemon/strong-baselines-20261112/mimicgen_v2/`，控制进程记录在父目录的mimicgen-v2-run-process.json（首次PID1745299），主日志mimicgen-v2-run.log。恢复会话先检查PID、子进程及最新训练/评测状态；进程存活时继续观察，不重复启动。训练状态每250更新保存，已完成训练和闭合episode复用。

执行顺序：一次48000更新的BC-Transformer-GMM训练 → Panda两个任务各40条60–99评测 → report。生成次数0，旧BC对照直接复用；不自动启动KITE。模型架构、随机初始参数、总更新/样本预算与旧BC匹配，生成数据按任务50/13，最终checkpoint固定48000步。代码已冻结，不能根据reference成功率调参。

完成标志需要同时检查 `status.json.component_status == BOUNDED_DOWNSTREAM_CONTROL_COMPLETE` 和 `delivery_acceptance.json.status == PASS_BOUNDED_COMPONENT_ONLY`。总体status仍为PARTIAL，因为原MimicGen50+50门槛失败、父示范为Panda且下游不是原论文BC-RNN；PARTIAL不能误判为需要重新训练。可用 `python -m baselines.strong.mimicgen_v2_progress` 读取当前快照；训练完成后它生成验证曲线和日志索引，组件完成后增加新旧真实成功次数对照图，不重训/仿真。完成后立即提交代码、完整训练/80条真实评测、资源和SHA索引，并从GitHub接口重新读取核验；之后才处理KITE。

当前新技术修复0次，允许一次记录在案的下游修复；原生成的修复和187条历史物理记录保持不动。全部177条增强记录/视频已完整审计，63条成功全部使用；父ID来自原训练身份。probe一次真实更新使用6个人类、10个生成样本，143个张量更新；两个完整物理smoke通过。

用户允许空间不足时使用/data。该挂载有约3.2TiB空闲，但当前lemon(uid1003)无法访问owner1009、mode700的/data，sudo -n不可用。当前/home仍有约157GiB，现有实验继续在原路径运行，尚无文件迁移。
