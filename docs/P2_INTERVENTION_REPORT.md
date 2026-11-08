# P2-B 实际执行报告（进行中）

本轮依据 `202611082145.md` 已真实执行物理仿真，不再处于此前暂停状态。原 P0/P1 和 P2-A 证据保留；此前未执行状态可在 commit `74e57e3` 查看。

## 已完成的开发实验

固定开发索引为 `0,1,2,4,6,10,12,18`。B1 阈值 0.85、0.90、0.93 各运行 8 次，均为 0/8。滞回宽度 0.03、最短保持 8 步。按预先确定的“成功数最大，平局取最低阈值”规则，唯一选择 0.85，并在新测试前写入 `frozen_protocol.json`。没有添加第四个阈值。

B1 只读取原始连续夹爪信号及历史，只改变执行动作第七维；其他维度与当次原 BARX 预测保持一致。成功标签沿用官方 `info[is_success][task]`。

B2 已完成 7 个原始/闭爪干预配对，共 14 次物理重放。6 个失败配对来自 3 条代表轨迹、各 2 个固定时刻，32 步强制闭爪均未救回最终任务；成功正对照的两个分支均成功。全部分支的关节、EEF、物体、夹爪、actuator ctrl、求解器 warmstart 和控制器状态核对通过，记录误差为 0。最初一次重放在事后审计遇到 NumPy bool 相减错误，已修复并另存 v2，原始两次物理记录没有删除，也没有混入可信配对计数。

这些是有限时刻因果探针，不是经验证的通用 Oracle。因此 B2 正式评估状态为 `ORACLE_NOT_VALIDATED`，不伪造 20 条 B2 新测试结果，先运行其余 80 次。

## 正式测试

官方 bundle 先经完整 loader 和语义验证，再统一选取原索引 20–39；entry、state、XML、condition_id、actual seed 和哈希一一保留。不是修改 start_seed 来选窗。

Joint Reps/UR5e 已完成 20 次，为 **8/20**。Panda Original Joint Reps、Original No Reps 和 B1 正在同一 20 条条件上运行，尚未出最终比较。精确状态见 [原始汇总](../evidence/p2/execution_2145/analysis.json)，不得将进行中计数视为最终成功率。

[开发及正式逐次 CSV](../evidence/p2/execution_2145/episodes.csv) · [B2 实际配对](../evidence/p2/execution_2145/b2_development.json) · [视频索引](../evidence/p2/execution_2145/video_index.json)

大型 HDF5、MP4、日志与 B2 分支快照位于 `/home/users/lemon/coft-p2/p2/physical-20261108/`。本阶段未新增机器人、任务、相机、奖励或成功判据，也没有开发 COFT 新网络。
