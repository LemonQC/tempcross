# ACT断点与验收入口

原主进程信息：`/home/users/lemon/strong-baselines-20261112/act-run-process.json`。该原版runner因decoder问题暂停，不能恢复它而启动原版参考评测。当前有效进程信息见修复根目录下`act-run-process.json`；先核对PID/命令，不能同时再启动一份。

修复后的输出根：`/home/users/lemon/strong-baselines-20261112/act_decoder_repair_01/act/`。原目录完整保留作为技术尝试证据。

- `probe/<setting>/status.json`：一次真实更新的独立探针，已完成的探针直接复用。
- `training/<setting>/status.json`、`latest.pt`：每250次更新保存模型、AdamW状态、CPU/CUDA/NumPy RNG与消费前沿。达到COMPLETE的设置不再训练。
- `training/<setting>/best.pt`：仅由冻结的目标示范验证损失选择的checkpoint。
- `physical/<smoke|formal>/<setting>/<task>/condition-*/episode.json`：已闭合物理episode，直接复用，不能重放。只有正式60–99行进入参考成功率。
- `acceptance.json`：保存的实际视频、动作、状态和官方成功判据的只读核查结果。

如进程因会话/连接中断但仍存活，继续监控即可。如进程已退出，应先检查原始错误记录与checkpoint前沿，保留失败文件；一个方法只允许一次有记录的技术修复。未闭合的物理目录会拒绝自动重试，不能用覆盖目录的办法隐藏失败尝试。

确认没有重复进程、没有尚待诊断的错误后，恢复入口是：

```bash
cd /home/users/lemon/CodexProject/tempcross
/home/users/lemon/coft-p2/barx/.venv/bin/python -m results.strong_baselines.act.run
```

执行上述Python命令时必须设置 `STRONG_BASELINE_RUN_ROOT=/home/users/lemon/strong-baselines-20261112/act_decoder_repair_01`；禁止把修复代码写入原版的vendor或训练输出。ACT唯一一次技术修复已用于decoder输出选择，后续实质阻碍须报告BLOCKED。

该入口只处理ACT，不启动VQ-BeT或后续方法。严格顺序为target-only训练与两任务评测，再source+target训练与两任务评测，最后只读验收/报告。断点只恢复已保存前沿，不因为会话切换重新从随机初始化训练。

训练原始日志是根目录下`act-training-*.log`；每次实际更新的损失、用时和每个示范ID/帧均另存CSV。`baselines.strong.consumption`可独立对账已保存前沿。它不调用网络或仿真，也不改变选择规则。

正式完成后必须核对两设置各24000更新、两任务各40条，生成最终SHA索引，提交GitHub并重新回读确认；仅有本地COMPLETE或脚本不算交付。历史BC/DP/BARX及MimicGen原报告不能修改。
