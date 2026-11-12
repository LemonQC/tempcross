# MimicGen有界Baseline：实际进度

状态：**IN_PROGRESS**。首次探针完成：生成4/5、动作重放4/5，共8/10，两任务均有生成与重放成功；视频、接触、动作、物理状态、重放初态/动作前缀及旧输入哈希核查通过。实际10条探针，无重复轨迹。

两阶段实现修复及记录器异常的恢复、NumPy RNG续跑差异见协议与原日志。探针数据均不用于下游训练。现仅完成下游方案冻结：每方向50条额外成功示范、最多100次尝试；唯一BC-Transformer配置、seed7、48000更新/768000 anchors、固定验证选模。

截至本提交：额外训练数据尚未生成、下游optimizer更新0、正式评测0。没有MimicGen下游成功率可填写。原1280＋160条物理实验未重跑。

协议：[MIMICGEN_BASELINE_PROTOCOL.md](MIMICGEN_BASELINE_PROTOCOL.md)。状态：[status.json](../results/strong_baselines/mimicgen/status.json)。原始文件根：`/home/users/lemon/mimicgen-baseline-20261112/`。
