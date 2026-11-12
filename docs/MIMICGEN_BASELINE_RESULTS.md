# MimicGen有界Baseline：实际进度

状态：**IN_PROGRESS**。已完成两任务10条固定源示范的官方datagen_info提取。物理探针已执行5条：Counter→Sink生成2/3成功，对应已完成重放2/2成功；第三条生成403步后未成功，其重放及另一任务尚未执行。没有下游训练或正式评测。

第三条生成的记录器出现NumPy int64 JSON序列化异常。原日志及异常状态保留；已只读核验并恢复其已有物理数据，不重跑任何轨迹。按原指令对物理探针单独允许的一次实现修复，修正索引序列化；本次修复及预检/探针预算勘误详见协议追加说明和`probe_logging_repair.json`。

探针总上限仍为10条、门槛仍为至少6条成功且两任务生成/重放均至少一次成功。剩余ID将在修复代码重新冻结后继续。所有探针轨迹不用于下游训练；无MimicGen下游策略成功率可报告。原1280＋160条实验未重跑。

协议：[MIMICGEN_BASELINE_PROTOCOL.md](MIMICGEN_BASELINE_PROTOCOL.md)。实时状态：[status.json](../results/strong_baselines/mimicgen/status.json)。原始文件根：`/home/users/lemon/mimicgen-baseline-20261112/`。
