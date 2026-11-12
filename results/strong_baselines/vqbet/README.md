# VQ-BeT

当前状态以status.json为准；未产生正式评测前，不报告成功率。已获取官方源码并实现必要的语言/canonical动作接口，核心保留和预算见[PROTOCOL.md](PROTOCOL.md)。

候选协议：candidate_protocol.json。正式训练须先完成两设置真实双阶段训练探针和四条完整物理smoke，再冻结protocol.json/freeze.json。唯一精度修复及原探针归档见technical_repair.json。修复版大文件服务器目录为 `/home/users/lemon/strong-baselines-20261112/vqbet_input_repair_01/vqbet/`，运行须把STRONG_BASELINE_RUN_ROOT设为其父目录；原尝试保留。完成后结果见RESULTS.md、results.json和episodes.csv，权重、视频及原始轨迹绝对路径和SHA见sha_index.json。
