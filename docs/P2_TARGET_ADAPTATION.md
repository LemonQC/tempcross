# P2-C 目标适配实际状态（进行中）

最新指令已授权训练。必要数据已实际下载并校验，匹配样本已生成；正在等待正式策略评估释放 GPU，尚未执行优化器 step。不能把等待中的模型记成 0% 成功率。

官方 Panda target_50/PnP 为两个方向各 50 条、共 100 条，人类遥操作仿真示范。原始元数据确认 PandaOmron + Robotiq85Gripper、第三人称 `robot0_agentview_left` 训练相机；评估保持官方固定 `barx_panda_agentview`。两者是官方数据/评估配置，不是新增 Eye-in-Hand。数据版本、文件大小及校验值见 [下载清单](../evidence/p2/execution_2145/data_manifest.json)。传输恢复使用公开镜像，原 revision、LFS SHA256 和实际内容不变。

目标 RLDS revision：`6e297509924c831c722180212d13f21686676e8e`；源 RLDS revision：`a2b5256a48428ea56064b0afb3dc184a045b10ae`；原始目标 HDF5 revision：`c8b3aba2dfbbfbafa67fa82466320945d560208f`。没有下载完整 284GiB 原始库。

已检查 100 条原始示范和新测试 20–39 的精确物理状态重复（忽略时间），未发现可检测重复。原始发布文件没有逐示范模型 XML，关节顺序及元数据缺失限制穷尽性身份证明；不将此检查冒称为完全证明无轨迹重叠。见 [来源与重叠审计](../evidence/p2/execution_2145/target_data_audit.json)。

两种方法将读取同一份官方源/目标混合样本序列：共 6000 个样本位置，其中源 2932、目标 3068。原 mixture 权重 0.05:1.0 按 transitions 再平衡；沿用源归一化统计与源 VQ，夹爪符号沿用官方转换。固定序列 SHA256 为 `88b7af02fd345baa1765b702d09d69eadd025e0ffef0d904f36d70afe5b8a62c`。[匹配样本记录](../evidence/p2/execution_2145/matched_samples.json)

预定先做真实 full-parameter forward/backward/optimizer step 和显存测试；必要时使用已有 upstream LoRA。最终预算两模型均为 6000 个相同样本位置，选择固定末步 checkpoint，不根据测试结果选模型。这是受限单卡适配，**不等价于官方 global batch 256 的训练暴露量**。受限适配失败不能排除充分标准适配。

实际脚本：`p2_validation/prepare_training.py`、`train_adapter.py`、`train_suite.py`、`merge_adapter.py`。日志保留 `/home/users/lemon/coft-p2/p2/physical-20261108/`；实际训练步数、显存和独立 rollout 将在完成后据日志更新。
