# P2-C 目标适配实际记录（202611082145 版本）

最新指令已授权训练。必要数据已实际下载并校验，匹配样本已生成。全参数 batch=1 的真实训练尝试触发 CUDA OOM：峰值分配 23,224,258,560 bytes，完成优化器更新 0 步。上游 LoRA 的真实 forward/backward/optimizer probe 通过。No Reps 的受限 LoRA 已完成 3000 步（batch=2，6000 样本位置），用时约 609 秒，峰值分配 6,687,184,896 bytes；末步 adapter 合并后的独立评测为 **2/20（10%）**。相对原始 No Reps，救回条件 34、37，破坏条件 35，净增加 1 次成功。Joint Reps 也已完成 3000 步、6000 样本位置，用时约 704 秒，峰值分配 7,444,677,632 bytes，末步 checkpoint 的新条件评测为 **3/20（15%）**。相对原始 Joint Reps 救回条件 34、37、39，破坏条件 35，净增加 2 次成功。两模型的 40 次新条件评测均已完成。

官方 Panda target_50/PnP 为两个方向各 50 条、共 100 条，人类遥操作仿真示范。原始元数据确认 PandaOmron + Robotiq85Gripper、第三人称 `robot0_agentview_left` 训练相机；评估保持官方固定 `barx_panda_agentview`。两者是官方数据/评估配置，不是新增 Eye-in-Hand。数据版本、文件大小及校验值见 [下载清单](../evidence/p2/execution_2145/data_manifest.json)。传输恢复使用公开镜像，原 revision、LFS SHA256 和实际内容不变。

目标 RLDS revision：`6e297509924c831c722180212d13f21686676e8e`；源 RLDS revision：`a2b5256a48428ea56064b0afb3dc184a045b10ae`；原始目标 HDF5 revision：`c8b3aba2dfbbfbafa67fa82466320945d560208f`。没有下载完整 284GiB 原始库。

已检查 100 条原始示范和新测试 20–39 的精确物理状态重复（忽略时间），未发现可检测重复。原始发布文件没有逐示范模型 XML，关节顺序及元数据缺失限制穷尽性身份证明；不将此检查冒称为完全证明无轨迹重叠。见 [来源与重叠审计](../evidence/p2/execution_2145/target_data_audit.json)。

两种方法已读取同一份官方源/目标混合样本序列：共 6000 个样本位置，其中源 2932、目标 3068；目标中 Counter→Sink 为 1488 个位置，Sink→Counter 为 1580 个位置。数据池是 50+50 条示范，不表示完整遍历了每条轨迹或证明每条都被本次有限采样访问。实际缓存沿用官方 action chunk 8、未来物体/轨迹窗口 40、stride 8、224px 图像及无图像增强；shuffle buffer 为 10000。训练读取有限 6000 样本缓存，保证方法间输入一致，不冒称完整官方无限 RLDS 流训练。原 mixture 权重 0.05:1.0 按 transitions 再平衡；沿用源归一化统计与源 VQ，夹爪符号沿用官方转换。固定序列 SHA256 为 `88b7af02fd345baa1765b702d09d69eadd025e0ffef0d904f36d70afe5b8a62c`。[匹配样本记录](../evidence/p2/execution_2145/matched_samples.json)

实际采用已有 upstream LoRA：rank 32、alpha 16、dropout 0，AdamW、固定学习率 2e-5、weight decay 0、BF16，单卡 batch=2、3000 步。两模型均为 6000 个相同样本位置，选择固定末步 checkpoint，不根据测试结果选模型。匹配的是更新数和样本暴露量；Joint Reps 的辅助 token 会增加算量，并非严格匹配 FLOPs 或 GPU 秒数。官方 frozen source `docs/experiments.md` 的适配为 3000 步、global batch 256，即 768000 样本位置。本轮暴露量为其 1/128，并且仅训练 LoRA 参数。因此这是受限单卡适配，**不等价于充分标准适配**。受限适配失败不能排除充分标准适配。

实际脚本：`p2_validation/prepare_training.py`、`train_adapter.py`、`train_suite.py`、`merge_adapter.py`。日志保留 `/home/users/lemon/coft-p2/p2/physical-20261108/`；实际逐步优化记录见 [训练完整性核验](../evidence/p2/execution_2145/training_integrity.json)、[No Reps 3000 步 CSV](../evidence/p2/execution_2145/training/no_reps-lora/optimizer_metrics.csv)；完整原始 JSONL 和视频保留服务器，最终索引包含 SHA256。FSDP 的 NO_SHARD hook 仅用于 full-parameter probe；LoRA 实际使用上游 DDP strategy。


## 实际显存失败与 checkpoint

全参数 probe 的真实命令由 `train_suite.train('joint_reps','full',1,1,'full-step-probe',cache)` 生成，核心命令为：

```bash
/home/users/lemon/coft-p2/barx/.venv/bin/python -m torch.distributed.run \
  --standalone --nnodes 1 --nproc_per_node 1 -m p2_validation.train_adapter \
  --source /home/users/lemon/coft-p2/barx \
  --checkpoint /home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt \
  --cache /home/users/lemon/coft-p2/p2/matched-samples-6000 \
  --out /home/users/lemon/coft-p2/p2/physical-20261108/training/full-step-probe \
  --method joint_reps --route full --steps 1 --batch-size 1
```

环境设置见 `train_suite.py`。原输出目录保留；复现失败必须使用新的空 `--out` 目录。记录的 CUDA OOM 为申请额外 520 MiB 时仅剩 259 MiB；[真实失败日志](../evidence/p2/execution_2145/training/full-step-probe/console.log) 和 [内存/更新状态](../evidence/p2/execution_2145/training/full-step-probe/training_status.json) 保留。随后 LoRA 独立 probe 成功完成 1 个 forward/backward/optimizer step，峰值分配 6,796,534,784 bytes；该 probe 权重被丢弃，两种正式适配都从各自原始 50k checkpoint 开始，分别恰好 3000 次更新。

[Checkpoint 索引](../evidence/p2/execution_2145/checkpoint_index.json) 列出原始来源、末步 adapter、合并模型绝对路径及 SHA256。No Reps 和 Joint Reps 的合并模型分别位于：

- `/home/users/lemon/coft-p2/p2/physical-20261108/adapted-checkpoints/no_reps/checkpoints/step-003000-epoch-00-loss=restricted.pt`
- `/home/users/lemon/coft-p2/p2/physical-20261108/adapted-checkpoints/joint_reps/checkpoints/step-003000-epoch-00-loss=restricted.pt`

No Reps 的训练后 adapter 含 383 个 LoRA B 张量，其中 370 个非零；训练日志、实际参数变化和成功加载后的 20 条独立仿真共同支持“确实训练和评测”，并非仅有脚本。Joint Reps 训练与 No Reps 后段评测有时间重叠；GPU 独占显存 probe 已事先完成，样本、优化器和更新数未因此改变。重用已完成任务时只校验并返回，不追加任何更新，原 Joint 训练 stdout 另存 `training/joint_reps-lora-actual.log`。


## 最终任务结果

| 模型 | 实际更新 | 相同样本位置 | 新条件官方成功 | 相对本方法原始模型救回/退化 |
| --- | --- | --- | --- | --- |
| No Reps + 受限 LoRA | 3000 | 6000 | 2/20 (10%，Wilson 95%：2.8–30.1%) | 2 / 1 |
| Joint Reps + 受限 LoRA | 3000 | 6000 | 3/20 (15%，Wilson 95%：5.2–36.0%) | 3 / 1 |

两模型均失去了原来成功的条件 35，表现为未闭爪。新救回条件 34 在两种原始模型和 B1 中均失败，但在两个适配模型中均官方成功，支持“已有目标适配能处理部分简单控制无效案例”。Joint 适配仍有 14/20 未闭爪、3/20 闭爪但无双指接触；No Reps 适配分别有 15/20、2/20，另有 1 条接触但无持续抬升。这些有限预算下的现象不能否定充分标准适配。

Joint 的 383 个 LoRA B 张量同样有 370 个非零。两模型各 3000 条优化器记录均完整、loss 为有限值，checkpoint 已加载执行各 20 条官方闭环仿真。[Joint 逐步记录](../evidence/p2/execution_2145/training/joint_reps-lora/optimizer_metrics.csv) · [完整配对](../evidence/p2/execution_2145/paired_comparisons.csv) · [大文件 SHA256 索引](../evidence/p2/execution_2145/external_files.csv)。训练 loss 包含不同辅助目标，不能跨方法直接比较，也不能代替任务成功。

“新条件评测”表示固定 20–39，没有用于阈值或 checkpoint 选择；公开示范缺少逐轨迹 XML 的重叠审计限制仍然成立。40–59 未使用。
