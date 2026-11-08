# P2-C：未训练、未评测

状态：`NOT_TRAINED`。实际训练 step 为 0；没有测量训练峰值显存，没有目标适配 checkpoint，没有适配后的独立 rollout。成功率为 null，不是 0%。

此前已完成代码和公开元信息审查：官方 `target_50/panda/pnp` 是两个方向各 50 条、合计 100 条；XP-900 适配混合 `mg_pnp_lite` 与 `panda_pnp`，沿用源统计和 VQ。原始示范与评测条件的精确重叠尚未审查。详见已经找回的 [执行前分析](P2_REVIEW_202611082025_ZH.md)。

当前服务器只有源模型：

- Joint Reps：`/home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt`
- No Reps：`/home/users/lemon/coft-p2/artifacts/runs/no_reps/checkpoints/step-050000-epoch-15-loss=0.8424.pt`

它们是原始预训练权重，不能称为 Panda 50 示范适配模型。此前的单卡命令 dry-run 只生成了命令文本，没有完成 forward/backward/optimizer step。本次用户明确要求先不运行训练或仿真，因此没有通过新训练来填补这个缺口。

| 方法 | 目标适配实际 steps | 适配后独立评测 episodes | 成功率 |
|---|---:|---:|---|
| No Reps + Panda 示范 | 0 | 0 | N/A |
| Joint Reps + Panda 示范 | 0 | 0 | N/A |

尚未训练不能解释为普通目标适配无效。当前也没有实测 OOM，不能宣称 RTX 3090 无法训练。机器可读状态见 [status.json](../evidence/p2/status.json)。
