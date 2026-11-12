# VQ-BeT阶段快照

这是断点快照，不是完整Baseline结果。最终状态以status.json和RESULTS.md为准。

已完成组件更新：30000；已经闭合的物理episode：41；通过完整40条单元验收的评测：40条。

| 设置 | 组件 | 实际更新 | 验证选择step | 验证损失 |
|---|---|---:|---:|---:|
| target_only | vqvae | 6000 | 4000 | 0.02146214 |
| target_only | bet | 24000 | 1000 | 19.65698469 |

| 设置 | 已验收任务 | 官方成功/实际次数 | 作用域 |
|---|---|---:|---|
| target_only | pnp_counter_to_sink | 0/40 | EXPOSED_REFERENCE |

未完成或未验收的单元不填写0%成功率。checkpoint和原始训练日志的绝对路径及SHA见model_index.json及training_artifact_index.json。

原TF32精度尝试和修复版探针分别归档，不计正式训练更新或正式成功率。已完成阶段复用，不重训。
