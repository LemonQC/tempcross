# VQ-BeT结果

COMPLETE_FAITHFUL_PORT：官方残差VQ、GPT码预测、连续偏移和两阶段更新保留；语言输入、canonical动作及有限计算预算差异见PROTOCOL.md。

| 设置 | 任务 | 官方成功/实际评测 | 成功率 | BeT checkpoint step |
|---|---|---:|---:|---:|
| target_only | pnp_counter_to_sink | 0/40 | 0.0% | 1000 |
| target_only | pnp_sink_to_counter | 0/40 | 0.0% | 1000 |
| source_target | pnp_counter_to_sink | 0/40 | 0.0% | 6000 |
| source_target | pnp_sink_to_counter | 0/40 | 0.0% | 6000 |

条件60–99标记为EXPOSED_REFERENCE，不是独立未见测试。只按原身份留出目标示范的验证损失选择RVQ和BeT checkpoint，未用物理成功率选模。

每设置RVQ实际6000次更新，BeT实际24000次更新；BeT前12000次同时更新码预测和偏移，后12000次只更新偏移。每个组件阶段384000个目标anchor；joint另有等量UR5e anchor。BeT每anchor含10个因果观测及对应动作标签，RVQ每anchor一个动作。

使用官方发布支持的冻结ResNet18特征模式；不声称复现视觉编码器微调或论文全部计算预算。动作窗口为官方UR3配方的1，每物理控制步重新预测。

唯一一次技术修复统一缓存和部署的FP32计算（关闭cuDNN及matmul TF32），在正式训练前完成。原四次组件探针更新/六次优化器调用及四条smoke另行归档，修复后四次组件探针更新/六次优化器调用及四条smoke单列；八条smoke都不计正式160条。全部视频、原始状态/动作/接触及权重留在服务器，绝对路径及哈希见sha_index.json及attempts/initial_tf32/sha_index.json；失败分类是描述性代理。
