# Baseline复现审查（训练前冻结）

审查日期2026-11-09。环境为已验证BARX/RoboCasa-X提交`0f435089f48b1f16d225f868dbef855a5e75d1e0`，不替换原MuJoCo环境、不改成功判据。状态描述的是代码复现关系；实际训练/评测数量另列，不能把代码可构建写成实验完成。

| 方法及论文 | 官方实现及固定版本 | 原输入／动作 | 预训练权重与原任务支持 | 本平台修改、3090可行性和状态 |
| --- | --- | --- | --- | --- |
| BC-GMM；[What Matters in Learning from Offline Human Demonstrations for Robot Manipulation，CoRL2021](https://arxiv.org/abs/2108.03298) | [robomimic](https://github.com/ARISE-Initiative/robomimic)，`ae5799f0fae05c4559ee1f9645b0f77eb5251929` | 可配置RGB/状态；GMM连续动作 | 没有与本次两任务、固定观测一致的已训练权重；支持robosuite HDF5 | **FAITHFUL_PORT**：直接调用官方BC_GMM，ResNet18/SpatialSoftmax，5成分GMM；仅移植数据、语言及统一动作/评估。CPU构建已通过，13,176,555参数；GPU真实训练待执行。 |
| BC-RNN；同上CoRL2021研究 | 同一robomimic固定提交 | 因果RGB/状态序列；LSTM-GMM动作 | 同上 | **FAITHFUL_PORT**：官方BC_RNN_GMM、400维双层LSTM、10步序列；14,484,763参数，CPU构建通过。非改写MLP。 |
| BC-Transformer；robomimic实现及[RoboCasa，RSS2024](https://robocasa.ai/docs/build/html/v0.2/use_cases/policy_learning.html) | 同一robomimic提交，官方BC-Transformer模板 | RGB历史＋可选状态；Transformer-GMM | 原模板不是BARX X*冻结任务 | **FAITHFUL_PORT**：官方BC_Transformer_GMM，512维/6层/8头/10帧，30,574,315参数。明确改用固定步数constant LR，非完整原论文配置。 |
| [Diffusion Policy，RSS2023／IJRR2025](https://diffusion-policy.cs.columbia.edu/) | [作者代码](https://github.com/real-stanford/diffusion_policy) `5ba07ac6661db573af695b419a7947ecb704690f`；实际运行其[robomimic官方移植](https://robomimic.github.io/docs/introduction/implemented_algorithms.html)，提交同上 | RGB/可选状态；条件UNet去噪动作序列 | 作者权重不是本次两任务；原生不含BARX冻结评估器 | **FAITHFUL_PORT**：直接用官方robomimic DiffusionPolicyUNet，100,271,015参数、100训练噪声步、EMA；选择其已支持的10步DDIM。保留2帧观测、16步预测、8步执行。CPU构建通过，真实GPU训练待执行。 |
| [RoboCasa Diffusion Policy](https://robocasa.ai/docs/build/html/benchmarking/policy_learning_algorithms.html) | [官方fork](https://github.com/robocasa-benchmark/diffusion_policy) `41212698b6f481ed92a55f0d5f1778ec56bea417` | RGB/状态，公开默认diffusion-transformer BS192 | 面向新版RoboCasa，不直接等于固定BARX X*任务 | **NOT_APPLICABLE**作为第二篇独立DP算法计数；保留为官方配置参考。本轮主实现采用上行UNet移植，不能把它宣称成这个新版官方完整复现。 |
| [BARX No Reps／Joint Reps，ICRA2026](https://arxiv.org/abs/2607.27549) | [BARX](https://github.com/ajaysridhar0/barx) `0f435089f48b1f16d225f868dbef855a5e75d1e0` | RGB＋完整指令；源归一化VQ动作chunk8 | 原始50k模型及现有6k/96k LoRA真实权重可用，原生支持两任务 | **OFFICIAL_RUNNABLE**：模型核心/预处理复用，上层统一评估器为外部适配。Group A source prior与Group C历史LoRA分开，LoRA不是充分全参数官方适配。 |
| BARX Single Rep（bbox/language motion/EE trace）／ECoT | 同一官方源码有训练配置及transform | RGB＋指令；附加表征目标 | 截至审查，公开`ajaysri` BARX模型列表及项目artifact manifest未发现匹配Single Rep/ECoT权重 | **BLOCKED**：从头source VLA需要官方大规模预训练，不在核定单卡轻量Baseline预算内；不伪造权重，不填0%。实际公开模型清单保存于results/baselines/reproducibility。 |
| [MimicGen，CoRL2023](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf) | [官方源码](https://github.com/NVlabs/mimicgen)，使用BARX `third_party/mimicgen`固定快照 | 仿真对象位姿/操作片段；几何变换轨迹 | BARX附有X任务生成包装器；不是直接RGB学习策略权重 | **OFFICIAL_RUNNABLE**代码/生成接口，保留历史几何参考为`PRIVILEGED_GEOMETRY_REFERENCE`；本轮不重新生成示范，也不把P3 A3冒充完整MimicGen官方方法。 |
| [KITE，2026预印本](https://arxiv.org/abs/2606.22113) | [官方源码](https://github.com/helloqxwang/KITE) `136159544aea68f5e8b6cc64f17d221e825e3eee`，已实际取得 | RGB/点云、机器人状态；接触位置/法向intent及专用kinematic decoder | 有decoder/runtime/example archives；原生任务YCB Objects Placing等，不是BARX厨房X任务 | **BLOCKED（本平台移植）**，不是“未开源”：原依赖Torch2.9.1+cu128、MuJoCo3.8及PyTorch3D，和当前固定栈不同；还需RoboCasa-X接触意图数据/解码器映射，现有RLDS不含该监督。不得换环境任务来凑成功率。 |
| [CrossZero，CVPR Workshops2026](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ge_CrossZero_Zero-Shot_of_Unseen_Embodiment_Manipulation_via_Object-Centric_Hierarchical_Diffusion_CVPRW_2026_paper.html) | 从论文/主会开放页面尚未核实可执行作者仓库，source commit不适用 | 去机器人点云＋状态；分层EE/IK diffusion | 未核实可用于本任务的代码和权重 | **BLOCKED**：缺少核实的官方实现/资产且不满足纯RGB共同信息边界；不凭论文臆造实现。网页获取限制记录为获取问题，不当成算法失败。 |

## 数据和公平性

XP-900是IIWA、Kinova3、UR5e多源，每机器人每任务300条，总1800条。已逐条读取现有TFDS并核对这六个组，不使用P3的有限115条source缓存冒充完整多源池。

新学习Baseline统一只输入RGB与完整指令，以冻结CLIP-L/14文本投影作语言输入；不输入EE、夹爪状态、深度、物体真值或接触。观测历史长度遵循各模型结构。数据适配、128px resize/116px crop、固定步数优化和语言条件化与原论文设置的差异全部公开，所以标为FAITHFUL_PORT，而非“完全官方论文复现”。

依赖通过外部`deps`和PYTHONPATH隔离，固定diffusers0.11.1/huggingface_hub0.23.4，不改旧BARX虚拟环境。上游lang_utils在import时联网，外部shim只将相同CLIP-L/14投影与tokenization改为hash核验的本地权重；不改策略网络。

## 训练执行补充

上表的“构建通过/待GPU训练”记录的是训练前冻结时点。现已真实执行四类核心模型的forward/backward/optimizer核验及8个设置的首档96k训练，阶段日志、参数变化、采样覆盖和后续预算延长状态见[真实结果](BASELINE_RESULTS.md)及[机器状态](../results/baselines/status.json)。这不会将FAITHFUL_PORT改称完全官方论文复现，也不把首档完成等同于已完成正式测试。
