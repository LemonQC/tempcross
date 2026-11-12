# VQ-BeT移植及冻结协议

官方源码固定为 `jayLEE0301/vq_bet_official@09d4851288ca5deaaa1ab367a208e520f8ee9a84`，MIT。官方模板需用户填写任务维度和数据路径；本实验属于有明确I/O差异的faithful port，不声称原生UR3任务完整复现。

保留官方残差VQ-VAE（512 latent、2组16码）、六层六头120维GPT、focal码损失、随机码采样、连续偏移预测及官方前半段双优化器/后半段偏移优化器的训练安排。只在条件接口增加冻结CLIP-L语言的768→1024线性前缀投影，并注册到官方optimizer1；原生视觉、量化和码/偏移模块不替换。原生目标图像/目标状态监督没有引入。实际逐文件比对见official_source.json，三处接口变更见official_io.patch。

使用官方已发布支持的冻结ResNet18视觉特征模式。RGB先与已有packed数据一致缩为128×128，再按官方路径nearest缩为224；保留源码蓝通道std=.255，不悄悄替换为常见的.225。模型不观察关节、点云、真值物体状态或未来图像。动作保留原始canonical OSC[-1,1]，act_scale=1，不使用ACT的均值方差归一化；夹爪正值闭爪。历史10帧严格截至当前帧，动作窗口1采用官方UR3配置，控制20Hz，部署每步预测一次。

完整示范身份划分保持不变：每任务Panda45条训练、5条验证；source+target另外使用每任务270条UR5e训练示范，不使用IIWA/Kinova数据。两设置各阶段目标暴露相等，joint加入1:1源样本。验证仅使用原留出目标示范的256固定anchor，不使用60–99成功率。

候选预算在真实探针之前写入candidate_protocol.json：每设置RVQ6000次更新、目标batch64；BeT24000次更新、目标batch16；joint额外等量源batch。每个阶段每设置目标anchor384000。RVQ每500更新按验证重构L1选模；BeT每1000更新按固定随机种子的官方总验证损失选模，相等取更早。前12000次更新两个优化器，后12000次只更新偏移，显式冻结不更新参数以免无效梯度积累。预算是单卡有界移植预算，不等同论文各原生数据集的epoch/batch2048总计算量。

正式训练前，两设置分别进行一次真实RVQ更新及一次真实BeT更新，核对GPT、码、偏移、语言参数实际变化；然后在两任务各做condition0完整物理smoke。smoke检查有限动作、原始接口和官方成功谓词记录，不设置任务成功阈值。丢弃探针权重，通过后才生成protocol.json及freeze.json。正式每模型每任务40条，仅60–99，标记EXPOSED_REFERENCE。初态、控制器、完整placement AND retreat成功定义不变，原1280+160条不重跑。

停止条件：固定更新预算、非有限数、磁盘低于25GiB、每设置阶段12小时墙钟上限或实质障碍。最多一次记录在案的算法技术修复；所有失败保留。checkpoint按组件仅保留验证best和可恢复latest，探针另存。每250更新保存权重/优化器/RNG，SIGTERM在更新后保存；若日志发现未入checkpoint的已执行后缀，拒绝静默重跑。物理完整episode只读复用，不完整episode须保留并停止自动重试。不得依据reference结果选模型或调参。

正式训练前唯一技术修复：关闭cuDNN及matmul TF32，统一冻结视觉缓存与部署的计算精度。初始缓存NCHW/batch128与原生图像NHWC/batch10在TF32默认开启时有0.00654最大特征差；相同NCHW/batch128可精确复现原缓存，关闭TF32后不同布局差降至约1.4e-5。权重完全相同，未改变动作接口或调参。原四次组件探针更新（六次优化器调用）及四条完整smoke保留在原服务器路径和attempts/initial_tf32/；正式训练之前重建精度一致的缓存，并重新执行独立四条smoke。八条smoke均不计正式成功率。无剩余技术修复额度。

保留官方优化器分组细节：GPT衰减组weight_decay=.0002、无衰减组0；官方后加的码预测头和视觉header继承AdamW默认.01，语言header同样加入该组；偏移optimizer2显式.0002。不把YAML全局.0002误报为所有组的实际衰减。实际各组学习率、衰减、参数数目写入训练状态。官方码损失主码系数5、次码.5，偏移L1系数100，均保留。

运行前设置 `STRONG_BASELINE_RUN_ROOT=/home/users/lemon/strong-baselines-20261112/vqbet_input_repair_01`。然后 `python -m results.strong_baselines.vqbet.prepare` → `python -m results.strong_baselines.vqbet.run probes` → 测试及原生图像/缓存推理一致性验收 → `python -m results.strong_baselines.vqbet.run freeze` → `python -m results.strong_baselines.vqbet.run formal`。修复版大文件在上述根目录下的vqbet/；原路径 `/home/users/lemon/strong-baselines-20261112/vqbet/` 完整保留。代码、协议、更新/验证统计和索引提交GitHub。
