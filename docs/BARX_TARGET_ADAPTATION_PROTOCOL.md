# BARX 目标适配：正式协议

本阶段只补强 BARX Baseline，不开发 Method、网络或新任务。起点为 `ad14daaabdab8c4f3ceae97dd50b0a4882b01abb`，官方源码固定为 `0f435089f48b1f16d225f868dbef855a5e75d1e0`。机器可读配置：[protocol.json](../results/strong_baselines/barx_adaptation/protocol.json)；逐文件源码和旧配置核查：[recipe_audit.json](../results/strong_baselines/barx_adaptation/recipe_audit.json)。

## 正式对象与官方配方的关系

正式新训练只有**一个 Joint Reps 目标适配配置**，从原发布的 XP-900 Joint 50k 权重重新开始。比较对象为原始 Joint、旧 P2 Joint 6000位置、旧 P3 Joint 96000位置。旧 No Reps 配置也核查并保留其历史结果，但本阶段不另开 No Reps 新训练、不并行扩展其他方法。

官方文档参考为全参数、8卡×32、global batch256、3000更新，共768000样本位置；LR2e-5、constant、无图像增强、两方向联合训练、XP-900源权重0.05/目标1、沿用源归一化和VQ。原训练入口、训练策略工厂、LoRA、FSDP、数据转换、动作解码、配置均已检查。[固定版本的官方说明](https://github.com/ajaysridhar0/barx/blob/0f435089f48b1f16d225f868dbef855a5e75d1e0/docs/experiments.md)；本地原件与SHA256在源码审计中。

现有设备为一张24GB RTX3090。历史全参数batch1真实尝试在0更新时OOM，分配峰值23224258560 bytes；这是该实现路径的实际限制，不是所有CPU offload或其他硬件方案不可能的证明。本轮未引入量化、改结构或另一种算法，使用发布代码自带的 `ddp-lora`：rank32、alpha16、dropout0，原工厂选取的Linear/Embedding/Conv2d/Conv1D模块，AdamW、weight decay0、梯度裁剪1、BF16。LoRA构造器实际关闭activation checkpointing，即使旧配置字段写着开启；没有将配置字段误当成已生效优化。

新适配的**样本位置预算达到参考量，参数更新方式仍是LoRA**。它不等价于论文全参数复现，不称为“已穷尽充分标准适配”。旧两份受限配置也继续明确称为受限适配。

## GPU及真实运算检查

新单卡batch8官方训练探针完成1个forward/backward/optimizer更新，分配峰值19811245056 bytes；权重弃用。另完成实际累积探针：32个batch8 backward组成一个256样本更新，371个adapter张量改变，梯度有限；更新耗时17.62秒，对3000更新的粗估为14.68小时，不含正式数据管道、验证及物理评测开销。[累积探针](../results/strong_baselines/barx_adaptation/accumulation_probe.json)

官方VLA循环明确断言不支持累积。本轮新增的是可复核的训练执行循环：把256样本分成8个虚拟rank，每rank32样本；每个rank内部按实际监督token数加权四个batch8 loss，再对8个rank取平均，避免对不同长度回答的microbatch均值直接平均。使用同一官方模型、转换、loss、LoRA工厂、优化器及裁剪，没有新学习目标。

真实固定batch8与四个batch2的加权梯度比较，在关闭随机性的eval模式、仍开启梯度下进行：BF16相对梯度L2差约**7.13%**，loss约0.95434/0.95939，在检查前设定的10%梯度、2% loss相对容差内。不能将其说成数值严格等价于8卡FSDP或bitwise复现。正式训练恢复train模式。首次新执行包装器探针曾因裸模型输入未移至GPU而失败，0正式更新；已修复并保留完整技术记录，未将技术失败计成任务失败。

## 数据与验证隔离

沿用仓库既有、按原始示范身份hash分层的整条episode划分，禁止相邻帧随机切分：

| 数据 | 训练示范 | 验证示范 |
| --- | ---: | ---: |
| Panda Counter→Sink | 45 | 5 |
| Panda Sink→Counter | 45 | 5 |
| 既有XP-900源池，两任务合计 | 1620 | 180 |

源池保留原官方既有混合，其中含IIWA/Kinova3/UR5e；不下载或新增其他源数据。物理源域评测机器人仍只有UR5e，目标仍为Panda，两者均Robotiq85。保留原源池是为了保持BARX原co-finetuning数据设置，不把它隐写成“源训练只含UR5e”。

在原RLDS loader中先捕获规范原始动作/指令的轨迹内容身份，再在**归一化、未来动作和辅助轨迹窗口构造之前**按整条身份筛选，训练只保留train episode。沿用官方224px单外部RGB、无腕相机、无增强、8动作chunk、future object/EE窗口40及stride8。源/目标权重0.05:1按各自训练transition数再平衡；训练从实时原始RLDS流采样，逐位置保存episode、frame、变换类型和原始payload hash，不重复播放旧6000/96000缓存充当新数据。

归一化和VQ使用既有源prior，不重新拟合目标验证统计。只为采样计数更新训练池大小。完整[划分表](../results/strong_baselines/barx_adaptation/split_manifest.csv)、[划分说明](../results/strong_baselines/barx_adaptation/split_summary.json)和[256条实际训练输入管道检查](../results/strong_baselines/barx_adaptation/train_input_probe.json)均可核查。

目标验证固定为10条Panda验证示范每条32个均匀帧，共320；源诊断为60条UR5e验证示范各4帧，共240。均使用原RLDS转换，无物理step。主指标为两个任务分别计算的teacher-forced action-response NLL的平均，包含stop token；同时报告response token accuracy。它们不是自回归闭环成功率。

验证隔离只针对**本轮新适配**：旧P2实际消费过10条Panda验证示范中的1条，旧P3消费过全部10条。故旧adapter验证是带明确已见暴露标记的描述性对照。源验证也可能在原pretraining中被见过，仅诊断新增适配的遗忘。不能把这些指标宣称为所有模型共同的未见泛化测试。

## 冻结预算、选模和停止规则

正式预算固定为 **3000次optimizer更新 × effective batch256 = 768000样本位置、96000个microbatch backward**；单训练seed7，microbatch8、累积32。探针参数全部丢弃，正式模型从原始50k权重重载，不继承旧LoRA。

只在500/1000/1500/2000/2500/3000更新保存候选并进行固定验证。选Panda两任务平均action-response NLL最低的**已训练**checkpoint；并列选更早更新。step0仅作参考，不参与选模。源NLL、训练loss及任何物理成功率均不用于选checkpoint。

无验证驱动early stopping，训练完成既定3000更新；遇非有限loss/梯度、OOM、数据/源码hash不符、剩余磁盘低于25GiB或训练墙时超过36小时立即停止，保留技术证据。无自动降batch、换学习率、换算法、增预算或失败后隐式重训。不通过物理测试分数改模型；技术限制若触发，报告实际完成量，不能用计划量冒充。

配置、划分、验证帧索引及全部执行代码在正式训练前提交GitHub并读回；服务器 `frozen_publication.json` 记录远程SHA及文件哈希。训练入口强制核对冻结内容。协议冻结后不得按测试表现修改。

## 物理评测与历史对照

只评测验证选出的一个新checkpoint，固定**两任务×UR5e/Panda×40＝160条新增rollout**，使用已有60–99精确保存的模型和初态。它们标为 **EXPOSED_FROZEN_REFERENCE_NOT_INDEPENDENT_TEST**；不能再宣称独立未见测试。原有1280次物理实验零重跑，原始模型与旧LoRA的物理对照只读取历史结果。

评测复用原 `baselines/common/evaluator.py`、原动作转换、20Hz、8动作队列、原RoboCasa-X任务、原600/650步上限及官方 `info.is_success.task`（放置＋退离）。新wrapper只改变输出目录和诚实的评测角色元数据，并逐条核对恢复初态、动作映射、查询队列、H5有限值和完整视频。

旧Joint6000仅有Counter→Sink的20–39物理结果，不能与60–99当成同条件改进；旧Joint96000在60–99已有两任务结果，可与新模型作已暴露同条件描述性比较。数据划分、预算、有效batch同时改变，任何收益不能单独归因为某一因素。

失败分类在新旧已保存轨迹上用同一只读程序计算：无闭爪、闭爪无双侧接触、有接触无稳定抬升、稳定抬升无官方放置、放置无退离、官方成功。接触含左右指尖或指垫，抬升为body原点上升3cm并持续双侧接触10状态；这些只作阶段代理，不替代官方分数或因果结论。

## 执行与交付

全部新代码在 [code/](../results/strong_baselines/barx_adaptation/code/)：`prepare.py`/`data.py`数据、`train.py`真实训练与验证、`merge.py`合并、`evaluate.py`原管线评测、`baseline_validation.py`旧模型离线验证、`collect.py`汇总、`supervise.py`有限顺序执行。协议、旧配置、真实资源检查、划分、验证、每次更新、成功率、失败集合及路径校验均输出至 `results/strong_baselines/barx_adaptation/`。

大权重、逐样本完整trace、H5和视频留在 `/home/users/lemon/barx-target-adaptation-20261110/`，提交绝对路径、字节数及SHA256索引。新旧实验账本分开，所有原Git历史、Baseline结果及输入保持不变。本阶段完成后不进入其他方法或新Method。
