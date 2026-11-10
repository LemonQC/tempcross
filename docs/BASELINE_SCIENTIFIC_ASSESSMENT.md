# Baseline科研判断与交付范围

本轮实际完成8个新学习设置的训练、4份既有BARX权重复用、48次开发smoke，以及32格×40＝1280次正式物理评测；Panda960次、UR5e320次。正式条件为原官方60–99，训练/选模结束后才启动；checkpoint清单已在首次正式仿真前发布并远程读回。所有正式轨迹、原始动作及视频校验通过，正式训练和正式评测技术失败均为0；早期DP资源探针的零LR首步记录器断言错误已单独保留并修正。旧P0–P3证据保留。

## 运行的是什么实现

BC-GMM、BC-RNN-GMM、BC-Transformer-GMM、Diffusion Policy UNet直接使用固定robomimic核心，属于FAITHFUL_PORT。共用RGB与原文指令、CLIP语言投影、数据/动作适配和评估器；分辨率、裁剪、训练日程及DDIM10等区别已公开，不把本轮纯RGB配置当作原论文含机器人状态等配置的最优分数。BARX复用官方No Reps/Joint网络和预处理；历史LoRA不等于充分全参数适配。近期方法及原始论文链接见[复现矩阵](BASELINE_REPRODUCIBILITY_MATRIX.md)。

## 数据及实际预算

目标池每任务50条，共100；每任务45训练/5整示范验证。源池为IIWA/Kinova3/UR5e每机器人每任务300条，共1800；1620训练/180验证。所有新模型实际消费覆盖全部相应训练示范ID。

| 新设置 | target anchor | source anchor | 实际更新 | batch |
| --- | --- | --- | --- | --- |
| A Transformer / A DP（各自） | 0 | 192000 | 12000 | 16 |
| B GMM / B RNN / B DP（各自） | 384000 | 0 | 24000 | 16 |
| B Transformer | 768000 | 0 | 48000 | 16 |
| C Transformer | 768000 | 768000 | 48000 | 32 |
| C DP | 384000 | 384000 | 24000 | 32 |

全部按预先冻结的验证延长规则选择末步checkpoint。验证最近窗口不再改善不证明全局充分收敛；历史最小验证值和末步值均报告。训练循环计时合计约3.44小时，不含模型加载/构建、初始验证和保存哈希。原始更新CSV、各demo覆盖及参数变化见[训练预算](../results/baselines/training_budgets.csv)。

旧No Reps6k为3000更新、3068 target＋2932 source，精确缓存回溯实际18条target/19条source示范；95条是可采样目标池。旧Joint96k为12000更新、48412 target＋47588 source，实际100条target/115条source，源三具身均有。它们的划分、初始化、分辨率与预算不同，单列历史设置。

## 实际成功与最强观测值

两个任务的全部12个Panda设置及4个UR5e参考、Wilson95%CI见[主结果](BASELINE_RESULTS.md)与[原始逐格表](../results/baselines/cells.csv)。当前观测值不支持忽略预算的统一算法排名：

- 零目标示范组：Counter→Sink最高为BARX No Reps源先验5/40；Sink→Counter最高为BARX Joint源先验7/40。
- 仅目标组：Transformer两个任务均为1/40，其余三类均为各任务0/40。
- 新的源＋目标组：Transformer为4/40、0/40，DP为0/40、0/40。
- 历史源＋目标设置：No Reps6k为3/40、2/40；Joint96k为2/40、1/40，不能与新模型忽略预算比较。
- UR5e源参考：Joint为16/40、14/40，No Reps为6/40、5/40；新source-only Transformer及DP均为两个任务各0/40。

Source-only新模型在UR5e也未完成任务，因此其Panda零成功不能单独归因跨具身迁移。Joint源先验的UR5e/Panda落差值得记录；目标域纯RGB学习本身也较弱。本轮没有从这些现象确定必须由新算法解决的独立因果机制。

## 源数据是否稳定有益

新B/C的初始化文件及架构/优化配置SHA相同，实际target ID/anchor有序序列哈希与更新次数相同，见[匹配审计](../results/baselines/matched_data_audit.json)。C加入等量source，总数据与计算更多；随机训练配置相同，逐样本随机crop实现及混合batch统计并不相同。

Transformer Counter→Sink救回75、77、87号，没有退化，1/40→4/40，精确双侧McNemar p＝0.25；反向任务退化64号，1/40→0/40，p＝1。DP未出现成功变化。仅一个训练seed、每任务40条件，未得到跨任务一致且统计充分的源数据收益证据，也未证明普遍负迁移。检验未做多重校正，不能把观测最大值称为稳定最强方法。历史LoRA比较属于不同训练预算的适配前后记录。

## 操作断点与技术边界

Panda932次失败中，745次观察到闭爪但没有双指接触；其余包括未闭爪131次、接触后未稳定抬升25次等。阶段定义及全表见[失败分析](BASELINE_FAILURE_ANALYSIS.md)，[配对变化视频路径](../results/baselines/paired_discordant_cases.csv)可直接定位救回/退化案例。阶段代理说明观测断点，不识别视觉、控制、数据或时序建模的唯一根因；官方task success始终是主判据。表中的0/40均执行了完整40次。

KITE有官方源码，当前阻碍是固定旧栈与其Torch/CUDA/MuJoCo/PyTorch3D依赖及BARX接触意图/decoder数据接口；CrossZero尚未核实可执行作者实现和本任务资产，原点云输入也超出共同边界。BARX Single Rep/ECoT未核实匹配预训练权重，从头源VLA训练超出本轮预算。它们标为BLOCKED，未填成0%。

## 平台是否就绪

工程平台已就绪：数据ID与划分、动作接口、冻结状态、评价循环及原始证据可复用；新策略需实现BasePolicy并登记工厂/CLI后使用同一评价循环。当前32格全部通过校验，但一训练seed、两任务及已声明的移植/受限适配配置尚不足以证明稳定SOTA排名或T-RO科学创新。

[复现命令](BASELINE_REPRODUCTION.md)、[权重索引](../results/baselines/checkpoint_index.csv)、[1280条视频索引](../results/baselines/video_index.csv)、[所有外部文件绝对路径/大小/SHA256](../results/baselines/external_files.csv)均已提供。大型文件保留在`/home/users/lemon/baseline-benchmark/run-20261109`及索引列明的既有资产目录。
