# Baseline真实执行结果

状态：**ACTUAL_BENCHMARK_RUNNING**。已关闭 0 个正式评测格、0 个episode。训练日志设置 8/8，选模完成 0/8。

只统计新冻结官方60–99条件；每任务每机器人每设置40次。旧P0–P3和开发smoke不加入成功率。技术错误另列，不填0%。原始小型结果见[逐episode结果](../results/baselines/episodes.csv)、[主表](../results/baselines/cells.csv)、[训练预算](../results/baselines/training_budgets.csv)、[checkpoint索引](../results/baselines/checkpoint_index.csv)。

本文件是执行中的真实快照，不能作为完整Benchmark结论。

## Group A：零目标示范迁移

| 设置 | Counter→Sink | Sink→Counter |
| --- | --- | --- |
| BARX No Reps source | 未完成（不填0%） | 未完成（不填0%） |
| BARX Joint source | 未完成（不填0%） | 未完成（不填0%） |
| BC-Transformer source-only | 未完成（不填0%） | 未完成（不填0%） |
| Diffusion Policy source-only | 未完成（不填0%） | 未完成（不填0%） |

## Group B：仅目标学习

| 设置 | Counter→Sink | Sink→Counter |
| --- | --- | --- |
| BC-GMM target-only | 未完成（不填0%） | 未完成（不填0%） |
| BC-RNN target-only | 未完成（不填0%） | 未完成（不填0%） |
| BC-Transformer target-only | 未完成（不填0%） | 未完成（不填0%） |
| Diffusion Policy target-only | 未完成（不填0%） | 未完成（不填0%） |

## Group C：源+目标学习

| 设置 | Counter→Sink | Sink→Counter |
| --- | --- | --- |
| BC-Transformer source+target | 未完成（不填0%） | 未完成（不填0%） |
| Diffusion Policy source+target | 未完成（不填0%） | 未完成（不填0%） |
| BARX No Reps legacy LoRA6k | 未完成（不填0%） | 未完成（不填0%） |
| BARX Joint legacy LoRA96k | 未完成（不填0%） | 未完成（不填0%） |

Group C历史BARX使用原LoRA权重，其目标划分、初始化、分辨率、样本量与新BC/DP不同，不构成等预算排名。

## UR5e源具身参考

| Source-only设置 | Counter→Sink | Sink→Counter |
| --- | --- | --- |
| BARX No Reps source | 未完成（不填0%） | 未完成（不填0%） |
| BARX Joint source | 未完成（不填0%） | 未完成（不填0%） |
| BC-Transformer source-only | 未完成（不填0%） | 未完成（不填0%） |
| Diffusion Policy source-only | 未完成（不填0%） | 未完成（不填0%） |

## 真实训练预算

新模型共享100条Panda示范池（每任务50），实际90条训练、10条整示范验证；源池1800条为IIWA/Kinova3/UR5e每任务每机器人300条，实际1620/180。B/C架构配对的目标采样序列、更新次数和目标暴露相同，C另加同量source，因此总样本/计算不相等。anchor不等于展开后的图像/标签总数；精确覆盖见各training目录的demonstration_coverage.csv与consumption_summary.json。

| 设置 | target anchor | source anchor | 实际更新 | 实际独立demo | GPU占用墙钟秒 | 峰值GiB | 终止/状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_bc_transformer | 0 | 96000 | 6000 | {'source': 1620, 'target': 0} | 330.1 | 1.56 | TIER_COMPLETE |
| A_diffusion_policy | 0 | 96000 | 6000 | {'source': 1620, 'target': 0} | 269.9 | 2.31 | TIER_COMPLETE |
| B_bc_gmm | 200000 | 0 | 12500 | {'source': 0, 'target': 90} | 149.2 | 0.44 | TRAINING |
| B_bc_rnn | 96000 | 0 | 6000 | {'source': 0, 'target': 90} | 334.7 | 1.32 | TIER_COMPLETE |
| B_bc_transformer | 96000 | 0 | 6000 | {'source': 0, 'target': 90} | 328.5 | 1.56 | TIER_COMPLETE |
| B_diffusion_policy | 96000 | 0 | 6000 | {'source': 0, 'target': 90} | 336.0 | 2.31 | TIER_COMPLETE |
| C_bc_transformer | 96000 | 96000 | 6000 | {'source': 1620, 'target': 90} | 635.0 | 2.52 | TIER_COMPLETE |
| C_diffusion_policy | 96000 | 96000 | 6000 | {'source': 1620, 'target': 90} | 331.4 | 2.47 | TIER_COMPLETE |

GPU秒是该模型单GPU占用的训练墙钟时间（包含数据等待和验证），不是CUDA kernel累计时间。验证每12k固定256 anchors，最终checkpoint来自预先冻结的验证延长/末步规则，未用最终测试挑选。预算到顶仍改善须标为BUDGET_LIMITED_NOT_CONVERGED；验证平台也不证明全局充分收敛。

BARX source prior复用官方50k更新权重，不把其原始大规模预训练当作本轮训练。No Reps6k旧LoRA为3000更新/batch2、3068 target+2932 source；逐图像SHA回溯全部3068 target实际覆盖18条示范（95条eligible池）。Joint96k旧LoRA为12000更新/batch8、48412 target+47588 source，实际100 target及115 source demos，source三具身均有。旧LoRA不是官方充分全参数适配。

## 同条件配对与稳定性

配对正式结果尚未完成。

新BC/DP每设置只有一个训练seed，40条件CI衡量本次测试集不确定性，不能据此宣称跨训练seed的稳定排序或跨场景泛化。配对检验为描述性、未做多重校正；加入source的结论只适用于该共享动作/观测、目标暴露匹配且C计算更多的设定。

## 本次观测到的最佳值


BARX历史LoRA在C表单独保留，不能和新C模型忽略预算差异宣称总体最强。

## 推理与资产

独占GPU、同一旧开发图像的16次warmup+80次策略调用见[latency](../results/baselines/latency/)。平均每控制步和fresh-query时间分开，chunk8缓存不会被当作每步完整推理；正式并发评测耗时不作独占延迟比较。

服务器执行根目录：`/home/users/lemon/baseline-benchmark/run-20261109`。完整视频、原生输出、12维实际控制、TCP、夹爪、物体轨迹及接触均在formal/，仓库[视频索引](../results/baselines/video_index.csv)和[所有外部文件SHA256](../results/baselines/external_files.csv)保留绝对路径。

![Official success](../results/baselines/official_success.svg)

![Validation](../results/baselines/validation_progress.svg)
