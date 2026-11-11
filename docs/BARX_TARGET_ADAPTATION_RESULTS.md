# BARX目标适配：实际执行结果

执行状态：**COMPLETE**，阶段 `collect`。
实际完成 **3000/3000 次正式optimizer更新、96000 次正式backward、768000/768000 样本位置**。
冻结提交：[6a25f330270b5fc81ce2ff0705d12d442dfc3598](https://github.com/LemonQC/tempcross/commit/6a25f330270b5fc81ce2ff0705d12d442dfc3598)。
协议与全部训练/评测控制代码在首个正式更新前已远程读回；本报告脚本只汇总实际文件，不改变训练、选模或物理评测。

训练配置为发布版Joint Reps LoRA，rank32/alpha16、AdamW 2e-5常数、BF16、batch8累积32；达到参考样本位置预算时仍不等价于论文全参数适配。单训练seed；没有新Method。

补充实现核查：shuffle buffer为**10000**，与两轮旧受限LoRA相同，低于上游所选配置类默认的256000。10000在训练前冻结代码中已固定，初版协议JSON未单列，此处补录，未改变执行配置。
实际采用BOUNDS_Q99；生效的q01、q99、mask及用于零维处理的min/max，与发布先验逐值一致。未采用的mean/std存在约3.28e-7/6.56e-6的数值差异，没有据此修改既有统计。
验证包在正式训练前保存并逐次核对SHA256；准备阶段的TensorFlow RNG状态未单独保存，精确重放以保留的验证pickle及其索引为准，不承诺仅靠seed7重新生成相同验证包。详见 `implementation_details.json`。

旧模型固定离线验证如下。Panda的320帧来自10条示范；旧6000适配实际见过其中1条，旧96000见过全部10条，因此旧adapter分数不当成干净未见验证。UR5e验证也可能在源pretraining中被见过，只诊断新增适配的遗忘。

| 旧模型 | Panda两任务平均action-response NLL | UR5e两任务平均NLL |
| --- | ---: | ---: |
| original_joint | 6.486815 | 0.279929 |
| restricted_joint_6000 | 3.566034 | 0.411181 |
| restricted_joint_96000 | 2.863985 | 0.720276 |

NLL为teacher-forced action-response（含stop token），不是闭环成功率。不得据离线loss宣称任务成功。

验证选中 **step 2500**，Panda两任务平均NLL **2.920817**。实际总训练更新数仍按上文报告，不把选中的较早step当成训练只进行了该步数。
**被物理评测的模型自身经历 2500 次更新、640000 个样本位置**；768000是完整训练运行的预算，不能写成该较早checkpoint已见过768000位置。候选仅为500/1000/1500/2000/2500/3000，按预定验证指标选模，未使用物理成功率。

选中模型输入前缀的实际域/任务计数：`[{"dataset": "mg_pnp_lite", "task": "pnp_counter_to_sink", "positions": 148689}, {"dataset": "mg_pnp_lite", "task": "pnp_sink_to_counter", "positions": 169847}, {"dataset": "panda_pnp", "task": "pnp_counter_to_sink", "positions": 149583}, {"dataset": "panda_pnp", "task": "pnp_sink_to_counter", "positions": 171881}]`；唯一示范数：`{"mg_pnp_lite": 723, "panda_pnp": 90}`。完整运行与部署checkpoint的暴露分别见 `consumption.json` 和 `selected_checkpoint_exposure.json`。

已完成 **160条新增物理rollout**。均为已暴露60–99参考条件，不能宣称独立未见测试；旧1280次实验零重跑。RoboCasa-X环境、两任务、动作接口、20Hz、600/650步上限和官方放置＋退离成功定义均未修改。

| 模型 / 已暴露60–99 | C→S Panda | S→C Panda | C→S UR5e | S→C UR5e |
| --- | ---: | ---: | ---: | ---: |
| A_barx_joint | 1/40 (2.5%) | 7/40 (17.5%) | 16/40 (40.0%) | 14/40 (35.0%) |
| C_barx_joint_96000 | 2/40 (5.0%) | 1/40 (2.5%) | 未评测 | 未评测 |
| 新适配（运行768k；选中checkpoint 640k） | 5/40 (12.5%) | 2/40 (5.0%) | 17/40 (42.5%) | 15/40 (37.5%) |

旧Joint6000在不同20–39条件为Counter→Sink Panda 3/20；不与本表当成同条件收益。数据划分、暴露预算及有效batch同时改变，当前对照不能单独归因某一个因素。

同条件逐条变化如下；“救回”为旧失败/新成功，“退化”为旧成功/新失败。它们是固定已暴露条件下的描述，不是严格因果配对或未见测试。

| 历史对照 | 任务 / 机器人 | 救回 | 退化 | 均成功 | 均失败 |
| --- | --- | ---: | ---: | ---: | ---: |
| A_barx_joint | pnp_counter_to_sink / panda | 5 | 1 | 0 | 34 |
| A_barx_joint | pnp_counter_to_sink / ur5e | 6 | 5 | 11 | 18 |
| A_barx_joint | pnp_sink_to_counter / panda | 2 | 7 | 0 | 31 |
| A_barx_joint | pnp_sink_to_counter / ur5e | 6 | 5 | 9 | 20 |
| C_barx_joint_96000 | pnp_counter_to_sink / panda | 3 | 0 | 2 | 35 |
| C_barx_joint_96000 | pnp_sink_to_counter / panda | 2 | 1 | 0 | 37 |

本配方在Panda两方向合计官方成功 **7/80**。逐任务结果应分别判断，不能用验证NLL下降替代闭环收益，也不能把增加暴露预算直接表述为跨具身问题已解决。

| 新模型 | 官方成功 | 发出闭爪 | 稳定抬升代理 | 失败阶段计数 |
| --- | ---: | ---: | ---: | --- |
| pnp_counter_to_sink / panda | 5/40 | 11 | 8 | {"NO_CLOSE_COMMAND": 29, "CONTACT_WITHOUT_STABLE_LIFT": 3, "STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT": 3} |
| pnp_counter_to_sink / ur5e | 17/40 | 36 | 19 | {"CLOSE_WITHOUT_BILATERAL_CONTACT": 9, "NO_CLOSE_COMMAND": 4, "CONTACT_WITHOUT_STABLE_LIFT": 8, "STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT": 2} |
| pnp_sink_to_counter / panda | 2/40 | 36 | 3 | {"CLOSE_WITHOUT_BILATERAL_CONTACT": 33, "NO_CLOSE_COMMAND": 4, "STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT": 1} |
| pnp_sink_to_counter / ur5e | 15/40 | 35 | 26 | {"STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT": 10, "PLACEMENT_WITHOUT_RETREAT": 1, "CONTACT_WITHOUT_STABLE_LIFT": 2, "NO_CLOSE_COMMAND": 5, "CLOSE_WITHOUT_BILATERAL_CONTACT": 7} |

关键观察：新Panda Counter→Sink仍有29/40条全程未发闭爪；Sink→Counter则有33/40条发闭爪但未形成双侧接触。此次适配未消除方向相关的启动/接触阶段缺口；阶段统计不能单独识别视觉、运动历史或动作预测的根因。
本轮唯一建议是保留此已完成的参考预算LoRA作为Baseline，如实报告收益与退化，在此停止本阶段。不能把这些结果包装成新算法必要性的证明，也不能据此否定尚未实现的充分全参数适配。

失败阶段统一重读新旧原始H5/接触JSONL，含指尖及指垫。稳定抬升仅为接触＋body原点抬升3cm持续10状态的代理；不能充作官方成功或物理根因。完整逐条结果和救回/退化集合在 `episodes.csv`、`paired_cases.csv`。

实际输入域/任务计数：`[{"dataset": "mg_pnp_lite", "task": "pnp_counter_to_sink", "positions": 175249}, {"dataset": "mg_pnp_lite", "task": "pnp_sink_to_counter", "positions": 207079}, {"dataset": "panda_pnp", "task": "pnp_counter_to_sink", "positions": 179108}, {"dataset": "panda_pnp", "task": "pnp_sink_to_counter", "positions": 206564}]`；唯一示范数：`{"mg_pnp_lite": 867, "panda_pnp": 90}`；唯一帧位置418956。实际读取trace逐条检查均来自本轮train episode，无验证示范输入。
正式训练用时15.699小时；GPU分配/预留峰值20169629184/23955767296 bytes。训练和模型选择均已完成，原物理账本哈希保持不变。

选中模型及大文件：

- 原权重：`/home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt`，SHA256 `ffce76802a8e75d4806b30dee43624accf2c4de4213c17059457e34a8243efce`。
- 选中adapter：`/home/users/lemon/barx-target-adaptation-20261110/training/checkpoints/step-002500`，SHA256 `2bf53be7da389ffa29e20733794998f99f9a14ae8671354cc86cb6f82b7a91a1`。
- 实际物理评测合并权重：`/home/users/lemon/barx-target-adaptation-20261110/adapted/checkpoints/step-002500-epoch-00-loss=validationselected.pt`，SHA256 `8bfcc5cef31bea99df8c5128491dafe7329560acde1589e61526c57496108f25`。
- 原始H5、视频、完整训练trace及验证结果的绝对路径与SHA256见 `artifact_index.json`、`physical_jobs.json`、`consumption.json`；大型文件留在服务器。

这些结果只回答此固定训练配方的适配表现；不能据单seed、已暴露条件或LoRA路线排除充分全参数适配。本阶段仅交付BARX适配Baseline，不据此提出新算法或启动其他方法。

完整协议：[BARX_TARGET_ADAPTATION_PROTOCOL.md](BARX_TARGET_ADAPTATION_PROTOCOL.md)。原始运行目录：`/home/users/lemon/barx-target-adaptation-20261110/`。

[训练与验证曲线](../results/strong_baselines/barx_adaptation/figures/training_and_validation.svg)。

[官方成功率对照图](../results/strong_baselines/barx_adaptation/figures/exposed_reference_success.svg)；误差棒为各40条条件的描述性Wilson区间，不作为独立未见泛化或因果效应检验。

交付核查：`PASS`；从160条新H5重算官方放置＋退离，核对640条新旧逐条记录与各cell/失败计数。`raw_rollout_index.json`保留全部新H5、接触JSONL和视频的绝对路径、字节数及SHA256。该核查没有策略推理或物理step。
