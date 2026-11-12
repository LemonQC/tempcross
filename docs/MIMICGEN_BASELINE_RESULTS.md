# MimicGen有界Baseline：实际结果

状态：**BLOCKED**。完整MimicGen学习Baseline尚未形成；未把几何生成器或单条成功轨迹当成部署视觉策略。

阻碍发生在额外训练数据生成阶段：冻结目标为两任务各50条成功示范，每任务最多100次尝试。当前固定源池/配方在Sink→Counter未达到门槛，未降低数据目标、扩大生成量或改训练配置。原始阶段错误与精确退出信息见`phase2_execution.json`。

| 任务 | 官方成功 / 实际额外生成 | 目标成功数 |
| --- | ---: | ---: |
| pnp_counter_to_sink | 50/77 (64.9%) | 50 |
| pnp_sink_to_counter | 13/100 (13.0%) | 50 |

额外生成共177条物理轨迹、63条官方成功，合计76828个控制步。全部成功与失败都保留，所用为既有开发条件0–19的变体，不是独立测试样本。

## 已完成与未完成

- 首轮生成4/5、对应动作重放4/5，共8/10；五对相关轨迹仅用于可行性门槛。逐条视频、动作、接触、物理状态及重放初态/动作前缀通过核查。
- 已审查两项官方生成接口，准备每方向固定5条Panda human train的原始状态注释；没有使用人类validation示范生成数据。源身份与全部父片段选择分别见`source_seed_manifest.json`、每次lineage原件。
- 已冻结唯一BC-Transformer配置、seed7、48000更新/768000 anchors、固定验证和checkpoint选择规则。它们只是已准备的执行代码/配置，不能写成已训练。
- 实际下游optimizer更新：**0**；checkpoint：**未产生**；选模：**未执行**；正式视觉策略评测：**未执行，成功率N/A**。
- 原1280＋160条Baseline物理实验零重跑，原输入和原HDF5保持SHA256不变。没有启动其他Baseline或新Method。

## 实现修复及复现边界

预检发现原源动作反解缺少base→world旋转，完成一次隔离注释修复；纯数学往返测试不当成物理成功率。探针第三条的NumPy int64 JSON索引记录错误使用物理探针阶段单独允许的一次修复，已有403步失败轨迹只读恢复，未重新执行。首次协议将两阶段修复合并的勘误在剩余探针前公开提交。详见`probe_logging_repair.json`和协议追加说明。
崩溃时NumPy RNG状态未保存，剩余探针按预注册seed7重启，未声称与无中断执行逐位一致；额外生成则从开始就按每attempt固定种子记录。没有根据成功率换seed、源池、噪声或控制器。

源示范是Panda目标域human train每方向5条，而非零目标示范的UR5e迁移。生成器使用训练时物体/目标/EE位姿和接触阶段；原定部署模型仅RGB＋原文指令。额外生成成本应与BC/DP/BARX分列，不能声称等数据公平。

## 已有生成轨迹的只读失败诊断

| 任务 | 成功/已完成生成 | 物理步数 | 失败阶段计数 |
| --- | ---: | ---: | --- |
| pnp_counter_to_sink | 50/77 | 27895 | {"CONTACT_WITHOUT_STABLE_LIFT": 17, "SUCCESS": 50, "CLOSE_WITHOUT_BILATERAL_CONTACT": 4, "STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT": 6} |
| pnp_sink_to_counter | 13/100 | 48933 | {"CONTACT_WITHOUT_STABLE_LIFT": 27, "STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT": 23, "CLOSE_WITHOUT_BILATERAL_CONTACT": 37, "SUCCESS": 13} |

阶段代理是已保存实际动作、双侧接触、物体body原点上升3cm持续10状态及官方放置/退离的描述，不识别严格物理根因。生成率低不能归因于视觉策略，因为根本未进入学习阶段；也不能据此断言MimicGen一般无效或证明需要新算法。
本轮科学结论限定为：当前固定源池、原任务配置及有界数据预算未形成预注册的两任务强对照。保留这些真实结果并在此停止，不用替代规则控制器、少量成功示范或未运行评测填补完整Baseline。

原始目录：`/home/users/lemon/mimicgen-baseline-20261112`。实际物理账本：`physical_launches.jsonl`；逐条结果：`generation_and_replay_episodes.csv`、`generation_failure_episodes.csv`；分条件结果：`generation_condition_cells.csv`。视频、动作、接触和状态的绝对路径、字节数/SHA256见`external_file_index.json`，大文件留服务器。
协议：[MIMICGEN_BASELINE_PROTOCOL.md](MIMICGEN_BASELINE_PROTOCOL.md)。本次汇总没有新物理step或模型调用。

## 数据完整性

177条额外生成轨迹全部通过原开发条件初态逐值比对、动作与telemetry一致性、有限物理状态、官方placement AND retreat、接触状态数量及父示范谱系核查。每条额外视频核对容器帧数和首帧解码；首轮10条探针视频另做了逐帧完整解码，未把两种检查范围混同。证据见`generation_integrity.json`、`probe_validation.json`。

完整额外生成谱系已汇总为`all_generation_lineage.json`，包含每个实际选择的父示范ID、子任务索引、生成seed及原始路径。
