# MimicGen v2结果

**PARTIAL；有界下游组件完成**。不等量50/13生成数据和Panda源身份使本实验不能代替完整MimicGen跨具身基线；原50+50门槛BLOCKED保持不变。

| 任务 | 旧人类BC对照（复用） | 新human+generated策略 | 新评测次数 |
|---|---:|---:|---:|
| pnp_counter_to_sink | 1/40 | 2/40 | 40 |
| pnp_sink_to_counter | 1/40 | 0/40 | 40 |

全部均标EXPOSED_REFERENCE，不是独立未见测试。新策略仅一次48000更新、固定最终checkpoint，与复用对照的架构、随机初始参数、总batch/更新预算匹配；不把旧自适应停止程序与新固定预算称严格相同的因果对照。

实际768000训练anchor中，人类479469，生成288531；每anchor监督最后一步动作，加载10帧因果历史。原始消耗逐行核对，见actual_exposures.json。

177条既有生成记录全部核查并解码视频，63条成功示范全部使用；没有新增生成、重跑旧BC物理评测或增加目标人类示范。新策略80条正式评测与两条smoke分开，失败阶段为描述性代理，完整目标仍为placement AND retreat。

权重、原始轨迹、视频和训练日志的服务器绝对路径与SHA见model_index.json、sha_index.json；既有生成大文件见existing_generation_index.json。
