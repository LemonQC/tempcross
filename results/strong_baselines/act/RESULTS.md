# ACT结果

**COMPLETE_FAITHFUL_PORT**：官方CVAE/动作分块/时序集成核心保留；RGB＋语言到canonical OSC接口适配差异见PROTOCOL.md。

| 设置 | 任务 | 官方成功/实际评测 | 成功率 | checkpoint step |
|---|---|---:|---:|---:|
| target_only | pnp_counter_to_sink | 5/40 | 12.5% | 17000 |
| target_only | pnp_sink_to_counter | 4/40 | 10.0% | 17000 |
| source_target | pnp_counter_to_sink | 8/40 | 20.0% | 11000 |
| source_target | pnp_sink_to_counter | 2/40 | 5.0% | 11000 |

唯一一次技术修复把官方第一层decoder中间输出改为最后层输出，实际七层均更新。原版target-only的24000次训练、2次探针更新和4条smoke完整保留并单列，不进入修复版正式成功率。

条件60–99全部为EXPOSED_REFERENCE，不是独立未见测试。模型只由留出示范验证损失选择；没有根据这里的成功率调参。

每设置实际24000次更新，384000个目标anchor暴露；joint另有384000个UR5e anchor。训练合计5527.6秒。探针2次更新和4条物理smoke单列，不计正式160条。

全部失败轨迹保留在episodes.csv。原始视频、状态、动作、接触及checkpoint的服务器绝对路径和SHA256见sha_index.json。失败阶段为描述性代理，不推断严格物理根因。

原1280＋160条结果及其启动账本哈希不变；TOCAT分支未使用。
