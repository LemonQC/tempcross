# ACT：完整faithful port结果

**COMPLETE_FAITHFUL_PORT**。本目录包含两种设置的实际训练与160条真实官方完整任务评测；没有声称原平台/原生观测完全复现。

[最终结果](RESULTS.md) · [协议及差异](PROTOCOL.md) · [验收](delivery_acceptance.json) · [checkpoint与实际暴露](selected_checkpoint_exposures.json)

| 设置 | 任务 | 官方成功/次数 | 成功率 |
|---|---|---:|---:|
| target_only | pnp_counter_to_sink | 5/40 | 12.5% |
| target_only | pnp_sink_to_counter | 4/40 | 10.0% |
| source_target | pnp_counter_to_sink | 8/40 | 20.0% |
| source_target | pnp_sink_to_counter | 2/40 | 5.0% |

全部正式条件为EXPOSED_REFERENCE。checkpoint只由留出示范验证损失选择。

每设置实际24000更新，目标anchor暴露384000；joint另有384000个UR5e source anchors。所选checkpoint的实际更新数和暴露另列，不能用完整训练成本替代。

唯一一次技术修复为读取最后decoder输出。此前上游首层输出版本的24000更新、2次更新探针和4条smoke完整归档，未混入修复版160条正式结果；修复版另有2次更新探针和4条smoke。

历史阶段快照`PARTIAL_RESULTS.md`及`partial_results.json`保留；当前完整状态以本README、RESULTS.md和status.json为准。

大文件的绝对路径和SHA见sha_index.json。旧BC/DP/BARX、BARX适配结果及TOCAT独立分支均保留。此入口不启动VQ-BeT或其他算法。
