# VQ-BeT

**COMPLETE_FAITHFUL_PORT**：两设置均完成原生RVQ及BeT训练和两个任务各40条正式物理评测。

结果：[RESULTS.md](RESULTS.md)；协议：[PROTOCOL.md](PROTOCOL.md)；论文/发布代码差异：[PAPER_CODE_DIFFERENCES.md](PAPER_CODE_DIFFERENCES.md)。

原始逐episode结果：episodes.csv。权重/训练日志：model_index.json、training_artifact_index.json；选中checkpoint样本暴露：selected_checkpoint_exposures.json。全部大文件在服务器，路径与SHA见sha_index.json；原TF32尝试见attempts/initial_tf32/。

formal训练两组件合计60000次阶段更新、84000次优化器调用；加两轮探针合计60008次阶段更新、84012次优化器调用。八条smoke与160条正式评测严格分列。

仅使用EXPOSED_REFERENCE条件60–99，不称独立未见测试；未用物理成功率选模。数据/输入/真实参数更新/码本冻结/优化器计数/物理视频动作和原始基准保护均已验收，见delivery_acceptance.json。历史PARTIAL文件保留为当时快照，不代替最终结果。
