# MimicGen v2

当前状态见status.json。原MimicGen50+50门槛保持BLOCKED；本目录是使用既有50/13成功生成示范的有界下游组件对照，不是完整跨具身复现。

协议见[PROTOCOL.md](PROTOCOL.md)，既有数据全量审计见data_audit.json及existing_generation_records.csv。大文件根目录 `/home/users/lemon/strong-baselines-20261112/mimicgen_v2/`；未增强BC对照复用旧权重与结果，不重跑。

顺序：audit_inventory → prepare → test_data及一次真实更新/两条完整smoke → run freeze → run formal。最终需要RESULTS.md、results.json、episodes.csv、训练及原始大文件SHA索引；未评测单元不填写0%。
