# P3-B：96000样本位置的增强受限Joint LoRA实际完成

已完成 **12000次真实optimizer更新 × batch 8 = 96000样本位置**，随后仅用预指定最终checkpoint完成独立40–59的20次物理评测，官方成功为 **2/20（10%）**，Wilson 95%区间为2.8%–30.1%。成功条件为[47, 58]；相对原始Panda救回[47]、退化[55]，不以平均数掩盖失败集合变化。

增强普通适配为 2/20，未超过原始Panda的 2/20，低于源机器人 6/20。本次受限配方没有解决完整任务差距；它仍只有官方暴露量的1/8，不能据此否定充分标准适配或所有标准方法。

## 实际训练、来源与资源

batch=8的独占GPU单步forward/backward/optimizer探针真实通过，探针权重丢弃；正式模型从原始XP-900 Joint Reps 50k重新开始。实际路线是单GPU上游DDP LoRA，rank32、alpha16、dropout0、AdamW LR2e-5常数、weight decay0、BF16，没有新架构。固定最终更新数12000，没有根据40–59挑checkpoint。用时7622.8秒，主训练峰值分配20086709760 bytes（18.71 GiB）、预留23702011904 bytes（22.07 GiB）。没有资源OOM，因此未继续尝试batch4/2。

全部12000次backward和optimizer记录完整、损失有限、nonfinite计数0。前500步原始loss均值0.816815，末500步0.968202；训练loss不能代替闭环成功率。初始/最终adapter逐张量比较：768个保存张量中 **742个真实改变**，最终值均有限。探针的一步参数核验与正式模型分开保存，不将非零初始化误当成更新。

源/目标实际输入为47588/48412位置；源池1800、实际涉及115条；目标两个方向各50条、全部100条原始示范均被访问。目标方向实际位置数：`{"PnPCounterToSink": 22878, "PnPSinkToCounter": 25534}`。88261个唯一原始帧位置、7739次重复帧、138次预处理内容完全重复；目标40673/41065原始帧被采到，59/100目标示范帧覆盖完整。它不是机械重复旧6000缓存。完整实际消费CSV与准备序列SHA256逐字节相同：`9edb3945712159549d52f3708f1d181f8305fe5ac29cf45eedc5cd731c57d1df`；每个输入的原始payload hash和episode/frame身份在读取时均核对。

沿用官方RLDS源/目标transition混合权重0.05:1、源归一化与VQ；归一化统计与P2逐值相同。本轮显式使用全train split（100目标/1800源），P2默认95%，同时batch和更新数均增加，因此不能把效果单独归因为训练步数。这仍是 **增强预算的受限LoRA，只有官方参考768000样本位置的1/8**，并非完整标准微调。仅训练Joint Reps。

## 最终checkpoint和真实加载评测

- 原始checkpoint：`/home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt`，SHA256 `ffce76802a8e75d4806b30dee43624accf2c4de4213c17059457e34a8243efce`。
- 最终adapter权重：`/home/users/lemon/coft-p3/p3-20261109/training/joint-reps-96000-b8/runs/joint_reps--lora/checkpoints/step-012000-epoch-01-loss=0.7951/adapter_model.safetensors`，SHA256 `dea19a43fb9dbbe8d14e119c87e550339913a9e202e50e7ee67a834020f9fb25`。
- 实际评测的合并checkpoint：`/home/users/lemon/coft-p3/p3-20261109/adapted/checkpoints/step-012000-epoch-01-loss=restricted96000.pt`，SHA256 `c6fb093b9071886f93aac233f76bf3333cb98b09f959930fd5c3e469fd738b5c`。

评测job中的路径/文件hash、合并provenance和training suite已强制核对一致；它不是仅写出了脚本。最终模型20次视频/状态全部校验，稳定接触抬升为2/20，官方成功为2/20，两者分开报告。阶段观测为：**15条整段未发闭爪命令，2条闭爪但无双指接触，1条有接触但无稳定抬升，0条抓持后失败，2条官方成功**。未发闭爪命令是实际执行层观测，不能未经进一步验证就断言神经网络没有抓取意图或表达能力。

[每步原始损失与优化器统计](../evidence/p3/execution/training/) · [实际参数变化](../evidence/p3/execution/parameter_changes.json) · [输入及示范覆盖](../evidence/p3/execution/data/) · [同条件救回/退化](../evidence/p3/execution/paired_cases.csv) · [完整权重路径/大小/SHA256](../evidence/p3/execution/model_artifacts.csv) · [训练曲线](../evidence/p3/execution/plots/)。

## 技术恢复与可解释性限制

已保留两次技术恢复证据：96000缓存写完后覆盖CSV字段不一致，仅恢复统计而未重采样；追踪器首次接入受上游argv重写影响，0更新/0样本即失败，修复固定run label后batch8探针及正式训练成功。这些不是资源OOM，也不是任务0%成功。所有原始日志保留服务器。

目标数据与40–59检查未发现可检测的精确物理状态重叠，但缺少逐示范XML/关节排序，不能穷尽证明轨迹独立。单训练seed、20个新条件的结果也不能证明充分标准适配已经穷尽。
