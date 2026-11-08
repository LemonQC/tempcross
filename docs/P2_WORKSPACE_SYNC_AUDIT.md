# P2 本地工作区与远程同步审计

本次审计确认：P2 新物理干预和目标适配尚未执行。此前交付的是分析与 dry-run，不是 P2-B/P2-C 实验结果。本地四份分析产物此前没有提交或推送；这是实际的同步缺口。

用户本次明确要求先不重新运行训练或仿真。本次仅核查、同步、整理证据，并继续不涉及新训练/仿真的离线分析。机器可读状态见 [status.json](../evidence/p2/status.json)。

## Git 核查

- 实际项目目录：`/home/users/lemon/CodexProject/tempcross`；原始会话目录 `chatgpts-main` 是历史 IHCL 项目。
- remote fetch/push：`https://github.com/LemonQC/tempcross.git`。
- 审计开始时：本地 `main` 和过期的本地 `origin/main` 都为 `aa23600dc2e41d1ca3aab8e40c4caf66cb529fd9`。
- `git ls-remote` 与 GitHub connector 独立确认远程 main 为 `c8ecb6ef36bed4e06a109795636f62ca990ff52b`，新增文件仅 `202611082025.md`。
- 已 fetch 并 fast-forward 到该远程提交，没有 reset、force push 或覆盖原 80 次结果。
- 只有一个 tempcross 本地分支和一个注册 worktree。检查了 reflog、不可达 commits、`CodexProject` 和外部 `coft-p2` 工作区；其他 Git checkout 是 BARX 与依赖检查副本，没有找到隐藏的 P2 实验分支。

## 找回的实际文件

以下四份是此前未提交的分析证据，现在一并入库：

- `docs/EXPERT_REVIEW_20261108_ZH.md`
- `docs/EXPERT_REVIEW_20261108_AUDIT.json`
- `docs/P2_REVIEW_202611082025_ZH.md`
- `docs/P2_REVIEW_202611082025_EVIDENCE.json`

其中模型哈希、旧 80 次轨迹复核、82 段视频完整性、17 条未闭爪信号与公开数据元信息属于已有数据审查。下载/训练命令 dry-run 没有执行训练，不产生新的成功率。

## 实验实际执行情况

| 阶段 | 已有内容 | 尚缺内容 |
|---|---|---|
| P2-A | 原始夹爪信号与命令检查；17 条未闭爪案例阈值统计；数据/代码审查 | 完整指垫几何、可见性诊断和真实反事实验证尚未全部完成 |
| P2-B | 专家定义与分析报告中的建议 | 没有找到已完成 B1/B2 物理干预；新测试 B0/B1/B2 均未运行 |
| P2-C | 官方数据来源、规模、混合规则及单卡命令 dry-run | 未运行训练 step，未完成目标适配，没有适配 checkpoint 或独立物理评测 |

当前外部 `rollouts/` 只有旧 `formal/` 受阻记录和 `user-authorized-resume/` 的 P0/P1。后者为 2 次 P0 与 80 次 P1，不是 P2 的 100 次新测试。旧结果为 No Reps UR5e 5/20、Panda 1/20，Joint Reps UR5e 9/20、Panda 0/20。不能将它们填入 P2 新条件 20–39 的表格。

真实旧数据路径：`/home/users/lemon/coft-p2/rollouts/user-authorized-resume/`；旧逐文件索引为 `results/simulation/closed_loop/external_files.csv`。目前新测试集原始结果路径为 null，目标适配独立评测成功率为 null。

## 科研结论

P2 当前判断为 `INSUFFICIENT_EVIDENCE`：既未证明简单控制足够，也未证明标准适配足够，更未证明存在必须用新网络解决的残余问题。已有源—目标差距仅支持继续机制验证。

用户此次的同步审计和禁止新运行指令优先于任务文档中的立即运行要求。新的离线产物会明确注明其来源是旧开发轨迹，不会包装成新仿真。提交后将通过 GitHub 远程接口读取实际 commit 和新增文件，确认内容已离开本地工作区。
