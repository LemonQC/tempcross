# COFT-P2 官方 BARX 闭环评估执行报告

日期：2026-11-07；基于 `808f0ab19b1fb0be6bbc9f20445744ab8db8b87a`。

**实际结果：P0_BLOCKED。两轮集中修复后，官方锁定运行环境仍未安装成功。没有完成任何机器人 rollout，没有启动 P1 的 80 episodes，不能作出跨具身性能或新算法判断。** 这是技术前置条件受阻，不是 COFT 假设失败，也不是两种策略的任务成功率为零。

本报告记录实际执行，而非实验计划。已执行官方源码获取、模型／动作／相机／固定条件核查、依赖安装、两轮修复、官方单元测试和正式 gate 命令。未运行的内容明确标为未运行。

## 1. 固定官方来源与协议核对

官方源码 [ajaysridhar0/barx](https://github.com/ajaysridhar0/barx) 固定在
`0f435089f48b1f16d225f868dbef855a5e75d1e0`，本地 `/home/users/lemon/coft-p2/barx`，源工作树未修改。只读取 XML／官方源码，无 ROS，无真机接口，无专家数据生成，无训练，也没有创建新任务或几何场景。

| 项目 | 实际核查结果 |
|---|---|
| 任务 | 官方 `pnp_counter_to_sink` → `XPnPCounterToSink` |
| 物理成功判据 | 官方 `_check_success`：`OU.obj_inside_of(env, "obj", sink, partial_check=True)` 且 `OU.gripper_obj_far(env, th=0.1)`；评估器使用 `info["is_success"]["task"]`，在首次满足时结束。不是 COFT 几何标签，也不是自行增强后的“完全进入容器”判据 |
| 来源机器人 | `UR5eOmron`，官方 `gripper_types="default"` 在固定 robosuite 源码中解析为 `Robotiq85Gripper` |
| 目标机器人 | `PandaOmron` + `Robotiq85Gripper`；没有换成 Panda 原生手 |
| 相机 | 保持官方 `barx_ur5e_agentview` / `barx_panda_agentview`，320×180，单幅历史；没有 Eye-in-Hand 输入或新相机 |
| 动作接口 | 策略 7-D；官方 normalize/invert gripper 后由 `pad_action_robocasa` 扩展至仿真动作；固定动作块 8，max_steps=600，settle_steps=10，global seed=7 |
| 归一化与 VQ | 两 checkpoint 的统计 key 都是 `mg_pnp_lite`，tokenizer 都是 `mg_pnp_lite_vq_extra_action_tokenizer`；实际 VQ config input_dim_w=7、input_dim_h=8 |
| 推理模式 | 两种训练方法都保持官方 action-only 推理：`inference_representation=none`，不额外给 Joint Reps 表征提示 |
| 固定条件 | 两臂各有官方 100 条 bundle；已固定前 20 条并保存 condition/model/state hash；没有重采样、重排或换失败种子 |

两种机器人前 20 条的实际 seeds 和 layout/style 在索引上均 20/20 相同，但本轮没有在物理环境中恢复并逐一核对所有对象的世界位姿与布局。因此只承诺**方法之间同一机器人条件严格一致**；跨机器人按未完成严格配对核验处理，不能作逐场景因果归因。详见 `results/simulation/condition_pairing_audit.json`。

官方冻结条件采用 `visible-target-v1` 筛选，前 20 条实际 seeds 是：
`1000,1002,1004,1005,1009,1010,1011,1012,1013,1014,1015,1016,1017,1019,1024,1025,1026,1028,1029,1030`。
官方评估器写 episode 日志时仍使用 `start_seed + trial_idx`，不一定是真实条件 seed。本项目不改变条件选择，只在观察记录器中保存原始 entry，供将来按 condition_id/真实 seed 联接，避免错误配对。

## 2. 模型版本与训练来源

| 方法 | 固定 Hugging Face revision | 固定 checkpoint |
|---|---|---|
| [Joint Reps XP-900](https://huggingface.co/ajaysri/barx-joint-reps-xp900-pnp-pretrain) | `191da4c06c1050767a0379feb67a5c20e85bd703`，来自官方 public_artifacts.json | `step-050000-epoch-15-loss=0.2577.pt` |
| [No Reps XP-900](https://huggingface.co/ajaysri/barx-no-reps-xp900-pnp-pretrain) | `9ebc6f6fb8311f3c4394cd77811a4025b26d0a62`，核查时公开 revision | `step-050000-epoch-15-loss=0.8424.pt` |
| XP-900 PnP VQ | `7d50d28628237c499eb8488e1cf91aab8acbaec6` | `checkpoints/model.pt` |

两组模型卡声明 source embodiments 为 IIWA、Kinova3、UR5e，PnP 双方向共 1800 training episodes；Panda 为目标机器人，模型卡的论文评价包含适配。这里预定评测的是**源预训练策略直接迁移**，不是目标微调，也不能拿原论文适配结果作本轮直接迁移成功参考。没有重新核验作者训练日志或重训模型。

已实际下载并核对两组 config.json 与 dataset_statistics.json，模型、VQ、统计接口在定义层面匹配；**没有完成策略权重加载和数值推理匹配验证**。当前公开 Joint Reps 仓库 metadata HEAD 为 `583a13…`，本轮仍使用官方 manifest 的上述固定 revision，不随 metadata HEAD 改选 checkpoint。所有元数据与选择信息入库。

VQ 权重已完整取得并核验：9,519,250 bytes，SHA256
`cbf410959839f99ee131aa8c7d4c2dc2116d134f23ee9e88e90fa089270e7378`。
Joint Reps 预期 5,554,882,540 bytes，官方 SHA256
`ffce76802a8e75d4806b30dee43624accf2c4de4213c17059457e34a8243efce`；当前只有部分下载，不是有效 checkpoint。No Reps 大权重没有下载；按 P0 门槛不提前启动其评测。

下载过滤排除了 RLDS 和不需要的 base VLM checkpoint；官方 `load_vla` 从 registry 建架构并加载完整源策略，仍需要 DINO/SigLIP/Qwen 运行时资源。这些资源未完整准备，不能声称模型可推理。

## 3. 宿主机与独立环境

实际 `nvidia-smi`：**RTX 3090 24,576 MiB，driver 535.230.02，CUDA driver capability 12.2**。当前执行器没有检测到 RTX 4090，也没有找到 `cbl4090_clean`。已向用户指出执行器差异；没有修改任何已有环境，也没有访问其他主机或修改 Docker 权限。

新建环境仅在 `/home/users/lemon/coft-p2/`：

- `bootstrap/`：独立 Python 3.10.22，uv 0.13.7、CMake 4.4.4、下载工具和最小源码测试依赖。
- `barx/.venv/`：官方 `uv.lock` 目标环境；安装不完整，不能用于策略推理。
- 官方锁定 Torch 2.2.0+cu121、MuJoCo 3.1.1、NumPy 1.23.5 等，未更换成宿主机其他版本。
- 检测到 nvcc 11.3，与 Torch 的 CUDA 12.1 编译环境不同。官方要求推理安装 FlashAttention 2.5.5；本轮未完成该依赖安装，既未尝试用其他 attention 实现替代，也没有把这个潜在编译前提写成已观察到的推理报错。

**单张 RTX 4090 推理能力：NOT_TESTED。** 本机不是 4090，且模型没有加载，不能用显存估算替代实测，也没有 CUDA OOM 数据。

## 4. 实际执行与严格两轮修复

| 阶段 | 实际执行 | 结果 |
|---|---|---|
| 初始安装 | 官方 `uv sync --locked --no-dev` | `nvidia-cusolver-cu12==11.4.5.107` 下载四次传输尝试后超时，报 `operation timed out` |
| 修复轮 1 | 复用 34 个与锁文件版本及 SHA256 完全一致的本地 wheel；降低并发为 2、HTTP timeout=180；同一锁文件重试 | 官方 `vq-bet` Git 获取失败，exit 128：远端挂断、EOF、index-pack 失败。没有改策略、版本、镜头或任务 |
| 轮 1 资源传输 | 原 VQ HTTP 下载在 4,436,298 bytes 处出现 IncompleteRead；改用现有固定范围下载工具并保留官方 SHA256 校验 | VQ 最终完整验证通过；不等于模型推理通过 |
| 修复轮 2 | 两官方依赖仅浅获取固定 commits，清理上一失败任务遗留的自有 orphan Git 传输；继续相同锁文件 | 两 commit 获取 exit 0，但 `uv` 在 VQ 工作树执行 `git reset --hard 09d485…` 时 exit 128：`不能解析对象`。更深层缓存／工作树兼容原因未继续修补 |
| 正式 gate | `BARX_RUN_ROOT=/home/users/lemon/coft-p2/rollouts/formal scripts/run_barx_gate.sh` | exit 2；官方测试因锁定环境不完整失败；`gate.json` 为 P0_BLOCKED、p1_started=false |

另有一次安装命令从目标项目目录执行，报 `No pyproject.toml found`；已显式指定 `--project /home/users/lemon/coft-p2/barx` 纠正，单独保留工程错误日志。它没有改变环境版本或实验条件，也不掩盖上述两轮修复的实际结果。

停止时没有继续第三轮修复；自有大资源下载进程已停止，部分分块保留，未留一个后台“仍在跑”的评测。两臂 rollout 均未进入有效执行，未创建成功或失败视频。

测试必须分开报告：

- 原项目加新增 gate 单元测试：**13 passed**，实际输出 `local_tests.txt`。
- 官方源码测试在独立最小依赖环境：**76 tests，OK (skipped=1)**；跳过的是完整 simulator 注册测试，不能当作物理仿真运行通过。
- 正式锁定环境的官方测试：**52 tests，6 errors，1 skipped，exit 1**，主要为缺 NumPy/h5py 导入；这是安装未完成的后果。没有将最小环境测试结果替换成正式环境 gate 通过。
- 测试中 downloader 的“offline / rate limited / retry”文本属于官方 mock 测试输出，不能误当真实模型下载成功或实际网络重试；实际网络错误看对应安装／VQ 原始日志。

## 5. 四组完整执行状态

| 方法 | 机器人 | 计划 | 已开始有效 episode | 已完成 | 成功数／成功率／CI |
|---|---|---:|---:|---:|---|
| No Reps XP-900 | UR5e | 20 | 0 | 0 | N/A |
| No Reps XP-900 | Panda | 20 | 0 | 0 | N/A |
| Joint Reps XP-900 | UR5e | 20 | 0 | 0 | N/A |
| Joint Reps XP-900 | Panda | 20 | 0 | 0 | N/A |

P0 的 Joint Reps UR5e/Panda 各一次完整 rollout：均 **NOT_RUN**，有效次数 0。P1 80 次完成数为 0；JSON rate=null，不是 0/20，更不是 0% 成功率。无 success once、任务进度或失败阶段的可用物理记录；不能构造置信区间。

## 6. 失败机制与算法判断

本轮唯一可重复失败是软件物料／环境构建失败，不是跨具身策略失效。因此：

1. **官方仿真是否真正运行成功？** 否，未通过 P0。
2. **四组结果？** 上表全部未运行；不存在任务成功率。
3. **源／目标主要失败阶段？** 无操作轨迹，均不可判定；失败发生在依赖构建／导入阶段。
4. **是否存在可复现跨具身迁移失效？** 本轮没有建立证据；可复现安装错误不能替代迁移失效。
5. **能否由简单控制／规划解释？** 无法判断。没有动作—状态记录，不把依赖错误归为 IK、可达性、抓取偏差、碰撞或相机遮挡。
6. **是否有必要研究具身条件残差？** 当前没有实验依据作这个决定。严格停在 P0_BLOCKED，不进入 A/B/C/D 的性能归因，也不启动残差网络研究。不是已经证明残差策略没有必要。

完整失败分类与空案例状态见 [TRANSFER_FAILURES.md](TRANSFER_FAILURES.md)，算法决定见 [NEXT_ALGORITHM_DECISION.md](NEXT_ALGORITHM_DECISION.md)。本轮没有 Eye-in-Hand 输入和闭环数据，不能声称验证 COFT 可观测性或联合学习。

## 7. 已交付代码、原始错误和本地文件

- `simulation_validation/`：固定协议核查、官方评估入口、只读 telemetry、P0→P1 硬门槛、推理物料过滤下载、外部清单与视频索引工具。**完整集成没有验证通过**，代码编译／单元测试不构成 rollout 证据。
- `scripts/run_barx_gate.sh`：一键启动已准备环境上的 P0/P1；P0 不完整立即拒绝 P1，技术异常不转换成失败 episode。
- `results/simulation/`：正式 gate.json、四组未执行状态、固定前20条件清单、模型元数据／配置、硬件／物料清单、原始错误与测试输出；无大权重或视频。
- 全部实际大文件和部分分块：`/home/users/lemon/coft-p2/artifacts/`；逐文件绝对路径与 bytes 见 `external_file_inventory.json`。只有官方 SHA256 已验证的文件才可视为完整物料。
- 原始外部日志与下载文件：`/home/users/lemon/coft-p2/`；正式 gate：`/home/users/lemon/coft-p2/rollouts/formal/`。
- 视频索引为真实空列表；生成命令：`python -m simulation_validation.video_index --root /home/users/lemon/coft-p2/rollouts/formal --out results/simulation/video_index.json`。将来有真实视频时可加 `--sheets` 生成外部 contact sheets；本轮没有伪造示例视频或案例。
- 旧 COFT 仅复用通用 `camera_projection` 作为事后中心视锥诊断，不把 SR5 运动链套到 UR5e/Panda，不参与策略动作或成功判据。由于无 rollout，本轮诊断也未执行。
- v0/v1 原代码与结果未覆盖。

## 8. 最小恢复建议（未执行第三轮）

先确认正确 GPU 执行器并预备一个**全新独立的官方运行环境／缓存**，正常取得上述固定 commits，避免继续复用本次失败的 checkout。使用稳定网络或离线 wheel／模型／厨房资源包，校验官方 hashes 后一次性满足 `uv.lock`、FlashAttention 2.5.5 和全部推理资源。不要改机器人、相机、任务、模型精度或成功判据来规避安装失败。

准备完成后，从同一个协议与前20条件重新启动：

```bash
cd /home/users/lemon/CodexProject/tempcross
# 先按 simulation_validation/README.md 准备独立环境与物料。
BARX_RUN_ROOT=/home/users/lemon/coft-p2/rollouts/validated-environment scripts/run_barx_gate.sh
```

保留本次 P0_BLOCKED 记录，不覆盖。只有两臂 P0 真实完整运行后，才允许 P1 及之后的机制归因；本报告不把恢复建议包装成已完成实验。
