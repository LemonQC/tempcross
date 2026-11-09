# Baseline运行和复现

本平台复用原BARX物理环境，不覆盖P0–P3目录。执行根目录为`/home/users/lemon/baseline-benchmark/run-20261109`，固定BARX为`/home/users/lemon/coft-p2/barx`提交`0f435089f48b1f16d225f868dbef855a5e75d1e0`，Python为该目录`.venv/bin/python`。具体版本、GPU、驱动、软件路径见[environment.json](../results/baselines/environment.json)。旧环境依赖声明存在历史冲突，未以升级依赖“修复”已验证任务。

## 本机读取已完成实验

```bash
cd /home/users/lemon/CodexProject/tempcross
export BARX_SOURCE=/home/users/lemon/coft-p2/barx
export BASELINE_RUN_ROOT=/home/users/lemon/baseline-benchmark/run-20261109
export BASELINE_ROBOMIMIC_ROOT=/home/users/lemon/baseline-benchmark/vendor/robomimic
export BASELINE_DEPS_ROOT=/home/users/lemon/baseline-benchmark/deps
export BASELINE_ASSETS_ROOT=/home/users/lemon/baseline-benchmark/assets
export BARX_DATA_ROOT=/home/users/lemon/coft-p2/p2/data
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.collect --final
"$BARX_SOURCE/.venv/bin/python" scripts/summarize_baselines.py
```

上述只是读取、校验、汇总，不重新训练或仿真。`collect --final`需要完整正式运行ledger，否则拒绝声称完成。所有模型checkpoint、HDF5及视频的绝对路径、size和SHA256见`results/baselines/checkpoint_index.csv`、`video_index.csv`、`external_files.csv`。完整逐步采样trace及每档checkpoint保留服务器；每条示范覆盖和小型训练CSV纳入Git。

## 新的独立重复运行

先设置**全新的**`BASELINE_RUN_ROOT`，不删除已有证据。可复用同一份只读原始TFDS/官方HDF5、CLIP资产及上游源码。BARX首次安装参考已有`simulation_validation/README.md`与`docs/SIM_VALIDATION_REPORT.md`，需相同任务资产、MuJoCo与官方环境，不替换成新版RoboCasa。官方robomimic不是旧BARX环境内包：

```bash
# 在新位置准备固定核心；若目录已存在，先检查版本及工作区，不覆盖。
git clone https://github.com/ARISE-Initiative/robomimic.git "$BASELINE_ROBOMIMIC_ROOT"
git -C "$BASELINE_ROBOMIMIC_ROOT" checkout ae5799f0fae05c4559ee1f9645b0f77eb5251929
# 使用有pip的宿主Python，仅安装纯Python隔离依赖；不改BARX虚拟环境。
python -m pip install --target "$BASELINE_DEPS_ROOT" --no-deps diffusers==0.11.1 huggingface_hub==0.23.4
export PYTHONPATH="$BASELINE_ROBOMIMIC_ROOT:$BASELINE_DEPS_ROOT:$PWD:$BARX_SOURCE:$BARX_SOURCE/policy"
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.download_assets
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.prepare_data target
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.prepare_data source
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.dataset_adapter
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.freeze
```

CLIP固定`openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`，传输镜像内容必须匹配固定git blob或LFS SHA。source逐条读取1800原始TFDS，保留IIWA/Kinova3/UR5e与两任务身份；目标两HDF5各50条。打包拒绝已有输出，整demo划分和canonical动作标签不依赖最终测试。

冻结文件必须**先提交到GitHub并读回**，再执行训练/物理步骤。本次首次冻结已在`1611433f432a145b93752b6a2eb07e1987bee428`发布。如果复用完全相同的冻结文件，可运行：

```bash
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.verify_publication --commit 1611433f432a145b93752b6a2eb07e1987bee428
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.audit_semantics
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.capture_environment
bash scripts/run_baseline_suite.sh
```

如果原始绝对路径改变导致manifest变化，先提交新的冻结manifest，传入该真实远程SHA；不得伪造`github_readback_verified`。`freeze`在物理launch记录存在后拒绝覆盖协议。新ROOT需要原始数据和外部checkpoint可访问；BARX历史权重位置由`baselines/barx/registry.py`列明，可显式修改注册路径并核对SHA，不能用不同权重替代。

## 单项命令与运行顺序

```bash
# 一真实optimizer步；DP因官方zero-LR warmup另作第二步参数变化核验。
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.probe --group B --method bc_gmm --target-batch 16
# 首档真实训练；扩展与配对规则由train_suite自动完成。
"$BARX_SOURCE/.venv/bin/python" scripts/train_baseline.py --group B --method bc_gmm --end-positions 96000 --target-batch 16
# 用准确checkpoint单独检查旧开发条件0，方法/路径需与checkpoint一致。
"$BARX_SOURCE/.venv/bin/python" scripts/eval_baseline.py --label manual_selected_gmm --kind robomimic --checkpoint "$BASELINE_RUN_ROOT/training/B_bc_gmm/positions-096000.pt" --task pnp_counter_to_sink --robot panda --smoke
# 统一执行：train_suite → evaluate_suite等待选模关闭 → 选定权重smoke → 独占latency → 正式。
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.train_suite
"$BARX_SOURCE/.venv/bin/python" -m baselines.common.evaluate_suite
```

`positions-096000.pt`只是单项命令示例，不代表最终选中档位，真实路径读取各`training/<label>/selected.json`。最终训练规则是固定验证集的96k→192k→384k→768k阶梯；不能拿测试成功率决定是否延长或换checkpoint。B/C Transformer及DP分别共用目标序列、目标batch16及停止档位；C实际batch32。

评估器只接受RGB和原文指令，native7维输出经明确适配到12维OSC；真值/接触只写logger。完整100条件先由官方验证，再原样restore冻结60–99，每格40次，20Hz/600或650步。每episode初始flattened state逐值及官方SHA核验，跨方法共享同一任务/机器人条件；原XML哈希不变。正式4个独立进程并发仅提高吞吐，latency另在独占GPU测量。

所有训练与正式评测都拒绝覆盖失败目录。技术错误需先保存完整状态和日志、诊断后用独立attempt目录处理，不能将异常当作策略失败，也不能无记录重跑。日志按实际执行命令和optimizer更新保存；初次GMM trainer源快照、之后的每档执行源码hash均在服务器，复现报告披露HDF句柄关闭这一记录性修正，不改学习规则。

## 复现关系与扩展接口

BC四类直接调用固定官方robomimic核心，属于FAITHFUL_PORT：RGB128/crop116、冻结CLIP-L/14语言条件、时间窗口/数据/动作适配、训练日程及DDIM10均已声明。BARX复用原官方网络/预处理/权重，只包公共评估器。LoRA历史预算不等于完整官方适配；不同组和历史设置不得忽略数据及算力差异排名。

加入未来方法需实现`BasePolicy.reset/predict_action/load_checkpoint`，返回native action/representation/query，由同一动作适配和评估器调用。先核对其输入信息边界、真单步训练、旧条件完整smoke，再在不改旧结果的独立目录追加冻结评测。新增额外点云/真值方法需独立输入分组。当前工程平台是否完成见`platform_readiness.json`；T-RO主张仍需科学创新、更多训练seed和任务/场景泛化证据，不能由平台能运行直接保证发表水平。
