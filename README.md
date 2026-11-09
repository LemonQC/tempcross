# COFT SR5: Cross-Embodiment Executability × Observability Hypothesis Probe

> **P3 正在真实执行。** 已完成24次接触空间—时间干预：A3稳定抓持8/8、完整任务5/8；新40–59的80次基线完成。新96000样本Joint LoRA已通过batch8单步验证并持续训练，最后20次评测尚未开始。见 [因果报告](docs/P3_SPATIOTEMPORAL_CAUSAL_REPORT.md)、[增强适配](docs/P3_STRONGER_ADAPTATION.md)、[当前判断](docs/P3_SCIENTIFIC_DECISION.md)、[机器状态](evidence/p3/status.json)。

> **P2 真实实验完成（受限适配版本）。** 新条件正式评测 120 次：Joint Reps/UR5e 8/20、Panda 1/20，No Reps/Panda 1/20，B1/Panda 3/20；B1 救回 2 次、退化 0 次。两模型各完成 3000 步受限 LoRA，评测分别 2/20、3/20。B2 正式 Oracle 未验证。当前停止该具体新算法分支，不能据受限训练否定充分标准适配。见 [干预报告](docs/P2_INTERVENTION_REPORT.md)、[适配记录](docs/P2_TARGET_ADAPTATION.md)、[科研决定](docs/P2_SCIENTIFIC_DECISION.md) 和 [原始结果](evidence/p2/execution_2145/README.md)。

> **COFT-P2：P0通过，80/80次官方物理仿真完成。** No Reps：UR5e 5/20、Panda 1/20；Joint Reps：UR5e 9/20、Panda 0/20。观察到迁移差距，但未证明学习算法不可替代。见 [闭环验证报告](docs/SIM_VALIDATION_REPORT.md)、[完整失败审计](docs/TRANSFER_FAILURES.md) 和 [下一阶段决定](docs/NEXT_ALGORITHM_DECISION.md)。


> **2026-11-07 正式审计已完成：PROXY_ONLY / INSUFFICIENT_EVIDENCE。**
> 阅读 [验证报告](docs/VALIDATION_REPORT.md)、[创新性审计](docs/NOVELTY_AUDIT.md) 和 [下一阶段决定](docs/NEXT_STAGE.md)。
> v0 的 `analytic_and` 100% 是同标签筛选的覆盖／自洽检查，不能作为算法有效性或任务成功率。
> 原 v0 代码与配置保留。独立 v1：`coftvalidation/`，正式原始数据：`evidence/v1/`。
>
> 复现：`python -m venv .venv`；`.venv/bin/pip install -r requirements-validation.lock`；
> `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 scripts/run_validation.sh`；
> `.venv/bin/python scripts/audit_results.py evidence/v1`。
> 两条官方运动链已验证，但真实工具／相机、官方网格碰撞与独立闭环结局仍未通过证据门槛。


**Status: v0 analytical pilot only; NOT a trained model or a validated T-RO contribution.**

Hardware target: **ROKAE SR5-5/0.9C** 6-DoF, **PGIA-140-80-W** two-finger gripper, **Intel RealSense D435i** eye-in-hand. This experiment uses Python only. It does **not** need ROS, large GPU training, a custom robotics simulator, or direct real-robot control.

## Scope / research question

A source robot demonstration may be geometrically executable on the target arm yet leave the target wrist-mounted camera unable to see the object during approach. We test task-equivalent top-down grasp-yaw candidates on an axisymmetric cylindrical object, vary an obstacle counterfactually, and compare a motion-only selector with **equal-information** view-only, weighted, and analytic reachability-AND-visibility selectors.

**Important:** This is a *geometry-only* falsification pilot. No real scene measurements, gripper collision mesh, physical trials, source Panda model, or trained policy are included. We do not declare a learned COFT algorithm superior from these outputs.

## Run

Clone this repository (all source code is directly visible), then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
python -m coftprobe.experiment --mode demo --scenes 12 --yaws 8 --seed 20261107 --out results/proxy
cat results/proxy/REPORT.md
```

The scripts do **not** issue robot motion commands.

### Initial local synthetic-only results

| Metric | Result |
| --- | ---: |
| Fixed scene pairs × grasp orientations | 12 × 8 |
| Occluder scenes with a feasible-unobservable candidate and feasible-observable alternative | 4/12 (33.3%) |
| Motion-only selector selects feasible-and-observable | 91.7% |
| View-only selector selects feasible-and-observable | 100% |
| Weighted / analytic AND selector selects feasible-and-observable | 100% |

These metrics are only geometric proxies. **An equal-information analytic selector solves this toy scene set; this does not validate a new learning algorithm.** Do not interpret them as robot task success rates, transfer rates or measured camera performance.

## Measured mode

To use real parameters, fill `configs/measured_input_TEMPLATE.json` with **measured** camera intrinsics, hand-eye calibration `T_flange_camera_optical` and tool transform `T_flange_tcp`. Fill `configs/measured_scenes_TEMPLATE.json` with actual object center and obstacle AABBs in **SR5 base coordinates**, all in metres.

```bash
python -m coftprobe.experiment --mode measured \
  --calibration /path/to/measured_calibration.json \
  --scene-file /path/to/measured_scenes.json \
  --out results/measured
```

Only change the template status to `MEASURED` after actually replacing its example numbers with measurements. Measuring geometry is not the same as real manipulation evaluation; no policy or collision-free full-trajectory guarantee is implied.

## Source and limitations

Six revolute joint origins, axes and limits were transcribed from [ROKAE official xMateSR5 URDF](https://github.com/RokaeRobot/rokae_ros2/blob/main/rokae_description/urdf/xMateSR5.urdf.xacro) (upstream Apache-2.0). Exact variant, TCP and camera mounts must be checked against the actual robot. The tool model is a simple TCP line; robot/hand mesh collisions, gripper finger occlusion, dynamics, contact and history-dependent observability are out of scope.

This repo is a strictly **limited preliminary test** of an idea. Even if a geometric conflict exists, demonstrating T-RO novelty would additionally require strong joint-geometry planning baselines, real cross-embodiment source-to-target policy transfer, and closed-loop simulation/real robot results. See the code and tests in this repository for scientific gating, caveats and tests.
