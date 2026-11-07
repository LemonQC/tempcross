# COFT SR5: Cross-Embodiment Executability × Observability Hypothesis Probe

**Status: v0 analytical pilot only; NOT a trained model or a validated T-RO contribution.**

Hardware target: **ROKAE SR5-5/0.9C** 6-DoF, **PGIA-140-80-W** two-finger gripper, **Intel RealSense D435i** eye-in-hand. This experiment uses Python only. It does **not** need ROS, large GPU training, a custom robotics simulator, or direct real-robot control.

## Scope / research question

A source robot demonstration may be geometrically executable on the target arm yet leave the target wrist-mounted camera unable to see the object during approach. We test task-equivalent top-down grasp-yaw candidates on an axisymmetric cylindrical object, vary an obstacle counterfactually, and compare a motion-only selector with **equal-information** view-only, weighted, and analytic reachability-AND-visibility selectors.

**Important:** This is a *geometry-only* falsification pilot. No real scene measurements, gripper collision mesh, physical trials, source Panda model, or trained policy are included. We do not declare a learned COFT algorithm superior from these outputs.

## Run

Download the source archive in this repository, extract it, then run:

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

Measuring geometry is not the same as real manipulation evaluation; no policy or collision-free full-trajectory guarantee is implied.

## Source and limitations

Six revolute joint origins, axes and limits were transcribed from [ROKAE official xMateSR5 URDF](https://github.com/RokaeRobot/rokae_ros2/blob/main/rokae_description/urdf/xMateSR5.urdf.xacro) (upstream Apache-2.0). Exact variant, TCP and camera mounts must be checked against the actual robot. The tool model is a simple TCP line; robot/hand mesh collisions, gripper finger occlusion, dynamics, contact and history-dependent observability are out of scope.

This repo is a strictly **limited preliminary test** of an idea. Even if a geometric conflict exists, demonstrating T-RO novelty would additionally require strong joint-geometry planning baselines, real cross-embodiment source-to-target policy transfer, and closed-loop simulation/real robot results. See detailed README in the source archive for scientific gating, caveats and tests.
