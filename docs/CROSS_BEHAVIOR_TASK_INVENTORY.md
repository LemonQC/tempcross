# Pinned task inventory and implementation boundary

Branch: `cross-behavior-task-expansion`, base `92a6d212353f5e209d63534825b4763afe6893fd`. Main and the original two Pick-and-Place tasks are unchanged.

## Runtime actually inspected

- BARX / embedded RoboCasa-X: `/home/users/lemon/coft-p2/barx`, commit `0f435089f48b1f16d225f868dbef855a5e75d1e0`.
- Python: `/home/users/lemon/coft-p2/barx/.venv/bin/python`; installed robosuite 1.5.1 comes from checkout `2ebb2a0249f7a271a3ade56725d12e32d7da8898` under `/home/users/lemon/coft-p2/uv-cache/git-v1/checkouts/79abe6cb204f004b/2ebb2a0`.
- MuJoCo 3.1.1; native `mjSTATE_INTEGRATION` used for step-boundary recovery.
- Actual GPU: RTX 3090 / 24 GB. No VLA/network training or optimizer updates in this phase.
- BARX and RoboCasa code: MIT; installed robosuite LICENSE: MIT, SHA256 `177978cbece0a4c454c2aaec5b3f145b39270814874c43109da9e829c39d9cba`; MuJoCo: Apache-2.0. Sampled RoboCasa/Objaverse mesh licenses are not automatically covered by the code license. Existing asset usage is retained; meshes are not redistributed in Git.

## Native candidates and selection

| Behavior | Exact pinned native candidate | Native predicate | Decision |
|---|---|---|---|
| Push | `multi_stage/sanitize_surface/push_utensils_to_sink.py:PushUtensilsToSink` | Both utensils inside sink and gripper far | Not the requested supported-surface goal-region task; not relabelled |
| Stack | `multi_stage/washing_dishes/stack_bowls.py:StackBowlsInSink` | Both bowls in sink and nested, gripper far | Not block A stably supported on B; not relabelled |
| Stack | installed `robosuite/environments/manipulation/stack.py:Stack` | Released, lifted A in contact with B (`staged_rewards` stack term) | Reuse its native BoxObject sizes and success implementation inside Kitchen; explicitly a task port, not an existing native RoboCasa task |
| Drawer | `single_stage/kitchen_drawer.py:OpenDrawer` | Every native normalized opening state ≥0.95 | Use the native task and predicate, retain native fixtures/objects/initialization |

New classes exist only in `cross_behavior/tasks.py`: `CBPush`, `CBStack`, `CBDrawer`. Push is a thin new task using existing Kitchen and native generated BoxObject dynamics. Stack is a native-component task port: the Kitchen support surface replaces robosuite's standalone table; Kitchen contact parameters are retained, not tuned. Drawer inherits `BARXTaskMixin, OpenDrawer` directly. No native package files or existing PnP classes are edited.

The one new BoxObject sampler adapter exposes its analytic box bounds to the native RoboCasa collision-aware placement sampler. It does not compile a different body, override its friction/density, or disable initial overlap checking. The Push goal outline consists only of visible non-colliding geoms (`contype=conaffinity=0`). Blocks are positioned by native initial-state sampling before the frozen t=0; no object is repositioned by policy actions.

## Shared interface and observations

| Robot | Native robot | Gripper | Primary camera |
|---|---|---|---|
| UR5e | `UR5eOmron` | existing default Robotiq85 | `barx_ur5e_agentview` |
| Panda | `PandaOmron` | `Robotiq85Gripper` | `barx_panda_agentview` |

`BARXTaskMixin` preserves the existing Omron locks, robot alignment/init qpos and camera configurations. Images are 320×180. These task-gate experts additionally request depth from the same camera. Their sensor API contains only RGB-D, calibrated rays, robot FK/pad geometry and the task instruction. Dynamic object/fixture state, contact and success never enter `Expert.predict`.

Canonical action `[dx,dy,dz,dax,day,daz,grip]` maps to `[arm6,grip,base3=0,torso=0,mode=-1]` (12D). Positive closes, negative opens. OSC translation output scaling is ±0.05 m and angular scaling ±0.5 rad; its reference is **base**, so the expert rotates world deltas into the controller origin frame. Control remains 20 Hz. Base, torso, gripper mechanics, gains, termination semantics and all old action adapters are unchanged. New task-gate rollouts stop at success, 600 control steps, or a documented target-identification failure after 40 attempts.

IIWA/Kinova3 have pinned native robot definitions and BARX offsets, but their new task feasibility was **not tested**. Native class availability is not accepted embodiment support. No extra source demonstrations were collected.

## Frozen goals and restoration

See `configs/tasks/cross_behavior_contracts.json` and the protocol amendment for exact pre-rollout thresholds. Push requires full box footprint in its visible 12 cm region, ≥8 cm displacement, no fall/lift/grasp violation, support contact and bounded velocities for 20 ticks. Stack requires its native predicate plus XY ≤15 mm, height difference 45±12 mm, A–B support contact, B support, release, 10 cm retreat, bounded velocities and 20 ticks. Drawer retains native ≥0.95 opening plus observed handle contact, release, 10 cm retreat, bounded joint change and 20 ticks.

Every actual development attempt has serialized model XML, t=0 integration state, RGB-D, metadata, contract SHA, instruction, action/phase/contact logs, per-step states, full video and a SHA index. Resume has been exercised on real Push and Stack episodes; exact stored integration-state restoration and contiguous step/video indices are checked. This is not a claim that a separately rerun, uninterrupted physical trajectory would be bitwise identical: that duplicate experiment was not performed.

Native initialization performs 10 control-period-equivalent settling steps per reset; construction and the explicit initial reset both do this before t=0. Earlier per-condition metadata's `native_initialization_control_steps=10` denotes one reset, not total CPU work across construction/resume. Recreating an environment during recovery also executes native initialization before loading the saved state; it is recorded as recovery overhead, not another policy episode. Policy steps are counted only after the t=0 snapshot and never replayed silently.

No task has passed the dual-robot gate. `configs/tasks/cross_behavior_tasks.json` therefore has an empty **validated** registry; class import availability is not task acceptance. The actual outcomes and limitations are in [CROSS_BEHAVIOR_TASK_FEASIBILITY.md](CROSS_BEHAVIOR_TASK_FEASIBILITY.md).
