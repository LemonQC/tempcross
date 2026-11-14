# Cross-Behavior Task Expansion Protocol v0.1

**Base commit:** `97b7a21bbc29547cf94f6e7c2860fca10189e94c`  
**Branch:** `cross-behavior-task-expansion`  
**Status:** APPROVED TO START TASK/ACTION/DATA INTEGRATION; NO FORMAL TRAINING OR FINAL EVALUATION UNTIL FEASIBILITY GATES PASS.

## 1. Objective and scope

The existing benchmark has broad algorithm coverage but only two task definitions: Counter-to-Sink and Sink-to-Counter. Both are Pick-and-Place variants. Expand task coverage while preserving all existing code, checkpoints, datasets, manifests, reports and results.

Add three behavior families, in this priority order:

1. **Push** — non-prehensile contact interaction: move an object into a goal region without grasping it.
2. **Stack** — multi-object manipulation: pick one object and place it stably on a support object.
3. **Drawer** — articulated-object interaction: establish handle contact and open a drawer to a defined target state.

Keep the two existing PnP tasks as anchor tasks. Do not describe a collection of separate per-task policies as cross-behavior generalization. The first expansion measures cross-embodiment performance across multiple behavior families. A separate held-out-behavior transfer experiment may be designed only after the per-task benchmark is valid.

## 2. Non-negotiable preservation rules

- Do not modify, overwrite, rename or regenerate the existing PnP data, conditions 60–99, model weights, results, videos or SHA manifests.
- Do not change existing PnP task success predicates, robot dynamics, gripper parameters, control frequency, camera conventions or action normalization.
- All new tasks, data, conditions and results must live under new task-specific paths and have independent manifests and SHA256 indexes.
- Existing PnP results remain frozen and must pass the existing preservation audit before and after this work.
- Do not start a new algorithm/Method. This phase is task and benchmark expansion only.
- Do not claim any task or algorithm is complete based on import success, a single forward pass, an action replay, or an incomplete rollout.

## 3. Task inventory before implementation

Inspect the exact pinned local RoboCasa/RoboCasa-X/benchmark source used by the current environment (including the configured `BARX_SOURCE`, installed package paths, task registry, task XML/assets and success code). Do not guess task class names or assume a task exists because it appears in a paper.

For each candidate, report:
- native task class/name and source commit, if present;
- robot/embodiment support and gripper support;
- object/fixture assets and license/source;
- camera and observation keys;
- action space and controller;
- native success predicate and termination behavior;
- whether the task can be restored from a frozen initial state;
- whether source demonstrations can be collected for UR5e, IIWA, Kinova3 and target Panda.

Prefer a native official task. If Push is not available, a thin task wrapper may be added using existing RoboCasa/MuJoCo assets and dynamics. Do not introduce a new physics engine, hand-coded object teleportation, privileged policy inputs, or modified contact/friction parameters. If Stack or Drawer cannot be faithfully supported by the pinned environment, report the blocker instead of fabricating a task.

## 4. Frozen task contracts

Implement exactly one canonical task per new behavior family first. Avoid task variants until the canonical task passes feasibility.

### 4.1 Push

- Instruction: push a specified object from its initial counter/table region into a visible target region.
- The robot must not grasp/lift the object. Gripper remains in the declared open/non-grasping mode.
- Success must require the object to enter the goal region, remain stable for a predeclared number of simulation steps, and not fall from the support surface.
- Record first contact, continuous contact duration, object displacement, final pose, object velocity, collision and grasp/lift violations.
- If the native task has an official success predicate, retain it and document any additional diagnostic metrics separately.

### 4.2 Stack

- Instruction: place object A on object B (or the designated support object) and release.
- Success must require a stable supported configuration: predeclared XY alignment tolerance, vertical support/contact condition, bounded linear/angular velocity, and stability for a predeclared horizon.
- A transient overlap or simulator contact is not success.
- Record grasp, lift, transport, first support contact, release, stability and toppling.

### 4.3 Drawer

- Canonical task: open one specified drawer to a target joint displacement and release/retreat.
- Prefer the native RoboCasa articulated task and native success predicate.
- If a custom wrapper is necessary, define the target joint threshold, stable-hold duration, handle contact requirement and retreat condition before any evaluation.
- Record handle contact onset, contact continuity, drawer joint trajectory, final joint state, release and retreat.
- Do not silently combine open and close into one task. Treat close-drawer as a later separate task if needed.

For all three tasks, policy inputs must not include simulator object poses, contact forces, task success flags, future states or other privileged state. Such values may be logged by the evaluator for diagnostics only.

## 5. Robots and transfer setting

- Target embodiment: Panda with the same Robotiq85 two-finger gripper used by the current benchmark.
- Primary source embodiment: UR5e with the same gripper.
- IIWA and Kinova3 may be added as additional source embodiments only after their kinematics, gripper, controller and task feasibility pass the same acceptance tests.
- Keep the existing 20 Hz control convention and canonical 7D OSC action boundary where supported. If a method requires a native action representation, preserve its native policy and document the exact adapter; never silently change its action semantics.
- Keep RGB plus instruction as the common-information comparison tier. SmolVLA/Flow Matching and any other state-using method must remain separately labelled as RGB+state/native-information results. Do not call them RGB-only.

## 6. Scene and condition design

For each new task, create:
- **20 development conditions** for controller/data/adapter debugging;
- **40 frozen formal conditions** for the first formal comparison;
- distinct, deterministic condition IDs and seeds;
- distinct initial-state, model XML and metadata SHA256 values;
- object/fixture identity, target pose/region and task instruction;
- frozen restore/replay verification.

Development and formal conditions must be disjoint. Training demonstrations must use a separate seed/condition pool and must not contain formal test states or near-duplicates. Formal conditions must not be inspected for model selection or threshold tuning.

This creates 60 new condition records per task (180 for Push/Stack/Drawer), of which 120 are formal held-out records. Existing PnP 60–99 conditions remain untouched. Report unique task conditions separately from robot-specific restores and repeated policy trials.

## 7. Expert/controller and demonstration feasibility gate

Before adapting any learning algorithm:

1. Build or reuse a non-learning expert/controller for each task and verify it on both UR5e and Panda.
2. Run at most 8 development episodes per task per robot for the initial gate (Push/Stack/Drawer × UR5e/Panda = at most 48 episodes).
3. Require at least 6/8 official task successes on each robot, no object teleportation, no hidden state in policy actions, valid initial-state restoration, and complete telemetry/video.
4. If a task fails this gate, diagnose the task/controller/geometry and either repair once with documented evidence or mark that task BLOCKED. Do not collect large datasets or train policies for a blocked task.
5. Once passed, collect a 20-demo pilot per task per robot and audit success, action continuity, observation/action alignment, and data diversity.
6. Only after the pilot passes, scale demonstrations. Target data should initially match the existing per-task target budget (50 Panda demos, with a deterministic whole-demo 90/10 train/validation split). Source data should initially match the existing per-task source budget (300 demos per supported source embodiment). If collection cost makes this infeasible, report the actual budget and obtain a written protocol amendment before training; do not silently reduce data and claim a matched comparison.

## 8. Existing algorithms to integrate

Do not implement new model architectures. Reuse each method's existing source code, official checkpoints, training logic, normalization and native inference behavior. Add task support through dataset/task adapters only.

Mandatory existing policy families/configurations:
- BC-GMM;
- BC-RNN;
- BC-Transformer (Source-only, Target-only and Source+Target configurations where previously supported);
- Diffusion Policy (Source-only, Target-only and Source+Target where previously supported);
- BARX No Reps;
- BARX Joint Reps;
- existing BARX LoRA adaptation configurations (report separately);
- ACT (Target-only and Source+Target);
- VQ-BeT (Target-only and Source+Target);
- Octo-Small (existing Source+Target setting);
- SmolVLA (existing Source+Target state-information setting);
- HRI-EU visual Flow Matching Policy (existing Source+Target RGB+qpos setting).

MimicGen v2 is a data-generation/augmentation component, not a standalone policy. Keep it PARTIAL. It may be adapted as an optional augmentation comparison only if its official generation procedure supports the new task and all generated trajectories pass task-specific validation. Do not label it a complete policy baseline.

KITE is not a mandatory method in this phase; it remains a separate candidate pending native target-kinematics decoder validation. CrossZero and Cross-Behavior Object Flow remain archived and out of scope.

For each method and task, preserve the existing method-specific training regime. Do not force every method into a regime it did not support in the original benchmark. Record COMPLETE / PARTIAL / BLOCKED / NOT_APPLICABLE with an explicit reason.

## 9. Staged execution and resource limits

### Stage A — Task inventory and feasibility (start immediately)
- Produce the task inventory, exact task contracts, controller feasibility results, and a frozen task registry.
- Run only the 48-episode maximum feasibility gate above.
- No policy training or formal policy evaluation in this stage.

### Stage B — Demonstration/data pipeline
- Produce pilot demonstrations and validate them.
- Freeze the data manifests, splits, normalization, camera and action mappings.
- Preserve all old datasets.

### Stage C — Algorithm adapters and smoke tests
- Integrate every mandatory existing method with all three new tasks.
- For each method/task/configuration, run unit tests, one genuine optimizer update (where training is applicable), native action-interface checks, and at most two development rollouts.
- No formal test conditions may be used.
- If a method cannot support a task without changing its core algorithm, mark it BLOCKED or NOT_APPLICABLE; do not fake compatibility.

### Stage D — Freeze and formal training/evaluation
- Only after all task gates and smoke tests pass, freeze a per-task protocol and report its SHA before formal training.
- Run one method at a time on the single RTX 4090; no parallel large training jobs.
- Reuse the existing method-specific budget and checkpoint selection rules where meaningful. Select checkpoints using validation only, never formal success.
- Evaluate each accepted method on the same 40 formal conditions per task. Preserve all 40 outcomes, videos and telemetry; no condition dropping or success-criterion changes.
- First-pass comparison uses the existing training seed. Do not claim statistical robustness. Use at least three training seeds for the proposed method and the strongest 2–3 baselines in the final paper stage, after the first-pass results identify them.

## 10. Metrics

Report both official task success and task-specific subgoal metrics.

- Common: success, completion time, collision, out-of-bounds/fall, action clipping, inference latency, object final pose, stability, retreat.
- Push: first-contact success, contact continuity, displacement along intended direction, lateral deviation, goal-region entry and stability.
- Stack: grasp/lift success, support contact, XY alignment, release success, stable stack duration, topple rate.
- Drawer: handle acquisition, contact continuity, joint displacement, target opening success, release/retreat.
- Cross-embodiment: source-vs-target gap, source-only transfer, target-only, source+target where supported, success conditioned on subgoal completion.
- All stage/failure labels are diagnostic telemetry, not causal explanations unless supported by interventions.

## 11. Required first delivery

The first delivery is not a large training result. It must include:

1. `docs/CROSS_BEHAVIOR_TASK_INVENTORY.md`: exact native task availability and task contracts.
2. `docs/CROSS_BEHAVIOR_TASK_EXPANSION_PROTOCOL.md`: this protocol plus any evidence-based amendments.
3. `configs/tasks/cross_behavior_tasks.json`: frozen task registry for tasks that passed feasibility.
4. task expert/controller code and tests, isolated from old PnP implementation.
5. development condition manifests and SHA256 indexes.
6. up to 48 feasibility rollouts, videos and per-task reports.
7. a method integration matrix listing every mandatory method × task × configuration and current status.
8. preservation audit proving the old 2080 formal baseline episodes and all previous artifact hashes remain unchanged.
9. exact Git commit SHA and server paths for large videos/data/checkpoints.

Do not claim that all algorithms are connected until Stage C has passed for every method/task. Do not claim the new task benchmark is complete until Stage D has formal results.

## 12. Immediate instruction

Start Stage A now. Inspect the pinned local environment and task registry, implement the three task feasibility pilots in priority order Push → Stack → Drawer, and stop after the Stage A report. Do not train models or run the old PnP benchmark again.
