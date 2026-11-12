# Method v0: what must be learned for TOCAT?

Research branch: `method-v0/tocat-20261112`. Base: `571b1b0cf66eb20495a5cc3a6cdbb0da9d25834e`.

This is the formulation and evidence for the **single TOCAT candidate**, not a new BARX failure analysis. The proposed variable is a target-embodiment **continuation contract**: which subsequent task effects remain attainable after an interaction, under a specified continuation policy class, information set and remaining budget. The present work does **not** establish an original learning algorithm for that variable. The negative decision and a conditional data protocol are in [METHOD_V0_DECISION.md](METHOD_V0_DECISION.md).

## 1. The scientific distinction

Three quantities must not share one binary label:

1. **Local realization:** the robot actually realizes a specified contact or object-relative interaction.
2. **Factual continuation:** a particular recorded controller completes the official task after that interaction, with its actual stopping rule.
3. **Physical possibility:** there exists an admissible continuation that completes the task within the remaining benchmark budget.

For a physical state \(s\), robot \(e\), goal \(G\), continuation \(\pi\), and budget \(B\), write

\[
q^\pi(s,e,G,B)=\Pr_\pi(\tau_G\leq B\mid s,e),\qquad
F(s,e,G,B)=\mathbf 1\{\exists\pi\in\Pi_e:\tau_G(s,\pi)\leq B\}.
\]

An unsuccessful realization of \(q^\pi\) is not a negative label for \(F\). A successful recorded suffix supplies an existence witness for that exact state, model and budget; it does not establish robust success probability or transfer to another robot. A bilateral contact reading establishes neither force closure nor load-bearing attachment.

The initial expression \(C_{\rm continuation}(z,e,G)\) omits the continuation class, state uncertainty, horizon and termination rule. A model trained without these distinctions can learn the weaknesses of its logging controller and call them robot capability. Conversely, factual return prediction is a legitimate learnable target; the problem is claiming that it identifies physical impossibility or a new mechanism.

## 2. One candidate, and its falsifiable claim

TOCAT proposes to refine a task-effect representation by **embodiment-conditioned future distinctions**. Two interactions with the same present task effect may need different representations if a specified continuation test has different attainable future effects. Transfer should preserve the goal-relevant continuation distinctions, rather than merely matching present object displacement or contact geometry. [METHOD_V0_ALGORITHM.md](METHOD_V0_ALGORITHM.md) gives the variables, joint likelihood and inference rule.

This is stronger than writing three weighted costs, but it is not automatically new. Predictive-state representations, learned feasibility models, completion values and task planning already cover substantial parts of this formulation. The research claim would have to be an identifiable and more data-efficient way of learning the relevant cross-embodiment distinctions, or a justified computational improvement. A new name for a value head, phase classifier, contact trigger, or three-head critic would not satisfy the claim.

The formulation deliberately does not revive H1 load-belief, H2 timing/control, or H3 effect-Jacobian/visual-MPC hypotheses. Their existing results remain intact. It also does not use BARX's missed closing command as evidence for this candidate.

## 3. Training and deployment information

| Information | Offline factual labels / analysis | Proposed deployment input |
|---|---|---|
| RGB, instruction, causal observation history | Yes, when recorded | Yes |
| Own past commanded actions | Yes | Yes |
| Static robot structure, gripper geometry, calibration | Yes | Yes |
| Measured joint state | Available in some Panda logs, absent from source RLDS schema | **Not silently assumed** |
| Exact object pose, contact geometry, simulator state | Available in evaluation telemetry and some raw Panda files | No |
| Future official task outcome | Factual supervision only | No |
| Unexecuted alternative continuation outcome | Not supplied by these demonstrations | No |

The requested RGB-plus-structure policy does not observe an exact Markov physical state. Static kinematics cannot recover unobserved configuration, contact modes or controller state by itself. An oracle-state planning comparison is a separate information tier; it cannot be presented as an equal-information comparison to an RGB policy. At equal information, both methods may maintain the same belief or use the same learned observation model.

Whole-episode source/target demonstration partitions are inherited from the existing manifests. The Panda split is 90 training / 10 validation demonstrations across the two tasks. No result on already exposed conditions 60–99 is an independent unseen test. The outcome-selected historical mechanism conditions are used only for a label-validity counterexample, never model selection or performance estimation.

## 4. What the existing files actually supervise

The executable inventory is [supervision_inventory.json](../method_v0/results/supervision_inventory.json). It checks all manifest counts and the available storage schemas, rather than assuming that an RLDS `state` is full physical state.

| Existing resource | Count | Actual usable supervision | Missing for the proposed variable |
|---|---:|---|---|
| UR5e/IIWA/Kinova3 source demonstrations | 1,800 | Packed actions and RGB; original RLDS also has EEF/gripper state, 2D boxes and motion labels | Joint configuration, exact contact and alternative-continuation trials are not in the released RLDS schema |
| Original Panda demonstrations | 100 | Actions, RGB, EEF and joint states, raw states, boxes | All recorded rewards in the two raw files are zero; termination alone does not certify official full-task success |
| Original Baseline trajectories | 1,280 | Actual actions, states, contact logs and official outcomes | Failures are conditional on the executed policy |
| New BARX adaptation trajectories already present | 160 | Same factual telemetry and official outcomes | Same limitation; no rerun here |
| Existing MimicGen augmentation trajectories | 177 | Actual segment execution and official outcomes | 114 unsuccessful episodes end before the benchmark horizon |

The primary factual corpus contains **1,617** episodes, disjoint across the last three rows. All 177 augmentation episodes are included, not just successful generated demonstrations. The ten historical MimicGen probes are excluded because repaired/replayed attempts are not an additional independent sample. All 24 existing mechanism rows are scanned separately; five conditions contain the three comparable handoff suffixes used below.

An early stopped unsuccessful script is a valid negative outcome for that finite script, but is right-censored relative to a controller allowed to continue until the original benchmark horizon. No early stop is relabeled as physical infeasibility. The task effect sequence uses the **existing official placement and retreat predicates** and actual object displacement; no height, lift, or distance proxy is substituted for success.

## 5. A real counterexample to an invalid training label

[matched_continuation_witnesses.json](../method_v0/results/matched_continuation_witnesses.json) checks every available historical matched handoff group. Each group's three `handoff.npz` files match exactly across **149 numerical fields**, with maximum difference 0. The recorded prefix actions/states, model XML hash and six declared RNG/queue/mode entries also match. These are existing executed continuations; this work does not restore or step a simulator.

| Existing condition | M0 outcome | M1 outcome | M2 outcome | Opposite outcomes at verified handoff |
|---|---|---|---|---|
| 68 | Success | Success | Success | No |
| 71 | Unsuccessful, early stop | Success | Unsuccessful, early stop | Yes |
| 72 | Unsuccessful, early stop | Success | Unsuccessful, early stop | Yes |
| 88 | Success | Unsuccessful at full horizon | Success | Yes |
| 93 | Unsuccessful, early stop | Success | Unsuccessful, early stop | Yes |

Thus **seven unsuccessful suffixes occur at four states with an actual successful continuation witness**. Even condition 88, whose failing suffix consumes the full horizon, cannot receive an intrinsic-impossibility label. This establishes the supervision distinction on real robot-simulation records. It is not a new controller improvement, a newly generated counterfactual, or a generalization score. M0/M2 and M1 have different observation privileges; their success counts do not compare equally informed policies.

The 80 original Joint UR5e/Panda task/index pairs also have 30 different factual outcomes. All have matching object asset paths, maximum initial object position discrepancy 0.0611 mm and orientation discrepancy 0.002511 rad. They have different robot configurations, histories and policies. They **do not identify intrinsic cross-embodiment capability differences**. Matching object location is not a causal state match.

## 6. What remains unidentifiable

An untried continuation can succeed in one dynamics model and fail in another while both models reproduce every logged action/outcome. `symbolic_nonidentifiability_witness()` constructs this elementary two-world argument. It is a logical proof about missing observations, **not a physical counterfactual label**.

Existing trajectories support factual task effects and a small number of positive existence witnesses. They do not supply certified negative physical-feasibility labels, or sufficient coverage of alternative target continuations to identify the proposed cross-embodiment contract. Parametric assumptions might extrapolate beyond this coverage; an extrapolation is not an authenticated label. A conventional factual success predictor could still be fitted, but would not establish TOCAT's claimed distinction.

All work here is offline: **0 new physics steps, 0 Baseline reruns, 0 model updates**. The conclusion is a limitation of this candidate and these observations, not a theorem that no useful transfer algorithm can exist.
