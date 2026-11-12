# TOCAT v0: technical novelty and replacement checks

This comparison concerns a proposed learned continuation contract, not a claim that planning or existing methods ignore contact. Sources below are primary papers. Existing Baselines are preserved and were not retrained or ported. Native input differences matter: a method with point clouds/joint state or two cameras cannot be treated as the same RGB-plus-static-structure information tier without adaptation.

## Required comparisons

| Method | Relevant mechanism and native information | What a continuation contract would add | Boundary of the present candidate |
|---|---|---|---|
| **KITE**, v2 | Contact positions/normals encoded as shared intent, embodiment-conditioned kinematic decoding; original deployment uses point cloud and joint state | Explicit goal-conditioned future attainability after realizing an intent | Merely adding a sequence/phase or an embodiment-conditioned decoder is not a nontrivial distinction. No new intent-learning estimator is supplied here. |
| **Cross-Behavior Object Flow**, accepted T-RO version | Future object mask sequence represents task effect; target action prediction learns how to realize it; native policy uses two RGB cameras and proprioception | Preserve future distinctions between interactions producing similar current effects | Object-centered effects and learning different behaviors for a common effect are already present. A future-effect latent alone is not new. |
| **BARX** | Behavior alignment with action prediction and auxiliary object-box, motion-language and EEF-trace supervision | A supported test of downstream completion for an interaction | Additional auxiliary outcome prediction is insufficient unless it identifies a new decision variable and produces a justified benefit. |
| **MimicGen** | Object-relative segment transformation, interpolation and executed generation; successful generated demonstrations support policy learning | Learn when a locally transferable interaction cannot sustain later task completion | Geometry plus a full continuation check can already reject many such cases. Privileged generation geometry is distinct from deployed policy input. No port is attempted. |

Primary sources: [KITE v2](https://arxiv.org/html/2606.22113v2), [Cross-Behavior Object Flow author manuscript](https://eprints.whiterose.ac.uk/id/eprint/242108/1/26-0166_02_MS.pdf) and [publication record](https://eprints.whiterose.ac.uk/id/eprint/242108/), [BARX v1](https://arxiv.org/html/2607.27549v1), [MimicGen CoRL paper](https://proceedings.mlr.press/v229/mandlekar23a/mandlekar23a.pdf).

These are technical comparisons, not reproduced head-to-head performance results. The proposed strict deployment tier has no silently added measured joint state. Offline exact-state checks do not remove that limitation.

## Stronger overlaps than the original four comparisons

| Prior work | Specific overlap with the proposed learning variable | Consequence |
|---|---|---|
| **Foresight Residual RL**, v1 | A locally successful terminal state may be unsuitable for a downstream skill. A continuation value for fixed downstream policies is fitted from repeated handoffs and used for residual skill improvement. | A post-contact completion predictor is already an established mechanism. Conditioning it on robot identity or calling it a contract does not establish novelty. This paper's MAXQ/option-completion connection also precludes claiming continuation value itself as new. |
| **Active Learning of Abstract Plan Feasibility**, RSS 2021 | Complete plan feasibility is learned from actual executions, using prefix structure and active query acquisition. | A learned full-task feasibility model is not outside planning. Learning from failed full sequences or choosing informative continuation tests alone is insufficient. |
| **Predictive Representations of State**, NeurIPS 2001 | Action-observation continuation-test predictions define a state representation. | A vector of future-test probabilities is not a new representational principle. |
| **DeepMDP**, ICML 2019 | Joint reward/transition prediction motivates a latent model and value-preserving abstraction. | A shared latent outcome/transition model, without a new estimator or guarantee, is insufficient. |
| **Counterexample-Guided Abstraction Refinement** | Counterexamples split an abstraction when merged states cannot support the same property. | Splitting an effect class using future distinctions is not by itself an original refinement algorithm. |

Sources: [Foresight Residual RL v1](https://arxiv.org/html/2607.16506v1), [Abstract Plan Feasibility](https://arxiv.org/html/2107.00683v1), [Predictive Representations of State](https://proceedings.neurips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf), [DeepMDP](https://proceedings.mlr.press/v97/gelada19a/gelada19a.pdf), [CEGAR, CMU research record](https://doi.org/10.1184/R1/6604547).

The overlap is evaluated at the mechanism level. We do not claim that any cited paper already solves the exact RoboCasa-X/Panda few-demonstration benchmark. Conversely, a different benchmark is not a methodological distinction.

## Full task and motion planning under the same information

Consider the illustrative situation in which a grasp can lift an object, but cannot move it through a constrained destination while maintaining a reachable arm configuration. A full task/motion planner can carry the **same grasp and attachment variables** across pick, transport, placement and retreat constraints. It can reject that grasp using the entire skeleton. This example illustrates a standard constraint dependency; **no new robot counterexample was simulated or certified for it here**.

[PDDLStream](https://arxiv.org/html/1802.08705v5) provides conditional samplers/certified facts for continuous constraints inside task planning. A TOCAT score used only as an external cost would not certify feasibility. The comparison must distinguish: actual stream certificates, learned probabilistic feasibility, and absent evidence. The last is unknown, not a geometric impossibility certificate.

There are two legitimate comparison tiers:

1. **Oracle/model tier:** give both algorithms the same exact state, robot geometry, dynamics assumptions and budget. A contract recurrence over that model is backward reachability; complete forward search tests the same paths. Contacts, collision constraints and goal retreat remain in both formulations.
2. **Deployment tier:** give both the same RGB history, instruction, actions and static structure. Both must infer or maintain uncertain state. A planner may use the same learned observation/transition model. Giving only the planner privileged object/contact state is not an equal-information experiment; denying the planner the contract model supplied to TOCAT is also unfair.

`method_v0/contracts.py` implements the first tier's finite **witnessed evidence graph** specialization and an independent forward search. It yields identical supported options on 925,754 budget checks. It is not a continuous geometric planner benchmark, and its mostly single-edge graph does not measure complex task-planning efficiency. Its purpose is to expose whether the current decision operator adds anything beyond composing the same evidence.

Neither the label “TAMP” nor equal optimal decisions disproves algorithmic research value. A learned method could reduce expensive geometry queries, identify transferable abstractions with fewer target demonstrations, improve calibrated uncertainty, or bound planning complexity. Such a contribution needs an actual estimator and equal-information cost/generalization evidence. This delivery has none and does not infer one from an attractive objective.

## What can be replaced directly

- Contact-intent realization and robot-conditioned decoding overlap KITE and kinematics.
- Task-effect prediction overlaps object flow and behavior alignment.
- Fixed-policy post-contact completion prediction overlaps downstream completion values.
- Learned whole-sequence feasibility and query selection overlap abstract feasibility learning.
- Exact future feasibility from geometry/dynamics is within full task/motion planning.
- Phase-triggered closing, visual servoing and ordinary state-machine transitions are implementation tools, not an independent TOCAT contribution.

The only potentially distinct claim would be a new **identifiable cross-embodiment estimator/refinement rule** with demonstrated data or computational advantage after these replacements. That mechanism is not supplied by the current candidate. The unique recommendation is to retain the formulation and evidence as a **negative Method gate**, without training a renamed return head or starting another candidate.
