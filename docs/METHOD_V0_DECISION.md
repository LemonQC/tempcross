# Method v0 decision and conditional minimal experiment protocol

**Decision: `NO_GO_LEARNED_METHOD_IDENTIFIABILITY_AND_NOVELTY_NOT_ESTABLISHED`.**

The single TOCAT candidate is now specified as task-effect / embodiment-conditioned continuation-contract refinement. The existing evidence does not identify its claimed physical continuation variable, and the draft does not provide a learning mechanism substantively distinct from completion-value, predictive-state and learned plan-feasibility methods. **There is no trained original Method, new Panda checkpoint or new physical success rate.** Do not interpret this delivery as permission to run the next phase.

This is a completed negative research decision, not a claim that all future approaches are impossible. The reusable work is a formalized variable, supervision provenance, a real label counterexample, executable reference operators and a bounded follow-up protocol. See [scientific problem](METHOD_V0_SCIENTIFIC_PROBLEM.md), [algorithm](METHOD_V0_ALGORITHM.md), [technical comparison](METHOD_V0_COMPETITOR_COMPARISON.md) and [code/reproduction guide](../method_v0/README.md).

## Actual work and results

Base: `571b1b0cf66eb20495a5cc3a6cdbb0da9d25834e`. Independent branch: `method-v0/tocat-20261112`. All files preceding this branch remain unchanged. Machine-readable results: [summary.json](../method_v0/results/summary.json), [coverage.csv](../method_v0/results/coverage.csv).

| Existing factual corpus | Episodes | Official successes | Measured bilateral contact | Contact then not completed | Early stopped unsuccessful episodes |
|---|---:|---:|---:|---:|---:|
| Original Baselines | 1,280 | 69 | 187 | 118 | 0 |
| Existing new BARX adaptation runs | 160 | 39 | 69 | 30 | 0 |
| Existing MimicGen augmentation | 177 | 63 | 136 | 73 | 114 |
| Total | **1,617** | **171** | **392** | **221** | **114** |

These are corpus coverage counts, **not TOCAT evaluation results or a comparison between those methods**. The corpora have different experimental roles. All 221 contact-then-noncompletion observations remain policy/stopping-rule-conditioned; there are **0 certified intrinsic-infeasibility negatives**. The 114 early stopped unsuccessful augmentation episodes are censored relative to the full benchmark horizon.

- All five available matched handoff groups were checked. At four groups, seven failed suffixes have another actual successful suffix from the exact recorded handoff. This disproves the proposed shortcut label “failed continuation = physically infeasible interaction.” It does not compare equally informed controllers.
- All 80 original Joint UR5e/Panda task/index pairs were included. Their 30 factual outcome differences do not separate robot capability from policy/history differences. No real intrinsic cross-embodiment feasibility difference was identified.
- The finite-evidence continuation operator and independent forward planner used the same 1,622 roots and every remaining integer budget: **925,754 comparisons, 0 decision disagreements**. The 176 supported-positive and 1,446 unknown roots are support accounting, not success probabilities. This comparison does not claim to implement full continuous TAMP or measure its runtime.
- An abstract two-model test demonstrates missing-data nonidentifiability. It produces **no robot counterfactual labels**.
- **Training updates: 0. New physical steps/episodes: 0. Old Baseline reruns: 0. Full-VLA training: 0. KITE/MimicGen ports: 0.** The limit of one small prototype was a ceiling, not a requirement to fit a scientifically unjustified predictor.

The graph decision comparison alone is not a universal novelty veto: a new estimator might improve sample or compute efficiency while making the same optimal decisions. No such estimator or advantage is established here. The negative gate follows from that missing mechanism together with the supervision limitations, not merely from a planner's name or a lack of compute.

## Artifacts and limits

`method_v0/` contains extraction, partitioned factual records, measured contacts/predicate words, input hashes, exact-handoff checks, the reference contract operator, an independently coded planner and unit tests. Large extracted numeric arrays remain at:

`/home/users/lemon/method-v0-tocat-20261112/`

[server_artifacts.json](../method_v0/results/server_artifacts.json) indexes absolute paths, byte sizes and SHA-256 hashes. [input_index.index.json](../method_v0/results/input_index.index.json) indexes the 3,234 primary raw telemetry/contact inputs; the matched-handoff index adds the actual prefix/runtime/snapshot and suffix inputs. These paths refer to this server, not GitHub-hosted data. No videos, weights or raw images are newly generated.

The retained historical 60–99 conditions are exposed. Matched mechanism cases were originally outcome-selected. There is **no independent unseen test**, no validation-based checkpoint selection and no new physical success estimate in this delivery. Original source/Panda demonstration train/validation manifests are not altered.

## Conditional smallest additional-data protocol — NOT EXECUTED

This is a bounded protocol for researcher review, not an automatic next run or a promise that additional data resolves novelty. A distinct estimator must first be justified against the comparisons above. If the researcher instead wishes only to test whether continuation-contract supervision can be collected, the following pilot would address coverage without claiming an original algorithm:

1. **Scope and cap:** the same two tasks, UR5e and Panda, the same objects/interfaces/success definitions. Use existing development conditions **0 and 1**, chosen before observing this pilot's outcomes. Two tasks × two robots × two conditions gives **8 anchor acquisitions** maximum. No IIWA/Kinova3 simulator extension is required; their existing demonstrations remain available.
2. **Anchor:** first measured bilateral object contact under a predeclared, frozen prefix controller. If no contact occurs by the original task horizon, record missing anchor and stop that branch. Do not search replacement conditions or modify thresholds/controllers to create a favorable example.
3. **Snapshot certificate:** save model XML, complete integration/controller/warm-start state, object state, observation history, policy queue and all RNG states. Before each fork, verify the restored state against that snapshot. If restoration cannot be certified, do not call outcomes a matched causal pair.
4. **Continuation options:** exactly two frozen options, declared and hashed before collection: the selected existing policy's ordinary continuation and the existing object-relative complete place/retreat script where available. No tuning, new network, or success-selected option search. The script's privileged-state information must be declared; it supplies a physical existence witness, not an equal-RGB policy comparison.
5. **Replicates and budget:** two predeclared stochastic seeds per option per anchor, giving at most **32 suffix rollouts**. Together with at most eight prefix acquisitions, the cap is **40 new episodes**. These are future budgets, not executed counts. Deterministic duplicate suffixes are marked duplicates, not independent probability trials.
6. **Original success and horizon:** the original joint placement/retreat task check, 600 global actions for Counter→Sink and 650 for Sink→Counter. A suffix receives only `global_horizon - anchor_action_index`; no clock reset, extra actions or relaxed phase success. Record all outcomes, steps, contacts, task predicates, actions and termination reasons.
7. **Labels:** success is a positive physical-existence witness at the certified snapshot. Failure is a negative result for that option/seed only. Two failed options do not prove no feasible continuation exists. Any stronger negative requires a valid geometric/dynamic impossibility certificate with stated model assumptions.
8. **Splits and analysis:** all forks from a prefix, robot/task/condition lineage stay in the same whole-group split. Predeclare a development/validation assignment before any fit; with this small pilot, report label coverage only, not generalization success. No split or checkpoint selection on historical 60–99 outcomes. A later efficacy study would require separately authorized, predeclared conditions and adequate sample size.
9. **Information matching:** any future learned contract and planning comparator receive the same observations, geometry/model access and learned predictors. Report query cost, prediction calibration and decisions separately. Do not compare oracle planning against RGB learning as though the inputs were equal.

This pilot can reveal repeated policy-conditioned distinctions and the cost of obtaining labels. It cannot alone certify intrinsic target/source capability differences or establish novelty beyond existing learned feasibility and completion values. It is therefore **not a recommendation to execute it immediately**.

## Unique recommendation and stop condition

**Retain this candidate as a documented negative result and stop for researcher review.** Do not train a renamed outcome classifier, fabricate counterfactual labels, launch physical trials, expand Baselines or open a second candidate. Reopening TOCAT would require both an explicit new estimator with a defensible distinction and an authorized supervision protocol; neither is inferred from the current objective.
