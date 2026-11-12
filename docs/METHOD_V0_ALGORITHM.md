# TOCAT v0: continuation-contract refinement

Status: **explicit candidate specification; substantive new learned mechanism not established; no neural Method trained**. The implemented evidence operator is a reference specialization, not a deployed Panda policy. See [decision](METHOD_V0_DECISION.md) and [competitor comparison](METHOD_V0_COMPETITOR_COMPARISON.md).

## 1. Variables and semantics

Let \(h_t=(I_{\le t},l,a_{<t},m_e)\) contain causal RGB history, instruction, past commanded actions and static embodiment description. A learned belief \(b_\theta(s\mid h_t)\), if used, must use only these inputs. Let \(\zeta\) denote a candidate robot-object interaction and \(z\) its task effect. Neither is assumed directly observable from RGB. Let \(w\) be a finite word of task effects, ending in the official goal if attained.

Define a **continuation test** \(T=(\pi,B,w)\), with an explicit continuation controller, remaining global budget and effect event. The proposed variable is

\[
\kappa_{e,G}(h_t,\zeta)[T]
=\Pr(w,\tau_G\le B\mid h_t,\operatorname{realize}(\zeta),\operatorname{do}(\pi),e,G).
\]

The semantics require an actual realization and a well-defined intervention. Ordinary behavior-cloning samples do not by themselves identify this `do` quantity. A deterministic alternative is the set of witness-supported achievable effect words. This is existential support, not calibrated probability.

Two histories with the same present task effect may be merged only if their continuation tests agree in the relevant policy class and budget range:

\[
h\sim_{e,G}h'\ \Longleftrightarrow\ z(h)=z(h')\ \land\
\kappa_{e,G}(h)[T]=\kappa_{e,G}(h')[T]\quad\forall T\in\mathcal T.
\]

The candidate learning problem is to construct a compact, transferable **refinement of the task-effect quotient** that retains these distinctions. Changing the robot changes the contract; equal 3D object position or equal current predicate bits does not certify equivalence. The phrase “continuation contract” names this variable, not an established original algorithm.

## 2. Joint learning formulation, rather than three weighted scores

A possible parameterization uses one history/structure encoder and one conditional transition kernel

\[
p_\theta(\zeta_{k+1},z_{k+1},a_{k:k+d-1},d,c_{k+1}
\mid \zeta_k,z_k,h_k,m_e,G),
\]

where \(c\) records realized contact and termination information. A robot-conditioned action decoder can differ across embodiments, while the goal-effect variables share semantics. A chain of these kernels defines the **joint** distribution of task effects, interaction realizations, actions and subsequent completion. Continuation probabilities are marginals of that common distribution, not three independent heads added with hand-picked weights.

For an observed episode \(i\), with observed variable mask \(M_i\) and termination/censoring event \(E_i\), the basic objective would be

\[
\mathcal L(\theta)=-\sum_i\log\int
p_\theta(X_i\mid h_i,e_i,G_i)\,
\mathbf1\{X_{i,M_i}=x_{i,M_i},E_i\}\,dX_{i,\bar M_i}.
\]

Continuous observations require densities/noise models rather than literal point-mass indicators. For a completed task, \(E_i\) includes the observed goal time; an unsuccessful early stop contributes survival/censoring information only up to the observed time. Actions conditioned on the logging policy are not randomized interventions. A learned joint likelihood does not cure missing support or confounding.

The task/contact/continuation costs in the initial user expression could be derived negative log marginals of this model. Their sum is generally **not** the joint log likelihood unless the corresponding conditional independence assumptions hold. No such independence is asserted. A bottleneck or representation regularizer could control model capacity, but is not the proposed novelty.

### What source/target transfer would need to learn

Transfer must align task-effect semantics while preserving **target** continuation distinctions, allowing different joint configurations and actions. A source interaction is transferable only through a target realization whose contract supports the goal within budget. This is not an assumption that equal task effects imply equal continuation. The source logs contain 2D effect information but not the full source contact/configuration labels needed to supervise such alignment directly.

An actual algorithm would need to specify a statistically identifiable estimator, selection of informative tests and a source-to-target transport mechanism that improves on existing predictive-state/feasibility learning. The equations above define a research variable and an objective family. They do **not** supply that missing estimator; adding an embodiment token is insufficient.

## 3. Inference specification and information limits

1. Infer a belief and candidate interactions from RGB history, instruction and structure, using the frozen learned model.
2. For each candidate, compose the joint kernel over admissible continuations with the **same remaining benchmark budget**.
3. Retain candidates with supported goal-reaching continuation; optimize expected official outcome or a declared risk bound, without choosing a model using target test success.
4. Abstain or report unknown outside supported continuation tests. An unsupported candidate cannot be certified physically impossible merely because no logged success exists.

This is a conditional inference specification. There is no implemented image encoder, learned belief, target action decoder, new checkpoint or online TOCAT success rate in this delivery. The user required substantive distinctness before learning implementation; the comparison below fails to establish it. Training an ordinary success classifier and calling it TOCAT would not fix that failure.

## 4. Implemented finite-evidence specialization

`method_v0/contracts.py` contains actual working code for:

- An evidence graph whose edges are entire **recorded** continuation suffixes, with witness IDs and positive durations.
- A backward predecessor recursion returning goal-supported first options.
- An independently implemented forward exhaustive witnessed-path planner using exactly the same edges, state identity and budget.
- Feasibility intervals: `[1,1]` for existential feasibility with a witnessed successful path, `[0,1]` for unknown; `[0,0]` requires a separate impossibility certificate. These are logical support bounds, **not success probabilities**.

Only five verified historical handoff groups join multiple continuations at one root. Ordinary episodes are never spliced by a shared object pose, predicate label or category. Each suffix is one edge, so this is a deliberately small evidence model, not a discretized continuous TAMP solver. Abstract multi-edge examples are tested separately and never included as robot results.

Let \(E(s)\) be the common edge set, \(d(e)>0\), and \(G\) the goal nodes. The reference contract obeys

\[
R_0=G,\qquad
R_B=G\cup\{s:\exists e=(s,s'),\ d(e)\le B,\ s'\in R_{B-d(e)}\}.
\]

**Equivalence:** on this finite graph, the recursion supports a first option if and only if exhaustive forward search finds a goal path starting with that option within budget. Proof: the first edge of any path leaves a smaller nonnegative budget; induction on integer budget applies because durations are positive. Conversely, each recursive witness can be prepended to its successor path. There is no first-option decision advantage on identical evidence.

The code actually compared **1,622 roots across 925,754 integer root/budget queries**: 0 disagreements. There are 176 witness-positive roots and 1,446 unknown roots. These counts include five shared handoff roots; they are not independent samples or new physical trials. A local-contact-only rule would differ, but it is not the relevant full-task planning comparator.

## 5. Where an algorithmic contribution would have to survive

At exact known dynamics/state, this formulation reduces to reachability with goal and embodiment constraints. At an identical partial-information model/belief, a planner can consume the same learned contract. Equivalent decisions do not rule out a useful learning or computational algorithm: sample complexity, uncertainty calibration, representation size or search cost could improve. None is demonstrated here. The finite graph does not establish empirical equivalence to every continuous planner.

Predictive-state test signatures, joint latent transition prediction, downstream completion value, learned plan feasibility and counterexample refinement have substantial prior art. [METHOD_V0_COMPETITOR_COMPARISON.md](METHOD_V0_COMPETITOR_COMPARISON.md) identifies the specific overlaps. We have not found a justified estimator/refinement rule beyond those components. **No claim of originality is made for their assembly.**

## 6. Supervision and evaluation result

Actual labels come only from the saved actions, measured contacts, observed object motion and official predicate traces. The real matched-state counterexample disproves labeling any failed suffix intrinsically infeasible. The abstract two-world test proves that unexecuted alternatives can remain ambiguous; its worlds are not robot samples.

Fitting a factual policy-conditioned return model is possible in principle. Learning the claimed embodiment-dependent physical continuation contract from these logs is not identified without additional assumptions or experiments. Neither supervised-negative coverage nor a distinct learning mechanism passed the gate. Consequently **0 of the allowed at-most-one small training runs were used**. This is the explicit negative Method decision, not an unreported failed training run.
