# Pre-registered scientific decision rules (v0)

This file is a **project constraint**, not a claim of T-RO novelty. Do not move these goalposts after observing an experiment.

## What we have actually completed
- Reproducible Python-only SR5 kinematic FK/bounded IK and pinhole camera geometry code.
- A paired toy counterfactual: same target and 8 yaw choices, obstacle present vs absent.
- Equal-information motion-only, view-only, weighted and analytic-AND baselines.
- Unit tests and GitHub Actions checks.
- **No simulated physics, real robot, source-arm dataset or learned policy result.**

## Measurement prerequisites
Before evaluating a physical setup: (1) calibrate D435i optical-to-flange transform, (2) calibrate flange-to-TCP, (3) record measured object / obstacle positions in SR5 base frame, (4) confirm official kinematic file corresponds to the actual SR5 variant, (5) specify camera intrinsic/depth validity. A template's claimed status is *not* independent verification.

If prerequisites are missing: report **INSUFFICIENT_REAL_GEOMETRY** and do not call the research hypothesis false.

## Hypothesis A: Does body-specific geometry change action ordering?
Compare fixed same-goal candidate grasps with the actual source arm and SR5 target geometry, not two grippers on one Panda arm. Check success of *complete* candidate trajectories, not just isolated IK waypoints. Include collision/hand meshes and externally measured wrist extrinsics. If all actions remain interchangeable, the proposed *mechanism* lacks evidence in that scene distribution.

## Hypothesis B: Is any learned joint model necessary?
Compare on precisely the same candidate pool / sensing / planning compute:
1. IK + collision only.
2. Frustum + depth + ray visibility only.
3. Analytic feasibility AND visibility.
4. Independent weighted feasibility/visibility score.
5. Common-information one-step and receding-horizon MPC/CEM.
6. Learned joint predictor and policy, only after B is meaningfully nontrivial.

If analytic AND or information-equivalent MPC solves the real closed-loop task, **do not claim a learned COFT algorithm is necessary**. If a learned model only works because it sees more information or candidates, that is not a fair novelty claim.

## No-go granularity
- **Implementation blocked:** tool or calibration issue; not a scientific falsification.
- **Mechanism not observed:** reject this chosen scene regime; do not pronounce Cross-Embodiment manipulation invalid.
- **Mechanism present, analytic solution sufficient:** stop **this proposed learning innovation**; do not train a new large policy solely to gain a small toy score.
- **Mechanism present, simple controls fail in genuine task outcomes:** consider a measured multi-task learned-policy study with a release-quality benchmark.

Do not change held-out scenes/seeds after examining results. Record *all* failed attempts and the same baseline evaluation episodes.
