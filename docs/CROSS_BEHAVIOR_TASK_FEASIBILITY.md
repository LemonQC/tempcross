# Stage A actual results: tasks are not ready for learning

Status: **BLOCKED_PHYSICAL_GATE**, not a completed cross-behavior benchmark. No new model was trained, no old baseline was rerun, and no formal learned-policy result was generated. This is a bounded non-learning expert feasibility experiment, not an algorithm comparison.

## Actual trials

19 development episodes, **9,160 actual control steps**, at 20 Hz. All 19 have complete contiguous telemetry, per-control-step integration snapshots, 20 fps videos, initial XML/state/RGB-D, instructions and SHA indexes. Each task used exactly one documented expert/initial-geometry repair; original failed trials remain included. The 600-step horizon, physical parameters and frozen success contract were not relaxed.

| Task | UR5e full frozen contract | Panda full frozen contract | Native predicate diagnostic |
|---|---:|---:|---|
| Push | 0/4 executed; 4 slots not run | 0/3 executed; 5 slots not run | No native supported-surface goal-region predicate; new contract |
| Stack | 0/3 executed; 5 slots not run | 0/3 executed; 5 slots not run | Native Stack predicate observed in 2/3 UR5e and 3/3 Panda trajectories |
| Drawer | 0/3 executed; 5 slots not run | 0/3 executed; 5 slots not run | Native ≥0.95 opening never reached |

The acceptance threshold remains **6/8 for each robot**. After three retained failures that threshold is unreachable, so new conditions were not launched. UR5e Push has a fourth episode because it had already been launched when the runner was paused; it was completed from step 319 without rerunning earlier actions. None of the unrun slots is encoded as a failed trial. Native predicate diagnostics are not stable full-contract successes or formal evaluation success rates.

## What physically happened

**Push.** UR5e condition 0 places the block outside the primary view using the native default random counter segment; the expert cannot identify the target. The sole physical repair restricts native initial placement to the counter segment nearest the robot's placement reference. Subsequent UR5e trials establish real finger–object contact and displace the block about 13.6–13.9 cm. They miss the goal laterally by roughly 8 cm. RGB-D inspection shows the visible blue outline is partially occluded: its observed-pixel median is a biased goal-center estimate. Panda's three trials remain in approach; the generic red segmentation can latch onto a larger reddish background component. A controller that picks the wrong visual component is not evidence of Panda's inability to push. We did not supply simulator target/object poses to rescue it.

**Stack.** The first UR5e trial similarly selects a reddish cabinet rather than the block and never grasps. The one repair adds a depth-estimated horizontal support plane and block-sized candidate filtering, and moves only the initial sampling region to keep both blocks visible. Subsequent real trajectories grasp, lift, transport, contact the support block and release. UR5e conditions 1 and 2 reach the native Stack predicate, with millimetre-scale XY/height error; all three Panda traces reach that predicate. The original expert is slow, with a 6 mm maximum commanded translation increment and phase holds. It does not finish the extra frozen 10 cm retreat plus 20 stable ticks within 600 steps. At step 600 it is already in retreat, rather than failing the native physical stacking interaction. This is substantial evidence against calling Stack physically impossible. It is also not permission to remove retreat or lengthen the horizon after seeing results. The extra retreat requirement is stricter than the native task and must be reviewed explicitly in a later protocol.

**Drawer.** Native `OpenDrawer` and its actual door mechanism are used. UR5e condition 0 establishes handle contact for 45 control steps, reaches the pull phase very late and opens at most **0.01911 normalized travel**, far below the unchanged native 0.95 threshold. Its initial RGB-D median refers to the visible upper bar surface rather than the handle center, and the unconstrained plane estimate tilts the pull normal. The one repair uses a vertical-front fit, a cylinder-center fit with the known native 13 mm CAD radius, and a Drawer-only 20 mm maximum commanded translation increment. It leaves native OSC gains/scales and physics intact. The refined detector cannot consistently certify a handle in the remaining UR5e/Panda views. Those episodes retain `RGBD_HANDLE_NOT_IDENTIFIABLE` instead of inventing a target pose. Panda reaches at most **0.0003482 normalized opening**. No full opening/release/retreat success is claimed.

Failure classes and per-episode phase/contact/opening diagnostics are in `results/cross_behavior/episodes.json`. All labels are diagnostic; matching a few spatial coordinates across these different histories is not causal pairing. The raw Stack `toppled` geometric flag is not a valid topple-rate label before the first support/release event; no such rate is inferred from it here.

## Preservation, recovery and limits

- The narrow pre-rollout preservation audit verified **931 pre-existing files / 8.2247 GB** against published expected hashes and the old committed trees. Its final check also passes.
- A broader cached inventory covers **10,863 old files / 47.5318 GB**, including the original formal telemetry/video tree and all prior SHA-index references. Every present file matches its initial snapshot at the final recheck; missing files: **0**. One historical checksum discrepancy predates this task: `/home/users/lemon/strong-baselines-20261112/mimicgen-v2-report.log`. Its current content is also preserved unchanged. Thus we claim no changes in this run, not that every historical mutable report-log hash had always matched.
- Main stays separate and unchanged. No PnP condition, dataset, checkpoint, success predicate, result or old source file is written by the new runner. Existing result trees and old code are unchanged relative to the branch base.
- Real Push and Stack recovery records verify exact saved integration-state restoration. Each completed episode has exactly one committed step and one video frame per control index. The crash tests reject uncertain in-flight actions rather than replaying them. No duplicate uninterrupted rollout was run to claim future bitwise physical determinism.
- 64 local tests passed. `--status` works with `python -S`, without simulator/model dependencies. Terminal `--resume` returns the saved gate decision without rescanning data or launching physics. The background runner and per-episode locks handle chat interruption independently.

Large artifacts remain under `/home/users/lemon/cross-behavior-20261114/stage_a/{push,stack,drawer}/{ur5e,panda}/dev-NNN/`. The Git index lists every video, telemetry, initial state/XML and each server-side full per-frame/per-step SHA manifest. Reports distinguish **10 unique task condition IDs** from **19 robot-specific restores**. The full 20-development/40-formal banks, pilots, training splits, 300/50 demonstration pools, algorithm adapters and optimizer probes remain **NOT_RUN**. No condition 60–99 was used for this expansion.

## Decision

Keep all learned-policy training disabled. The current evidence identifies failures of expert perception, scene visibility and declared execution timing; it does **not** establish a new learning problem, an embodiment limitation or a policy ranking. The single next recommendation is to review and repair the non-privileged expert's observability and trajectory-time contract before authorizing another bounded task acceptance run. Preserve this run as failed feasibility evidence; do not relabel it, spend another automatic repair, or train networks on its incomplete demonstrations.
