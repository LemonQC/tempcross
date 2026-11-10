# Final mechanism validation: frozen before physics

Expert instruction: `202611101108.md`; completed benchmark baseline: `fe149302b6f9f00c96fca6236e22df4e4a72cb06`.

Only UR5e → Panda, `pnp_counter_to_sink`, official conditions **64,68,71,72,76,88,93,94**. Read-only verification reproduced the expert's ten source-success/all-twelve-target-failure cases exactly. These eight selected difficult cases are **development diagnostics, not an independent test**.

The executable specification is [`configs/mechanism_protocol.json`](../configs/mechanism_protocol.json). Exactly one first attempt per condition × M0/M1/M2; at most 24 episodes, 600 control actions each, 20Hz, original dynamics, collisions and official success. No training, new baseline, expert generation, physical development smoke or threshold search. Every launched attempt occupies a slot; technical failures cannot be retried automatically.

## Observed source evidence, before intervention

[`scene_audit.json`](../results/mechanism/scene_audit.json) records actual source/target object assets, object poses and separate sink transforms. Sink transforms agree numerically in these eight saved scenes; object settling differs by up to about 8.44 micrometres. The controller nevertheless maps through each robot's actual object/sink frames rather than assuming world-coordinate identity.

Source successes and source contact poses come from saved benchmark H5/contact logs. Synchronous body+XML site poses avoid cached-observation substep lag. The complete extracted source trajectories remain on the server; [`source_contact_bank.json`](../results/mechanism/source_contact_bank.json) records absolute paths, byte sizes, SHA256, extraction times and grasp-complete state.

Condition **94** has official success but zero old `_check_grasp` pad-only proxy states. Its raw collisions show left and right **fingertip** contacts. It also has no post-lift open command: the source release event is final two-sided contact loss, not deliberate commanded release. Both limitations are retained. A new, explicit diagnostic accepts existing fingerpad OR fingertip contacts on each side, penetration/touch distance ≤0; old pad-only readings are logged separately. This changes **no official task-success criterion**. Stable grasp means this physical two-sided contact plus ≥3cm object rise for ten consecutive states.

## Matched interventions

All three independently execute the same anchored object-relative grasp prefix: open clearance, pregrasp, descent, pose-gated close, stable lift. Arrival requires position ≤15mm and orientation ≤10° for five states, plus contact/lift dwell where applicable. Time caps abort; elapsed time never advances an unreached motion phase. Ordinary unchanged OSC uses position/rotation error gain .4, max metric delta 12mm/.12rad. Original OSC kp/damping remain unchanged.

- **M0, PRIVILEGED_REFERENCE:** fixed source-derived object transport/release references, converted once using measured handoff attachment, tracked with continuous TCP OSC feedback.
- **M1, privileged prefix then original policy:** clear original BARX Joint action queue at confirmed stable lift; resume eight-action chunks using **RGB and verbatim instruction only**. Logger-only ground truth is never passed to BARX. No adaptation.
- **M2, PRIVILEGED_REFERENCE:** same prefix, object goals, gains and release/retreat logic as M0; continuously re-observe object-to-TCP attachment to correct actual object-state error. This is ordinary object feedback, not a proposed algorithm.

Source transport midpoint and source release object poses are expressed in the target sink frame. Transport/preplace use the common clearance floor `max(handoff object z, release object z+.12m)`; place uses source release pose. Both ordinary methods open while holding achieved TCP for20 steps, then retreat15cm upward. Full grasp/handoff snapshots use MuJoCo `mjSTATE_INTEGRATION`, warm-start, numeric controller/gripper and observation fields, RNG and cleared policy queue. Snapshots are saved for auditing, **never restored to branch or teleport**.

Every prefix action and simulator state is numerically compared against M0, as are handoff controller fields (tolerance1e-8). Failed prefixes are reported without a fabricated handoff. Videos, raw actions, object/contact trajectories, phase errors, placement and official retreat subpredicates are retained. A contact/lift proxy never substitutes for official task success.

## Fixed decision rules

A passes only when the common prefix restores stable grasp in at least6/8 cases **and** at least one individual ordinary controller (M0 or M2) completes at least4/8 official tasks. Do not pool a union of successes across methods.

If A passes: M1≥4/8 selects pregrasp visual/proprio contact acquisition; M1≤2/8 with M2≥4/8 selects postgrasp visual/proprio task-effect feedback. Other combinations are ambiguous; no second threshold search. These are research-prototype gates, not paper statistics. Strong ordinary-controller performance must be acknowledged.

Only after a pass, compare the single selected direction against MimicGen, KITE, CrossZero, object-flow transfer and classical visual servo/MPC. Without a specific distinguishable decision mechanism, report algorithm-development NO-GO. A clear mechanism requires an isolated `research_v0/` design and genuine forward/backward and closed-loop smoke; the document's additional smoke versus strict24 cap needs explicit resolution before exceeding24.

## Commands and publication gate

Use `/home/users/lemon/coft-p2/barx/.venv/bin/python` from the repository root:

```bash
python -m unittest discover -s tests -p test_mechanism_controls.py
python -m mechanism_validation.extract   # read-only, forbidden after first launch
python -m mechanism_validation.freeze    # read-only, forbidden after first launch
python -m mechanism_validation.run
```

The runner requires `/home/users/lemon/mechanism-20261110/frozen_publication.json` with the GitHub commit SHA, locally matching protocol SHA256 and `github_readback_verified:true`. Write this marker only after publishing and independently reading the protocol/code through GitHub. `run.py` validates all frozen source/code hashes and original checkpoint before launching. It refuses an existing/incomplete launched attempt instead of overwriting evidence.
