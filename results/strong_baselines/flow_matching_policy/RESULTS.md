# Official visual Flow Matching Policy port

COMPLETE_FAITHFUL_PORT for one source+target visual/state/language setting. This is the official visual flow-component port, not the original Robomimic object-state/absolute-action recipe or the full affordance-prompt framework.

| Task | Official successes/episodes |
|---|---:|
| pnp_counter_to_sink | 0/40 |
| pnp_sink_to_counter | 0/40 |

Actual formal updates:24000; discarded diagnostic update:1; full-run anchors:768000.
Selected native EMA update:16000, effective anchors:512000, fixed-target-validation velocity MSE:0.04101308.
Checkpoint:`/home/users/lemon/strong-baselines-20261112/flow_matching_policy/training/source_target_visual/best.pt`; SHA256:`e3e3af29cf2feb659d2da937ab60ed5e0575ab4b5b951c88aff64833551afece`.

Only validation chooses the checkpoint. Both development smokes completed600/650 steps but failed task success; they passed interface acceptance and are excluded from80 formal episodes.
Author ResNet18/GN, original UNet widths/time embedding, TorchCFM sigma0 velocity MSE, native EMA and16-predict/8-execute mechanism are retained. Native Gaussian training versus uniform single-Euler deployment is preserved and disclosed, not silently corrected.
RGB96 plus real qpos9 and frozen full-instruction CLIP768 are an explicitly separate information tier. No object poses, added camera or generated training data. Dataset/tensor-dimension, terminal-padding, finite-budget and script-driver differences are in PROTOCOL.md and OFFICIAL_AUDIT.md.

| Task | Descriptive failure category | Count |
|---|---|---:|
| pnp_counter_to_sink | CLOSE_WITHOUT_BILATERAL_CONTACT | 26 |
| pnp_counter_to_sink | CONTACT_WITHOUT_STABLE_LIFT_PROXY | 2 |
| pnp_counter_to_sink | NO_CLOSE_COMMAND | 11 |
| pnp_counter_to_sink | STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT | 1 |
| pnp_sink_to_counter | CLOSE_WITHOUT_BILATERAL_CONTACT | 23 |
| pnp_sink_to_counter | NO_CLOSE_COMMAND | 17 |

Contact/lift classifications are telemetry proxies, not causal explanations. Official task success is unchanged placement AND retreat. All60–99 conditions are EXPOSED_REFERENCE.
Reproducibility: official_source.json, protocol.json, freeze.json, installed_runtime.txt, errors.json, optimizer_audit.json, sample_exposure.json, validation.csv, episodes.csv and artifact_index.json. Large weights, videos, telemetry and native action logs remain at the indexed absolute server paths.
All previous baseline evidence is frozen and verified. No next baseline or new Method is started. Stop after GitHub publication and readback.
