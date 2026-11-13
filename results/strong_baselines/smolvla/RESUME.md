# SmolVLA completed: stop this round

Status: COMPLETE_FAITHFUL_PORT, one source_target_state setting only.
Official native RGB+joint/gripper qpos input; not RGB-only. Target-only not run.

Formal training: 16000 updates,512000 anchors;9 diagnostic updates discarded.
Selected checkpoint: update2000,64000 seen anchors, target-validation
real7 flow MSE0.23551705992571065.
Checkpoint SHA256: 364b9c5dfc17996df412f8e64dfc7b7c1df710a1b35f9acadbd62776d3fd5bc8.

Formal EXPOSED_REFERENCE60–99: Counter→Sink1/40; Sink→Counter2/40.
Exactly80 formal episodes and2 prior development smokes. All official complete
successes, videos, raw native32→7 actions,50-step queue and49263 real qpos rows
passed acceptance. Native optimizer/scheduler counters16000;345 frozen tensors
unchanged. Original8 protected files,7 old result trees and251 Octo artifacts
verified unchanged. Full results and absolute paths/SHA are in RESULTS.md,
results.json, episodes.csv and artifact_index.json.

The pipeline has stopped. Do not launch run_frozen again, resume training,
add smokes, rerun old baselines or start Flow Matching Policy/new Method.
Only finish GitHub publication/readback if not yet recorded in server
smolvla/final_delivery.json. Then wait for a new researcher instruction.
