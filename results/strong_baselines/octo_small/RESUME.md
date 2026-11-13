# Octo-Small completed; do not retrain

Current result: COMPLETE_FAITHFUL_PORT for one UR5e+Panda source_target setting.
Formal training24000 updates,768000 anchors; Adam and schedule counters24000;
frozen T5 unchanged. Selected update11000 by fixed target-validation DDPM loss
0.7085606306791306. Both official Panda tasks:0/40, EXPOSED_REFERENCE.
Target-only was not run. This is a declared bounded-budget faithful port,
not the default256×50000 official generic finetuning budget.

The detached pipeline finished80 formal episodes,50000 control steps, plus
2 separate development smoke episodes1250steps. Do not restart it or rerun any
old baseline. Read RESULTS.md, results.json, training_state.json,
training_metrics.json, optimizer_resume_audit.json and artifact_index.json.
The files named *_snapshot.json retain earlier partial progress; they are
historical snapshots. Current progress.json/status.json are the final state.

All large files remain under
`/home/users/lemon/strong-baselines-20261112/octo_small/`.
Selected checkpoint:
`training/source_target/selected/11000/default/checkpoint`, SHA256
`628073b9e6439a2e800293e980dde908cf45bf92d3b3772571e0f4f9d6f99188`.
Full optimizer/RNG state: `training/source_target/resume.msgpack`.
Formal videos/telemetry: `physical/formal/source_target/<task>/condition-060..099/`.
Native normalized/physical/canonical queries: two JSONL files indexed by
native_action_audit.json. All12520 formal queries matched saved telemetry.

Official Octo source241fb3514b7c40957a86d869fecb7c7fc353f540, MIT;
Octo-Small1.5 weights dc9aa3019f764726c770814b27e4ab0fc6e32a58, MIT.
Runtime: isolated JAX0.4.20/Flax0.7.5, CUDA11/cuDNN8.9.6.50.
Training uses original540/60UR5e and90/10Panda full-demonstration split,
primary RGB256+native T5 instructions, history2,4-step diffusion actions.
No additional real camera, robot state or goal image; missing modalities use
native false masks. Canonical OSC7→12,20Hz, official placement AND retreat unchanged.

Current scope is results/strong_baselines/research_scope.json. CrossZero/Object
Flow are archived, no new budget. KITE deferred. Preserve MimicGen v2's48000
updates/80 trials and all BC/DP/BARX/ACT/VQ-BeT results. No new Method.
SmolVLA is next in the authorized sequential order; it has NOT_RUN status.
No SmolVLA or Flow Matching Policy model was started during this Octo stage.
Ordinary engineering repairs are allowed; a Python error alone is not permanent
BLOCKED. New methods still need official-code/data/update/smoke gates before
budget freeze. Never tune or select checkpoints using60–99 success rates.
