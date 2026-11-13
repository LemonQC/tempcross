# Octo-Small active baseline (2026-11-13)

Latest user instruction replaces the old mechanical one-error stop rule. Ordinary
engineering fixes are allowed; core algorithms/benchmark/success stay unchanged.
CrossZero and Object Flow are archived with no new budget. KITE is deferred.
MimicGen v2's48000 updates/80 trials and all BC/DP/BARX/ACT/VQ-BeT results must
not be rerun. Candidate order:Octo-Small→SmolVLA→Flow Matching Policy. Only Octo
is being handled now; no other large training may start in parallel.

Official checkout `/home/users/lemon/baseline-benchmark/vendor/octo`, SHA
`241fb3514b7c40957a86d869fecb7c7fc353f540`, MIT. Original checkpoint is
`rail-berkeley/octo-small-1.5@dc9aa3019f764726c770814b27e4ab0fc6e32a58`.
Native JAX0.4.20/Flax0.7.5 run in a separate prefix:
`/home/users/lemon/strong-baselines-20261112/octo_small/env`.
Raw large files, checkpoints and videos stay in the server method directory.

Current state:formal pipeline launched, PID1944947 (verify liveness, do not assume).
Read server `session_resume.json`, `octo_small/pipeline_process.json` and
`octo_small/live_progress.json`; run `python -m results.strong_baselines.octo_small.progress`
for persisted actual progress. Do not launch a second trainer while the pipeline
or its child lives. `run_frozen.py` holds an exclusive file lock and sequentially
trains, evaluates the80 fixed conditions, then audits results. It never starts
another model. Failed/partial stages must be reported honestly, with logs kept.

All preflight gates PASS; formal protocol frozen.6 actual discarded
probe updates,252 tensors changed with frozen T5 unchanged. Both complete task
smoke episodes accepted (600/650steps, success false). No formal result yet.
The640 native RGB caches are PASS. Formal training is active; do not retrain
passed probes or recreate cached images. See progress.json for the latest
published snapshot; it is not a completed baseline or a final success estimate.
Native model supports single primary RGB256,language,T5 frozen encoder,2-frame
history and4-step diffusion actions. Wrist inputs can be absent by the original
mask contract. Initial main setting is UR5e+Panda full finetuning, retaining
native transformer/diffusion/T5 and original540/60 source,90/10 target identity
split. Saved raw RGB is available directly; no camera re-rendering is needed.
Original canonical OSC7→12 and full placement AND retreat are unchanged.
If the pipeline has stopped, inspect its error/state before resuming
`python -m results.strong_baselines.octo_small.run_frozen` with the old simulation
Python. This resumes optimizer/RNG and skips fully completed jobs/episodes.
Native training uses `env/bin/python -m results.strong_baselines.octo_small.launch train`
from the scientific repository. This launcher sets isolated CUDA11 library search
paths BEFORE importing TensorFlow, preventing its system cuSOLVER from polluting
JAX. `train.py` checks protocol/code/data hashes and resumes optimizer+RNG.
Frozen batch32×24000,768000 anchors, native full finetuning, validation every1000,
minimum fixed256-anchor DDPM loss, checkpoint/resume every500,14400s wall cap.
All reference60–99 trials are EXPOSED_REFERENCE, never independent test.

The prior six-method delivery is historical, not the active stopping condition.
Read `results/strong_baselines/research_scope.json` and current server
`session_resume.json` before resuming. GPU has no competing compute job at start;
/home has131GiB free. /data remains inaccessible to this UID despite free space.
