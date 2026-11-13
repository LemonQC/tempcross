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

Current gate:official source and checkpoint acquisition plus isolated runtime
setup. No real training update or task smoke has passed yet. No formal budget
is frozen. Do not claim a completed baseline or start formal training prematurely.
Native model supports single primary RGB256,language,T5 frozen encoder,2-frame
history and4-step diffusion actions. Wrist inputs can be absent by the original
mask contract. Initial main setting is UR5e+Panda full finetuning, retaining
native transformer/diffusion/T5 and original540/60 source,90/10 target identity
split. Saved raw RGB is available directly; no camera re-rendering is needed.
Original canonical OSC7→12 and full placement AND retreat are unchanged.
Model/resource/update probe and full smoke precede final configuration/budget
freeze. All reference60–99 trials are EXPOSED_REFERENCE, never independent test.

The prior six-method delivery is historical, not the active stopping condition.
Read `results/strong_baselines/research_scope.json` and current server
`session_resume.json` before resuming. GPU has no competing compute job at start;
/home has131GiB free. /data remains inaccessible to this UID despite free space.
