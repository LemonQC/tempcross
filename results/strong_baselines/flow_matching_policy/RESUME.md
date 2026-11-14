# Last baseline only: native Flow Matching Policy

Baseline base3c9671e8ca50cfca06b8f6e46ff3e97d1461cb4c; every earlier baseline,
including accepted SmolVLA16k/1-of40/2-of40, remains frozen. This round never
starts another baseline or Method.

Read server flow_matching_policy/pipeline_process.json, training state and
session_resume.json before any action. Do not launch a second pipeline if alive.
The run_frozen entry has an exclusive lock and complete-stage guards; it trains
one frozen native visual CFM, audits optimizer/EMA, evaluates80 original trials,
accepts evidence, then stops awaiting publication. Checkpoints every500 preserve
raw model, optimizer, EMA, scheduler and CPU/CUDA/NumPy RNG. Resume, never restart.

All development gates passed: one genuine discarded optimizer update,4 native
data tests, readonly actions,17-call native/RPC check and2 complete physical
smokes. Both smoke task successesfalse; interface/state/video/goal acceptance
passed. No extra smoke or optimizer probe is allowed. Fixed EMA validation and
raw-weight/RNG restoration passed. Native Gaussian-training/uniform-deployment
behavior is intentionally preserved; do not silently tune it.

Frozen unique intended run:batch32,24000 calls,768000 immutable balanced anchors,
native EMA minimum fixed256-target-validation velocity MSE, earliest tie. The
selected checkpoint's effective training quantity must be reported separately.
40/task60–99 EXPOSED_REFERENCE after training; original official complete goal.
Full protocol, actual native variant and differences are in PROTOCOL.md,
protocol.json and OFFICIAL_AUDIT.md. Original Robomimic lowdim/absolute-action
and full affordance-prompt reproduction are not claimed.

Large artifacts stay at
`/home/users/lemon/strong-baselines-20261112/flow_matching_policy/`.
Source checkout is `/home/users/lemon/baseline-benchmark/vendor/hri-eu-flow-matching`.
Run the orchestration with the old simulator Python and isolated new policy
worker, not a changed benchmark environment. After final acceptance publish
code/results, verify remote main/files, record server final_delivery.json, stop.
