# Cross-behavior execution checkpoint

All new files are isolated from the old PnP benchmark. Current durable root:
`/home/users/lemon/cross-behavior-20261114`. Current branch:
`cross-behavior-task-expansion`.

```bash
cd /home/users/lemon/CodexProject/tempcross-cross-behavior
python scripts/run_cross_behavior_expansion.py --stage auto --resume --status
python scripts/run_cross_behavior_expansion.py --stage auto --resume --background
```

The first command is read-only and uses only the standard library. During an
active run the second command detaches a single locked runner and resumes its
current condition/step. Completed episodes are skipped. Inventory/protection
results are cached. It never starts training until all three robot-pair gates
pass. The delivered run is terminal `BLOCKED_PHYSICAL_GATE`; repeating the
command returns that state and spends no additional physical or training budget.

Per episode: `model.xml`, `initial.pkl`, `initial_rgbd.npz`, `condition.json`,
`cursor.json`, `steps/NNNNNN.pkl`, `frames/NNNNNN.png`, `telemetry.jsonl`,
`rollout.mp4`, `episode.json`, `artifact_index.json`. Pickles are local trusted
artifacts, not an external upload format. `source_blobs` / `source_versions`
retain executed code revisions; execution/resume records identify their hashes.

An intent is fsynced before physics and a complete integration/controller/expert
checkpoint is fsynced afterward. A committed checkpoint can repair a stale
cursor. An intent without a committed checkpoint is ambiguous and stops recovery;
it is never interpreted as permission to repeat physics. SIGTERM pauses at a
control-step boundary. Resume recreates the native environment, restores its
saved state and continues the next policy step, with native pre-restore
initialization explicitly separate from policy-step counts.

The task source, sensor boundary, expert, ground-truth evaluator and recovery
worker are independent modules. The RGB-D expert is not an RGB-only learned
baseline. Failed task gates are not algorithm success rates.

Offline validation and evidence compilation (no physics):

```bash
/home/users/lemon/coft-p2/barx/.venv/bin/python -m pytest -q tests
/home/users/lemon/coft-p2/barx/.venv/bin/python -m cross_behavior.report
```

`docs/CROSS_BEHAVIOR_TASK_FEASIBILITY.md` contains the actual 19-episode result
and the specific reason no data scaling/adapters/training were started. Stage
B/C/D implementations have not been claimed or activated. A future reviewed
protocol must keep these artifacts, the exhausted per-task repair counts and
all prior PnP results intact.
