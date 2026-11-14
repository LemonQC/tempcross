# Independent local analysis and GitHub handoff

This tool uses local Python, the existing NumPy/OpenCV environment, and normal
Git authentication. It does not call Codex, ChatGPT, a connected-app API, a model
server, MuJoCo, training code or demonstration collection.

From `/home/users/lemon/CodexProject/tempcross-cross-behavior`, run:

```bash
python scripts/cross_behavior_handoff.py run
```

Equivalent shell wrapper: `bash scripts/run_cross_behavior_handoff.sh`.

## One-time authentication and environment check

```bash
python scripts/cross_behavior_handoff.py doctor
```

The parent command requires only the standard library. Numerical analysis runs
in the existing Python configured in `configs/diagnostics/handoff_config.json`.
No large dependency installation is performed. Doctor checks NumPy/OpenCV,
the indexed data, current branch, repository identity, remote head and a genuine
non-writing `git push --dry-run`. It never prints credential values.

The initial inspected server had no credential helper, SSH key/agent, `gh`, or
successful native Git push authentication. Read access to a public repository
is not push authentication. A connected Codex GitHub app is not a future local
credential source. Before native automatic publishing, complete one ordinary
authentication setup on the server. With GitHub CLI installed:

```bash
gh auth login --hostname github.com --git-protocol https
gh auth setup-git
python scripts/cross_behavior_handoff.py doctor
```

If `gh` is absent, an administrator can install that small authentication tool,
or you can use an existing standard Git credential manager or SSH setup instead.
For an interactive HTTPS alternative, configure a standard credential helper
and perform a normal authenticated push in a trusted terminal. GitHub requires
a token rather than an account password for HTTPS Git. Enter it only at the
credential prompt; never put it in source, JSON configuration, a remote URL,
command-line arguments, logs, or a chat message. This program neither creates
nor exports credentials. Doctor returning 2 means some prerequisite is not
ready; inspect its named status rather than assuming upload succeeded.

## Commands

```bash
python scripts/cross_behavior_handoff.py analyze
python scripts/cross_behavior_handoff.py run --dry-run
python scripts/cross_behavior_handoff.py run --resume
python scripts/cross_behavior_handoff.py publish --resume
python scripts/cross_behavior_handoff.py status
```

`analyze` produces a local immutable package. `run` analyzes and publishes it.
`publish` retries the existing package without recomputing analysis.
`status` reports analysis and publication separately. `--dry-run` on run/analyze
performs read-only prerequisite checks without analysis, commits or push; on
publish it verifies an existing package and expected remote without a commit or
push. A small lock/state directory under Git's private worktree metadata is used.

All commands accept `--resume`, `--dry-run`, `--data-root`, `--branch`, and
`--config`. `--artifact-index` selects an explicit evidence index. `--run-id`
selects an existing package for publication/status. Neither `main` nor `master`
can be selected, and the current worktree must be on the requested research
branch. The program never checks out or modifies main.

## New data without source edits

```bash
python scripts/cross_behavior_handoff.py run --data-root /data/my-next-experiment
```

Place a compatible `artifact_index.json` in that directory or pass
`--artifact-index /absolute/path/to/index.json`. The default index is
`results/cross_behavior/artifact_index.json`. The schema is the actual existing
top-level object containing `server_root` and `episodes`; each episode has
`task`, `robot`, `condition`, and `files` entries with absolute paths/size/SHA256.
Required files are `episode.json`, `condition.json` and `telemetry.jsonl`.
Relocating a directory maps paths relative to the index's recorded `server_root`
onto `--data-root`. Paths escaping the declared data root are rejected.

Supported trajectory rows contain contiguous `step`, canonical `action`,
`decision` and evaluator `metrics`. Push/Stack/Drawer metrics are versioned to
the current JSONL format. New episodes in that format are discovered from the
index, not hardcoded to 19 or to fixed directory names. A new file format or
task schema stops explicitly; this tool does not guess unknown HDF5 fields.

## Durable stages and duplicate detection

Each distinct input/content/config/source fingerprint gets one unique directory:
`results/cross_behavior/diagnostics/runs/<run_id>/`. The Git HEAD alone is not an
analysis input: publishing unchanged data must not create another report.
Previously complete input hashes are reused only while device/inode, size,
mtime and ctime match. Original input SHA expectations are checked.

The private state catalog is under `git rev-parse --absolute-git-dir`, in
`cross_behavior_handoff/`. Do not delete it if you want resume/history. Finished
episode analyses are committed atomically and skipped on resume. Interrupted
initial snapshot creation and final package sealing are recoverable. SIGTERM
pauses at an analysis episode boundary. A hard crash before an atomic rename
does not turn partial output into COMPLETE. Incomplete runs require `--resume`.

For a controlled recovery check with a new input set:

```bash
python scripts/cross_behavior_handoff.py analyze --stop-after-episodes 1
python scripts/cross_behavior_handoff.py analyze --resume
```

`--stop-after-episodes` stops analysis only; it never launches a physical trial.
Exit 75 indicates a saved interruption. Exit 1 indicates a failure; the completed
stages and error remain available. Repeating identical complete analysis reuses
the existing package, rather than overwriting or creating another report.

## Package contents and interpretation

Each package contains the diagnostic Markdown report, episode and phase CSVs,
perception audit, summary, input manifest, code context, source patch, relevant
source snapshots, per-episode analysis records, bounded overlays and a checksum
manifest. Raw MP4/NPZ/PKL/HDF5/weights remain on the server. Their absolute paths,
sizes and hashes are listed, not their bytes. Full per-step artifact indexes
are themselves checked; their unused child files are referenced rather than
misrepresented as freshly consumed/rehashed inputs.

Push errors use saved estimated and privileged true positions offline. The
first JSONL state is post-action step 1, and the estimate precedes its action;
the report labels this timing limitation. Direction is undefined for negligible
motion. Stack phase boundaries use the executed `decision.phase`; native
predicate events are separate from the frozen full-contract success. Drawer
reports latched-estimate availability, contact timing and actual normalized
joint opening. Exact dynamic handle ground truth is not in the saved JSONL and
is not invented. Initial-frame overlays fit camera projection from recorded
RGB-D rays; fit error and unresolved/off-frame points are recorded. These are
visual diagnostics, not counterfactual success labels or causal conclusions.

Limits, source selection and the existing analysis Python are configurable in
the JSON file. The defaults allow 12 overlays, width 640 pixels, 2 MiB per file
and 16 MiB total. Forbidden raw formats, environment files, private-key/token
patterns, unexpected files, package tampering and oversize files stop publication.
The generated source patch covers only the listed relevant files, including
their untracked contents. Other user files are never staged or copied.

## Publication, conflicts and proof

Publication uses an isolated temporary Git index and `git commit-tree`. It
includes only this immutable package and the explicit `publish_programs`
snapshots. It does not run `git add .`, rewrite the user's index, change live
files, check out a branch, force push, auto-merge or auto-rebase. Consequently
the live HEAD may remain behind the tool's own published commit; the private
publication ledger supplies the correct parent on later runs. This preserves
unrelated staged and unstaged work. Do not reset/stash user changes just to
make this program work.

A candidate commit is saved before push. If a network failure occurs after the
server accepts it, retry checks that same commit rather than creating duplicates.
The program fetches the actual remote commit and reads back every published
file byte-for-byte before publication becomes COMPLETE. Commit SHA and report
URL are saved in the private run state and printed. They are not written into
their own immutable Git commit, which would be a self-reference.

If the remote branch differs from the saved expected parent/candidate, automatic
push stops and the local package remains intact. Inspect:

```bash
git fetch origin cross-behavior-task-expansion
git log --oneline --graph --all -12
python scripts/cross_behavior_handoff.py status
```

Review and reconcile the branch intentionally in a separate worktree if needed;
do not force push. The tool does not silently change a frozen package's parent.
After authentication/network recovery, use `publish --resume`; no analysis or
simulation needs to repeat. A one-time connected-app bootstrap, if used for the
first delivery, is recorded as such and does not count as native server
authentication being ready. Future automatic publishing still requires doctor
to confirm the local credential setup.
