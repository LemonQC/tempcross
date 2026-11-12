# Method v0: TOCAT specification and offline evidence

This directory is the executable minimal evidence for **one** candidate: task-effect / embodiment-conditioned continuation-contract refinement. It is **not a trained manipulation policy**. The decision is `NO_GO_LEARNED_METHOD_IDENTIFIABILITY_AND_NOVELTY_NOT_ESTABLISHED`; see the four [Method documents](../docs/METHOD_V0_DECISION.md).

## What actually ran

- Inventory of 1,800 source and 100 Panda demonstrations, with actual schema/reward inspection.
- Extraction of all 1,280 old Baseline, 160 existing adaptation and 177 augmentation trajectories: 1,617 episodes.
- Exact recorded-state/prefix/RNG checks for five existing handoff groups.
- Continuation-support recursion versus independent forward path search: 925,754 root/budget comparisons, 0 disagreements on the same factual graph.
- Unit tests for label distinctions, budgets, unsupported-state joins, unknown feasibility and evidence accounting.

No simulator was instantiated, restored or stepped. No policy inference, optimizer update, image generation, Baseline rerun or physical evaluation occurred. The prototype here checks the proposed variable and operator; it does not implement a learned TOCAT model. The abstract two-world test is never used as physical supervision.

## Reproduce offline

From the repository root on the original server:

```bash
PYTHON=/home/users/lemon/coft-p2/barx/.venv/bin/python
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m method_v0.extract
"$PYTHON" -m method_v0.organize
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m method_v0.analyze
"$PYTHON" -m unittest discover -s method_v0/tests -v
"$PYTHON" -m method_v0.finalize
```

`extract` requires the absolute raw data paths recorded in the repository and uses NumPy/h5py. `analyze` deserializes **trusted local** runtime pickles from the historical mechanism experiment; PyTorch is used only to read/compare saved RNG tensors on CPU. Do not deserialize untrusted replacement pickles. Private policy configuration values are never copied to the evidence. Tests run with standard Python on checked-in results and do not need the simulator, GPU, raw dataset or network.

The entry points install an import guard rejecting `mujoco`, `mujoco_py`, `robosuite`, `robocasa`, and `mimicgen`. This is a code-path safeguard, not a substitute for the preserved input/launch-ledger hashes. `finalize` verifies the original frozen input hashes and indexes server-only outputs. Rerunning extraction may rewrite this branch's derived records only; it must not change historical files.

## Files

| File | Role |
|---|---|
| `protocol.json` | Frozen corpus, information boundary, labels, resource ceiling and gate |
| `extract.py` | Factual contacts, official task-effect words, outcomes, raw input hashes |
| `organize.py` | Small partition files with content hashes; no episode filtering |
| `analyze.py` | Exact handoff checks, descriptive cross-robot pairs, support/planning comparison |
| `contracts.py` | Reference evidence operator, independent planner, logical missing-data witness |
| `tests/` | Semantic and evidence-accounting tests |
| `finalize.py` | Preservation and artifact indexing, no training or simulation |
| `results/summary.json` | Machine-readable actual outcome and explicit negative decision |

`results/factual_continuations.index.json` and `results/input_index.index.json` list every partition and SHA-256. `results/same_information_decisions.json` reports all 1,622 roots without treating absence of a witness as impossibility. Larger numeric arrays/full record copies and logs remain under `/home/users/lemon/method-v0-tocat-20261112/`; `results/server_artifacts.json` gives absolute paths and hashes. All physical success counts refer to **existing** executions.
