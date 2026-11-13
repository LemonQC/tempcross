# SmolVLA state-conditioned joint baseline results

COMPLETE_FAITHFUL_PORT for one source_target_state setting. Native RGB+joint/gripper state; not RGB-only. Target-only was not run.
Conditions60–99 are EXPOSED_REFERENCE; the original official placement AND retreat criterion is unchanged.

| Task | Official successes / episodes |
|---|---:|
| pnp_counter_to_sink | 1/40 |
| pnp_sink_to_counter | 2/40 |

Actual formal updates:16000; actual discarded probe updates:9; anchors:512000.
Selected update:2000; fixed target-validation real7 flow MSE:0.23551706. No test-based model selection.
The selected checkpoint has seen64000anchors; the full training run consumed512000. These are different budgets when selection precedes the last update.
Checkpoint:`/home/users/lemon/strong-baselines-20261112/smolvla/training/source_target_state/best.pt`; SHA256:`364b9c5dfc17996df412f8e64dfc7b7c1df710a1b35f9acadbd62776d3fd5bc8`.

The official network,50-step native queue,10-step flow generation,32-dimensional training loss, native optimizer and frozen VLM are retained. Declared data, processor, validation and finite-budget differences are in PROTOCOL.md.
Both development smokes failed task success but completed600/650 steps and passed interface/state/video acceptance; they are excluded from the80 formal episodes.
All actual policy state vectors match saved pre-action named joint qpos. Raw native32 actions, crop7, native unnormalization,50-step queues and executed OSC12 were checked. No generated success reference enters training.
Failure stages are descriptive sampled-contact/placement proxies, not causal explanations. No old baseline was retrained or evaluated. No new Method or next baseline started.

Code/config/provenance: official_source.json, protocol.json, freeze.json, errors.json. Evidence: training_state.json, training_metrics.json, optimizer_audit.json, sample_exposure.json, episodes.csv, per-task acceptance files, artifact_index.json. Large artifacts stay on the server.

## Failure classification and frozen reference

These are descriptive telemetry categories; contact/lift proxies do not establish failure causes.

| Task | Closed without bilateral contact | Contact without stable-lift proxy | Stable lift without official placement | Official success |
|---|---:|---:|---:|---:|
| pnp_counter_to_sink | 36 | 3 | 0 | 1 |
| pnp_sink_to_counter | 33 | 2 | 3 | 2 |

| Frozen/new joint baseline | Information | Counter to Sink | Sink to Counter |
|---|---|---:|---:|
| Octo-Small (frozen, no rerun) | RGB256 + language | 0/40 | 0/40 |
| SmolVLA | RGB512 + language + real joint/gripper qpos9 | 1/40 | 2/40 |

Both use EXPOSED_REFERENCE60–99. Input information, pretraining and budgets differ; this table does not establish an equal-input improvement. SmolVLA is a completed bounded faithful port, not the full official batch64/20000-update example recipe.
This round ends here. No extra training, physics, next baseline or new Method is authorized by these results.
