# Cross-behavior method integration status

The original 2,080 PnP formal outcomes remain frozen. This expansion has **zero policy training updates, zero learned-policy smokes, zero formal policy evaluations and zero collected demonstration pilots**. The mandatory task × method × configuration rows are recorded in `results/cross_behavior/method_integration_matrix.csv`; every row is `NOT_RUN`, with a physical-gate reason. Listing a method does not claim that its new task adapter exists.

| Existing family | Configurations reserved for later verification | Push | Stack | Drawer |
|---|---|---|---|---|
| BC-GMM, BC-RNN, BC-Transformer | source-only, target-only, source+target where historically supported | NOT_RUN | NOT_RUN | NOT_RUN |
| Diffusion Policy | source-only, target-only, source+target where historically supported | NOT_RUN | NOT_RUN | NOT_RUN |
| BARX No Reps / Joint Reps | existing source policies | NOT_RUN | NOT_RUN | NOT_RUN |
| BARX adaptation | both restricted LoRA configurations and separate standard target adaptation | NOT_RUN | NOT_RUN | NOT_RUN |
| ACT / VQ-BeT | target-only, source+target | NOT_RUN | NOT_RUN | NOT_RUN |
| Octo-Small | existing source+target | NOT_RUN | NOT_RUN | NOT_RUN |
| SmolVLA | existing source+target, RGB+instruction+qpos9 | NOT_RUN | NOT_RUN | NOT_RUN |
| HRI-EU visual FMP | existing RGB+CLIP768+qpos9 | NOT_RUN | NOT_RUN | NOT_RUN |

Historical settings, normalization, budgets, whole-demo splitting, native model cores and validation selection are not reset or reimplemented here. Their task adapters must receive genuine future probes before any row can change. No RGB+state method is to be merged into an RGB-only information tier.

MimicGen v2 stays its historical **PARTIAL** augmentation component; no new task generation or downstream training ran. KITE is deferred, not mandatory in this stage. CrossZero and Cross-Behavior Object Flow are archived with their old reports and no new budget. No method development was started.

The expert is a non-learning feasibility controller with RGB-D/robot FK, not an additional learned baseline. Its failures are not substituted for method success rates. The present decision is to withhold all training until the task/expert contract and observability issues are resolved under a new reviewed protocol.
