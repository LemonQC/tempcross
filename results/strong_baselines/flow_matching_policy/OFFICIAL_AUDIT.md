# Official implementation boundary

Repository: [HRI-EU/flow_matching](https://github.com/HRI-EU/flow_matching/tree/516e8e18875b27741bdbdb8a252e904032c90723),
master `516e8e18875b27741bdbdb8a252e904032c90723`, BSD-3-Clause.
TorchCFM1.0.7 and diffusers0.35.2 provide the native matcher/EMA/scheduler.
All imported core files are SHA256-pinned in official_source.json/protocol.json.
The upstream checkout remains unmodified. Existing local BC/DP code is not used
as a substitute network or learning objective.

## What the official code actually supplies

The author [visual example](https://github.com/HRI-EU/flow_matching/blob/516e8e18875b27741bdbdb8a252e904032c90723/examples/flow_pusht.py)
uses its own external/models ResNet18 wrapper, BatchNorm-to-GroupNorm replacement
and ConditionalUnet1D. The UNet defaults remain256/512/1024 channels, time
embedding256, kernel5,8 norm groups and native FiLM residual conditioning.
Predict16 actions with one observation; execute the first8 before querying again.

Its training samples Gaussian x0, calls the actual
`ConditionalFlowMatcher(sigma=0)` and minimizes mean squared velocity error over
the entire predicted action tensor. No DDPM denoising objective, diffusion noise
schedule or DDIM sampler is substituted. The public visual and Robomimic
deployment loops instead sample uniform[0,1] noise and take one Euler step at0.
**The Gaussian-training/uniform-deployment inconsistency is preserved.**
Changing this initialization distribution or selecting a different solver using
reference successes would create another experiment; neither is done here.

The author [Robomimic example](https://github.com/HRI-EU/flow_matching/blob/516e8e18875b27741bdbdb8a252e904032c90723/examples/flow_mimic.py)
is a low-dimensional object/robot-state experiment with absolute actions,
rotation6D conversion and hard-coded20-action/50-condition dimensions. It also
prints tensor shapes and exits before backward. Those dimensions and absolute
controller mode do not match the fixed single-arm OSC7/OSC12 benchmark.
The public scripts additionally have dataset/import-path and EMA bookkeeping
limitations. Our driver calls the real author classes directly, keeps their
mathematics, and explicitly stores/restores raw parameters around EMA validation.

## The one port being evaluated

Use the **official visual flow component**, without claiming reproduction of
the original Robomimic lowdim/absolute-action recipe or the full affordance-prompt
framework. This matches the existence of the separate visuomotor flow baseline
in the [paper](https://arxiv.org/html/2409.01083v2), rather than inventing a new
vision or affordance module.

| Part | Author visual reference | This fixed RoboCasa-X port |
|---|---|---|
| Encoder | native GN ResNet18, weights=None | same classes, trained from scratch |
| Flow network | native ConditionalUnet1D | same layers/default widths; dimensions adapted |
| RGB |96 square, float[0,1] | original180x320 resized to96, same camera |
| Proprioception | planar agent position2 | certified arm/gripper joint positions9 |
| Language | none, single task | existing frozen CLIP-L/14 vectors768, full16 instructions |
| Global condition |514 |1289=512+9+768, native concatenation/FiLM |
| Actions |2 | canonical OSC7, unchanged positive-close OSC12 conversion |
| Sequence |observe1,predict16,execute8 | same;20Hz execution gives0.4s chunks |
| Boundary repetition |up to7 tail slots | up to15 to preserve original anchor frames |
| Dataset |PushT-specific loader | same original700 complete IDs and split; train630 |
| Training |native Gaussian CFM/velocity MSE | same objective and matcher, no loss masks |
| Generation |uniform, single Euler | same public-code behavior |
| EMA |native EMAModel(power=.75) | same constructor; library defaults/version pinned |

There is no learned replacement language encoder, additional policy head,
hand-coded interaction stage, object pose input or controller tuning. Conditional
input/output dimension changes are real reproduction differences:94,874,439
trainable parameters here, not a claim of identical paper parameter count.
The pretrained benchmark checkpoints have incompatible observation/action
shapes; there is no misleading partial checkpoint loading.

Dependency recipe is also a declared difference: use the isolated pinned
Torch2.9.1+cu128/torchvision0.24.1 runtime and actual required native imports,
not a claim that the entire historic requirements.txt was reproduced. Its
Lightning<2/quality-tool requirements are not used by this driver; the resolved
TorchCFM/TorchDyn dependency graph includes Lightning2.6.6 without training a
Lightning model. requirements_runtime.txt and installed_runtime.txt distinguish
the used entrypoint requirements from the complete181-package shared/overlay
graph. Real native gradients, deployment and EMA were tested in this runtime.

All raw RGB/qpos/actions are original demonstration data in an existing read-only
cache, not SmolVLA predictions. Old baselines and their artifacts stay frozen.
The two existing generated-success references are command-interface checks only;
they are excluded from training. A complete faithful port denotes completion
under the declared protocol, not proof that this is the best possible FMP setup.
