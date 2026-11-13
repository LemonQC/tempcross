# KITE native-port preflight

Historical preflight notes; final closure is **BLOCKED** in RESULTS.md.
Assets, CUDA12.8 compiler and the native extension are now verified. The source
geometry checker subsequently failed at import, before any state extraction;
it has not been repaired or relaunched after the one-repair budget was used.

This is source/data/environment inspection, not a completed baseline. No KITE
training update, policy checkpoint selection, or physical evaluation is claimed.

- Official repository: https://github.com/helloqxwang/KITE, checked HEAD
  `136159544aea68f5e8b6cc64f17d221e825e3eee` (2026-11-13 UTC).
- Paper: https://arxiv.org/html/2606.22113v2; project:
  https://kite-manip.github.io/. The root checkout contains no top-level LICENSE;
  vendored iDP3 and PyTorch3D licenses must be recorded separately. No upstream
  source is copied into this repository.
- Preserve the shared contact-position/normal intent encoder, learned kinematic
  decoder, synthetic kinematic supervision and iDP3 interaction predictor.
  A canonical-action diffusion policy alone is not KITE.
- The released Objects Placing code uses object-only world-frame point clouds,
  previous intent, measured current configuration, and 20 Hz control. Its native
  six-object/bin scene and 120-frame trial are not our kitchen benchmark. Only
  the original RoboCasa-X tasks, frozen conditions and full official kitchen
  success predicate may be evaluated here.
- Zero-shot means **no Panda task demonstration is used for policy/decoder
  fitting or model selection**. Synthetic data from the target kinematic model
  is allowed by the paper. Adding target-task demonstration fine-tuning would
  require a separately named experiment and would not be zero-shot.
- Raw UR5e XP900 release is available: `ajaysri/barx-raw-hdf5` revision
  `c8b3aba2dfbbfbafa67fa82466320945d560208f`. Only the original six UR5e files for
  our two tasks were acquired (600 demos, about7.87 GiB), with published SHA256.
  `baselines/strong/raw_geometry_inventory.py` checks every action frame and
  instruction against the old 600-ID pool, preserving540/60 whole-demo splits.
  This is additional fields from the **same demonstrations**, not new examples.
- Saved raw fields include full simulator states, joint/gripper/EE states and
  portable episode/camera metadata. No depth dataset is present. Actual depth,
  contact geometry/normal reconstruction and source/target robot-model mappings
  must be tested; field presence alone does not certify those labels.
- Separate native runtime:
  `/home/users/lemon/strong-baselines-20261112/kite/env`; actual imports confirm
  Torch2.9.1+cu128, Hydra1.3.2, diffusers0.38.0, pytorch-kinematics0.7.6,
  MuJoCo3.8.0 and NumPy2.2.6. These are the upstream versions for the core stack.
  The existing RoboCasa-X evaluation environment is unchanged. Native runtime
  availability does not establish a valid deployment bridge.
- Native decoder/runtime archives are acquired through the author's checksum
  manifest. A separate CUDA12.8 compiler prefix is being prepared for the
  vendored PyTorch3D extension; the host default compiler toolkit is CUDA11.3.
  Acquisition logs retain dependency URL timeout/index-routing attempts and the
  failed Conda dependency resolutions. These are not training runs or successes.

Remaining gates: verified assets and native extension → source geometry / target
decoder correspondence → real learning update on valid data → full frozen-task
physical smoke → frozen final protocol → formal learning and evaluation.
No train budget or checkpoint rule is frozen yet; native feasibility must first
be established. No task success rate is available.
