# KITE result: BLOCKED, not a completed baseline

Official code136159544aea68f5e8b6cc64f17d221e825e3eee and both released asset
bundles were obtained and SHA256 verified. Native Torch2.9.1+cu128, CUDA12.8.93,
MuJoCo3.8.0 and the vendored PyTorch3D0.7.9 extension run in a separate prefix.
The real CUDA FPS operator and the original frozen contact encoder ran; the
encoder produced a finite256-dimensional latent from native composite-hand FK.
This is a prerequisite test, not KITE training or task evaluation.

All600 original UR5e raw demonstrations match the old action/instruction IDs
and every packed action frame;540/60 whole-demo training/validation is intact.
Full states, arm/gripper states and camera metadata exist. No raw depth field
exists. We did not establish calibrated depth/contact-label reconstruction;
missing processed RGB-only fields must not be mistaken for missing raw states.

The one technical repair supplied already installed pinned CUDA library header
paths to the extension compiler; that repair passed. A subsequent geometric
checker failed at import because my launcher omitted the existing vendored
MimicGen module path. **This is an agent integration error**, not missing public
MimicGen code, not a scientific limitation of KITE, and not a simulator failure.
No environment was constructed and no physics step occurred in that checker.
I have not repaired/relaunched it, respecting this round's one-repair ceiling.

The actual released decoder bank contains Wuji26,Allegro22,Barrett14,
Robotiq+Stick13 and MANO51 configurations, with no bare Robotiq85/Panda head.
Discarding the stick channels would not certify the target robot's kinematic
decoder. A valid future port must establish actual gripper FK/coordinate/action
correspondence, extract native source intent labels, train the required target
kinematic decoder with synthetic supervision, and pass a real update and full
smoke before freezing formal training. Existing source/Panda raw-data presence
and a successful frozen encoder do not complete those steps.

Actual new learning updates:0; task smoke/formal episodes:0; success rate:N/A;
task failure videos:N/A. No Panda task supervision, canonical diffusion-policy
substitute, manual grasp trigger or invented label was used. No conclusion
about KITE's task performance is supported. See errors.json,checks.json and
sha_index.json for reproducible evidence and all retained acquisition logs.
