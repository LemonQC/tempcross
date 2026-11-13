# Reimplementation assessment

Primary paper: [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ge_CrossZero_Zero-Shot_of_Unseen_Embodiment_Manipulation_via_Object-Centric_Hierarchical_Diffusion_CVPRW_2026_paper.html).
The cached PDF/text and current publisher snapshot are indexed by SHA256.

| Core requirement | Available specification | Current platform gate |
|---|---|---|
| Robot-masked point cloud | XYZRGB1024; PointNet64/128/256/512 | Original RGB/states exist; calibrated depth/mask conversion not validated |
| Object functional pose | Auxiliary L2 and joint objective,Sec3.2 | Native annotation schema/frame/loss coefficient not certified |
| EE diffusion | FiLM1D U-Net,horizon8,10DDIM; several optimizer settings | Single-arm mapping of published16D schema not certified |
| IK diffusion | EE action and current q→q sequence,Sec3.3 | Actual joint fields exist; no valid joint-target deployment bridge |
| Unseen-robot adaptation |50 EE-policy rollouts then teacher-IK distillation,Sec3.3.3–4 | No new CrossZero EE policy or teacher rollouts; old models cannot be relabelled |
| Checkpoint merging | Mentioned inSec3.3.5 | Exact operation unspecified in inspected material |

These are uncertified implementation differences, not proof that the algorithm
cannot be implemented. A paper-only candidate would need explicit fixed choices
and verification. The unresolved joint-controller bridge independently prevents
a faithful full-method result in this attempt. No core module was dropped.
