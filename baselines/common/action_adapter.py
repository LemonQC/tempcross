"""Verified boundary between native policy representation and normalized OSC."""
import numpy as np


def canonical_action(native,representation='canonical_osc'):
 a=np.asarray(native,dtype=np.float64).copy()
 if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Invalid native action: expected seven finite values')
 if representation=='barx_unnormalized':
  # Exact official normalize_gripper_action then invert_gripper_action.
  a[6]=-np.sign(2*a[6]-1)
 elif representation!='canonical_osc':raise ValueError(representation)
 clipped=np.clip(a,-1.,1.)
 return np.r_[clipped,np.zeros(4),-1.],dict(preclip_canonical=a.tolist(),clipped_dimensions=int(np.count_nonzero(a!=clipped)),native_representation=representation)
