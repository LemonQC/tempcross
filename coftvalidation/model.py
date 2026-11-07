from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from coftprobe.geometry import transform
ROOT=Path(__file__).resolve().parents[1]

class Arm:
    def __init__(self,name):
        self.spec=json.loads((ROOT/f'configs/validation/{name}.json').read_text())
        self.joints=self.spec['joints']; self.home=np.array(self.spec['home'])
        self.lo=np.array([j['lower'] for j in self.joints]); self.hi=np.array([j['upper'] for j in self.joints])
        self.origins=[transform(j['xyz'],Rotation.from_euler('xyz',j['rpy']).as_matrix()) for j in self.joints]
        self.axes=[np.array(j['axis']) for j in self.joints]
        self.fixed=transform(self.spec['flange_fixed_xyz'])
        self.calls=0; self.cache={}

    def frames(self,q):
        t=np.eye(4); frames=[t.copy()]
        for v,o,a in zip(q,self.origins,self.axes):
            t=t@o@transform(rotation=Rotation.from_rotvec(a*v).as_matrix()); frames.append(t.copy())
        frames.append(t@self.fixed)
        return frames

    def fk(self,q): return self.frames(q)[-1]

    def ik(self,target,previous=None):
        key=(target.tobytes(),None if previous is None else np.asarray(previous).tobytes())
        if key in self.cache: return self.cache[key]
        best=None
        for seed in ([previous,self.home] if previous is not None else [self.home,(self.lo+self.hi)/2]):
            def residual(q):
                f=self.fk(q)
                return np.r_[f[:3,3]-target[:3,3],.22*Rotation.from_matrix(target[:3,:3]@f[:3,:3].T).as_rotvec()]
            sol=least_squares(residual,np.clip(seed,self.lo+1e-8,self.hi-1e-8),bounds=(self.lo+1e-8,self.hi-1e-8),
                              max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            self.calls+=sol.nfev
            f=self.fk(sol.x); p=float(np.linalg.norm(f[:3,3]-target[:3,3])); a=float(np.linalg.norm(Rotation.from_matrix(target[:3,:3]@f[:3,:3].T).as_rotvec()))
            r=dict(q=sol.x.tolist(),position_error_m=p,angle_error_rad=a,found=bool(p<=.003 and a<=.04))
            if best is None or p+.22*a<best['position_error_m']+.22*best['angle_error_rad']: best=r
            if r['found']: break
        self.cache[key]=best
        return best
