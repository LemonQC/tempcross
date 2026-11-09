"""Frozen, identical Cartesian waypoint schedule for A2 and A3."""
import numpy as np
from p3_validation.geometry import interpolate


def shifted(pose,z):
    result=pose.copy();result[2,3]+=z;return result


class WaypointPlan:
    def __init__(self,branch_tcp,grasp,place,config):
        c=config['controls'];self.grasp=grasp.copy();self.place=place.copy()
        ends=[shifted(branch_tcp,c['approach_up_m']),shifted(grasp,c['pregrasp_up_m']),grasp,grasp,
              shifted(grasp,c['lift_m']),shifted(place,c['preplace_up_m']),place,place,shifted(place,c['retreat_up_m'])]
        self.segments=[];start=branch_tcp.copy();offset=0
        for (name,n),end in zip(c['phases'],ends):
            self.segments.append((name,offset,offset+n,start.copy(),end.copy()))
            offset+=n;start=end
        self.duration=offset;self.final=start.copy()

    def goal(self,step):
        for name,first,last,start,end in self.segments:
            if first<=step<last:return name,interpolate(start,end,(step-first+1)/(last-first)),end.copy()
        if step>=self.duration:return 'post_retreat',self.final.copy(),self.final.copy()
        raise ValueError(step)
