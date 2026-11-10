"""Reach-terminated standard OSC controls; no learned or novel controller."""
import numpy as np
from p3_validation.geometry import inverse, error
from p3_validation.intervene import osc_action


def up(pose, distance):
    result=pose.copy(); result[2,3]+=distance
    return result


def reached(current,goal,c):
    p,r=error(current,goal)
    return p<=c['position_tolerance_m'] and r<=np.deg2rad(c['rotation_tolerance_degrees'])


class Prefix:
    phases=['clearance','pregrasp','descend','close','lift']
    def __init__(self,tcp,obj,relative,c):
        self.c=c;self.initial_object_z=float(obj[2,3]);self.grasp=obj@relative
        self.goals=[up(tcp,c['clearance_up_m']),up(self.grasp,c['pregrasp_up_m']),self.grasp,self.grasp,up(self.grasp,c['lift_m'])]
        self.stage=0;self.age=0;self.pose_run=0;self.contact_run=0;self.lift_run=0
        self.done=False;self.failed=None;self.reached_phases=[]

    def command(self,tcp,obj,contact):
        if self.done or self.failed: return None
        phase=self.phases[self.stage];goal=self.goals[self.stage]
        self.pose_run=self.pose_run+1 if reached(tcp,goal,self.c) else 0
        self.contact_run=self.contact_run+1 if contact else 0
        stable=contact and obj[2,3]-self.initial_object_z>=self.c['stable_rise_m']
        self.lift_run=self.lift_run+1 if stable else 0
        ready=self.pose_run>=self.c['pose_dwell_states']
        if phase=='close': ready=ready and self.age>=self.c['close_min_steps'] and self.contact_run>=self.c['contact_dwell_states']
        if phase=='lift': ready=ready and self.lift_run>=self.c['stable_dwell_states']
        if ready:
            self.reached_phases.append(phase)
            if phase=='lift': self.done=True;return None
            self.stage+=1;self.age=0;self.pose_run=0
            phase=self.phases[self.stage];goal=self.goals[self.stage]
        if self.age>=self.c['phase_caps'][phase]:
            self.failed=phase+'_TIMEOUT';return None
        self.age+=1
        return phase,goal,1. if phase in ['close','lift'] else -1.


class Postgrasp:
    """M0 fixes the handoff attachment; M2 re-observes that attachment each step.

    Both use the same source-derived object goals, OSC gains, arrival tolerance,
    release dwell, retreat rule and original physical model.
    """
    phases=['transport','preplace','place','release','retreat','wait']
    def __init__(self,tcp,obj,template,c,method):
        self.c=c;self.method=method;self.relative=inverse(obj)@tcp
        sink=np.asarray(template['T_target_sink'])
        _,middle,release=[sink@np.asarray(p) for p in template['postgrasp_object_in_sink']]
        height=max(float(obj[2,3]),float(release[2,3]+c['preplace_up_m']))
        middle=middle.copy();middle[2,3]=max(float(middle[2,3]),height)
        preplace=release.copy();preplace[2,3]=height
        self.object_goals=[middle,preplace,release]
        self.stage=0;self.age=0;self.pose_run=0;self.failed=None;self.release_tcp=None
        self.reached_phases=[]

    def command(self,tcp,obj):
        if self.failed:return None
        phase=self.phases[self.stage]
        if self.stage<=2:
            desired=self.object_goals[self.stage]
            attachment=self.relative if self.method=='M0' else inverse(obj)@tcp
            goal=desired@attachment
            arriving=reached(tcp,desired@self.relative,self.c) if self.method=='M0' else reached(obj,desired,self.c)
            self.pose_run=self.pose_run+1 if arriving else 0
            if self.pose_run>=self.c['pose_dwell_states']:
                self.reached_phases.append(phase)
                if phase=='place':self.release_tcp=tcp.copy()
                self.stage+=1;self.age=0;self.pose_run=0
                return self.command(tcp,obj)
        elif phase=='release':
            goal=self.release_tcp
            if self.age>=self.c['release_steps']:
                self.reached_phases.append(phase);self.stage+=1;self.age=0
                return self.command(tcp,obj)
        else:
            goal=up(self.release_tcp,self.c['retreat_up_m'])
            self.pose_run=self.pose_run+1 if reached(tcp,goal,self.c) else 0
            if phase=='retreat' and self.pose_run>=self.c['pose_dwell_states']:
                self.reached_phases.append(phase);self.stage+=1;self.age=0
                phase='wait'
        if phase in self.c['phase_caps'] and self.age>=self.c['phase_caps'][phase]:
            self.failed=phase+'_TIMEOUT';return None
        self.age+=1
        return phase,goal,1. if self.stage<=2 else -1.


def command12(controller,tcp,goal,gripper,c):
    action=np.r_[osc_action(controller,tcp,goal,c),gripper,np.zeros(4),-1.]
    assert np.isfinite(action).all() and np.max(np.abs(action[:7]))<=1.+1e-12
    return action
