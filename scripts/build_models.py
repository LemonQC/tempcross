"""Extract pinned public kinematic sources; no xacro/ROS runtime needed."""
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def number(s):
    return {'${pi/2}':math.pi/2, '${-pi/2}':-math.pi/2}.get(s, None) if 'pi' in s else float(s)

def extract():
    for robot, filename, commit, repo in [
        ('sr5','rokae/xMateSR5.urdf.xacro','d6a508c38513d4c39b2f10f14275d0e26df7fb43','RokaeRobot/rokae_ros2'),
        ('panda','franka/franka_arm.xacro','ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98','frankarobotics/franka_ros')]:
        tree=ET.parse(ROOT/'vendor'/filename)
        limits=[]
        if robot=='panda':
            text=(ROOT/'vendor/franka/joint_limits.yaml').read_text()
            limits=[(float(a),float(b)) for a,b in re.findall(r'lower:\s*([-\d.]+)\s+upper:\s*([-\d.]+)',text)]
        joints=[]
        for node in tree.iter('joint'):
            if node.get('type')!='revolute': continue
            ori=node.find('origin'); axis=node.find('axis'); lim=node.find('limit')
            lo,hi=limits[len(joints)] if robot=='panda' else (float(lim.get('lower')),float(lim.get('upper')))
            joints.append(dict(name=node.get('name'),xyz=[number(x) for x in ori.get('xyz').split()],
                               rpy=[number(x) for x in ori.get('rpy').split()],axis=[float(x) for x in axis.get('xyz').split()],lower=lo,upper=hi))
        spec=dict(robot=robot,source_repo=repo,source_commit=commit,joints=joints,
                  flange_fixed_xyz=[0,0,.107] if robot=='panda' else [0,0,0],
                  flange_frame='panda_link8' if robot=='panda' else 'xMateSR5_link6',
                  home=[0,-.5,0,-1.8,0,1.5,.7] if robot=='panda' else [0,-.7,1.1,0,.7,0])
        (ROOT/f'configs/validation/{robot}.json').write_text(json.dumps(spec,indent=2)+'\n')
if __name__=='__main__': extract()
