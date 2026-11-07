import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from coftprobe.geometry import transform

def scenes(seed,n=9):
    rng=np.random.default_rng(seed); out=[]
    for i in range(n):
        family=['asymmetric_grasp','clutter_grasp','oriented_place'][i%3]
        target=[float(rng.uniform(.32,.74)),float(rng.uniform(-.22,.22)),.035]
        yaw=float(rng.uniform(-np.pi,np.pi)); boxes=[]
        if family!='asymmetric_grasp':
            for j in range(2):
                a=rng.uniform(-np.pi,np.pi); d=rng.uniform(.07,.18); h=rng.uniform(.08,.30)
                boxes.append(dict(center=[target[0]+d*np.cos(a),target[1]+d*np.sin(a),h/2],half_extent=[.025,.035,h/2]))
        out.append(dict(id=f'{seed}-{i}',seed=seed,split='dev' if seed==314159 else 'test',family=family,
                        target=target,yaw=yaw,half_extent=[.035,.022,.035],boxes=boxes,
                        place=[.42,-.20,.035],place_yaw=0.))
    return out

def pose(p,yaw): return transform(p,Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('x',np.pi).as_matrix())

def candidates(sc):
    p=np.array(sc['target']); out=[]
    for flip in range(2):
        yaw=sc['yaw']+flip*np.pi
        for route in range(4):
            high=.42 if route==3 else .30
            offset=np.array([0, .12 if route==1 else -.12 if route==2 else 0,0])
            # First observation and pregrasp shared; route is a transit offset, returns before contact.
            positions=[p+[0,0,.34],p+[0,0,high]+offset,p+[0,0,.20],p,p+[0,0,high]]
            yaws=[yaw]*len(positions); carried=[False]*4+[True]
            if sc['family']=='oriented_place':
                goal=np.array(sc['place']); endyaw=sc['place_yaw']+flip*np.pi
                positions += [goal+[0,0,high]+offset,goal+[0,0,.20],goal,goal+[0,0,.30]]
                yaws += [endyaw]*4; carried += [True,True,True,False]
            out.append(dict(id=flip*4+route,flip=flip,route=route,poses=[pose(x,y) for x,y in zip(positions,yaws)],carried=carried))
    return out

def interpolate(a,b,steps):
    rotations=Slerp([0,1],Rotation.from_matrix([a[:3,:3],b[:3,:3]]))
    for u in np.linspace(0,1,steps+1)[1:]:
        yield transform(a[:3,3]*(1-u)+b[:3,3]*u,rotations([u]).as_matrix()[0]),float(u)

def dense_path(candidate,step=.025):
    poses=candidate['poses']; out=[(poses[0],False)]
    for i,(a,b) in enumerate(zip(poses,poses[1:])):
        angle=np.linalg.norm(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).as_rotvec())
        n=max(1,int(np.ceil(np.linalg.norm(a[:3,3]-b[:3,3])/step)),int(np.ceil(angle/.10)))
        out.extend((t,candidate['carried'][i+1]) for t,u in interpolate(a,b,n))
    return out
