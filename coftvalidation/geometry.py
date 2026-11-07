"""Assumed geometry only. Finite checking is never a continuous certificate."""
import numpy as np
from coftprobe.geometry import transform, inverse, apply, camera_projection, segment_box_intersects
TCP=transform([0,0,.12])
CAMERA=dict(width=640,height=480,fx=420,fy=420,cx=320,cy=240,near_m=.25,far_m=2.)
MOUNTS={'mount_a':transform([.065,0,.025]),'mount_b':transform([-.065,.025,.025])}
TABLE=dict(center=[.5,0,-.025],half_extent=[.55,.60,.025])

def hand_boxes():
    # Parallel jaws open around a 44 mm wide object; flange coordinates. Not PGIA/Panda CAD.
    return [
            dict(center=[0,0,.045],half_extent=[.045,.045,.025]),
            dict(center=[0,-.034,.095],half_extent=[.04,.008,.025]),
            dict(center=[0,.034,.095],half_extent=[.04,.008,.025])]

def balls(frames,radius=.025):
    groups=[]
    for a,b in zip(frames,frames[1:]):
        a=a[:3,3]; b=b[:3,3]; n=max(1,int(np.ceil(np.linalg.norm(b-a)/.025)))
        groups.append(np.linspace(a,b,n+1))
    # Radius inflated by half sample spacing to cover proxy centerline segment.
    return groups,radius+.0125

def point_box(points,box):
    delta=np.maximum(np.abs(points-np.array(box['center']))-np.array(box['half_extent']),0)
    return np.linalg.norm(delta,axis=-1)

def collision(frames,flange,boxes,carried,sc,radius=.025):
    groups,r=balls(frames,radius); reasons=set(); world_boxes=boxes+[TABLE]
    for i,g in enumerate(groups):
        for b in world_boxes:
            # Base mounted through table; base segment allowed table contact.
            if i==0 and b is TABLE: continue
            if np.min(point_box(g,b))<r: reasons.add('arm_environment_proxy')
    for i,g in enumerate(groups):
        for j in range(i+3,len(groups)):
            if np.min(np.linalg.norm(g[:,None,:]-groups[j][None,:,:],axis=-1))<2*r:
                reasons.add('self_collision_proxy')
    # Oriented hand boxes checked with corner/center balls enclosing each hand box.
    for h in hand_boxes():
        center=apply(flange,h['center']); hr=np.linalg.norm(h['half_extent'])
        for b in world_boxes:
            if float(point_box(center,b))<hr: reasons.add('hand_environment_proxy')
        for g in groups[:-3]:
            if np.min(np.linalg.norm(g-center,axis=-1))<hr+r: reasons.add('hand_arm_proxy')
    if carried:
        center=apply(flange@TCP,[0,0,0]); cr=float(np.linalg.norm(sc['half_extent']))
        for b in boxes:
            if float(point_box(center,b))<cr: reasons.add('carried_object_proxy')
    return sorted(reasons)

def surface(sc):
    # Equal-area top face quadrature, not exact projected visible area.
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('z',sc['yaw']).as_matrix(); p=np.array(sc['target']); ex=sc['half_extent']
    return np.array([p+r@np.array([x,y,ex[2]]) for x in np.linspace(-ex[0],ex[0],5) for y in np.linspace(-ex[1],ex[1],5)])

def visibility(flange,mount,sc,boxes,frames=None):
    camera=flange@mount; origin=camera[:3,3]; inv=inverse(flange); visible=0; occluded_hand=0; robot_occ=0
    groups,r=balls(frames) if frames is not None else ([],0)
    for p in surface(sc):
        proj=camera_projection(camera,p,CAMERA)
        hand=any(segment_box_intersects(apply(inv,origin),apply(inv,p),h['center'],h['half_extent']) for h in hand_boxes())
        env=any(segment_box_intersects(origin,p,b['center'],b['half_extent']) for b in boxes)
        # Ray vs ball proxy for arm self-occlusion, ignoring terminal flange sphere containing camera.
        d=p-origin; den=float(d@d); rob=False
        for g in groups[:-2]:
            u=np.clip(((g-origin)@d)/den,0,1)
            if np.any(np.linalg.norm(origin+u[:,None]*d-g,axis=1)<r): rob=True; break
        occluded_hand+=int(hand); robot_occ+=int(rob)
        visible+=int(proj['in_frustum'] and not (hand or env or rob))
    return dict(fraction=visible/25,hand_occlusion=occluded_hand/25,robot_occlusion=robot_occ/25)
