"""Diagnostic observers are isolated from the policy information boundary."""
import numpy as np

def gripper_pad_distance(raw):
 gripper=raw.robots[0].gripper
 if isinstance(gripper,dict):gripper=gripper['right']
 points=[]
 for side in ['left_fingerpad','right_fingerpad']:
  ids=[raw.sim.model.geom_name2id(name) for name in gripper.important_geoms[side]]
  points.append(np.asarray(raw.sim.data.geom_xpos[ids]).mean(0))
 return float(np.linalg.norm(points[0]-points[1]))


def physical_closure(actions,pad_distance):
 actions=np.asarray(actions);gap=np.asarray(pad_distance)
 assert gap.shape==(len(actions)+1,) and np.isfinite(gap).all()
 # This measures actual jaw movement after a close command, not a command
 # alone and not a claim of full closure around differently sized objects.
 prior_max=np.maximum.accumulate(gap)[:-1]
 return bool(np.any((actions[:,6]>0)&(prior_max-gap[1:]>=.005)))

def observe_subpredicates(raw,task):
 from robocasa.utils import object_utils as OU
 if task=='pnp_counter_to_sink':
  in_goal=bool(OU.obj_inside_of(raw,'obj',raw.sink,partial_check=True));support=True
 else:
  in_goal=bool(OU.check_obj_in_receptacle(raw,'obj','container'));support=bool(raw.check_contact(raw.objects['container'],raw.counter))
 return dict(object_in_goal=in_goal,goal_supported=support,placement=in_goal and support,retreat=bool(OU.gripper_obj_far(raw,th=.1)))


def summarize_states(tcp,obj,contact,actions,placement,retreat,official_success,pad_distance=None):
 from p2_validation.offline_audit import longest_run
 tcp=np.asarray(tcp);obj=np.asarray(obj);contact=np.asarray(contact,dtype=bool);actions=np.asarray(actions)
 rise=obj[:,2]-obj[0,2];stable=contact & (rise>=.03);first=None;run=0
 for i,v in enumerate(stable):
  run=run+1 if v else 0
  if run==10:first=i-9;break
 horizontal=np.linalg.norm(obj[:,:2]-obj[0,:2],axis=1)
 transported=bool(np.any(contact[first:] & (horizontal[first:]>=.10))) if first is not None else False
 return dict(approached_center_8cm_proxy=bool(np.linalg.norm(tcp-obj,axis=1).min()<=.08),issued_close=bool((actions[:,6]>0).any()),physical_jaw_closure_5mm=physical_closure(actions,pad_distance) if pad_distance is not None else None,bilateral_contact=bool(contact.any()),stable_contact_lift=first is not None,
  transport_10cm_contact_proxy=transported,placement_official_subpredicate=bool(np.any(placement)),retreat_after_placement=bool(np.any(np.asarray(placement,dtype=bool)&np.asarray(retreat,dtype=bool))),ever_gripper_object_far=bool(np.any(retreat)),official_task_success=bool(official_success),
  longest_bilateral_contact_states=longest_run(contact),longest_contact_lift_states=longest_run(stable),maximum_object_rise_m=float(rise.max()))
