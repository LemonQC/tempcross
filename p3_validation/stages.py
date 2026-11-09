"""Post-hoc stage measurements from recorded states; no task relabeling.

State s is the pre-state of action s and the post-state of action s-1.
Only the already frozen contact/lift rule produces a stage confirmation.
Transport and retreat are continuous measurements, not new success tests.
"""
import numpy as np


def measure_stages(object_positions,eef_positions,contact,gripper):
    obj=np.asarray(object_positions);eef=np.asarray(eef_positions)
    contact=np.asarray(contact,dtype=bool);gripper=np.asarray(gripper)
    n=len(gripper)
    assert obj.shape==eef.shape==(n+1,3) and contact.shape==(n+1,)
    assert np.isfinite(obj).all() and np.isfinite(eef).all() and np.isfinite(gripper).all()
    rise=obj[:,2]-obj[0,2];stable=contact & (rise>=.03)
    run=0;confirmation=None;start=None
    for s,valid in enumerate(stable):
        run=run+1 if valid else 0
        if run==10 and confirmation is None:confirmation=s;start=s-9
    contacts=np.flatnonzero(contact)
    release=(np.flatnonzero((gripper[:-1]>0)&(gripper[1:]<0))+1).tolist()
    later=[t for t in release if confirmation is not None and t>=confirmation]
    first_release=later[0] if later else None
    # Include the first state of the confirmed run and retain contact requirement.
    carry=np.flatnonzero(contact & (np.arange(n+1)>=start)) if start is not None else np.array([],dtype=int)
    horizontal=float(np.linalg.norm(obj[carry,:2]-obj[start,:2],axis=1).max()) if len(carry) else None
    distances=np.linalg.norm(eef-obj,axis=1)
    return dict(first_bilateral_contact_state_index=int(contacts[0]) if len(contacts) else None,
        first_stable_lift_start_state_index=start,first_stable_lift_confirmation_state_index=confirmation,
        stable_lift_confirmed=confirmation is not None,
        maximum_horizontal_displacement_from_lift_start_while_in_contact_m=horizontal,
        release_command_action_indices=release,first_release_after_confirmed_lift_action_index=first_release,
        maximum_eef_object_distance_after_release_m=float(distances[first_release+1:].max()) if first_release is not None else None,
        final_eef_object_distance_m=float(distances[-1]),
        transport_and_retreat_are_continuous_measurements_not_new_success_rules=True)
