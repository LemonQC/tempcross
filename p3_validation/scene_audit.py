"""Read-only cross-embodiment scene audit; preserve all official condition IDs."""
import copy
import json
import numpy as np
from p3_validation.common import ROOT,SOURCE


def canonical_metadata(meta):
    meta=copy.deepcopy(meta)
    # This field lists category memberships, not ordered geometry or actions.
    for obj in meta['object_cfgs']:
        info=obj.get('info',{})
        if 'groups_containing_sampled_obj' in info:
            info['groups_containing_sampled_obj']=sorted(info['groups_containing_sampled_obj'])
    return meta


def audit():
    import h5py
    folder=SOURCE/'evaluation/conditions/pnp_counter_to_sink'
    source=json.loads((folder/'ur5e.json').read_text())['episodes']
    target=json.loads((folder/'panda.json').read_text())['episodes'];rows=[]
    for index in range(40,60):
        a=source[index];b=target[index];ma=a['ep_meta'];mb=b['ep_meta']
        paths=[ROOT/'formal'/f'original-joint_reps-{robot}'/'telemetry'/f'episode-{index-40:03d}.h5' for robot in ['ur5e','panda']]
        with h5py.File(paths[0]) as fa,h5py.File(paths[1]) as fb:
            na=json.loads(fa.attrs['body_names']);nb=json.loads(fb.attrs['body_names'])
            objects=[o['name'] for o in ma['object_cfgs']];fixtures=sorted(ma['fixtures'])
            object_errors=[];fixture_errors=[];angles=[]
            for names,errors in [(objects,object_errors),(fixtures,fixture_errors)]:
                for name in names:
                    body=name if name in na and name in nb else name+'_main'
                    if body not in na or body not in nb:continue
                    ia=na.index(body);ib=nb.index(body)
                    errors.append(float(np.linalg.norm(fa['sim/body_xpos'][0,ia]-fb['sim/body_xpos'][0,ib])))
                    if name in objects:
                        qa=fa['sim/body_xquat'][0,ia];qb=fb['sim/body_xquat'][0,ib]
                        dot=abs(float(np.dot(qa,qb)/(np.linalg.norm(qa)*np.linalg.norm(qb))))
                        angles.append(float(np.rad2deg(2*np.arccos(np.clip(dot,0.,1.)))))
        assets=lambda m:{o['name']:o['info']['mjcf_path'] for o in m['object_cfgs']}
        rows.append(dict(official_condition_index=index,source_condition_id=a['condition_id'],target_condition_id=b['condition_id'],
            original_seeds_match=a['seed']==b['seed'],actual_seed=a['seed'],raw_metadata_identical=ma==mb,
            metadata_identical_after_sorting_category_memberships=canonical_metadata(ma)==canonical_metadata(mb),
            original_object_assets_identical=assets(ma)==assets(mb),object_bodies_compared=len(object_errors),fixture_bodies_compared=len(fixture_errors),
            maximum_initial_object_position_difference_m=max(object_errors),maximum_initial_object_rotation_difference_degrees=max(angles),
            maximum_initial_fixture_position_difference_m=max(fixture_errors)))
    result=dict(role='POSTHOC_SCENE_COMPARABILITY_AUDIT_NO_NEW_PHYSICS',records=rows,
        original_hashes_and_condition_ids_unchanged=True,
        category_sorting_used_only_for_comparison_not_loading_or_hash_verification=True,
        interpretation='Cross-robot comparisons share original index and seed, not identical robot states. Small recorded object settling differences are reported; all Panda method pairs are checked separately for exact initial-state equality.')
    (ROOT/'cross_embodiment_scene_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(audit(),indent=2))
