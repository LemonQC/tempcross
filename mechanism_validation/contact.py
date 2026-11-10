"""Physical two-sided contact, including fingertips omitted by pad-only proxy."""
def bilateral_pairs(pairs, target_names):
    sides=set(); target_names=set(target_names)
    for a,b in pairs:
        for finger,other in [(a,b),(b,a)]:
            if other not in target_names or not finger:
                continue
            for side in ['left','right']:
                if finger in {f'gripper0_right_{side}_fingerpad_collision', f'gripper0_right_{side}_fingertip_collision'}:
                    sides.add(side)
    return len(sides)==2


def physical_bilateral(raw):
    sim=raw.sim
    pairs=[(sim.model.geom_id2name(sim.data.contact[i].geom1),sim.model.geom_id2name(sim.data.contact[i].geom2)) for i in range(sim.data.ncon) if sim.data.contact[i].dist<=0]
    return bilateral_pairs(pairs,raw.objects['obj'].contact_geoms)
