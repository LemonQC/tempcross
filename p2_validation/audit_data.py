"""Audit target demonstration provenance and detectable frozen-state overlap."""
import json
from pathlib import Path
import sys

import numpy as np


def audit(source,data,out):
    import h5py
    source=Path(source);data=Path(data);sys.path.insert(0,str(source))
    from barx.evaluation_conditions import load_bundle,model_sha256,state_sha256
    payload,frozen,models=load_bundle(source/'evaluation/conditions',task='pnp_counter_to_sink',embodiment='panda')
    tests=payload['episodes'][20:40];test_models=[model_sha256(x) for x in models[20:40]]
    rows=[];matches=[]
    for path in sorted((data/'raw_target').rglob('*.hdf5')):
        with h5py.File(path,'r') as f:
            groups=f['data'];assert len(groups)==50
            env_args=groups.attrs.get('env_args','{}')
            if isinstance(env_args,bytes):env_args=env_args.decode()
            env=json.loads(env_args)
            for name in sorted(groups):
                demo=groups[name];meta=json.loads(demo.attrs['ep_meta']);states=demo['states'][:]
                actions=demo['actions'][:];assert actions.ndim==2 and actions.shape[1]==12
                xml=demo.attrs.get('model_file',groups.attrs.get('model_file',''))
                if isinstance(xml,bytes):xml=xml.decode()
                model_hash=model_sha256(xml) if '<mujoco' in xml else None
                exact_states=[]
                # Compare complete physical state sequences (ignoring simulator clock),
                # not just language, kitchen layout, or object category.
                if 'PnPCounterToSink' in str(path):
                    for i,state in enumerate(frozen[20:40]):
                        if states.shape[1]!=len(state):continue
                        found=np.flatnonzero(np.max(np.abs(states[:,1:]-state[None,1:]),axis=1)<1e-10)
                        for t in found:
                            match=dict(demo=name,raw_file=str(path),test_condition_id=tests[i]['condition_id'],
                                       demo_step=int(t),same_model=model_hash==test_models[i]);exact_states.append(match);matches.append(match)
                rows.append(dict(file=str(path),demo=name,steps=len(actions),task=path.parent.name,
                    layout_id=meta.get('layout_id'),style_id=meta.get('style_id'),seed=meta.get('seed'),
                    instruction=meta.get('lang'),model_hash=model_hash,model_hash_matches_test=model_hash in test_models,
                    state_shape=list(states.shape),gripper_action_values=np.unique(actions[:,6]).tolist(),
                    camera_image_shape=list(demo['obs/agentview_rgb'].shape),
                    declared_env=env.get('env_name'),declared_robot=env.get('env_kwargs',{}).get('robots'),
                    declared_gripper=env.get('env_kwargs',{}).get('gripper_types'),exact_state_matches=exact_states))
    if len(rows)!=100:raise RuntimeError(f'Expected official 50-per-direction target demos, got {len(rows)}')
    result=dict(status='COMPLETE',target_demonstrations=len(rows),per_direction=50,
        frozen_test_indices=[20,39],exact_physical_state_matches=matches,
        overlap_conclusion='NO_DETECTABLE_EXACT_STATE_OVERLAP' if not matches else 'POTENTIAL_OVERLAP_REQUIRES_REVIEW',
        overlap_limitations='Exact physical-state comparison ignores simulator clock. Different model/joint ordering or incomplete metadata limits exhaustive identity proof; same kitchen/object class is not trajectory leakage.',
        action_mapping='12D canonical simulator -> first7 -> official libero transform (gripper open=1 close=0); mg_pnp_lite prior normalization and VQ reused',
        mixture={'mg_pnp_lite':.05,'panda_pnp':1.,'balance_weights_by_transitions':True},demonstrations=rows)
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
    if matches:raise RuntimeError('Detectable train/test state overlap requires resolution before adaptation')
    return result
