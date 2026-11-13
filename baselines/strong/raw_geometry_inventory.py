"""Verify raw UR5e provenance against the frozen RGB/canonical-action pool.

Reads saved states only. It neither steps a simulator nor infers contact or
counterfactual labels. Availability is distinct from reconstruction/validation.
"""
import collections
import hashlib
import json
import struct
from pathlib import Path
from baselines.strong.common import PROJECT, ROOT, OLD, read, write, digest, now


def identity(actions, instruction):
    import numpy as np
    a = np.ascontiguousarray(actions, dtype='<f4')
    return hashlib.sha256(struct.pack('<Q', len(a))+a.tobytes()+instruction.encode()).hexdigest()


def main():
    import h5py
    import numpy as np
    manifest = PROJECT/'configs/datasets/baseline_source_manifest.json'
    expected = {r['episode_id']: r for r in read(manifest) if r['embodiment']=='ur5e'}
    assert len(expected)==600
    acquired = read(ROOT/'raw_source_ur5e/acquisition.json')
    assert acquired['status']=='COMPLETE' and len(acquired['verified'])==6
    rows=[]
    with h5py.File(OLD/'data/source.h5', 'r') as packed:
        for file in acquired['verified']:
            path=Path(file['absolute_path'])
            assert path.stat().st_size==int(file['bytes']) and digest(path)==file['sha256']
            with h5py.File(path, 'r') as f:
                assert len(f['data'])==100
                env_args=json.loads(f['data'].attrs['env_args'])
                for name,g in f['data'].items():
                    raw=g['actions'][:]
                    assert raw.shape[1]==12 and np.isfinite(raw).all()
                    assert np.all(raw[:,7:11]==0) and np.all(raw[:,11]==-1)
                    a=raw[:,:7].astype(np.float32)
                    a[-1]=a[-2];a[-1,:6]=0
                    meta=json.loads(g.attrs['ep_meta'])
                    eid=identity(a,meta['lang']);old=expected[eid]
                    assert '/mg/'+str(path).split('/mg/',1)[1] in old['original_file']
                    assert old['frames']==len(a) and old['instruction']==meta['lang']
                    assert np.array_equal(a,packed[eid]['actions'][:])
                    assert g['obs/agentview_rgb'].shape==(len(a),180,320,3)
                    fields={}
                    for key in ['states','obs/joint_states','obs/gripper_states','obs/ee_states']:
                        value=g[key][:]
                        assert len(value)==len(a) and np.isfinite(value).all()
                        fields[key]=dict(shape=list(value.shape),sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                    rows.append(dict(episode_id=eid,task=old['task'],split=old['split'],frames=len(a),
                        raw_file=str(path),raw_demo=name,fields=fields,
                        ep_meta_has_camera_configs='cam_configs' in meta,
                        raw_camera_depths=env_args['env_kwargs'].get('camera_depths'),
                        raw_depth_dataset_present=any('depth' in k for k in g['obs']),
                        contacts_extracted=False,geometry_reconstruction_validated=False))
    assert len(rows)==600 and {r['episode_id'] for r in rows}==set(expected)
    assert len({r['episode_id'] for r in rows})==600
    counts=collections.Counter((r['task'],r['split']) for r in rows)
    assert sorted(counts.values())==[30,30,270,270]
    summary=dict(status='PASS_RAW_IDENTITY_AND_FIELD_AVAILABILITY',utc=now(),
        demonstrations=600,train=540,validation=60,actions_matched_all_frames=True,
        new_demonstrations=0,new_training_updates=0,new_policy_rollouts=0,
        source_manifest_sha256=digest(manifest),acquisition=acquired,
        depth_present_episodes=sum(r['raw_depth_dataset_present'] for r in rows),
        contacts_extracted=False,geometry_reconstruction_validated=False,
        limitation='Saved state and camera metadata availability do not certify contact or depth reconstruction.')
    folder=ROOT/'shared_geometry';write(folder/'ur5e_raw_identity.json',rows);write(folder/'ur5e_raw_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='acquisition'},indent=2))


if __name__=='__main__':
    main()
