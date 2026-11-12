"""Validate existing generated files without creating an environment or stepping physics."""
import json
from pathlib import Path
from common import OUT, ROOT, SOURCE, TASKS, install_paths, read, write


def main():
    import cv2
    import h5py
    import numpy as np
    assert read(OUT/'status.json')['status'] in ['COMPLETE','BLOCKED']
    install_paths()
    from barx.evaluation_conditions import load_bundle
    bundles={task:load_bundle(SOURCE/'evaluation/conditions',task=task,embodiment='panda') for task in TASKS}
    parent_ids={r['episode_id'] for r in read(OUT/'source_seed_manifest.json')}
    rows=[]
    for p in sorted((ROOT/'augmentation/probe').glob('*/episode.json')):
        r=read(p);n=r['steps'];i=r['condition']
        payload,states,_=bundles[r['task']]
        assert i<20 and r['condition_id']==payload['episodes'][i]['condition_id']
        with h5py.File(r['trajectory']) as f,h5py.File(r['telemetry']) as t:
            assert np.array_equal(f['states'][0],states[i])
            assert np.array_equal(f['actions'][:],t['actions/environment'][:])
            assert len(f['actions'])==n and len(f['rgb'])==n and len(f['states'])==n+1
            assert len(t['sim/qpos'])==n+1 and len(t['target/position'])==n+1
            for key in ['actions','states']:assert np.isfinite(f[key][:]).all()
            for key in ['sim/qpos','sim/qvel','target/position']:assert np.isfinite(t[key][:]).all()
            success=bool(np.any(t['mimicgen/post_action_placement'][:]&t['mimicgen/post_action_retreat'][:]))
            assert success==r['success']
        contact=[json.loads(x) for x in Path(r['contact_log']).read_text().splitlines()]
        assert len(contact)==n+1 and all(isinstance(x['contacts'],list) for x in contact)
        cap=cv2.VideoCapture(r['video'])
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==n
        ok,frame=cap.read();cap.release()
        assert ok and frame.shape==(180,320,3)
        lineage=read(ROOT/'augmentation/lineage'/(r['id']+'.json'))
        assert lineage['selected_sources'] and all(x['episode_id'] in parent_ids for x in lineage['selected_sources'])
        attempt=int(r['id'].rsplit('-',1)[1]);assert i==attempt%20
        assert lineage['seed']==700000+1000*TASKS.index(r['task'])+attempt
        rows.append(dict(id=r['id'],steps=n,success=success,exact_frozen_development_state=True,
                         finite_saved_actions_and_states=True,video_header_frames=n,first_frame_decodable=True,
                         contact_states=n+1,source_lineage_verified=True))
    assert len(rows)==177 and sum(r['success'] for r in rows)==63
    write(OUT/'generation_integrity.json',dict(status='PASS',checked_trajectories=len(rows),
        official_successes=sum(r['success'] for r in rows),records=rows,
        video_check_scope='Every augmentation video: container frame count and first frame decoding; full decoding was performed for all ten initial probe videos.',
        simulator_environments_created=0,physics_steps=0,model_calls=0))
    print('PASS: 177 saved augmentation trajectories, 63 official successes; zero new physics/model calls.')


if __name__=='__main__':main()
