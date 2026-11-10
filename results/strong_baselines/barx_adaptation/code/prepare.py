"""Prepare episode split evidence and fixed official-transform validation frames."""
import argparse
from collections import Counter
import csv
import json
import pickle
import numpy as np
from common import OUT, PROJECT, ROOT, digest, manifests, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['manifest', 'validation_target', 'validation_source_ur5e','train_probe'], required=True)
    a = parser.parse_args()
    rows = manifests()
    ids = [r['episode_id'] for r in rows]
    assert len(rows) == 1900 and len(set(ids)) == 1900
    if a.split == 'manifest':
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'split_manifest.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader(); w.writerows(rows)
        counts = Counter((r['domain'], r['embodiment'], r['task'], r['split']) for r in rows)
        write(OUT / 'split_summary.json', dict(
            split_rule='Reuse unchanged frozen baseline episode split: SHA256(20261109:episode_id), per task/embodiment lowest 10% held out',
            no_frame_level_split=True, target_train=90, target_validation=10, source_train=1620, source_validation=180,
            source_prior_contains_existing_iiwa_kinova3_ur5e=True,
            target_validation_seen_by_old_P3_adapter=True,
            pretraining_source_validation_not_unseen_to_original_prior=True,
            counts=[dict(domain=k[0], robot=k[1], task=k[2], split=k[3], episodes=v) for k,v in sorted(counts.items())],
            original_manifests=[dict(path=str(PROJECT / f'configs/datasets/baseline_{d}_manifest.json'),
                                      sha256=digest(PROJECT / f'configs/datasets/baseline_{d}_manifest.json')) for d in ['source','target']]))
        return
    from data import install, dataset_kwargs
    if a.split=='train_probe':
        allowed=install('train')
        from prismatic.vla.datasets.datasets import RLDSDataset
        ds=RLDSDataset(__import__('common').DATA,'panda_xp_900_pnp',[(lambda x:x,1.)],train=True,**dataset_kwargs())
        checked=[]
        for i,raw in enumerate(ds.dataset.as_numpy_iterator()):
            eid=raw['formal_episode_id'].decode();assert eid in allowed and allowed[eid]['split']=='train'
            assert raw['action'].shape==(8,7) and np.isfinite(raw['action']).all()
            assert raw['observation']['image_primary'].shape==(1,224,224,3)
            checked.append(dict(index=i,dataset=raw['dataset_name'].decode(),episode_id=eid,
                                task=allowed[eid]['task'],frame=int(raw['formal_frame_index'])))
            if len(checked)==256:break
        assert len(checked)==256
        write(OUT/'train_input_probe.json',dict(status='COMPLETE',samples=256,policy_calls=0,
              all_samples_train_episodes=True,action_shape=[8,7],rgb_shape=[1,224,224,3],
              sample_weights=ds.dataset.sample_weights,records=checked))
        print('Actual filtered RLDS input probe passed',flush=True)
        return
    allowed = install(a.split)
    from prismatic.vla.datasets.datasets import EpisodicRLDSDataset
    from prismatic.util.data_utils import tree_map
    domain = 'panda_pnp' if a.split == 'validation_target' else 'mg_pnp_lite'
    n = 32 if domain == 'panda_pnp' else 4
    ds = EpisodicRLDSDataset(__import__('common').DATA, domain, [(lambda x:x,1.)], train=False, **dataset_kwargs())
    directory = ROOT / a.split
    directory.mkdir(parents=True, exist_ok=False)
    index = []; seen = set()
    for trajectory in ds.dataset.as_numpy_iterator():
        eid = trajectory['formal_episode_id'][0].decode()
        assert eid in allowed and eid not in seen
        seen.add(eid)
        length = len(trajectory['action'])
        anchors = np.linspace(0, length-1, n, dtype=int)
        assert len(np.unique(anchors)) == n
        for t in anchors:
            raw = tree_map(lambda x:x[t], trajectory)
            path = directory / f'{eid}-{int(t):04d}.pkl'
            path.write_bytes(pickle.dumps(raw, protocol=4))
            index.append(dict(path=str(path), sha256=digest(path), episode_id=eid, frame=int(t),
                              task=allowed[eid]['task'], role='FIXED_OFFLINE_VALIDATION_NOT_PHYSICAL_TEST'))
    assert seen == set(allowed)
    write(OUT / f'{a.split}_index.json', dict(samples=len(index), episodes=len(seen),
            frames_per_episode=n, episode_disjoint_from_new_training=True,
            old_adapters_may_have_trained_on_these_episodes=True, frames=index))
    print(json.dumps(dict(split=a.split, samples=len(index), episodes=len(seen))), flush=True)


if __name__ == '__main__':
    main()
