"""Episode-held-out official RLDS pipeline; no changed objectives or images.

The runtime patch captures raw trajectory identity, loads the complete published
split, filters whole episodes BEFORE windowing, and reuses frozen prior stats.
Upstream normalization, task annotations, future windows and RGB transforms stay
in the original code. No changes are written into the official checkout.
"""
import copy
import hashlib
import inspect
import pickle
from collections import Counter
import numpy as np
from common import DATA, OLD_STATS, OUT, ROOT, manifests, setup_source, write


def install(split):
    setup_source()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    from p3_validation.prepare_data import fingerprint
    import prismatic.vla.datasets.rlds.dataset as rd
    rows = manifests()
    selected = [r for r in rows if r['split'] == ('train' if split == 'train' else 'validation')]
    if split == 'validation_source_ur5e':
        selected = [r for r in selected if r['domain'] == 'source' and r['embodiment'] == 'ur5e']
    elif split == 'validation_target':
        selected = [r for r in selected if r['domain'] == 'target']
    allowed = {r['episode_id'] for r in selected}
    keys = sorted(allowed)
    lookup = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, [True] * len(keys)), False)
    sizes = {}
    for dataset, domain in [('mg_pnp_lite', 'source'), ('panda_pnp', 'target')]:
        rr = [r for r in selected if r['domain'] == domain]
        sizes[dataset] = (sum(r['frames'] for r in rr), len(rr))
    original = getattr(rd, '_adaptation_original_loader', rd.make_dataset_from_rlds)
    rd._adaptation_original_loader = original
    text = inspect.getsource(original)
    anchor = '        # apply a standardization function, if provided\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '        formal_fields = _formal_capture(traj)\n' + anchor)
    anchor = '        if absolute_action_mask is not None:\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '        traj.update(formal_fields)\n\n' + anchor)
    anchor = 'split = "train[:95%]" if train else "train[95%:]"'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'split = "train"')
    anchor = '    dataset = dataset.traj_map(restructure, num_parallel_calls)\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '    dataset = dataset.filter(lambda x: _formal_lookup.lookup(x["formal_episode_id"][0]))\n')
    anchor = '    # load or compute dataset statistics\n'
    assert text.count(anchor) == 1
    # Always use the existing source-prior statistics, including for Panda.
    # Only counts change for sampling; no held-out target statistics are fit.
    text = text.replace(anchor, '    dataset_statistics = _formal_stats(name)\n' + anchor)

    def capture(traj):
        n = tf.shape(traj['action'])[0]
        def identify(a, length, lang):
            return fingerprint(a.numpy(), int(length.numpy()), lang.numpy()).encode()
        identity = tf.py_function(identify, [traj['action'], n, traj['language_instruction'][0]], tf.string)
        identity = tf.ensure_shape(identity, [])
        return dict(formal_episode_id=tf.repeat(identity, n), formal_frame_index=traj['_frame_index'],
                    formal_file_path=traj['traj_metadata']['episode_metadata']['file_path'])

    prior = __import__('json').loads(OLD_STATS.read_text())['mg_pnp_lite']
    def statistics(name):
        value = copy.deepcopy(prior)
        value['num_transitions'], value['num_trajectories'] = sizes[name]
        return value

    rd.__dict__.update(_formal_capture=capture, _formal_lookup=lookup, _formal_stats=statistics)
    snapshot = ROOT / 'pipeline_snapshots' / f'{split}.py'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(text)
    exec(compile(text, str(snapshot), 'exec'), rd.__dict__)
    return {r['episode_id']: r for r in selected}


def dataset_kwargs():
    return dict(resize_resolution=(224, 224), shuffle_buffer_size=10000, image_aug=False,
                future_action_window_size=7, future_obj_pose_window_size=40,
                future_2D_trace_window_size=40, obj_pose_stride=8, ee_pose_2D_stride=8,
                image_window_size=1, dataset_statistics_map={'panda_pnp': 'mg_pnp_lite'})


def trace_iterator(dataset, transform_rng, stream, role='train'):
    """Trace actual sampled frames and transform choices; no finite-cache replay."""
    import csv
    rows = {r['episode_id']: r for r in manifests()}
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['index', 'dataset', 'task', 'episode_id', 'frame', 'transform_index', 'raw_sha256'])
    counts = Counter()
    for i, raw in enumerate(dataset.dataset.as_numpy_iterator()):
        eid = raw['formal_episode_id'].decode()
        assert rows[eid]['split'] == role
        name = raw['dataset_name'].decode()
        transforms, weights = zip(*dataset.batch_transforms)
        index = int(transform_rng.choice(len(transforms), p=weights))
        hashed = hashlib.sha256(pickle.dumps(raw, protocol=4)).hexdigest()
        writer.writerow([i, name, rows[eid]['task'], eid, int(raw['formal_frame_index']), index, hashed])
        counts[name] += 1
        if (i + 1) % 256 == 0:
            stream.flush()
        yield transforms[index](raw)
