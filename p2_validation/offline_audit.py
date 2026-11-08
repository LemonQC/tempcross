"""Reconstruct fingerpad poses from existing P1 telemetry without simulation.

The center prism is a deliberately limited geometric diagnostic. Neither it nor
the contact/lift proxy is a legal-grasp oracle or a replacement success label.
No MuJoCo, policy, simulator or training package is imported by this module.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def rotation(quaternion):
    """Convert simulator wxyz quaternions, with arbitrary leading dimensions."""
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-12):
        raise ValueError('Invalid simulator quaternion')
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def multiply(first, second):
    a, b = np.broadcast_arrays(first, second)
    w, x, y, z = np.moveaxis(a, -1, 0)
    v, i, j, k = np.moveaxis(b, -1, 0)
    return np.stack([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v], axis=-1)


def relative_pose(origin_position, origin_quaternion, position, quaternion):
    inv = np.asarray(origin_quaternion).copy()
    inv[..., 1:] *= -1
    offset = np.einsum('...ji,...j->...i', rotation(origin_quaternion),
                       np.asarray(position)-origin_position)
    return offset, multiply(inv, quaternion)


def fingerpad_specs(xml):
    # The official portable XML contains package path markers inside attributes.
    for name in ('ROBOCASA', 'ROBOSUITE'):
        xml = xml.replace('<' + name + '>', '__' + name + '__')
    root = ET.fromstring(xml)
    specs = {}
    for body in root.iter('body'):
        for geom in body.findall('geom'):
            for side in ('left', 'right'):
                if geom.get('name') == f'gripper0_right_{side}_fingerpad_collision':
                    if geom.get('type') != 'box':
                        raise ValueError('Expected the fixed Robotiq85 box fingerpad')
                    specs[side] = dict(body=body.attrib['name'],
                                       position=np.fromstring(geom.get('pos', '0 0 0'), sep=' '),
                                       quaternion=np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '),
                                       half_size=np.fromstring(geom.attrib['size'], sep=' '))
    if set(specs) != {'left', 'right'}:
        raise ValueError('Cannot reconstruct both official Robotiq85 fingerpads')
    return specs


def world_pad(body_position, body_quaternion, spec):
    position = body_position + np.einsum('...ij,j->...i', rotation(body_quaternion), spec['position'])
    quaternion = multiply(body_quaternion, spec['quaternion'])
    return position, quaternion


def center_prism(left, right, object_position, specs):
    """Measure object center in the moving pad prism, ignoring object extent.

    A false result cannot rule out an off-center grasp of an extended object.
    A true result cannot establish collision clearance, force closure or success.
    """
    lp, lq = left
    rp, rq = right
    lr, rr = rotation(lq), rotation(rq)
    center = (lp + rp) / 2
    separation = rp - lp
    width = np.linalg.norm(separation, axis=-1)
    y_axis = separation / np.maximum(width[..., None], 1e-12)
    lx, rx = lr[..., :, 0], rr[..., :, 0]
    sign = np.where(np.sum(lx*rx, axis=-1) < 0, -1., 1.)
    x_axis = lx + rx*sign[..., None]
    x_axis -= y_axis*np.sum(x_axis*y_axis, axis=-1)[..., None]
    x_norm = np.linalg.norm(x_axis, axis=-1)
    x_axis /= np.maximum(x_norm[..., None], 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    frame = np.stack([x_axis, y_axis, z_axis], axis=-1)
    local = np.einsum('...ji,...j->...i', frame, object_position-center)
    thickness = sum(np.sum(np.abs(np.einsum('...i,...ij->...j', y_axis, rot))*specs[side]['half_size'], axis=-1)
                    for side, rot in [('left', lr), ('right', rr)])
    gap = width-thickness
    normal_alignment = np.minimum(np.abs(np.sum(lr[..., :, 1]*y_axis, axis=-1)),
                                  np.abs(np.sum(rr[..., :, 1]*y_axis, axis=-1)))
    dimensions = (specs['left']['half_size'] + specs['right']['half_size']) / 2
    center_inside = ((np.abs(local[..., 0]) <= dimensions[0]) &
                     (np.abs(local[..., 1]) <= gap/2) &
                     (np.abs(local[..., 2]) <= dimensions[2]) &
                     (gap > 0) & (x_norm > 1e-9) &
                     (normal_alignment >= np.cos(np.deg2rad(15))))
    return local, gap, normal_alignment, center_inside


def expand_chunks(chunks, starts, executed_steps):
    """Align pre-normalization predictions to actual zero-based action steps."""
    if len(chunks) != len(starts):
        raise ValueError('Chunk/start counts differ')
    raw = np.full((executed_steps, 7), np.nan)
    covered = np.zeros(executed_steps, dtype=bool)
    for chunk, start in zip(chunks, starts):
        start = int(start)-1  # Recorder increments after writing state zero.
        length = min(len(chunk), executed_steps-start)
        if start < 0 or length <= 0 or covered[start:start+length].any():
            raise ValueError('Invalid/overlapping chunk timeline')
        raw[start:start+length] = chunk[:length]
        covered[start:start+length] = True
    if not covered.all() or not np.isfinite(raw).all():
        raise ValueError('Incomplete/nonfinite policy action timeline')
    return raw


def longest_run(mask):
    best = current = 0
    for yes in mask:
        current = current + 1 if yes else 0
        best = max(best, current)
    return best


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit_episode(directory, record, condition, xml, external):
    import h5py
    if type(record.get('success')) is not bool:
        raise ValueError('Official success must be a JSON boolean')
    episode = record['episode']
    path = directory/'telemetry'/f'episode-{episode:03d}.h5'
    case = '/'.join(directory.parts[-2:]) + f'/{episode:03d}'
    with h5py.File(path, 'r') as f:
        observed = json.loads(f.attrs['condition'])
        if (any(observed[k] != condition[k] for k in
                ['condition_id', 'seed', 'state_sha256', 'model_sha256', 'metadata_sha256'])
                or record['condition_id'] != condition['condition_id']):
            raise ValueError('Telemetry/official/source condition mismatch')
        names = json.loads(f.attrs['body_names'])
        positions, quaternions = f['sim/body_xpos'][:], f['sim/body_xquat'][:]
        target, target_q = f['target/position'][:], f['target/quaternion'][:]
        environment = f['actions/environment'][:]
        if len(environment) != record['steps'] or len(target) != record['steps']+1:
            raise ValueError('State/action/official length mismatch')
        raw = expand_chunks(f['actions/policy_chunks'][:], f['actions/policy_chunk_step'][:], len(environment))
        expected = raw.copy()
        expected[:, 6] = -np.sign(2*expected[:, 6]-1)
        error = float(np.abs(expected-environment[:, :7]).max())
        if error > 1e-12:
            raise ValueError('Official action mapping mismatch')
        specs = fingerpad_specs(xml)
        pads = {side: world_pad(positions[:, names.index(spec['body'])],
                               quaternions[:, names.index(spec['body'])], spec)
                for side, spec in specs.items()}
        local, gap, normal, inside = center_prism(pads['left'], pads['right'], target, specs)
        tcp_id = names.index('gripper0_right_eef')
        tcp_relative, tcp_relative_q = relative_pose(positions[:, tcp_id], quaternions[:, tcp_id], target, target_q)
        finger_qpos = f['obs/robot0_gripper_qpos'][:]
    telemetry = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    if len(telemetry) != len(target):
        raise ValueError('Contact/state timeline mismatch')
    contact = np.asarray([row['grasp_contact'] is True for row in telemetry])
    rise = target[:, 2]-target[0, 2]
    contact_lift_run = longest_run(contact & (rise >= .03))
    closing = environment[:, 6] > 0
    # Pre-action state t corresponds to action t and video frame t.
    opportunities = inside[:-1]
    first_close = int(np.flatnonzero(closing)[0]) if closing.any() else None
    videos = list((directory/'official').rglob(f'*--episode={episode+1}--success=*--task=xpnpcountertosink.mp4'))
    if len(videos) != 1:
        raise ValueError('Missing or ambiguous existing video')
    row = dict(case_id=case, condition_id=condition['condition_id'], actual_seed=condition['seed'],
               original_condition_index=condition['episode'], success=record['success'], steps=record['steps'],
               raw_gripper_min=float(raw[:, 6].min()), raw_gripper_median=float(np.median(raw[:, 6])),
               raw_gripper_max=float(raw[:, 6].max()), first_close_action_index=first_close,
               close_command_steps=int(closing.sum()), center_in_pad_prism_pre_action_steps=int(opportunities.sum()),
               center_prism_without_close_steps=int((opportunities & ~closing).sum()),
               center_prism_with_close_steps=int((opportunities & closing).sum()),
               minimum_inner_pad_gap_m=float(gap.min()), maximum_inner_pad_gap_m=float(gap.max()),
               bilateral_contact_state_steps=int(contact.sum()),
               maximum_contact_and_3cm_lift_run_state_steps=contact_lift_run,
               sustained_contact_and_lift_proxy=contact_lift_run >= 10,
               maximum_object_rise_m=float(rise.max()), action_mapping_max_error=error,
               legal_grasp_opportunity='UNKNOWN_REQUIRES_PHYSICAL_COUNTERFACTUAL',
               video_path=str(videos[0]), telemetry_path=str(path))
    output = external/(case.replace('/', '_') + '.csv.gz')
    with gzip.open(output, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pre_action_index', 'raw_gripper', 'environment_gripper', 'inner_pad_gap_m',
                         'center_in_pad_prism', 'bilateral_contact', 'object_rise_m'] +
                        [f'tcp_object_{k}' for k in ['x','y','z','qw','qx','qy','qz']] +
                        [f'{side}_pad_world_{k}' for side in ['left','right'] for k in ['x','y','z','qw','qx','qy','qz']] +
                        [f'finger_joint_{i}' for i in range(finger_qpos.shape[1])])
        for t in range(len(environment)):
            writer.writerow([t, raw[t, 6], environment[t, 6], gap[t], bool(inside[t]), bool(contact[t]), rise[t]] +
                            tcp_relative[t].tolist()+tcp_relative_q[t].tolist() +
                            pads['left'][0][t].tolist()+pads['left'][1][t].tolist() +
                            pads['right'][0][t].tolist()+pads['right'][1][t].tolist() + finger_qpos[t].tolist())
    row['external_timeseries_path'] = str(output)
    row['timeseries_sha256'] = sha256(output)
    return row, [dict(absolute_path=str(p), size_bytes=p.stat().st_size, sha256=sha256(p))
                 for p in [path, path.with_suffix('.jsonl')]]


def audit(root, source, out, external):
    root, source, out, external = [Path(p).resolve() for p in (root, source, out, external)]
    for directory in [out, external]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'Refusing to replace evidence: {directory}')
    for directory in [out, external]:
        directory.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    bundles = {}
    for arm in ['ur5e', 'panda']:
        base = source/'evaluation/conditions/pnp_counter_to_sink'
        metadata = json.loads((base/f'{arm}.json').read_text())
        with np.load(base/f'{arm}.npz', allow_pickle=False) as archive:
            xmls = archive['model_xmls'].tolist()
        bundles[arm] = metadata['episodes'], xmls
        for path in [base/f'{arm}.json', base/f'{arm}.npz']:
            inputs.append(dict(absolute_path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path)))
    for method in ['no_reps', 'joint_reps']:
        for arm in ['ur5e', 'panda']:
            directory = root/'p1'/method/arm
            official = list((directory/'official').rglob('episodes.jsonl'))
            if len(official) != 1:
                raise ValueError('Expected one completed original official evaluation')
            inputs.append(dict(absolute_path=str(official[0]), size_bytes=official[0].stat().st_size,
                               sha256=sha256(official[0])))
            records = [json.loads(line) for line in official[0].read_text().splitlines()]
            if len(records) != 20 or [r['episode'] for r in records] != list(range(20)):
                raise ValueError('Only the original fixed first 20 development cases are supported')
            entries, xmls = bundles[arm]
            for record, condition, xml in zip(records, entries[:20], xmls[:20]):
                row, input_files = audit_episode(directory, record, condition, xml, external)
                rows.append(row)
                inputs.extend(input_files)
    never_close = [r for r in rows if r['case_id'].startswith('joint_reps/panda/') and not r['close_command_steps']]
    result = dict(status='PARTIAL_P2_A_OFFLINE_OLD_TRAJECTORIES',
                  new_physical_episodes=0, training_steps=0, input_episodes=len(rows),
                  input_condition_indices=[0,19], panda_joint_reps_never_close_cases=len(never_close),
                  never_close_center_in_pad_prism_cases=sum(r['center_in_pad_prism_pre_action_steps'] > 0 for r in never_close),
                  never_close_without_center_in_pad_prism_cases=sum(r['center_in_pad_prism_pre_action_steps'] == 0 for r in never_close),
                  legal_grasp_opportunities_causally_verified=False, visibility_audit_completed=False,
                  interpretation='Object-center prism is a geometric diagnostic with no object extent, collision-clearance or force-closure guarantee. Neither positive nor negative labels determine legal grasp opportunities.',
                  configuration=dict(pad_normal_tolerance_degrees=15, contact_lift_proxy_height_m=.03,
                                     contact_lift_proxy_consecutive_state_steps=10),
                  episodes=rows)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    write_csv(out/'episodes.csv', rows)
    write_csv(out/'input_files.csv', inputs)
    write_csv(out/'external_files.csv', [dict(absolute_path=r['external_timeseries_path'],
                                             size_bytes=Path(r['external_timeseries_path']).stat().st_size,
                                             sha256=r['timeseries_sha256']) for r in rows])
    print(json.dumps({k:v for k,v in result.items() if k != 'episodes'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'source', 'out', 'external']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    audit(args.root, args.source, args.out, args.external)
