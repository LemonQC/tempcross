"""Read-only verification of the completed evidence; no policy or simulator calls."""
import csv
from collections import Counter
from pathlib import Path
import sys

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from common import OUT, ROOT, digest, read, write


def main():
    execution = read(ROOT / 'execution.json')
    assert execution['status'] == 'COMPLETE'
    rows = list(csv.DictReader((OUT / 'episodes.csv').open()))
    new = [r for r in rows if r['new_physical_run'] == 'True']
    assert len(rows) == 640 and len(new) == 160
    cells = read(OUT / 'results.json')['cells']
    inventory = []
    official = Counter()
    no_close_contact_flags = []
    for r in rows:
        if r['issued_close'] == 'False' and r['bilateral_contact_after_close'] == 'True':
            no_close_contact_flags.append(dict(label=r['label'], task=r['task'],
                                               robot=r['robot'], condition_id=r['condition_id']))
    for r in new:
        path = Path(r['telemetry_path'])
        with h5py.File(path, 'r') as f:
            task_success = bool(np.any(f['baseline/post_action_placement'][:] &
                                       f['baseline/post_action_retreat'][:]))
            assert task_success == (r['success'] == 'True')
            closed = np.flatnonzero(f['actions/environment'][:, 6] > 0)
            assert bool(len(closed)) == (r['issued_close'] == 'True')
            assert (str(int(closed[0])) if len(closed) else '') == r['first_close']
        official[r['task'], r['robot']] += task_success
        for kind, raw in [('telemetry', path), ('contact_log', path.with_suffix('.jsonl')),
                          ('video', Path(r['video_path']))]:
            assert raw.is_absolute() and raw.is_file() and raw.stat().st_size > 0
            inventory.append(dict(kind=kind, path=str(raw), bytes=raw.stat().st_size,
                                  sha256=digest(raw)))
    for cell in cells:
        rr = [r for r in rows if (r['label'], r['task'], r['robot']) ==
              (cell['label'], cell['task'], cell['robot'])]
        assert len(rr) == cell['episodes']
        assert sum(r['success'] == 'True' for r in rr) == cell['successes']
        assert dict(Counter(r['failure_stage'] for r in rr)) == cell['failure_counts']
        if cell['new_physical_run']:
            assert len(rr) == 40
            assert sorted(int(r['official_condition_index']) for r in rr) == list(range(60, 100))
            assert official[cell['task'], cell['robot']] == cell['successes']
    # This is a semantic check, not a change to the frozen stage classifier.
    # If nonempty, after-close contact flags require a disclosed correction;
    # the NO_CLOSE_COMMAND priority and official task scores are unaffected.
    write(OUT / 'raw_rollout_index.json', inventory)
    write(OUT / 'delivery_checks.json', dict(
        status='PASS' if not no_close_contact_flags else 'CONTACT_FIELD_REQUIRES_DISCLOSURE',
        checked_new_rollouts=len(new), checked_all_comparison_rows=len(rows),
        new_raw_files=len(inventory), raw_bytes=sum(r['bytes'] for r in inventory),
        official_scores_recomputed_from_saved_placement_and_retreat=True,
        exact_per_cell_conditions_60_to_99=True,
        cell_counts_and_failure_counts_match_per_episode_rows=True,
        after_close_flags_on_episodes_without_close=no_close_contact_flags,
        policy_calls=0, physics_steps=0))
    print('Verified 160 new saved rollouts and 640 comparison rows; no policy/physics calls.')


if __name__ == '__main__':
    main()
