import unittest
from method_v0.common import OUT, read
from method_v0.organize import load_rows


class EvidenceTest(unittest.TestCase):
    def test_all_registered_primary_records_are_present_no_double_count(self):
        rows=load_rows('factual_continuations')
        self.assertEqual(len(rows),1617)
        self.assertEqual(len({r['uid'] for r in rows}),len(rows))
        self.assertFalse(any(r['certified_physical_negative'] for r in rows))

    def test_official_goal_requires_joint_placement_and_retreat(self):
        rows=load_rows('factual_continuations')
        self.assertTrue(all(('11' in r['continuation_effect_word'])==r['official_success'] for r in rows))

    def test_existing_opposite_outcomes_are_state_verified_not_spatially_matched(self):
        groups=read(OUT/'matched_continuation_witnesses.json')
        self.assertEqual(len(groups),5)
        for r in groups:
            self.assertEqual(r['numeric_fields_compared'],149)
            self.assertEqual(r['max_numeric_error'],0)
            self.assertTrue(r['prefix_actions_and_states_exact'])
            self.assertTrue(all(r['recorded_rng_checks'].values()))
        self.assertEqual(sum(r['opposite_factual_outcomes'] for r in groups),4)

    def test_no_unseen_test_or_trained_algorithm_claim(self):
        s=read(OUT/'summary.json')
        self.assertFalse(s['independent_unseen_test'])
        self.assertFalse(s['learned_policy_or_new_mechanism_implemented'])
        self.assertEqual(s['training_updates'],0)
        self.assertEqual(s['new_physics_steps'],0)
        self.assertEqual(s['old_baseline_reruns'],0)


if __name__=='__main__':unittest.main()
