import unittest
from method_v0.contracts import (Edge,EvidenceGraph,continuation_contract,witnessed_task_planner,
    physical_feasibility_interval,symbolic_nonidentifiability_witness)


class ContractsTest(unittest.TestCase):
    def test_local_contact_does_not_imply_supported_continuation(self):
        graph=EvidenceGraph({'contact-a':[], 'contact-b':[Edge('continue','goal',3,'real-path')]},{'goal'})
        self.assertEqual(physical_feasibility_interval(continuation_contract(graph,'contact-a',5)),(0,1))
        self.assertEqual(physical_feasibility_interval(continuation_contract(graph,'contact-b',5)),(1,1))

    def test_goal_budget_requires_whole_path_not_just_first_edge(self):
        graph=EvidenceGraph({'start':[Edge('grasp','held',2,'w1')],
                             'held':[Edge('place','goal',5,'w2')]},{'goal'})
        for budget in range(12):
            expected=frozenset({'grasp'}) if budget>=7 else frozenset()
            self.assertEqual(continuation_contract(graph,'start',budget),expected)
            self.assertEqual(witnessed_task_planner(graph,'start',budget),expected)

    def test_shared_effect_label_is_not_a_state_join_certificate(self):
        graph=EvidenceGraph({'episode-a/contact':[],
            'episode-b/contact':[Edge('finish','goal',1,'actual-b')]},{'goal'})
        self.assertFalse(continuation_contract(graph,'episode-a/contact',10))
        self.assertTrue(continuation_contract(graph,'episode-b/contact',10))

    def test_failed_logged_option_cannot_label_unexecuted_option(self):
        witness=symbolic_nonidentifiability_witness()
        for key,value in witness['observations'].items():
            self.assertEqual(witness['world_a'][key],value)
            self.assertEqual(witness['world_b'][key],value)
        self.assertNotEqual(witness['feasible_a'],witness['feasible_b'])
        self.assertIn('NOT_PHYSICAL',witness['kind'])

    def test_no_witness_no_certified_negative(self):
        self.assertEqual(physical_feasibility_interval([]),(0,1))
        self.assertEqual(physical_feasibility_interval([],certified_impossible=True),(0,0))
        with self.assertRaises(ValueError):physical_feasibility_interval(['actual'],True)

    def test_invalid_edges_cannot_create_phantom_success(self):
        for edge in [Edge('a','goal',0,'w'),Edge('a','goal',1,'')]:
            with self.assertRaises(ValueError):EvidenceGraph({'root':[edge]},{'goal'}).validate()

    def test_both_searches_support_all_real_alternatives(self):
        graph=EvidenceGraph({'s':[Edge('a','g1',2,'w1'),Edge('b','g2',4,'w2'),Edge('c','unknown',1,'w3')]},{'g1','g2'})
        self.assertEqual(continuation_contract(graph,'s',4),frozenset({'a','b'}))
        self.assertEqual(witnessed_task_planner(graph,'s',4),frozenset({'a','b'}))


if __name__=='__main__':unittest.main()
