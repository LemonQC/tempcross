"""Identity isolation and actuator convention tests; native parity is optional."""
import unittest
import numpy as np
from baselines.strong.common import read,digest,OLD
from .data import DIRECTORY,SCALE,encode_action,decode_action,normalized_action

class DataContract(unittest.TestCase):
    def test_actuator_roundtrip_and_gripper_polarity(self):
        a=np.random.default_rng(7).uniform(-1,1,(100,7)).astype(np.float32)
        a[:2,6]=[-1,1]
        native=encode_action(a)
        np.testing.assert_allclose(native[...,:6],a[...,:6]*SCALE,rtol=0,atol=0)
        np.testing.assert_array_equal(native[:2,6],[1,0])
        np.testing.assert_allclose(decode_action(native),a,rtol=0,atol=1e-7)
    def test_complete_identities_and_validation_isolation(self):
        rows=read(DIRECTORY/'manifest.json')
        self.assertEqual(len(rows),700);self.assertEqual(len({r['episode_id'] for r in rows}),700)
        train=np.load(DIRECTORY/'train.npy');val=np.load(DIRECTORY/'validation.npy')
        self.assertEqual(digest(DIRECTORY/'validation.npy'),digest(OLD/'data/target_validation_anchors.npy'))
        self.assertTrue(all(rows[j]['split']=='train' for j in set(train[:,0])))
        self.assertTrue(all(rows[j]['split']=='validation' and rows[j]['domain']=='target' for j in set(val[:,0])))
        self.assertFalse(set(train[:,0]) & set(val[:,0]))
        self.assertTrue(np.all(train[:,1]<np.array([rows[j]['frames'] for j in train[:,0]])))
        prefix=train[:768000]
        from collections import Counter
        counts=Counter((rows[j]['domain'],rows[j]['task']) for j in prefix[:,0])
        self.assertEqual(len(counts),4);self.assertEqual(set(counts.values()),{192000})
    def test_train_statistics_and_unscaled_openness(self):
        s=read(DIRECTORY/'action_statistics.json');rows=read(DIRECTORY/'manifest.json')
        self.assertEqual(s['num_transitions'],sum(r['frames'] for r in rows if r['split']=='train'))
        self.assertEqual(s['num_trajectories'],630);self.assertEqual(s['mask'],[True]*6+[False])
        a=encode_action(np.zeros((3,7),np.float32));out=normalized_action(a,s)
        np.testing.assert_array_equal(out[:,6],a[:,6])

if __name__=='__main__':unittest.main()
