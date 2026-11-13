"""Meaningful native adapter checks; no policy model or physical rollout."""
import unittest
import numpy as np
import torch
import h5py
from .runtime import config,processors,prepare_batch
from .data import Dataset,DIRECTORY
from baselines.strong.common import read,digest,OLD
from baselines.common.action_adapter import canonical_action

class NativeDataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pre,cls.post=processors(config('cpu'))
    def test_same_split_and_no_reference_inputs(self):
        data=Dataset();train=set(map(int,data.anchors[:,0]));val=np.load(DIRECTORY/'validation.npy')
        self.assertEqual(len(train),630);self.assertTrue(all(data.rows[j]['split']=='train' for j in train))
        self.assertFalse(train & set(map(int,val[:,0])))
        self.assertEqual(digest(DIRECTORY/'validation.npy'),digest(OLD/'data/target_validation_anchors.npy'))
        batch,_=data.batch(0,4)
        self.assertEqual(set(batch),{'observation.images.camera1','observation.state','action','actions_id_pad','task'})
        data.close()
    def test_actual_qpos_not_misnamed_velocity(self):
        data=Dataset()
        for domain in ['source','target']:
            j=next(i for i,r in enumerate(data.rows) if r['domain']==domain and r['split']=='train')
            data.anchors=np.array([[j,0]],np.int32);r=data.rows[j];item=data.item(0);n=6 if domain=='source' else 7
            with h5py.File(r['raw_file'],'r') as f:
                g=f['data'][r['raw_demo']];q=g['states'][0,5:5+n].astype(np.float32);v=g['obs/joint_states'][0]
                np.testing.assert_array_equal(item['state'][:n],q)
                self.assertFalse(np.allclose(q,v));np.testing.assert_array_equal(item['state'][7:],g['obs/gripper_states'][0].astype(np.float32))
            if domain=='source':self.assertEqual(item['state'][6],0)
        data.close()
    def test_native_preprocessing_keeps_real_tail_padding(self):
        data=Dataset();j=next(i for i,r in enumerate(data.rows) if r['domain']=='target' and r['split']=='train')
        data.anchors=np.array([[j,data.rows[j]['frames']-1]],np.int32)
        raw,_=data.batch(0,1);batch=prepare_batch(self.pre,raw)
        self.assertEqual(batch['action'].shape,(1,50,7));self.assertEqual(batch['actions_id_pad'].sum().item(),49)
        self.assertFalse(batch['actions_id_pad'][0,0]);self.assertTrue(batch['actions_id_pad'][0,1:].all())
        data.close()
    def test_native_action_roundtrip_and_close_sign(self):
        data=Dataset();raw,_=data.batch(0,1);raw['action'][0,0,6]=-1;raw['action'][0,1,6]=1
        original=raw['action'].clone();batch=prepare_batch(self.pre,raw)
        restored=self.post(batch['action']);torch.testing.assert_close(restored,original,rtol=1e-6,atol=1e-6)
        self.assertLess(float(restored[0,0,6]),0);self.assertGreater(float(restored[0,1,6]),0)
        self.assertEqual(canonical_action(restored[0,0].numpy())[0][6],-1)
        self.assertEqual(canonical_action(restored[0,1].numpy())[0][6],1)
        data.close()

if __name__=='__main__':unittest.main()
