"""Native CFM direction, immutable identities, true qpos and OSC roundtrip."""
import unittest
import inspect
import numpy as np
import torch
import h5py
from .native import SOURCE,ConditionalUnet1D,ConditionalFlowMatcher,normalize_data,unnormalize_data
from .data import Dataset,DIRECTORY
from baselines.strong.common import ROOT,read,digest
from baselines.common.action_adapter import canonical_action

class NativeFlowContract(unittest.TestCase):
    def test_native_source_and_velocity_target(self):
        self.assertEqual(inspect.getfile(ConditionalUnet1D),str(SOURCE/'external/models/unet.py'))
        x0=torch.arange(112,dtype=torch.float32).reshape(1,16,7)/100;x1=-x0
        t,xt,ut=ConditionalFlowMatcher(sigma=0.).sample_location_and_conditional_flow(x0,x1)
        torch.testing.assert_close(ut,x1-x0)
        torch.testing.assert_close(xt,t[:,None,None]*x1+(1-t[:,None,None])*x0)
    def test_split_and_complete_language_preserved(self):
        d=Dataset()
        self.assertEqual(digest(DIRECTORY/'train.npy'),digest(ROOT/'smolvla/data/train.npy'))
        self.assertEqual(digest(DIRECTORY/'validation.npy'),digest(ROOT/'smolvla/data/validation.npy'))
        self.assertEqual(len({int(i) for i in d.anchors[:,0]}),630)
        self.assertEqual(len(d.language),16)
        self.assertEqual(len({tuple(v) for v in d.language.values()}),16)
        batch,_=d.batch(0,1);self.assertTrue(batch['image'].is_contiguous())
        self.assertTrue(all(r['split']=='train' for r in [d.rows[int(i)] for i in d.anchors[:,0]]))
        d.close()
    def test_real_qpos_and_native_terminal_padding(self):
        d=Dataset()
        for domain in ['source','target']:
            j=next(i for i,r in enumerate(d.rows) if r['domain']==domain and r['split']=='train');r=d.rows[j]
            d.anchors=np.array([[j,r['frames']-1]],np.int32);item=d.item(0)
            raw_state=unnormalize_data(item['state'],d.stats['state']);n=6 if domain=='source' else 7
            with h5py.File(r['raw_file'],'r') as f:
                g=f['data'][r['raw_demo']]
                np.testing.assert_allclose(raw_state[:n],g['states'][-1,5:5+n],rtol=1e-6,atol=5e-7)
                self.assertFalse(np.allclose(raw_state[:n],g['obs/joint_states'][-1]))
            np.testing.assert_array_equal(item['action'],np.repeat(item['action'][:1],16,axis=0))
            self.assertEqual(item['trace']['real_future_frames'],1);self.assertEqual(item['image'].shape,(3,96,96))
        d.close()
    def test_action_inverse_and_gripper_sign(self):
        stats={k:np.array(v,np.float32) for k,v in read(DIRECTORY/'statistics.json')['action'].items()}
        a=np.zeros((2,7),np.float32);a[:,6]=[-1,1]
        restored=unnormalize_data(normalize_data(a,stats),stats)
        np.testing.assert_allclose(restored,a,atol=2e-7,rtol=1e-6)
        self.assertEqual(canonical_action(restored[0])[0][6],-1)
        self.assertEqual(canonical_action(restored[1])[0][6],1)

if __name__=='__main__':unittest.main()
