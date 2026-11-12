"""Critical boundaries of the new faithful ports, not model-output snapshots."""
import unittest


class StrongDataTest(unittest.TestCase):
    def test_whole_episode_source_target_split_and_no_extra_robot(self):
        from baselines.strong.common import ROOT,PROJECT,read
        rows=read(ROOT/'act/data/manifest.json')
        original={r['episode_id']:r for d in ['source','target'] for r in read(PROJECT/f'configs/datasets/baseline_{d}_manifest.json')}
        self.assertEqual(len(rows),700)
        self.assertEqual(len({r['episode_id'] for r in rows}),700)
        for r in rows:
            self.assertEqual(r['split'],original[r['episode_id']]['split'])
            self.assertEqual(r['embodiment'],'ur5e' if r['domain']=='source' else 'panda')

    def test_validation_cannot_use_training_demonstrations(self):
        from baselines.strong.data import Dataset
        d=Dataset('act','source_target','validation')
        try:
            self.assertEqual(len(d),256)
            for row,frame in d.anchors['target']:
                self.assertEqual(d.rows[int(row)]['split'],'validation')
                self.assertEqual(d.rows[int(row)]['domain'],'target')
        finally:d.close()

    def test_terminal_chunk_mask_has_one_label_and_no_future_image(self):
        import numpy as np
        from baselines.strong.data import Dataset
        d=Dataset('act','target_only',limit=1)
        try:
            row_index=int(d.anchors['target'][0,0]);r=d.rows[row_index]
            d.anchors['target']=np.asarray([[row_index,r['frames']-1]])
            b=d[0]
            self.assertEqual(b['image'].shape,(1,3,128,128))
            self.assertEqual(b['actions'].shape,(100,7))
            self.assertEqual(np.flatnonzero(~b['is_pad']).tolist(),[0])
        finally:d.close()

if __name__=='__main__':unittest.main()
