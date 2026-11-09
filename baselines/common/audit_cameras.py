"""Verify actual demonstration ep_meta calibration, not default camera names."""
import json,sys
import h5py,numpy as np
from baselines.common.paths import ROOT,DATA,PROJECT,SOURCE,write_json

def main():
 sys.path.insert(0,str(SOURCE))
 from robocasa.utils.camera_utils import BARX_CAMERA_CONFIGS
 expected=BARX_CAMERA_CONFIGS['barx_panda_agentview'];records=[]
 for path in sorted((DATA/'raw_target/human/PandaOmron').glob('*/demo_gentex_im320.hdf5')):
  with h5py.File(path) as f:
   args=json.loads(f['data'].attrs['env_args']);kw=args['env_kwargs'];assert kw['camera_names']==['robot0_agentview_left'] and kw['camera_widths']==320 and kw['camera_heights']==180
   for name,g in sorted(f['data'].items()):
    camera=json.loads(g.attrs['ep_meta'])['cam_configs']['robot0_agentview_left'];q=np.asarray(camera['quat']);qe=np.asarray(expected['quat']);q=q/np.linalg.norm(q);qe=qe/np.linalg.norm(qe)
    assert np.allclose(camera['pos'],expected['pos'],atol=1e-8) and np.isclose(abs(q@qe),1.,atol=1e-8) and camera['parent_body']==expected['parent_body'] and float(camera['camera_attribs']['fovy'])==float(expected['camera_attribs']['fovy'])
    assert g['obs/agentview_rgb'].shape[1:]==(180,320,3)
    records.append(dict(file=str(path),demo=name,training_camera='robot0_agentview_left',evaluation_camera='barx_panda_agentview',actual_ep_meta_calibration_matches=True))
 assert len(records)==100
 result=dict(status='VERIFIED_ALL_100_TARGET_CAMERA_CALIBRATIONS',new_training_or_physics=False,actual_episode_metadata_used=True,default_camera_name_alone_is_not_calibration_evidence=True,expected_base_relative_camera=expected,rgb_resolution=[180,320,3],records=records)
 write_json(ROOT/'camera_calibration_audit.json',result);write_json(PROJECT/'results/baselines/camera_calibration_audit.json',result);print('Verified all100 target ep_meta camera calibrations and image resolutions')
if __name__=='__main__':main()
