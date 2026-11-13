"""Frozen upstream prerequisites only; never train or run task physics."""
import json
import sys
import traceback
from pathlib import Path

UPSTREAM=Path('/home/users/lemon/baseline-benchmark/vendor/KITE')
DEST=Path('/home/users/lemon/strong-baselines-20261112/kite/runtime_audit.json')


def main():
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0,str(UPSTREAM))
    result=dict(status='RUNNING',torch=torch.__version__,cuda_build=torch.version.cuda,
        actual_training_updates=0,task_physical_episodes=0,task_success_rate=None)
    try:
        from pytorch3d.ops import sample_farthest_points
        cloud=torch.arange(192,device='cuda',dtype=torch.float32).reshape(1,64,3)/192
        points,indices=sample_farthest_points(cloud,K=8)
        assert points.shape==(1,8,3) and torch.isfinite(points).all()
        result['native_cuda_fps']='PASS'
        from paper_config import runtime_config
        from experiment.high_level_policy.contact_encoding import _load_cotrain_probe_policy,_PairedBottleneckEncoderAdapter
        from robot_model.robot_model import create_robot_model
        cfg=runtime_config(OmegaConf.load(UPSTREAM/'conf/kite.yaml'))
        metadata=torch.load(UPSTREAM/'checkpoints/decoder/dataset_meta.pt',map_location='cpu',weights_only=False)
        bank,_,_= _load_cotrain_probe_policy(cfg,checkpoint_path=str(UPSTREAM/'checkpoints/decoder/release.pt'),
            device=torch.device('cpu'),prefer_ema=True,cotrain_meta=metadata)
        result['released_decoder_bank']=metadata['manipulators']
        result['released_action_dimensions']={k:v['action_dim'] for k,v in metadata['per_manipulator'].items()}
        result['has_ur5e_or_panda_robotiq_only_head']=any(k in metadata['manipulators'] for k in ['ur5e','panda','robotiq_2f85','tabletop_franka_robotiq_right'])
        model=create_robot_model('robotiq_gripper_stick',device=torch.device('cpu'),num_points=128,load_surface_assets=False)
        q=model.get_canonical_q();p,n=model.get_surface_points_normals(q)
        contacts=torch.cat([p[:6],n[:6]],dim=-1)[None]
        head=bank.head_for(metadata['manipulators'][0])
        encoder=_PairedBottleneckEncoderAdapter(head.contact_only_encoder).eval()
        with torch.inference_mode():latent=encoder(contacts)
        assert latent.shape==(1,256) and torch.isfinite(latent).all()
        result.update(status='PASS_FROZEN_PREREQUISITES_ONLY',latent_shape=list(latent.shape),
            physical_synthetic_geometry_points=contacts.tolist(),frozen_latent=latent.tolist(),
            target_decoder_mapping_validated=False,source_task_intents_extracted=False,
            note='Native composite-hand geometry and encoder only. Does not establish UR5e/Panda deployment.')
    except BaseException as e:
        result.update(status='FAILED_PREREQUISITE',error=repr(e),traceback=traceback.format_exc())
    DEST.parent.mkdir(parents=True,exist_ok=True);DEST.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['physical_synthetic_geometry_points','frozen_latent','traceback']},indent=2))


if __name__=='__main__':main()
