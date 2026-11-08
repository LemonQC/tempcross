"""Verify the installed BARX runtime without inventing a manipulation task."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

EXPECTED={
 'torch':'2.2.0+cu121','torchvision':'0.17.0+cu121','torchaudio':'2.2.0+cu121',
 'mujoco':'3.1.1','numpy':'1.23.5','scipy':'1.13.1','transformers':'4.40.1',
 'timm':'0.9.10','h5py':'3.13.0','tensorflow':'2.15.0','vq-bet':'0.1.0',
}

def verify(out):
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    result={'status':'RUNNING','checks':{},'manipulation_episodes':0}
    try:
        versions={name:importlib.metadata.version(name) for name in EXPECTED}
        assert versions==EXPECTED,(versions,EXPECTED)
        result['versions']=versions
        import torch
        assert torch.cuda.is_available(),'CUDA unavailable in the BARX environment'
        x=torch.randn(512,512,device='cuda');y=x@x.T;torch.cuda.synchronize()
        assert torch.isfinite(y).all().item()
        result['checks']['cuda_matmul']=True
        result['gpu']=torch.cuda.get_device_name(0)
        result['torch_cuda_build']=torch.version.cuda
        result['bf16_supported']=torch.cuda.is_bf16_supported()
        import flash_attn
        from flash_attn import flash_attn_func
        assert flash_attn.__version__.split('+')[0]=='2.5.5'
        q=torch.randn(1,32,4,32,device='cuda',dtype=torch.float16)
        k=torch.randn_like(q);v=torch.randn_like(q)
        with torch.inference_mode():
            actual=flash_attn_func(q,k,v,dropout_p=0.)
            reference=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        torch.cuda.synchronize()
        error=(actual-reference).abs().max().item()
        assert torch.isfinite(actual).all().item() and error<.01,error
        result['checks']['flash_attention_cuda_kernel']=True
        result['flash_attention_version']=flash_attn.__version__
        result['flash_attention_max_abs_difference']=error
        import mujoco
        from OpenGL import GL
        context=mujoco.GLContext(64,64);context.make_current()
        try:
            renderer=GL.glGetString(GL.GL_RENDERER)
            assert renderer is not None
            result['egl_renderer']=renderer.decode()
            result['checks']['mujoco_egl_context']=True
        finally:context.free()
        import robocasa
        import robosuite
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
        assert 'UR5eOmron' in REGISTERED_ROBOTS and 'PandaOmron' in REGISTERED_ROBOTS
        result['checks']['official_robot_registration']=True
        result['robocasa_module']=str(Path(robocasa.__file__).resolve())
        result['robosuite_module']=str(Path(robosuite.__file__).resolve())
        result['status']='RUNTIME_VERIFIED'
    except Exception as error:
        result.update(status='RUNTIME_FAILED',error=repr(error),traceback=traceback.format_exc())
    output=Path(out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result['status']=='RUNTIME_VERIFIED'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    raise SystemExit(0 if verify(a.out) else 1)
