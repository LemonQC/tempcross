"""Official ACTPolicy with the recorded dimensional I/O patch."""
import sys
from baselines.strong.common import ROOT,read,out

def make():
    repair=read(out('act')/'technical_repair.json')
    assert str(ROOT)==repair['corrected_server_root'],'Set STRONG_BASELINE_RUN_ROOT to the corrected ACT root; original attempt remains separate.'
    sys.path[:0]=[str(ROOT/'vendor/act_port'),str(ROOT/'vendor/act_port/detr')]
    from policy import ACTPolicy
    previous=sys.argv
    try:
        sys.argv=['act','--ckpt_dir','unused','--policy_class','ACT','--task_name','robocasa_x','--seed','7','--num_epochs','0']
        return ACTPolicy(dict(lr=1e-5,lr_backbone=1e-5,num_queries=100,kl_weight=10,
                              hidden_dim=512,dim_feedforward=3200,backbone='resnet18',enc_layers=4,dec_layers=7,nheads=8,camera_names=['rgb']))
    finally:sys.argv=previous

class Policy:
    def __init__(self,checkpoint):
        import torch,numpy as np
        self.model=make();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model.load_state_dict(saved['model']);self.model.eval();self.model.requires_grad_(False)
        self.language=read(ROOT/'act/data/language.json');self.stats=saved['stats'];self.mean=np.asarray(self.stats['mean']);self.std=np.asarray(self.stats['std'])
        self.reset();self.runtime_metadata=dict(model='ACTPolicy_IO_port_with_declared_final_decoder_repair',
            decoder_output='last_of_7',technical_repair_number=1,temporal_aggregation=True,queries=100,k=.01,information=['RGB','frozen CLIP instruction'])
    def reset(self):self.forecasts=[];self.t=0
    def predict_action(self,obs):
        import cv2,numpy as np,torch
        image=cv2.resize(obs['rgb'],(128,128),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(image.transpose(2,0,1).copy()).float().cuda()[None,None]/255
        lang=torch.tensor(self.language[obs['instruction']],device='cuda')[None]
        with torch.inference_mode():future=self.model(lang,x)[0].cpu().numpy()
        self.forecasts.append((self.t,future));self.forecasts=[(t,f) for t,f in self.forecasts if self.t-t<len(f)]
        candidates=np.asarray([f[self.t-t] for t,f in self.forecasts]);weights=np.exp(-.01*np.arange(len(candidates)));weights/=weights.sum()
        action=(candidates*weights[:,None]).sum(0)*self.std+self.mean
        self.t+=1
        return dict(action=action,representation='canonical_osc',new_query=True,native_query=future*self.std+self.mean)
