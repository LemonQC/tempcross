"""Official residual VQ + BeT; declared language-prefix I/O port only."""
import sys
from collections import deque
from baselines.strong.common import ROOT,out,read,digest

def imports():
    import torch
    assert str(ROOT)==read(out('vqbet')/'technical_repair.json')['corrected_server_root'],'Use corrected STRONG_BASELINE_RUN_ROOT; preserve original attempt.'
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    for path,record in read(out('vqbet')/'official_source.json')['files'].items():
        assert digest(ROOT/'vendor/vq_bet_port'/path)==record['port_sha256'],path
    sys.path.insert(0,str(ROOT/'vendor/vq_bet_port'))
    from vqvae.vqvae import VqVae
    from vq_behavior_transformer import BehaviorTransformer,GPT,GPTConfig
    return VqVae,BehaviorTransformer,GPT,GPTConfig

def make_vq(training=False):
    VqVae,*_=imports()
    return VqVae(input_dim_h=1,input_dim_w=7,n_latent_dims=512,vqvae_n_embed=16,vqvae_groups=2,eval=not training,device='cuda',encoder_loss_multiplier=1.,act_scale=1.)

def make(vq):
    _,BehaviorTransformer,GPT,GPTConfig=imports()
    model=BehaviorTransformer(obs_dim=1024,act_dim=7,goal_dim=1024,
        gpt_model=GPT(GPTConfig(input_dim=1024,block_size=30,n_layer=6,n_head=6,n_embd=120,output_dim=256,dropout=.1)),
        vqvae_model=vq,offset_loss_multiplier=100,secondary_code_multiplier=.5,gamma=2.,obs_window_size=10,act_window_size=1,
        sequentially_select=False,visual_input=True,finetune_resnet=False).cuda()
    model.resnet.eval()
    return model

class Policy:
    def __init__(self,checkpoint):
        import torch
        vq=make_vq();saved=torch.load(checkpoint,map_location='cpu',weights_only=False);vq.load_state_dict(saved['vqvae'])
        self.model=make(vq);self.model.load_state_dict(saved['model']);self.model.eval();self.model.requires_grad_(False)
        self.language=read(ROOT/'vqbet/data/language.json');self.reset()
        self.runtime_metadata=dict(model='official_VQ_BeT_language_IO_port',obs_window=10,action_window=1,codebooks=2,codebook_size=16,
            visual_encoder='frozen ResNet18',stochastic_codes=True,information=['RGB','frozen CLIP instruction'],temporal_aggregation=False)
    def reset(self):self.history=deque(maxlen=10)
    def predict_action(self,obs):
        import cv2,numpy as np,torch
        rgb=cv2.resize(obs['rgb'],(128,128),interpolation=cv2.INTER_AREA)
        x=torch.tensor(rgb.transpose(2,0,1).copy(),device='cuda').float()[None]/255
        with torch.inference_mode():
            x=torch.nn.functional.interpolate(x,size=224,mode='nearest');feat=self.model.resnet(self.model.transform(x)).flatten(1)[0]
            self.history.append(feat);padded=[self.history[0]]*(10-len(self.history))+list(self.history)
            features=torch.stack(padded)[None];lang=torch.tensor(self.language[obs['instruction']],device='cuda')[None,None]
            action=self.model(features,lang,None)[0][-1].cpu().numpy()
        return dict(action=action[0],representation='canonical_osc',new_query=True,native_query=action)
