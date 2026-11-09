"""Construct the pinned official robomimic models without rewriting their cores."""
from collections import OrderedDict
from pathlib import Path
import json
from baselines.common.language import install_core
from baselines.common.paths import VENDOR
METHODS=['bc_gmm','bc_rnn','bc_transformer','diffusion_policy']


def config_for(method,max_steps=48000):
 install_core()
 from robomimic.config import config_factory
 name='diffusion_policy' if method=='diffusion_policy' else 'bc'
 c=config_factory(name)
 with c.unlocked():
  c.train.num_epochs=max_steps;c.experiment.epoch_every_n_steps=1
  c.algo.optim_params.policy.num_train_batches=1;c.algo.optim_params.policy.num_epochs=max_steps
  c.observation.modalities.obs.low_dim=['lang_emb'];c.observation.modalities.obs.rgb=['rgb']
  c.observation.modalities.obs.depth=[];c.observation.modalities.obs.scan=[]
  c.observation.modalities.goal.low_dim=[];c.observation.modalities.goal.rgb=[]
  c.observation.encoder.rgb.core_class='VisualCore'
  c.observation.encoder.rgb.core_kwargs=dict(feature_dimension=64,backbone_class='ResNet18Conv',backbone_kwargs=dict(pretrained=False,input_coord_conv=False),pool_class='SpatialSoftmax',pool_kwargs=dict(num_kp=32,learnable_temperature=False,temperature=1.,noise_std=0.))
  c.observation.encoder.rgb.obs_randomizer_class='CropRandomizer'
  c.observation.encoder.rgb.obs_randomizer_kwargs=dict(crop_height=116,crop_width=116,num_crops=1,pos_enc=False)
  if name=='bc':
   c.algo.gmm.enabled=True;c.algo.gmm.num_modes=5
   if method=='bc_rnn':
    c.algo.rnn.enabled=True;c.algo.rnn.horizon=10;c.algo.rnn.hidden_dim=400;c.algo.rnn.num_layers=2;c.algo.actor_layer_dims=[]
   if method=='bc_transformer':
    c.algo.transformer.enabled=True;c.algo.transformer.context_length=10;c.algo.transformer.embed_dim=512;c.algo.transformer.num_layers=6;c.algo.transformer.num_heads=8;c.algo.actor_layer_dims=[]
    c.algo.optim_params.policy.optimizer_type='adamw';c.algo.optim_params.policy.regularization.L2=.01
  else:
   # The upstream supports both DDPM and DDIM. Keep its UNet, 100 training
   # noise steps and EMA, and predeclare 10-step DDIM for affordable control.
   c.algo.ddpm.enabled=False;c.algo.ddim.enabled=True;c.algo.ddim.num_inference_timesteps=10
 return c


def make_model(method,device='cuda',config=None,max_steps=48000):
 install_core()
 import robomimic
 assert Path(robomimic.__file__).resolve().parent==(VENDOR/'robomimic').resolve(),'An incompatible robomimic was imported before the pinned core'
 from robomimic.algo import algo_factory
 from robomimic.config import config_factory
 import robomimic.utils.obs_utils as ObsUtils
 c=config_for(method,max_steps) if config is None else config_factory('diffusion_policy' if method=='diffusion_policy' else 'bc',dic=config)
 ObsUtils.initialize_obs_utils_with_config(c)
 m=algo_factory(c.algo_name,c,OrderedDict(rgb=(3,128,128),lang_emb=(768,)),7,device)
 return m,c
