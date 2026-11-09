"""Existing neural policies behind the common RGB/language boundary."""
from collections import deque
import json,sys
import cv2,numpy as np
from baselines.common.policy_interface import BasePolicy
from baselines.common.paths import SOURCE


class RobomimicPolicy(BasePolicy):
 def __init__(self,path):self.load_checkpoint(path)
 def load_checkpoint(self,path):
  import torch
  from baselines.common.model_factory import make_model
  p=torch.load(path,map_location='cpu');self.method=p['method'];self.model,self.config=make_model(self.method,config=p['config']);self.model.deserialize(p['model']);self.model.set_eval()
  self.context=10 if self.method=='bc_transformer' else 2 if self.method=='diffusion_policy' else 1
  self.history=deque(maxlen=self.context);self.query=None
  if self.method=='diffusion_policy':
   original=self.model._get_action_trajectory
   def capture(*args,**kwargs):
    result=original(*args,**kwargs);self.query=result[0].detach().cpu().numpy().copy();return result
   self.model._get_action_trajectory=capture
  self.reset()
 def reset(self):self.model.reset();self.history.clear();self.query=None
 def predict_action(self,observation):
  import torch
  from baselines.common.language import encode
  assert set(observation)=={'rgb','instruction'}
  image=cv2.resize(observation['rgb'],(128,128),interpolation=cv2.INTER_AREA)
  self.history.append(np.transpose(image,(2,0,1)).astype(np.float32)/255.)
  images=[self.history[0]]*(self.context-len(self.history))+list(self.history)
  language=encode(observation['instruction'])
  if self.context==1:
   obs=dict(rgb=torch.as_tensor(images[-1][None],device='cuda'),lang_emb=language[None].to('cuda'))
  else:
   obs=dict(rgb=torch.as_tensor(np.stack(images)[None],device='cuda'),lang_emb=language[None,None].expand(1,self.context,-1).to('cuda'))
  self.query=None
  with torch.inference_mode():action=self.model.get_action(obs).detach().float().cpu().numpy()[0]
  return dict(action=action,representation='canonical_osc',native_query=self.query if self.query is not None else action[None],new_query=self.query is not None if self.method=='diffusion_policy' else True)


class BarxPolicy(BasePolicy):
 def __init__(self,path):self.load_checkpoint(path)
 def load_checkpoint(self,path):
  sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
  from experiments.robot.robocasa_x.evaluate import GenerateConfig
  from experiments.robot.robot_utils import get_model
  from experiments.robot.robocasa_x.utils import patch_model_for_generation
  self.cfg=GenerateConfig(pretrained_checkpoint=str(path),unnorm_key='mg_pnp_lite',act_horizon=8)
  self.model=patch_model_for_generation(get_model(self.cfg));self.queue=deque();self.reset()
 def reset(self):self.queue.clear()
 def predict_action(self,observation):
  assert set(observation)=={'rgb','instruction'}
  query=None
  if not self.queue:
   from experiments.robot.robot_utils import get_action
   from barx.names import canonicalize_prediction_keys
   result=canonicalize_prediction_keys(get_action(self.cfg,self.model,dict(full_image=[observation['rgb']]),observation['instruction'],aux_task_types=None,processor=None))
   query=np.asarray(result['action']).copy();assert query.shape==(8,7);self.queue.extend(query)
  action=self.queue.popleft().copy()
  return dict(action=action,representation='barx_unnormalized',native_query=query if query is not None else action[None],new_query=query is not None)
