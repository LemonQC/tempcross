"""Local materialization of robomimic's frozen CLIP-L/14 language input."""
import functools,sys,types,os
from baselines.common.paths import ASSETS,VENDOR,DEPS
_MODEL=None;_TOKENIZER=None

@functools.lru_cache(maxsize=512)
def encode(text):
 global _MODEL,_TOKENIZER
 install_core()
 import torch
 from transformers import CLIPTextModelWithProjection,AutoTokenizer
 if _MODEL is None:
  path=str(ASSETS/'clip-vit-large-patch14')
  _MODEL=CLIPTextModelWithProjection.from_pretrained(path,local_files_only=True).eval().requires_grad_(False)
  _TOKENIZER=AutoTokenizer.from_pretrained(path,local_files_only=True)
 tokens=_TOKENIZER(text=text,add_special_tokens=True,max_length=25,padding='max_length',return_attention_mask=True,return_tensors='pt')
 with torch.no_grad():return _MODEL(**tokens)['text_embeds'][0].detach().cpu()

def install_core():
 os.environ['USE_TF']='0'
 sys.path[:0]=[str(VENDOR),str(DEPS)]
 # Upstream lang_utils downloads at import time. This transport shim keeps the
 # same public CLIP projection/tokenization but uses the hash-checked local copy.
 mod=types.ModuleType('robomimic.utils.lang_utils');mod.LANG_EMB_OBS_KEY='lang_emb'
 mod.get_lang_emb=lambda text:None if text is None else encode(text)
 mod.get_lang_emb_shape=lambda:[768]
 sys.modules['robomimic.utils.lang_utils']=mod
