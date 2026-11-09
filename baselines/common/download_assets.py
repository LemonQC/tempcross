"""Acquire pinned public CLIP weights without changing the BARX environment."""
import concurrent.futures,hashlib,json,urllib.request,time
from pathlib import Path
import requests
from huggingface_hub import HfApi
from baselines.common.paths import ASSETS,write_json
from simulation_validation.download_ranges import download

def main():
 info=HfApi().model_info('openai/clip-vit-large-patch14',revision='32bd64288804d66eefd0ccbe215aa642df71cc41',files_metadata=True)
 names={s.rfilename:s for s in info.siblings}
 weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
 wanted=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json',weight]
 rows=[];root=ASSETS/'clip-vit-large-patch14';root.mkdir(parents=True,exist_ok=True)
 for name in wanted:
  if name not in names:continue
  item=names[name];rows.append(dict(filename=name,size=item.size,sha256=(item.lfs or {}).get('sha256'),git_blob=item.blob_id,path=str(root/name)))
 write_json(ASSETS/'clip_manifest.json',dict(repo='openai/clip-vit-large-patch14',revision=info.sha,files=rows))
 urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
 def fetch(row):
  url=f"https://hf-mirror.com/openai/clip-vit-large-patch14/resolve/{info.sha}/{row['filename']}";p=Path(row['path'])
  if row['sha256']:download(url,p,row['size'],row['sha256'],workers=8)
  else:
   s=requests.Session();s.trust_env=False
   for attempt in range(6):
    candidate=url if attempt<3 else url.replace('/'+info.sha+'/','/main/')
    r=s.get(candidate,timeout=(20,90))
    if r.status_code==200:break
    time.sleep(min(2**attempt,15))
   r.raise_for_status();b=r.content
   assert len(b)==row['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob'];p.write_bytes(b)
  print(json.dumps(dict(downloaded=str(p),bytes=p.stat().st_size)),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,rows))
 write_json(ASSETS/'clip_complete.json',dict(status='COMPLETE',revision=info.sha))
if __name__=='__main__':main()
