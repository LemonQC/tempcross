"""Verify the actual GitHub files before allowing physical benchmark launches."""
import argparse,json,urllib.request
from baselines.common.paths import ROOT,PROJECT,digest,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 assert len(a.commit)==40
 for name in ['configs/baselines/protocol.json','configs/evaluation/baseline_conditions_60_99.json','configs/datasets/baseline_target_manifest.json','configs/datasets/baseline_source_manifest.json']:
  url=f'https://api.github.com/repos/LemonQC/tempcross/contents/{name}?ref={a.commit}'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept':'application/vnd.github.raw+json','User-Agent':'tempcross-baseline-verification'}),timeout=30) as response:raw=response.read()
  assert raw==(PROJECT/name).read_bytes(),f'GitHub differs from local freeze: {name}'
 write_json(ROOT/'frozen_publication.json',dict(remote_commit=a.commit,github_readback_verified=True,protocol_sha256=digest(PROJECT/'configs/baselines/protocol.json'),condition_manifest_sha256=digest(PROJECT/'configs/evaluation/baseline_conditions_60_99.json')))
 print('GitHub read-back matches all four frozen files',a.commit)
if __name__=='__main__':main()
