"""Bounded primary-site availability checks; never infer absence from errors."""
import concurrent.futures
import datetime
import json
import re
from pathlib import Path

DEST=Path('/home/users/lemon/strong-baselines-20261112/crosszero')
PAPER_URL='https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ge_CrossZero_Zero-Shot_of_Unseen_Embodiment_Manipulation_via_Object-Centric_Hierarchical_Diffusion_CVPRW_2026_paper.html'


def main():
    import requests
    def get(kind,url):
        result=dict(kind=kind,url=url,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            response=requests.get(url,timeout=(10,20));result['http_status']=response.status_code
            if response.status_code==200:
                if kind=='huggingface_models':result['returned_model_ids']=[x['id'] for x in response.json()]
                else:
                    result['github_links']=re.findall(r'href=[\"\'](https://github.com/[^\"\']+)',response.text)
                    (DEST/'publisher_page.html').write_text(response.text)
            else:result['error_type']='HTTP_ERROR_NOT_PROOF_OF_ABSENCE'
        except requests.RequestException as e:
            result.update(error_type='NETWORK_ERROR_NOT_PROOF_OF_ABSENCE',error=repr(e))
        return result
    calls=[('publisher_page',PAPER_URL),('huggingface_models','https://huggingface.co/api/models?search=CrossZero&limit=30')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda x:get(*x),calls))
    result=dict(checks=rows,prior_direct_huggingface_request='ConnectTimeout, recorded separately; no model absence inferred',
        official_weights_verified=False,search_is_bounded=True)
    DEST.mkdir(exist_ok=True,parents=True);(DEST/'primary_availability.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
