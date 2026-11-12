"""Verify cached versus native raw-image inference on the same saved RGB."""
from baselines.strong.common import ROOT,out,read,write,digest

def main():
    import h5py,numpy as np,torch
    from results.strong_baselines.vqbet.model import Policy
    from results.strong_baselines.vqbet.data import Dataset
    torch.set_num_threads(4)
    checkpoint=ROOT/'vqbet/probe/target_only/bet/latest.pt';policy=Policy(checkpoint);model=policy.model
    dataset=Dataset('target_only',split='validation',limit=1);batch=dataset[0]
    row=next(r for r in dataset.rows if r['episode_id']==batch['episode_id']);t=batch['frame']
    with h5py.File(row['packed_file'],'r') as f:rgb=f[row['episode_id']]['rgb'][t-9:t+1]
    cached=torch.from_numpy(batch['features']).cuda()[None];raw=torch.from_numpy(rgb).cuda()[None]
    language=torch.from_numpy(batch['language']).cuda()[None]
    with torch.inference_mode():
        torch.manual_seed(20261112);a=model(cached,language,None)[0]
        torch.manual_seed(20261112);b=model(raw,language,None)[0]
        x=raw.float().permute(0,1,4,2,3).reshape(10,3,128,128)/255
        features=model.resnet(model.transform(torch.nn.functional.interpolate(x,size=224,mode='nearest'))).flatten(1)
    fd=float((features-cached[0]).abs().max());ad=float((a-b).abs().max())
    assert torch.allclose(features,cached[0],atol=2e-4,rtol=1e-4),(fd,ad)
    assert torch.allclose(a,b,atol=5e-4,rtol=1e-4),(fd,ad)
    dataset.close()
    write(out('vqbet')/'input_acceptance.json',dict(status='PASS',episode_id=batch['episode_id'],frame=t,split='validation',
        checkpoint_sha256=digest(checkpoint),max_feature_absolute_difference=fd,max_native_action_absolute_difference=ad,
        same_rgb_and_language=True,same_code_sampling_seed=True,causal_window_only=True,new_optimizer_updates=0,new_physics_steps=0,
        raw_image_path='unmodified official RGB ResNet path',cache_dtype='float32'))
    print('PASS cached/native raw-image parity',fd,ad)

if __name__=='__main__':main()
