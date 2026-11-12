"""Reproduce the archived TF32/layout diagnosis, without training or physics."""
from pathlib import Path
from baselines.strong.common import ROOT,out,read,write

def main():
    import h5py,numpy as np,torch
    from torchvision.models import resnet18,ResNet18_Weights
    from results.strong_baselines.vqbet.model import Policy
    prior=Path(read(out('vqbet')/'technical_repair.json')['prior_attempt'])
    policy=Policy(prior/'probe/target_only/bet/latest.pt');model=policy.model
    rows=read(prior/'data/manifest.json');j,t=map(int,np.load(prior/'data/vq_validation.npy')[0]);r=rows[j]
    with h5py.File(r['packed_file'],'r') as f:rgb=f[r['episode_id']]['rgb'][t-9:t+1]
    cached=np.load(prior/'data/features'/f'{j:04d}.npy',mmap_mode='r')[t-9:t+1]
    encoder=torch.nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]).cuda().eval()
    equal=all(torch.equal(v,encoder.state_dict()[k]) for k,v in model.resnet.state_dict().items());results={};vectors={}
    with torch.inference_mode():
        for tf32 in [True,False]:
            torch.backends.cudnn.allow_tf32=tf32
            for layout in ['channels_last','contiguous']:
                for batch in [10,128]:
                    x=torch.from_numpy(rgb).cuda().permute(0,3,1,2).float()/255
                    if batch==128:x=torch.cat([x,x[:1].expand(118,-1,-1,-1)],0)
                    x=x.contiguous(memory_format=torch.channels_last if layout=='channels_last' else torch.contiguous_format)
                    y=model.resnet(model.transform(torch.nn.functional.interpolate(x,size=224,mode='nearest'))).flatten(1)[:10].cpu().numpy()
                    key=f'{tf32}_{layout}_{batch}';vectors[key]=y
                    results[key]=dict(cache_max_abs=float(np.max(np.abs(y-cached))),cache_relative_l2=float(np.linalg.norm(y-cached)/np.linalg.norm(y)),strides=list(x.stride()))
        reference=vectors['False_contiguous_10']
        for key,y in vectors.items():results[key]['precise_reference_max_abs']=float(np.max(np.abs(y-reference)))
    torch.backends.cudnn.allow_tf32=False
    write(out('vqbet')/'numeric_diagnosis_reproduction.json',dict(pretrained_weights_exactly_equal=equal,comparisons=results,new_optimizer_updates=0,new_physics_steps=0))
    print(results)

if __name__=='__main__':main()
