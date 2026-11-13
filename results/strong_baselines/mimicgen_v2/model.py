"""Exact existing official BC-Transformer-GMM architecture; learned deployment."""
from baselines.strong.common import ROOT,out,read,digest,setup

def make():
    setup()
    import torch
    from baselines.common.model_factory import make_model
    for path,sha in read(out('mimicgen_v2')/'official_source.json')['sha256'].items():assert digest(path)==sha,path
    control=read(out('mimicgen_v2')/'reused_control.json')
    assert digest(control['config'])==control['config_sha256'] and digest(control['initial_weights'])==control['initial_weights_sha256']
    config=read(control['config']);model,cfg=make_model('bc_transformer',config=config)
    initial=torch.load(control['initial_weights'],map_location='cpu',weights_only=False)
    named={k:v for k,v in model.nets.named_parameters() if v.requires_grad};assert set(initial)==set(named)
    with torch.no_grad():
        for name,value in named.items():value.copy_(initial[name].to(value.device))
    assert sum(v.numel() for v in model.nets.parameters())==30574315
    return model,cfg

class Policy:
    def __init__(self,checkpoint):
        setup()
        from baselines.common.policies import RobomimicPolicy
        self.policy=RobomimicPolicy(checkpoint);self.runtime_metadata=dict(self.policy.runtime_metadata,
            scope='MimicGen v2 unequal Panda augmentation component',hand_coded_deployment_rules=False,information=['RGB','instruction'])
    def reset(self):self.policy.reset()
    def predict_action(self,obs):return self.policy.predict_action(obs)
