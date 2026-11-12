"""CPU-only evidence that RVQ learned and stayed frozen during native BeT."""
from baselines.strong.common import ROOT,out,read,write,digest

def optimizer_steps(state):
    return sorted({int(v['step']) for v in state['state'].values() if 'step' in v})

def main():
    import torch
    folder=out('vqbet');cfg=read(folder/'protocol.json')
    assert str(ROOT)==read(folder/'technical_repair.json')['corrected_server_root']
    groups=[]
    for group in cfg['groups']:
        files=[folder/(group+'_'+stage+'_selection.json') for stage in ['vqvae','bet']]
        if not all(p.exists() for p in files):continue
        ae,be=[read(p) for p in files]
        for selection in [ae,be]:assert digest(selection['checkpoint'])==selection['sha256']
        a=torch.load(ae['checkpoint'],map_location='cpu',weights_only=False)
        b=torch.load(be['checkpoint'],map_location='cpu',weights_only=False)
        for section in ['encoder','decoder','vq_embedding']:
            assert set(a['vqvae'][section])==set(b['vqvae'][section])
            assert all(torch.equal(v,b['vqvae'][section][k]) for k,v in a['vqvae'][section].items()),section
        ema=[]
        for key,value in a['vqvae']['vq_embedding'].items():
            assert torch.isfinite(value).all()
            if key.endswith('cluster_size'):
                assert float(value.sum())>0
                ema.append(dict(key=key,shape=list(value.shape),positive_entries=int((value>0).sum()),ema_mass_sum=float(value.sum()),note='EMA mass, not cumulative sample count'))
        assert len(ema)==2
        assert optimizer_steps(a['vqvae']['optimizer'])==[ae['selected']['step']]
        counters={k:optimizer_steps(v) for k,v in b['optimizers'].items()}
        assert counters['optimizer1']==[min(be['selected']['step'],cfg['bet_full_optimizer_updates'])]
        assert counters['optimizer2']==[be['selected']['step']]
        latest=torch.load(ROOT/'vqbet/training'/group/'bet/latest.pt',map_location='cpu',weights_only=False)
        latest_counters={k:optimizer_steps(v) for k,v in latest['optimizers'].items()}
        assert latest_counters=={'optimizer1':[cfg['bet_full_optimizer_updates']],'optimizer2':[cfg['bet_updates']]}
        groups.append(dict(group=group,status='PASS',vqvae_selected_step=ae['selected']['step'],bet_selected_step=be['selected']['step'],
            quantizer_exactly_unchanged_during_bet=True,two_trained_ema_codebooks=ema,selected_optimizer_step_counters=counters,
            final_optimizer_step_counters=latest_counters,vqvae_checkpoint_sha256=ae['sha256'],bet_checkpoint_sha256=be['sha256']))
    assert groups
    write(folder/'quantizer_acceptance.json',dict(status='PASS' if len(groups)==2 else 'PARTIAL',accepted_groups=groups,
        scope='Actual saved optimizer counters and frozen RVQ tensor equality; not a physics success claim',new_optimizer_updates=0,new_physics_steps=0,device='CPU'))
    print('Accepted quantizer/optimizer evidence for',len(groups),'setting(s)')

if __name__=='__main__':main()
