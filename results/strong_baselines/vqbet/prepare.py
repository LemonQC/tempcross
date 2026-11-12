"""Pinned official VQ-BeT, minimal language input port, immutable feature cache."""
import difflib
import json
import shutil
import subprocess
from pathlib import Path
from baselines.strong.common import PROJECT, ROOT, out, read, write, digest, resource

COMMIT = '09d4851288ca5deaaa1ab367a208e520f8ee9a84'
UPSTREAM = ROOT/'vendor/vq_bet_official'
PORT = ROOT/'vendor/vq_bet_port'

def source():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=UPSTREAM,text=True).strip()
    for name in ['vq_behavior_transformer','vqvae','vector_quantize_pytorch']:
        shutil.copytree(UPSTREAM/name, PORT/name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(UPSTREAM/'LICENSE',PORT/'LICENSE')
    path=PORT/'vq_behavior_transformer/bet.py'; original=path.read_text(); patched=original
    changes=[
        ('        self._collected_actions = []', '        self._language_header = nn.Linear(768, 1024)  # RGB-language I/O port\n        self._collected_actions = []'),
        ('                    goal_seq = goal_seq.clone().detach()\n                    goal_seq = self._resnet_header(goal_seq)', '                    goal_seq = goal_seq.clone().detach()\n                    goal_seq = self._language_header(goal_seq)  # frozen CLIP language input'),
        ('        self.optimizer1 = optimizer1', '        optimizer1.add_param_group({"params": self._language_header.parameters()})\n        self.optimizer1 = optimizer1')]
    for old,new in changes:
        assert patched.count(old)==1,old
        patched=patched.replace(old,new)
    path.write_text(patched)
    (out('vqbet')/'official_io.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='official/vq_behavior_transformer/bet.py',tofile='port/vq_behavior_transformer/bet.py')))
    files={}
    for name in ['vq_behavior_transformer','vqvae','vector_quantize_pytorch']:
        for p in sorted((UPSTREAM/name).glob('*.py')):
            r=str(p.relative_to(UPSTREAM));q=PORT/r
            files[r]=dict(upstream_sha256=digest(p),port_sha256=digest(q),identical=p.read_bytes()==q.read_bytes())
    assert [p for p,v in files.items() if not v['identical']]==['vq_behavior_transformer/bet.py']
    write(out('vqbet')/'official_source.json',dict(repository='https://github.com/jayLEE0301/vq_bet_official',commit=COMMIT,license='MIT',upstream_path=str(UPSTREAM),port_path=str(PORT),files=files,
        core_changes=False,io_changes=['768-dimensional frozen CLIP language replaces future-goal image/state with a learned 768→1024 prefix projection; registered in official optimizer1','canonical OSC 7-dimensional actions through native act_dim parameter','released frozen ResNet18 visual-feature mode'],
        native_preserved=['residual VQ-VAE with EMA codebooks','six-layer causal GPT','native focal code loss and sampled code selection','native continuous offset loss','first half both optimizers, second half offset optimizer only'],
        native_preprocessing=dict(resize_224='nearest',mean=[.485,.456,.406],std=[.229,.224,.255],note='Official blue-channel std .255 deliberately retained'),
        not_claimed='Native UR3 environment, future-goal-image conditioning, paper-scale compute or fine-tuned visual encoder'))

def prepare_data():
    import h5py,numpy as np,torch
    from baselines.strong.data import prepare
    from torchvision.models import resnet18,ResNet18_Weights
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dest=ROOT/'vqbet/data';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        for p,sha in read(dest/'complete.json')['sha256'].items():assert digest(p)==sha
        return
    prepare('vqbet')
    # Shared preparation's complete marker covers original identity/anchor files only.
    (dest/'complete.json').rename(dest/'shared_complete.json')
    rows=read(dest/'manifest.json');torch.set_num_threads(4)
    encoder=torch.nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]).cuda().eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.255],device='cuda')[None,:,None,None]
    features=dest/'features';features.mkdir(exist_ok=True)
    handles={}
    for i,r in enumerate(rows):
        path=features/f'{i:04d}.npy'
        if path.exists():
            cached=np.load(path,mmap_mode='r');assert cached.shape==(r['frames'],512) and np.isfinite(cached).all()
            continue
        if r['domain'] not in handles:handles[r['domain']]=h5py.File(r['packed_file'],'r')
        rgb=handles[r['domain']][r['episode_id']]['rgb'];parts=[]
        with torch.inference_mode():
            for t in range(0,len(rgb),128):
                x=torch.from_numpy(rgb[t:t+128].transpose(0,3,1,2).copy()).cuda().float()/255
                x=torch.nn.functional.interpolate(x,size=224,mode='nearest')
                parts.append(encoder((x-mean)/std).flatten(1).cpu().numpy())
        values=np.concatenate(parts);assert values.shape==(r['frames'],512) and np.isfinite(values).all()
        tmp=path.with_suffix('.tmp.npy');np.save(tmp,values);tmp.replace(path)
        if i%25==0:print('FEATURES',i+1,len(rows),flush=True)
    for f in handles.values():f.close()
    for stage,seed in [('vqvae',2026111201),('bet',2026111202)]:
        for domain in ['target','source']:
            rng=np.random.default_rng(seed+(domain=='source'))
            groups={task:[i for i,r in enumerate(rows) if r['domain']==domain and r['split']=='train' and r['task']==task] for task in sorted({r['task'] for r in rows})}
            a=np.empty((384000,2),np.int32)
            for k in range(len(a)):
                j=int(rng.choice(list(groups.values())[k%2]));a[k]=j,int(rng.integers(9,rows[j]['frames']))
            np.save(dest/f'{stage}_{domain}.npy',a)
    val=np.load(dest/'validation.npy').copy()
    for a in val:a[1]=max(9,min(int(a[1]),rows[int(a[0])]['frames']-1))
    np.save(dest/'vq_validation.npy',val)
    files=[p for p in dest.rglob('*') if p.is_file() and p.name!='complete.json']
    write(dest/'complete.json',dict(status='COMPLETE',identity_counts=read(dest/'shared_complete.json')['counts'],
        source_robot='ur5e',target_robot='panda',train_target_per_task=45,validation_target_per_task=5,train_source_per_task=270,
        feature_shape=512,feature_dtype='float32',feature_encoder='frozen ImageNet ResNet18 native preprocessing',
        observation_window=10,action_window=1,train_anchors_per_domain_per_stage=384000,validation_anchors=len(val),
        no_reference_conditions_used=True,sha256={str(p):digest(p) for p in files}))
    write(out('vqbet')/'data_inventory.json',read(dest/'complete.json'))

def candidate():
    from baselines.strong.common import TASKS
    cfg=dict(method='VQ_BET_RGB_LANGUAGE_CANONICAL_OSC_PORT',planned_classification='COMPLETE_FAITHFUL_PORT',official_commit=COMMIT,
        groups=['target_only','source_target'],source_robots=['ur5e'],target_robot='panda',human_target_demonstrations_per_task=50,
        train_target_demonstrations_per_task=45,validation_target_demonstrations_per_task=5,train_source_demonstrations_per_task=270,
        preserve_original_demo_identity_splits=True,observation=['128x128 RGB, resized nearest to224 for native frozen ResNet18','frozen CLIP-L instruction768, learned prefix projection'],privileged_observation=False,
        obs_window=10,action_window=1,gpt=dict(layers=6,heads=6,embedding=120,output=256,block_size=30,dropout=.1),
        vqvae=dict(latent=512,groups=2,codebook_size=16,encoder_loss_multiplier=1.,act_scale=1.,lr=.001,weight_decay=.0001),
        bet=dict(lr=.000055,weight_decay=.0002,betas=[.9,.999],offset_loss_multiplier=100,secondary_code_multiplier=.5,gamma=2.,sequentially_select=False),
        seed=7,vqvae_updates=6000,vqvae_target_batch=64,bet_updates=24000,bet_target_batch=16,joint_source_to_target_batch_ratio=1,
        bet_full_optimizer_updates=12000,bet_offset_only_updates=12000,
        target_anchor_exposures_per_stage_per_group=384000,validation_anchors=256,validation_every_vqvae_updates=500,validation_every_bet_updates=1000,
        vqvae_selection='minimum deterministic held-out target reconstruction L1; ties earlier',
        bet_selection='minimum fixed-seed held-out target native total loss, among 1000..24000; ties earlier',
        physical_success_used_for_selection=False,formal_indices=list(range(60,100)),formal_role='EXPOSED_REFERENCE',physical_episodes_per_group_per_task=40,
        control_frequency_hz=20,horizons=dict(zip(TASKS,[600,650])),success='unchanged official placement AND retreat',
        probe='one native VQ-VAE update plus one BeT update per setting; then condition0 full physical smoke each task; discard all probe weights',
        deployment='native stochastic codes; last causal observation prediction; one action per query, no temporal aggregation for action_window=1',
        technical_repairs_allowed=1,max_wall_hours_per_training_group=12,disk_floor_gib=25,checkpoint_retention=['best per component','latest resumable per component','probe provenance'],
        technical_repair_number=1,remaining_technical_repairs=0,server_root=str(ROOT),tf32=False,
        stop='exact per-component update budget, nonfinite error, disk floor or wall cap; no extension or reference-based tuning',
        recipe_differences=['UR5e/Panda canonical OSC and language I/O rather than native UR3 joint/goal inputs','bounded updates and small batches instead of native dataset-specific epochs/batch2048','released frozen ResNet visual feature mode, not visual encoder fine-tuning','explicit train/eval modes and freeze unused non-offset parameters in latter half, avoiding native unused gradient accumulation','technical repair1: disable cuDNN/matmul TF32 consistently to align cached and native-image inference'])
    write(out('vqbet')/'candidate_protocol.json',cfg)
    write(out('vqbet')/'preflight_resources.json',resource())
    write(out('vqbet')/'status.json',dict(status='PARTIAL',stage='CORRECTED_INPUT_PREPARATION',actual_training_updates=0,physical_evaluation_episodes=0,success_rate=None,technical_repairs=1,prior_probe_stage_updates=4,prior_probe_optimizer_calls=6,prior_smoke_episodes=4))

def main():
    assert str(ROOT)==read(out('vqbet')/'technical_repair.json')['corrected_server_root'],'Use corrected STRONG_BASELINE_RUN_ROOT'
    if (out('vqbet')/'freeze.json').exists():
        assert digest(out('vqbet')/'protocol.json')==read(out('vqbet')/'freeze.json')['protocol_sha256']
        print('Frozen preparation reused; candidate/status not overwritten.');return
    source();candidate();prepare_data()

if __name__=='__main__':main()
