"""Persist official-code and old-recipe inspection without publishing secrets."""
from collections import Counter
import subprocess
from common import CHECKPOINT, OUT, PROJECT, ROOT, SOURCE, digest, manifests, read, write


def main():
    files=['docs/experiments.md','configs/experiments.toml','scripts/train.py','policy/vla-scripts/train.py',
           'policy/prismatic/training/materialize.py','policy/prismatic/training/strategies/lora_ddp.py',
           'policy/prismatic/training/strategies/fsdp.py','policy/prismatic/training/strategies/base_strategy.py',
           'policy/prismatic/vla/materialize.py','policy/prismatic/vla/datasets/datasets.py',
           'policy/prismatic/vla/datasets/rlds/dataset.py','policy/prismatic/conf/vla.py',
           'policy/prismatic/models/load.py','policy/prismatic/models/vlas/openvla.py']
    old=[]
    for label,path in [
        ('p2_joint_6000','/home/users/lemon/coft-p2/p2/physical-20261108/training/joint_reps-lora'),
        ('p2_no_reps_6000','/home/users/lemon/coft-p2/p2/physical-20261108/training/no_reps-lora'),
        ('p3_joint_96000','/home/users/lemon/coft-p3/p3-20261109/training/joint-reps-96000-b8')]:
        path=__import__('pathlib').Path(path)
        cfgpath=next((path/'runs').glob('*/config.json'));cfg=read(cfgpath)
        whitelist=['global_batch_size','per_device_batch_size','max_steps','learning_rate','weight_decay',
                   'train_strategy','transform_types','freeze_vision_backbone','freeze_llm_backbone','enable_gradient_checkpointing']
        old.append(dict(label=label,training_status=read(path/'training_status.json'),
                        safe_config={k:cfg['vla'].get(k) for k in whitelist},config_sha256=digest(cfgpath),config_path=str(cfgpath)))
    val={r['episode_id'] for r in manifests() if r['domain']=='target' and r['split']=='validation'}
    old6k=set(read(PROJECT/'results/baselines/legacy_target_provenance_summary.json')['resolved_episode_ids'])
    resource=read(ROOT/'probe-b8/training_status.json')
    assert resource['status']=='COMPLETE' and resource['completed_optimizer_steps']==1
    write(OUT/'recipe_audit.json',dict(
        official_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        official_source_clean=not bool(subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE).strip()),
        reviewed_files=[dict(path=str(SOURCE/p),sha256=digest(SOURCE/p)) for p in files],
        official_reference=dict(global_batch=256,updates=3000,sample_positions=768000,full_parameter=True,
                                lr=2e-5,source_weight=.05,target_weight=1.,prior_normalization_and_vq=True),
        upstream_vla_gradient_accumulation_supported=False,
        lora_factory=dict(rank=32,alpha=16,dropout=0,targets='Linear/Embedding/Conv2d/Conv1D across model',
                          activation_checkpointing_actual=False,
                          note='LoRA constructor hardcodes enable_gradient_checkpointing=False even when old TrainConfig says True'),
        old_recipes=old,old_6000_actual_target_validation_demonstrations_seen=len(val&old6k),
        old_96000_actual_target_validation_demonstrations_seen=10,new_target_validation_demonstrations=10,
        fresh_single_microbatch_probe=resource,
        historical_full_parameter_oom=read('/home/users/lemon/coft-p2/p2/physical-20261108/training/full-step-probe/training_status.json'),
        checkpoint_path=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),
        limitation='Reference exposure LoRA is not paper full-parameter adaptation and cannot establish that sufficient full-parameter adaptation fails.'))
    import shutil
    shutil.copyfile(ROOT/'technical_attempt_01.json',OUT/'technical_attempt_01.json')


if __name__=='__main__':main()
