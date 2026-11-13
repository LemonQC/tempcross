"""Freeze a bounded unequal-data augmentation control; no new generation."""
import ast
from collections import Counter
from pathlib import Path
import subprocess
from baselines.strong.common import PROJECT,ROOT,SOURCE,OLD,TASKS,out,read,write,digest,resource

UPSTREAM=ROOT/'vendor/mimicgen_official'

def function_ast(path,owner,name):
    tree=ast.parse(path.read_text());scope=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner) if owner else tree
    node=next(n for n in scope.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return ast.dump(node,include_attributes=False)

def selection_numerical_check(official,bundle):
    import importlib.util,types,numpy as np
    spec=importlib.util.spec_from_file_location('mimicgen_original_pose_math',official/'utils/pose_utils.py');poses=importlib.util.module_from_spec(spec);spec.loader.exec_module(poses)
    def load(path,owner):
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='select_source_demo')
        namespace=dict(np=np,PoseUtils=poses);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        return namespace['select_source_demo']
    first=load(official/'datagen/selection_strategy.py','NearestNeighborRobotDistanceStrategy')
    second=load(bundle/'mimicgen/datagen/selection_strategy.py','NearestNeighborInterpolationStrategy')
    rng=np.random.default_rng(20261113);previous=np.random.get_state();checks=0
    def pose():
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        return poses.make_pose(rng.normal(size=3),q)
    try:
        for count in [3,5,8]:
            for seed in range(40):
                sources=[types.SimpleNamespace(eef_pose=pose()[None],object_poses={'obj':pose()[None]}) for _ in range(count)]
                kwargs=dict(eef_pose=pose(),object_pose=pose(),src_subtask_datagen_infos=sources,nn_k=5)
                np.random.seed(seed);a=first(None,**kwargs)
                np.random.seed(seed);b=second(None,**kwargs)
                assert a==b,(count,seed,a,b);checks+=1
    finally:np.random.set_state(previous)
    return dict(status='PASS',matched_cases=checks,source_counts=[3,5,8],nn_k=5,no_simulator_created=True)

def source_audit():
    bundle=SOURCE/'third_party/mimicgen';official=UPSTREAM/'mimicgen'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()=='0f435089f48b1f16d225f868dbef855a5e75d1e0'
    upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()
    equivalent=function_ast(official/'datagen/selection_strategy.py','NearestNeighborRobotDistanceStrategy','select_source_demo')==function_ast(bundle/'mimicgen/datagen/selection_strategy.py','NearestNeighborInterpolationStrategy','select_source_demo')
    numerical=selection_numerical_check(official,bundle)
    transform_equal=function_ast(official/'utils/pose_utils.py',None,'transform_source_data_segment_using_object_pose')==function_ast(bundle/'mimicgen/utils/pose_utils.py',None,'transform_source_data_segment_using_object_pose')
    assert transform_equal
    files=[bundle/'mimicgen/datagen'/name for name in ['data_generator.py','selection_strategy.py','waypoint.py']]
    files+=[bundle/'mimicgen/utils/pose_utils.py',bundle/'mimicgen/env_interfaces/robosuite.py',bundle/'mimicgen/env_interfaces/robocasa/single_stage/mg_pnp.py',bundle/'LICENSE',
            PROJECT/'results/strong_baselines/mimicgen/common.py',PROJECT/'results/strong_baselines/mimicgen/probe.py',PROJECT/'results/strong_baselines/mimicgen/augment.py',
            PROJECT/'baselines/common/model_factory.py',PROJECT/'baselines/common/policies.py',PROJECT/'baselines/common/train.py']
    vendor=Path('/home/users/lemon/baseline-benchmark/vendor/robomimic')
    files += [vendor/'robomimic/algo/bc.py',vendor/'robomimic/models/policy_nets.py',vendor/'robomimic/models/obs_core.py',vendor/'LICENSE']
    write(out('mimicgen_v2')/'official_source.json',dict(official_repository='https://github.com/NVlabs/mimicgen',official_commit=upstream_commit,
        used_distribution='BARX bundled MimicGen1.0.1+barx.1',used_repository='https://github.com/ajaysridhar0/barx',used_commit='0f435089f48b1f16d225f868dbef855a5e75d1e0',
        generator_license='NVIDIA Source Code License, research/evaluation use',policy_repository='https://github.com/ARISE-Initiative/robomimic',policy_commit='ae5799f0fae05c4559ee1f9645b0f77eb5251929',policy_license='MIT',
        source_selection_function_ast_identical=equivalent,source_selection_numerical_equivalence_check=numerical,object_frame_pose_transform_ast_identical=transform_equal,
        reproduction_differences=['BARX RoboCasa-X interfaces and canonical12-action padding','BARX nearest_neighbor_interpolation differs in variable names and an unused nn_k=None branch; same object-frame distance/top-k rule,120 controlled numerical cases match upstream nearest_neighbor_robot_distance','Bundled valid-demo filtering and optional stuck detector; generated records used default end_if_stuck=False','Historical local base/world annotation repair retained; no new generation in v2','BC-Transformer-GMM downstream matches existing baseline, not original paper BC-RNN','Panda seeds only; no UR5e-to-Panda generation demonstrated','Existing50+13 successes, not original50+50 target'],
        sha256={str(p):digest(p) for p in files}))

def candidate():
    cfg=dict(method='MIMICGEN_V2_UNEQUAL_PANDA_AUGMENTATION_BC_TRANSFORMER_COMPONENT',planned_final_status='PARTIAL',
        component_completion_label='BOUNDED_DOWNSTREAM_CONTROL_COMPLETE',original_mimicgen_generation_gate='BLOCKED_PRESERVED',
        source_robots=['panda_seed_demonstrations_for_generation'],cross_embodiment_generation=False,target_robot='panda',
        groups=['human_plus_generated'],human_target_demonstrations_per_task=50,train_target_demonstrations_per_task=45,validation_target_demonstrations_per_task=5,
        existing_generated_train_counts={TASKS[0]:50,TASKS[1]:13},new_generation_attempts=0,probe_generations_excluded=5,
        sampling='task-balanced, then uniform demonstration and frame over human45+generated50/13 per task',anchor_seed=20261113,
        seed=7,policy='official robomimic BC_Transformer_GMM, exact existing B_bc_transformer config',
        initialization='exact saved random initial trainable tensors of old B_bc_transformer, not its learned weights',
        optimizer_updates=48000,batch=16,training_anchor_exposures=768000,observation_window=10,supervised_actions_per_anchor=1,
        lr=.0001,weight_decay=.01,validation_anchors=256,validation_every_updates=750,
        checkpoint_selection='fixed final update48000, matched to reused human-only control; validation diagnostic only',physical_success_used_for_selection=False,
        reused_control='B_bc_transformer original 48000-update model and80 original episodes; no control training or simulation rerun',
        observation=['128x128 RGB','frozen CLIP-L instruction'],deployed_privileged_observation=False,
        privileged_generation_inputs=['object pose','goal pose','eef pose','source actions','contact-stage annotations'],
        formal_indices=list(range(60,100)),formal_role='EXPOSED_REFERENCE',physical_episodes_per_task=40,new_formal_episodes=80,
        control_frequency_hz=20,horizons=dict(zip(TASKS,[600,650])),success='unchanged official placement AND retreat',
        probe='one real training update on the mixed pool, discard probe weights, full condition0 smoke each task; interface/finite/official-goal recording acceptance, no success threshold',
        technical_repairs_allowed=1,max_training_wall_hours=4,disk_floor_gib=25,checkpoint_retention=['probe','resumable latest','fixed final checkpoint'],
        stop='fixed48000 updates or nonfinite/resource/technical obstruction; no extra configurations, test-based selection or generation expansion',
        server_root=str(ROOT),data_directory=str(ROOT/'mimicgen_v2/data'))
    write(out('mimicgen_v2')/'candidate_protocol.json',cfg)
    write(out('mimicgen_v2')/'status.json',dict(status='PARTIAL',phase='DATA_PREPARATION',actual_training_updates=0,physical_evaluation_episodes=0,
        success_rate=None,technical_repairs=0,new_generation_attempts=0,original_full_mimicgen_status='BLOCKED',scope=cfg['method']))

def data():
    import cv2,h5py,numpy as np
    from p3_validation.prepare_data import fingerprint
    destination=ROOT/'mimicgen_v2/data';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'complete.json').exists():
        for p,sha in read(destination/'complete.json')['sha256'].items():assert digest(p)==sha
        return
    audit=read(out('mimicgen_v2')/'data_audit.json');assert audit['status']=='PASS'
    records=read(audit['source_record_file']);assert digest(audit['source_record_file'])==audit['source_record_sha256']
    rows=[dict(r,origin='human',packed_file=str(OLD/'data/target.h5')) for r in read(PROJECT/'configs/datasets/baseline_target_manifest.json')]
    ids={r['episode_id'] for r in rows};pack=destination/'generated.h5'
    if pack.exists():raise FileExistsError('Preserve incomplete pack before diagnosis: '+str(pack))
    with h5py.File(pack,'x') as target:
        for r in records:
            if not r['success']:continue
            with h5py.File(r['trajectory'],'r') as source:
                actions=source['actions'][:,:7].astype(np.float32);assert np.isfinite(actions).all() and np.max(abs(actions))<=1
                eid=fingerprint(actions,len(actions),r['instruction']);assert eid not in ids;ids.add(eid)
                g=target.create_group(eid);g.create_dataset('actions',data=actions)
                rgb=g.create_dataset('rgb',shape=(len(actions),128,128,3),dtype='uint8',chunks=(1,128,128,3),compression='lzf')
                for t,image in enumerate(source['rgb']):rgb[t]=cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
                rows.append(dict(domain='target',origin='generated',embodiment='panda',task=r['task'],episode_id=eid,frames=len(actions),instruction=r['instruction'],split='train',packed_file=str(pack),
                    original_file=r['trajectory'],original_demo=r['id'],condition_id=r['condition_id'],generation_seed=r['generation_seed'],source_lineage=r['selected_sources']))
    assert len(rows)==163
    from baselines.common.language import encode
    write(destination/'language.json',{text:encode(text).numpy().tolist() for text in sorted({r['instruction'] for r in rows})})
    write(destination/'manifest.json',rows)
    groups={task:[i for i,r in enumerate(rows) if r['task']==task and r['split']=='train'] for task in TASKS}
    assert [len(groups[t]) for t in TASKS]==[95,58]
    rng=np.random.default_rng(20261113);anchors=np.empty((768000,2),np.int32)
    for k in range(len(anchors)):
        j=int(rng.choice(groups[TASKS[k%2]]));anchors[k]=j,int(rng.integers(rows[j]['frames']))
    np.save(destination/'train.npy',anchors)
    validation=np.load(OLD/'data/target_validation_anchors.npy');assert validation.shape==(256,2)
    assert all(rows[int(j)]['origin']=='human' and rows[int(j)]['split']=='validation' for j,t in validation)
    np.save(destination/'validation.npy',validation)
    assert digest(destination/'validation.npy')==digest(OLD/'data/target_validation_anchors.npy')
    files=[p for p in destination.iterdir() if p.is_file()]
    write(destination/'complete.json',dict(status='COMPLETE',human_train=90,human_validation=10,generated_train=63,generated_counts={t:sum(r['origin']=='generated' and r['task']==t for r in rows) for t in TASKS},
        canonical_actions_unmodified=True,generated_terminal_actions='preserve actual final successful action',human_terminal_actions='unchanged historical RLDS convention',
        original_validation_anchors_exact=True,sha256={str(p):digest(p) for p in files}))
    write(out('mimicgen_v2')/'data_inventory.json',read(destination/'complete.json'))

def control():
    selected=read(OLD/'training/B_bc_transformer/selected.json');assert selected['completed_optimizer_steps']==48000
    assert digest(selected['checkpoint'])==selected['checkpoint_sha256']
    analysis=read(PROJECT/'results/baselines/analysis.json');cells=[c for c in analysis['cells'] if c['label']=='B_bc_transformer' and c['robot']=='panda'];assert len(cells)==2
    for c in cells:
        assert c['episodes']==40 and c['checkpoint_sha256']==selected['checkpoint_sha256']
        job=read(OLD/'formal/B_bc_transformer'/c['task']/'panda/job.json');assert job['status']=='COMPLETE' and job['indices']==list(range(60,100))
    write(out('mimicgen_v2')/'reused_control.json',dict(status='REUSED_HISTORICAL_NO_RERUN',role='EXPOSED_REFERENCE',cells=cells,training_updates=48000,anchor_exposures=768000,
        checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],initial_weights=str(OLD/'training/B_bc_transformer/initial_trainable.pt'),
        initial_weights_sha256=digest(OLD/'training/B_bc_transformer/initial_trainable.pt'),config=str(OLD/'training/B_bc_transformer/config.json'),config_sha256=digest(OLD/'training/B_bc_transformer/config.json'),
        original_selection='validation-only tier stop, final48000 update checkpoint; new run uses same final budget, not an identical adaptive stopping experiment',new_control_updates=0,new_control_physical_episodes=0))

def main():
    if (out('mimicgen_v2')/'freeze.json').exists():
        assert digest(out('mimicgen_v2')/'protocol.json')==read(out('mimicgen_v2')/'freeze.json')['protocol_sha256'];return
    source_audit();candidate();control();data();print('Prepared bounded MimicGen v2 mixed pool; no new generation')

if __name__=='__main__':main()
