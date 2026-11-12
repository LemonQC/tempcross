"""Read-only MimicGen results, provenance and historical comparisons."""
import csv
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import shutil
from common import PROJECT, OUT, ROOT, SOURCE, TASKS, digest, read, write


def table(path,rows):
    with Path(path).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def classify(row):
    import h5py
    import numpy as np
    spec=importlib.util.spec_from_file_location('saved_interaction_audit',PROJECT/'evidence/interaction_initiation/audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with h5py.File(row['telemetry_path']) as f:
        action=f['actions/environment'][:];closed=np.flatnonzero(action[:,6]>0)
        first=int(closed[0]) if len(closed) else None
        bodies=json.loads(f.attrs['body_names']);names=json.loads(f.attrs['geom_names'])
        geoms={g for g,b in zip(names,f['model/geom_bodyid'][:]) if bodies[b].startswith('obj_')}
        logs=[json.loads(x) for x in Path(row['telemetry_path']).with_suffix('.jsonl').read_text().splitlines()]
        contact=module.physical_contact(logs,geoms)
        if first is None:contact[:]=False
        else:contact[:first+1]=False
        rise=f['target/position'][:,2]-f['target/position'][0,2]
        lift=module.longest(contact & (rise>=.03))>=10
        placement=bool(f['baseline/post_action_placement'][:].any())
        success=bool(np.any(f['baseline/post_action_placement'][:]&f['baseline/post_action_retreat'][:]))
        assert success==row['success']
    stage=('SUCCESS' if success else 'NO_CLOSE_COMMAND' if first is None else
        'CLOSE_WITHOUT_BILATERAL_CONTACT' if not contact.any() else
        'CONTACT_WITHOUT_STABLE_LIFT' if not lift else
        'PLACEMENT_WITHOUT_RETREAT' if placement else 'STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT')
    return dict(row,issued_close=first is not None,first_close=first,
        bilateral_contact_after_close=bool(contact.any()),stable_lift_proxy=lift,failure_stage=stage)


def main():
    status=read(OUT/'status.json')
    complete=(ROOT/'evaluation/complete.json').exists() and (OUT/'selection.json').exists()
    if not complete:
        stages=read(ROOT/'phase2_execution.json') if (ROOT/'phase2_execution.json').exists() else {}
        write(OUT/'partial_results.json',dict(status='BLOCKED',actual_status=status,execution=stages,
            downstream_success_rate=None,unrun_evaluations_not_counted_as_failures=True))
        (PROJECT/'docs/MIMICGEN_BASELINE_RESULTS.md').write_text(
            '# MimicGen有界Baseline：实际结果\n\n状态：**BLOCKED**。\n\n'+
            '具体阶段、实际已完成量及技术原因：\n\n```json\n'+json.dumps(status,ensure_ascii=False,indent=2)+'\n```\n\n'+
            '未运行评测不填写为0%成功。原始日志与已生成数据保留于`'+str(ROOT)+'`。\n')
        return
    generation=read(ROOT/'augmentation/complete.json')
    training=read(ROOT/'downstream/training/status.json')
    selection=read(OUT/'selection.json')
    assert generation['successful_demos']==100 and training['actual_optimizer_updates']==48000
    assert training['sample_positions']==768000 and not selection['used_physical_success']
    rows=[];raw=[];jobs=[]
    for task in TASKS:
        directory=ROOT/'evaluation/formal/mimicgen_bc_transformer'/task/'panda'
        job=read(directory/'job.json');validation=read(directory/'validation.json')
        assert job['status']=='COMPLETE' and validation['all_verified'] and validation['exact_original_initial_states_verified']
        jobs.append(job)
        records=[json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()]
        assert len(records)==40 and [r['official_condition_index'] for r in records]==list(range(60,100))
        for r in records:
            r.update(task=task,robot='panda',label='mimicgen_bc_transformer',role='EXPOSED_REFERENCE')
            rows.append(classify(r))
            for kind,path in [('telemetry',Path(r['telemetry_path'])),('contacts',Path(r['telemetry_path']).with_suffix('.jsonl')),('video',Path(r['video_path']))]:
                raw.append(dict(kind=kind,path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
    table(OUT/'episodes.csv',rows)
    cells=[]
    for task in TASKS:
        rr=[r for r in rows if r['task']==task]
        cells.append(dict(task=task,robot='panda',episodes=40,successes=sum(r['success'] for r in rr),
            success_rate=sum(r['success'] for r in rr)/40,issued_close=sum(r['issued_close'] for r in rr),
            stable_lift=sum(r['stable_lift_proxy'] for r in rr),failure_counts=dict(Counter(r['failure_stage'] for r in rr))))
    table(OUT/'cells.csv',cells)
    manifest=read(ROOT/'downstream/data/manifest.json');by_id={r['episode_id']:r for r in manifest}
    full=Counter();selected=Counter();unique=defaultdict(set);n=0
    trace=ROOT/'downstream/training/consumed_anchors.csv'
    with trace.open() as f:
        for r in csv.DictReader(f):
            assert int(r['index'])==n and by_id[r['episode_id']]['split']=='train'
            assert 0<=int(r['frame'])<by_id[r['episode_id']]['frames']
            key=(r['origin'],r['task']);full[key]+=1
            if n<selection['anchor_positions']:selected[key]+=1
            unique[r['origin'],r['episode_id']].add(int(r['frame']));n+=1
    assert n==768000
    consumption=dict(run_sample_positions=n,selected_checkpoint_sample_positions=selection['anchor_positions'],
        full_counts=[dict(origin=k[0],task=k[1],positions=v) for k,v in sorted(full.items())],
        selected_counts=[dict(origin=k[0],task=k[1],positions=v) for k,v in sorted(selected.items())],
        unique_episodes={origin:sum(k[0]==origin for k in unique) for origin in ['human','generated']},
        unique_frame_positions=sum(len(v) for v in unique.values()),no_validation_episodes_consumed=True,
        supervised_actions_per_anchor=1,loaded_history_images_per_anchor=10,
        trace_path=str(trace),trace_bytes=trace.stat().st_size,trace_sha256=digest(trace))
    write(OUT/'consumption.json',consumption)
    updates=list(csv.DictReader((ROOT/'downstream/training/updates.csv').open()))
    assert len(updates)==48000 and all(int(r['update'])==i+1 and math.isfinite(float(r['loss'])) for i,r in enumerate(updates))
    table(OUT/'training_curve.csv',[r for i,r in enumerate(updates) if i==0 or (i+1)%100==0])
    write(OUT/'training_status.json',training)
    write(OUT/'data_summary.json',read(ROOT/'downstream/data/complete.json'))
    write(OUT/'generation_summary.json',{k:v for k,v in generation.items() if k!='rows'})
    table(OUT/'generation_episodes.csv',[{k:r[k] for k in ['id','task','condition','condition_id','generation_seed','success','steps','trajectory','video','lineage']} for r in generation['rows']])
    write(OUT/'source_lineage.json',[read(r['lineage']) for r in generation['rows']])
    old=read(PROJECT/'results/baselines/analysis.json')['episodes']
    barx=read(PROJECT/'results/strong_baselines/barx_adaptation/results.json')['cells']
    comparison=[];paired=[]
    for task in TASKS:
        for label in ['B_bc_transformer','C_bc_transformer']:
            rr=[r for r in old if r['label']==label and r['task']==task and r['robot']=='panda']
            assert len(rr)==40
            comparison.append(dict(label=label,task=task,successes=sum(r['success'] for r in rr),episodes=40,source='HISTORICAL_READ_ONLY'))
            for r in rows:
                if r['task']!=task:continue
                previous=next(v for v in rr if v['condition_id']==r['condition_id'])
                paired.append(dict(baseline=label,task=task,condition_id=r['condition_id'],baseline_success=previous['success'],candidate_success=r['success'],
                    rescued=not previous['success'] and r['success'],regressed=previous['success'] and not r['success']))
        b=next(r for r in barx if r['label']=='new_joint_adaptation_768000' and r['task']==task and r['robot']=='panda')
        comparison.append(dict(label='BARX_reference_budget_LoRA',task=task,successes=b['successes'],episodes=40,source='HISTORICAL_READ_ONLY'))
        c=next(r for r in cells if r['task']==task)
        comparison.append(dict(label='MimicGen_BCTransformer',task=task,successes=c['successes'],episodes=40,source='NEW_EXPOSED_REFERENCE'))
    table(OUT/'comparison.csv',comparison);table(OUT/'paired_cases.csv',paired)
    for r in generation['rows']:
        raw.extend(dict(item,stage='AUGMENTATION') for item in r['artifacts'])
    for p in [Path(selection['path']),ROOT/'downstream/training/updates.csv',trace,
              ROOT/'downstream/data/generated.h5',ROOT/'downstream/data/train_anchors.npy']:
        raw.append(dict(kind='model_or_training_data',path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(OUT/'artifact_index.json',raw)
    for path,sha in status['input_sha256'].items():assert digest(path)==sha
    result=dict(status='COMPLETE',probe=read(OUT/'probe_validation.json'),new_formal_episodes=80,
        evaluation_role='EXPOSED_REFERENCE',independent_unseen_test=False,old_rollouts_rerun=0,
        augmentation_attempts=generation['attempts'],augmentation_successes=100,training=training,
        selection=selection,cells=cells,comparison=comparison,old_inputs_and_source_files_unchanged=True)
    write(OUT/'results.json',result)
    lines=['# MimicGen有界Baseline：实际结果','','状态：**COMPLETE**。完成官方MimicGen生成→既有官方BC-Transformer视觉策略训练→原任务物理评测。没有新Method。','',
        '首次探针为生成4/5、对应动作重放4/5，总计8/10；这五对轨迹相关，只用于可行性门槛。预检动作帧修复和一次探针记录器修复、崩溃后RNG重启差异均保留在协议/日志中，没有重跑任何已执行轨迹。','',
        f"额外生成{generation['attempts']}次，收集两方向各50条成功示范（100条），物理{generation['actual_physics_steps']}步，耗时{generation['elapsed_seconds']/3600:.3f}小时。源片段来自每方向固定5条Panda人类训练示范；属于目标域特权几何数据增强，不能称零目标示范跨具身生成。",'',
        '下游使用原90条人类train＋100条生成train，共190条训练；10条原人类示范验证。全部探针轨迹排除，所有生成谱系只来自human train。生成时使用物体/EE真值，部署仅RGB＋完整指令。没有把生成成本或额外数据隐藏为等数据公平。','',
        f"真实训练48000次optimizer更新、768000个anchor；seed7，batch16；原官方BC-Transformer/GMM5/512d/6层/8头/10帧、AdamW1e-4常数、wd0.01、128px与116px crop。固定总anchor预算与历史B-BCTransformer相同，但原人类数据暴露量、采样序列和额外生成数据不同。",'',
        f"验证选中step {selection['step']}、{selection['anchor_positions']} anchors，expected-action MSE={selection['validation_expected_action_mse']:.8f}，NLL={selection['validation_nll']:.8f}。在8个预先固定候选中选最低MSE；测试成功率不用于选模。完整运行预算与较早checkpoint实际暴露分开记录。",'',
        '| 模型 | Counter→Sink Panda | Sink→Counter Panda |','| --- | ---: | ---: |']
    for label in ['B_bc_transformer','C_bc_transformer','BARX_reference_budget_LoRA','MimicGen_BCTransformer']:
        rr=[next(r for r in comparison if r['label']==label and r['task']==task) for task in TASKS]
        lines.append(f"| {label} | {rr[0]['successes']}/40 ({rr[0]['successes']*2.5:.1f}%) | {rr[1]['successes']}/40 ({rr[1]['successes']*2.5:.1f}%) |")
    lines+=['','新80条仅为已暴露60–99参考条件。环境、20Hz、控制器、动作接口、600/650步上限及官方放置＋退离成功定义不变；原1280＋160条零重跑。没有独立未见测试，不据单训练seed作跨seed稳定排序。','',
        '| 新模型任务 | 闭爪次数 | 稳定抬升代理 | 失败阶段计数 |','| --- | ---: | ---: | --- |']
    for c in cells:lines.append(f"| {c['task']} | {c['issued_close']}/40 | {c['stable_lift']}/40 | {json.dumps(c['failure_counts'])} |")
    total=sum(r['successes'] for r in cells)
    lines+=['',f'本次固定条件共成功{total}/80。失败阶段仅是保存轨迹的描述，不等于物理根因；稳定抬升代理为双侧接触、物体body原点升高3cm持续10状态。不能把数据增强收益单独归因于新交互算法，也不能把本轮有界规模当成MimicGen论文全部预算。','',
        '完整逐条结果：`episodes.csv`；历史对照及救回/退化：`comparison.csv`、`paired_cases.csv`；生成成本与实际父示范：`generation_episodes.csv`、`source_lineage.json`；训练输入计数：`consumption.json`；大文件绝对路径/字节数/SHA256：`artifact_index.json`。','',
        f'服务器根：`{ROOT}`。完整协议：[MIMICGEN_BASELINE_PROTOCOL.md](MIMICGEN_BASELINE_PROTOCOL.md)。本阶段完成后停止，不启动其他Baseline或Method。']
    (PROJECT/'docs/MIMICGEN_BASELINE_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
