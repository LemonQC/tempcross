"""Final acceptance and resource accounting; never trains or steps physics."""
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,TASKS,out,read,write,digest,resource,now

def main():
    folder=out('vqbet');result=read(folder/'results.json');assert result['status']=='COMPLETE_FAITHFUL_PORT'
    assert len(result['cells'])==4 and sum(c['episodes'] for c in result['cells'])==160
    from baselines.strong.vqbet_progress import main as index
    from baselines.strong.vqbet_quantizer_audit import main as quantizer
    from baselines.strong.vqbet_plots import main as plots
    index();quantizer();plots()
    assert read(folder/'quantizer_acceptance.json')['status']=='PASS'
    frozen=read(folder/'freeze.json')
    assert str(ROOT)==frozen['server_root'] and digest(folder/'protocol.json')==frozen['protocol_sha256']
    for name,sha in frozen['code_sha256'].items():assert digest(folder/name)==sha,name
    for name,sha in frozen['shared_code_sha256'].items():assert digest(PROJECT/name)==sha,name
    assert digest(ROOT/'vqbet/data/complete.json')==frozen['data_sha256']
    assert digest(folder/'official_source.json')==frozen['source_sha256']
    for name,record in read(folder/'official_source.json')['files'].items():
        assert digest(ROOT/'vendor/vq_bet_port'/name)==record['port_sha256'],name
    physical=[];jobs=[]
    for group in ['target_only','source_target']:
        for task in TASKS:
            accepted=read(folder/f'evaluation_{group}_{task}.json');assert accepted['status']=='PASS' and accepted['episodes']==40
            records=[r['episode'] for r in accepted['rows']]
            assert {r['official_condition_index'] for r in records}==set(range(60,100))
            job=read(ROOT/'vqbet/physical/formal'/group/task/'job.json')
            assert job['status']=='COMPLETE' and job['episodes']==40 and job['role']=='EXPOSED_REFERENCE'
            jobs.append(job);physical+=records
    assert len(physical)==160
    launches=[read_json_line(line) for line in (ROOT/'vqbet/physical/physical_launches.jsonl').read_text().splitlines()]
    assert sum(r['role']=='EXPOSED_REFERENCE' for r in launches)==160
    assert sum(r['role']=='DEVELOPMENT_SMOKE' for r in launches)==4
    prior=Path(read(folder/'technical_repair.json')['prior_attempt'])
    prior_launches=[read_json_line(line) for line in (prior/'physical/physical_launches.jsonl').read_text().splitlines()]
    assert len(prior_launches)==4 and all(r['role']=='DEVELOPMENT_SMOKE' for r in prior_launches)
    for row in read(ROOT/'preservation_freeze.json')['files']:assert digest(row['path'])==row['sha256']
    resources=read(folder/'resources.json')
    resources.update(formal_job_wall_seconds=sum(j['duration_seconds'] for j in jobs),formal_control_steps=sum(r['steps'] for r in physical),
        formal_policy_inference_wall_seconds=sum(r['mean_prediction_seconds']*r['policy_calls'] for r in physical),
        inference_wall_is_not_gpu_active_seconds=True,dedicated_server_artifact_bytes=sum(p.stat().st_size for p in (ROOT/'vqbet').rglob('*') if p.is_file()),
        prior_attempt_artifact_bytes=sum(p.stat().st_size for p in prior.rglob('*') if p.is_file()),
        shared_raw_pack_bytes_not_newly_created=True,corrected_smoke_episodes=4,prior_smoke_episodes=4,total_smoke_episodes=8)
    assert resources['formal_stage_updates']==60000 and resources['formal_optimizer_calls']==84000
    assert resources['total_executed_stage_updates']==60008 and resources['total_executed_optimizer_calls']==84012
    write(folder/'resources.json',resources)
    write(folder/'delivery_acceptance.json',dict(status='PASS',utc=now(),formal_episodes=160,accepted_cells=4,full_official_goal=True,
        frozen_core_code_unchanged=True,trained_residual_quantizer_and_native_optimizer_counters=True,
        original_1280_plus_160_hashes_unchanged=True,old_baseline_reruns=0,prior_smoke_episodes=4,corrected_smoke_episodes=4,
        total_executed_stage_updates=60008,total_executed_optimizer_calls=84012,reference_role='EXPOSED_REFERENCE',independent_unseen_test=False,
        final_model_components=len(read(folder/'model_index.json')),all_large_files_remain_on_server=True,final_resources=resource()))
    readme=['# VQ-BeT','','**COMPLETE_FAITHFUL_PORT**：两设置均完成原生RVQ及BeT训练和两个任务各40条正式物理评测。',
        '', '结果：[RESULTS.md](RESULTS.md)；协议：[PROTOCOL.md](PROTOCOL.md)；论文/发布代码差异：[PAPER_CODE_DIFFERENCES.md](PAPER_CODE_DIFFERENCES.md)。',
        '', '原始逐episode结果：episodes.csv。权重/训练日志：model_index.json、training_artifact_index.json；选中checkpoint样本暴露：selected_checkpoint_exposures.json。全部大文件在服务器，路径与SHA见sha_index.json；原TF32尝试见attempts/initial_tf32/。',
        '', 'formal训练两组件合计60000次阶段更新、84000次优化器调用；加两轮探针合计60008次阶段更新、84012次优化器调用。八条smoke与160条正式评测严格分列。',
        '', '仅使用EXPOSED_REFERENCE条件60–99，不称独立未见测试；未用物理成功率选模。数据/输入/真实参数更新/码本冻结/优化器计数/物理视频动作和原始基准保护均已验收，见delivery_acceptance.json。历史PARTIAL文件保留为当时快照，不代替最终结果。']
    (folder/'README.md').write_text('\n'.join(readme)+'\n')
    from baselines.strong.matrix_details import main as matrix
    matrix();print('PASS final VQ-BeT delivery: 160 formal, 8 smoke, 60008 stage updates, 84012 optimizer calls')

def read_json_line(line):
    import json
    return json.loads(line)

if __name__=='__main__':main()
