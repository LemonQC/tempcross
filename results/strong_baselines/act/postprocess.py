"""Final ACT delivery gate after the frozen runner has accepted all 160 trials.

Only reads saved checkpoints/logs and creates reports/figures/indices. It never
trains, infers policy actions, starts an environment, or launches another method.
"""
from pathlib import Path
from baselines.strong.common import ROOT,out,read,write,digest

def main():
    folder=out('act');status=read(folder/'status.json');results=read(folder/'results.json')
    assert status['status']==results['status']=='COMPLETE_FAITHFUL_PORT'
    assert status['actual_training_updates']==48000 and status['physical_evaluation_episodes']==160
    assert len(results['cells'])==4 and all(c['episodes']==40 for c in results['cells'])
    freeze=read(folder/'freeze.json');assert str(ROOT)==freeze['server_root']
    assert digest(folder/'protocol.json')==freeze['protocol_sha256']
    for path,sha in freeze['code_sha256'].items():assert digest(path)==sha,path
    from baselines.strong.consumption import audit_act
    exposures=[audit_act(g) for g in ['target_only','source_target']]
    assert all(r['training_status']=='COMPLETE' and r['atomically_saved_optimizer_updates']==24000 and r['target_anchor_exposures']==384000 for r in exposures)
    assert exposures[0]['source_anchor_exposures']==0 and exposures[1]['source_anchor_exposures']==384000
    write(folder/'actual_exposure_reconciliation.json',exposures)
    from results.strong_baselines.act.selected_exposure import main as selected
    from results.strong_baselines.act.source_fidelity import main as fidelity
    from results.strong_baselines.act.plot import main as plot
    selected();fidelity();plot()
    models={g:read(folder/(g+'_selection.json')) for g in ['target_only','source_target']}
    assert all(not s['used_physical_success'] and digest(s['checkpoint'])==s['sha256'] for s in models.values())
    write(folder/'model_index.json',dict(status='COMPLETE_FAITHFUL_PORT',models=models,
        role='EXPOSED_REFERENCE',physical_success_used_for_selection=False,weights_on_server_only=True))
    test_logs=[ROOT/'act-data-tests.log',ROOT/'act-attempt-boundary-tests.log']
    for p in test_logs:assert p.read_text().rstrip().endswith('OK')
    write(folder/'test_status.json',dict(status='PASS',data_boundary_tests=3,attempt_root_boundary_tests=1,
        logs=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in test_logs],
        actual_seven_layer_gradient_and_physical_probe_receipt=read(folder/'decoder_repair_acceptance.json')))
    from baselines.strong.matrix_details import main as matrix
    matrix()
    write(folder/'delivery_acceptance.json',dict(status='PASS',method_status='COMPLETE_FAITHFUL_PORT',
        accepted_formal_episodes=160,roles=['EXPOSED_REFERENCE'],independent_unseen_test=False,
        completed_optimizer_updates_per_setting=24000,formal_run_updates_total=48000,
        preserved_upstream_technical_attempt_updates=24000,total_optimizer_calls_including_probes=72004,
        physical_smoke_episodes_total=8,technical_repairs_used=1,old_baseline_reruns=0,
        frozen_protocol_and_executable_hashes_verified=True,consumed_identities_and_checkpoints_verified=True,
        source_fidelity='Exact declared dimensions and final-decoder-output repair; other upstream core files unchanged',
        results_sha256=digest(folder/'results.json'),weights_videos_and_raw_logs_on_server_only=True))
    lines=['# ACT：完整faithful port结果','',
           '**COMPLETE_FAITHFUL_PORT**。本目录包含两种设置的实际训练与160条真实官方完整任务评测；没有声称原平台/原生观测完全复现。',
           '', '[最终结果](RESULTS.md) · [协议及差异](PROTOCOL.md) · [验收](delivery_acceptance.json) · [checkpoint与实际暴露](selected_checkpoint_exposures.json)',
           '', '| 设置 | 任务 | 官方成功/次数 | 成功率 |', '|---|---|---:|---:|']
    for c in results['cells']:lines.append(f"| {c['group']} | {c['task']} | {c['successes']}/{c['episodes']} | {100*c['success_rate']:.1f}% |")
    lines+=['','全部正式条件为EXPOSED_REFERENCE。checkpoint只由留出示范验证损失选择。',
            '', '每设置实际24000更新，目标anchor暴露384000；joint另有384000个UR5e source anchors。所选checkpoint的实际更新数和暴露另列，不能用完整训练成本替代。',
            '', '唯一一次技术修复为读取最后decoder输出。此前上游首层输出版本的24000更新、2次更新探针和4条smoke完整归档，未混入修复版160条正式结果；修复版另有2次更新探针和4条smoke。',
            '', '历史阶段快照`PARTIAL_RESULTS.md`及`partial_results.json`保留；当前完整状态以本README、RESULTS.md和status.json为准。',
            '', '大文件的绝对路径和SHA见sha_index.json。旧BC/DP/BARX、BARX适配结果及TOCAT独立分支均保留。此入口不启动VQ-BeT或其他算法。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    print('PASS: complete ACT delivery accepted; commit and remotely reread before starting VQ-BeT.')

if __name__=='__main__':main()
