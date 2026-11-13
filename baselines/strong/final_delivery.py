"""Close this bounded six-method pass, without claiming six complete baselines."""
import collections
import subprocess
from baselines.strong.common import ROOT,PROJECT,METHODS,out,read,write,digest,initialize,now,resource


def main():
    statuses={m:read(out(m)/'status.json') for m in METHODS}
    assert [statuses[m]['status'] for m in METHODS]==['COMPLETE_FAITHFUL_PORT','COMPLETE_FAITHFUL_PORT','PARTIAL','BLOCKED','BLOCKED','BLOCKED']
    assert read(out('mimicgen_v2')/'delivery_acceptance.json')['status']=='PASS_BOUNDED_COMPONENT_ONLY'
    assert statuses['mimicgen_v2']['component_status']=='BOUNDED_DOWNSTREAM_CONTROL_COMPLETE'
    assert statuses['kite']['technical_repairs']==1 and statuses['object_flow']['technical_repairs']==1
    completed_acceptances=[]
    for method in ['act','vqbet']:
        acceptance_path=out(method)/'delivery_acceptance.json'
        acceptance=read(acceptance_path);assert acceptance['status']=='PASS'
        completed_acceptances.append(dict(method=method,path=str(acceptance_path),sha256=digest(acceptance_path),status='PASS'))
    counts=collections.Counter(s['status'] for s in statuses.values())
    cells=[]
    for method in METHODS:
        r=read(out(method)/'results.json')
        for cell in r.get('cells',[]):
            assert cell['role']=='EXPOSED_REFERENCE' and cell['episodes']==40
            cells.append(dict(method=method,**cell))
    assert len(cells)==10 and sum(c['episodes'] for c in cells)==400
    expected={('act','target_only','pnp_counter_to_sink'):5,('act','target_only','pnp_sink_to_counter'):4,
        ('act','source_target','pnp_counter_to_sink'):8,('act','source_target','pnp_sink_to_counter'):2,
        ('vqbet','target_only','pnp_counter_to_sink'):0,('vqbet','target_only','pnp_sink_to_counter'):0,
        ('vqbet','source_target','pnp_counter_to_sink'):0,('vqbet','source_target','pnp_sink_to_counter'):0,
        ('mimicgen_v2','human_plus_generated','pnp_counter_to_sink'):2,('mimicgen_v2','human_plus_generated','pnp_sink_to_counter'):0}
    assert {(c['method'],c['group'],c['task']):c['successes'] for c in cells}==expected
    assert all(statuses[m]['success_rate'] is None for m in ['kite','crosszero','object_flow'])
    act=read(out('act')/'resources.json');vq=read(out('vqbet')/'resources.json');mg=read(out('mimicgen_v2')/'resources.json')
    optimizer_calls=act['formal_optimizer_updates']+act['prior_attempt_optimizer_updates']+act['total_probe_optimizer_updates']+vq['total_executed_optimizer_calls']+mg['total_new_optimizer_updates']
    assert optimizer_calls==204017
    smoke=statuses['act']['physical_smoke_episodes']+statuses['act']['prior_physical_smoke_episodes']+vq['total_smoke_episodes']+mg['new_smoke_episodes']
    assert smoke==18
    protected=initialize();assert all(digest(r['path'])==r['sha256'] for r in protected['files'])
    retained_commits={'act':'8a7265114c8a01a9ff7a83cb6239bde3ee285a4e','vqbet':'a817d714a312f7392de4762722ebc798ebe97a26','mimicgen_v2':'983ee8d344c0abe654561c6d4809a4166cf79e32'}
    for method,sha in retained_commits.items():
        assert not subprocess.check_output(['git','diff',sha,'--','results/strong_baselines/'+method],cwd=PROJECT,text=True).strip(),method+' evidence changed'
    active=[]
    for row in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        if ' -m ' not in row:continue
        command=row.strip().split(None,1)[-1]
        if not command.startswith('/home/users/lemon/coft-p2/barx/.venv/bin/python'):continue
        if any(name in command for name in ['baselines.strong.act_train','results.strong_baselines.vqbet.run','results.strong_baselines.mimicgen_v2.run','results.strong_baselines.object_flow.label_probe']):active.append(row.strip())
    assert not active,active
    summary=dict(status='BOUNDED_PASS_CLOSED_FULL_BASELINE_TARGET_NOT_ACHIEVED',utc=now(),
        complete_methods=2,partial_methods=1,blocked_methods=3,six_full_baselines_completed=False,
        completed_method_acceptances=completed_acceptances,
        statuses=statuses,new_accepted_formal_episodes=400,new_smoke_episodes_including_prior_technical_attempts=smoke,
        formal_control_steps=act['formal_physical_control_steps']+vq['formal_control_steps']+mg['formal_control_steps'],
        all_executed_optimizer_calls_including_probes_and_discarded_attempt=optimizer_calls,
        reference_role='EXPOSED_REFERENCE',independent_unseen_test_claim=False,
        old_baseline_reruns=0,old1280_and_barx160_protection='PASS',original_full_mimicgen_gate='BLOCKED_PRESERVED',
        new_mimicgen_generation_attempts=0,tocat_branch_merged=False,new_method_developed=False,
        object_flow_bootstrap_physics='Occurred; native step1/step2 total not instrumented; not included as a policy episode or claimed zero',
        active_training_or_evaluation_processes=active,resource_snapshot=resource(),
        earlier_completed_method_trees_unchanged_since_verified_publications=retained_commits,
        cells=cells,unrun_methods_never_filled_as_zero_success=True,
        protected_files=protected['files'])
    write(PROJECT/'results/strong_baselines/final_delivery.json',summary)
    write(ROOT/'session_resume.json',dict(utc=now(),phase='BOUNDED_PASS_CLOSED_DO_NOT_AUTORESTART',current_method=None,
        active_training_pid=None,active_evaluation_pid=None,full_baseline_target_achieved=False,
        complete=['act','vqbet'],partial=['mimicgen_v2'],blocked=['kite','crosszero','object_flow'],
        reason='Per-method stop limits reached; no further repair, experiment or new method started',
        publication='Read latest publication api_state.json and verify GitHub main; do not retrain'))
    from baselines.strong.matrix_details import main as matrix
    matrix();print(dict(status=summary['status'],counts=dict(counts),formal_episodes=400,smokes=smoke,optimizer_calls=optimizer_calls))


if __name__=='__main__':main()
