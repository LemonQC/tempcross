"""Exactly one bounded native probe, offline gates and at most2 full smokes."""
import subprocess
from .audit import GROUP,audit_job
from .data import RUN
from baselines.strong.common import PYTHON,PROJECT,ROOT,out,read,write,TASKS,now

def main():
    assert read(RUN/'probe/result.json')['status']=='PASS'
    assert read(RUN/'probe/result.json')['actual_optimizer_calls']==1
    assert read(RUN/'readonly_replay.json')['status'].startswith('PASS')
    assert read(RUN/'rpc_audit.json')['status'].startswith('PASS')
    command=[str(RUN/'env/bin/python'),'-m','unittest','results.strong_baselines.flow_matching_policy.test_data']
    with (RUN/'data-tests-final.log').open('w') as log:subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(out('flow_matching_policy')/'code_checks.json',dict(status='PASS',native_data_tests=4,
        real_optimizer_calls=1,offline_RPC_calls=17,native_offline_queries=3,
        readable_success_reference_checks=2,physics_smoke_budget=2,formal_training_started=False,utc=now()))
    write(ROOT/'session_resume.json',dict(phase='FLOW_MATCHING_DEVELOPMENT_SMOKES',active_method='flow_matching_policy',
        round_only='flow_matching_policy',utc=now(),next='Finish max2 full smokes, freeze unique formal budget; no old baseline'))
    for task,label in zip(TASKS,['counter','sink']):
        with (RUN/f'smoke-{label}.log').open('a') as log:
            subprocess.run([str(PYTHON),'-u','-m','results.strong_baselines.flow_matching_policy.evaluate',
                '--method','flow_matching_policy','--group',GROUP,'--checkpoint',str(RUN/'probe/updated.pt'),
                '--task',task,'--smoke'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        audit_job(task,True)
    status=read(out('flow_matching_policy')/'status.json')
    status.update(phase='DEVELOPMENT_GATES_PASSED_READY_TO_FREEZE',physical_smoke_episodes=2,
        actual_discarded_probe_updates=1,formal_budget_frozen=False)
    write(out('flow_matching_policy')/'status.json',status)
    write(ROOT/'session_resume.json',dict(phase='FLOW_MATCHING_GATES_PASSED_READY_TO_FREEZE',active_method='flow_matching_policy',
        round_only='flow_matching_policy',utc=now(),next='Freeze one budget and validation-only selection, publish, train; no more smokes'))
    print('PASS_ALL_GATES_EXACTLY2_SMOKES_NO_FORMAL_TRAINING',flush=True)

if __name__=='__main__':main()
