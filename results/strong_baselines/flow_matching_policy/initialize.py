"""Freeze prior evidence before the last, Flow Matching Policy-only round."""
import subprocess
from baselines.strong.common import ROOT,PROJECT,out,read,write,digest,now,resource

BASE='3c9671e8ca50cfca06b8f6e46ff3e97d1461cb4c'
METHOD='flow_matching_policy'

def main():
    run=ROOT/METHOD;run.mkdir(parents=True,exist_ok=True)
    path=run/'preservation_freeze.json'
    if path.exists():
        print('EXISTING_PRESERVATION_NO_RESET');return
    old=read(ROOT/'smolvla/preservation_freeze.json')
    methods=['act','vqbet','mimicgen_v2','kite','crosszero','object_flow','octo_small','smolvla']
    trees=[dict(path='results/strong_baselines/'+m,tree=subprocess.check_output(
        ['git','rev-parse',BASE+':results/strong_baselines/'+m],cwd=PROJECT,text=True).strip()) for m in methods]
    smol=read(out('smolvla')/'artifact_index.json')['files']
    data=read(out('smolvla')/'protocol.json')['data_files']
    files={r['path']:r for r in old['protected_old_files']+old['octo_artifacts']+smol+data}
    for r in files.values():assert digest(r['path'])==r['sha256'],r['path']
    snapshot=dict(base_commit=BASE,utc=now(),trees=trees,files=list(files.values()),
        protected_smolvla_artifacts=len(smol),protected_octo_artifacts=len(old['octo_artifacts']),
        protected_legacy_files=len(old['protected_old_files']),resources=resource())
    write(path,snapshot);write(out(METHOD)/'prior_preservation_snapshot.json',snapshot)
    scope=read(PROJECT/'results/strong_baselines/research_scope.json')
    scope.update(revision='20261113_flow_matching_last_baseline',base_commit=BASE,active_method=METHOD,
        current_round_only=METHOD,last_baseline_this_round=True,stop_after_completion_or_substantive_blocker=True)
    scope['preserve_without_rerun']=list(dict.fromkeys(scope['preserve_without_rerun']+['octo_small','smolvla']))
    write(PROJECT/'results/strong_baselines/research_scope.json',scope)
    write(out(METHOD)/'status.json',dict(status='PARTIAL',method=METHOD,phase='OFFICIAL_IMPLEMENTATION_AUDIT',
        actual_training_updates=0,physical_evaluation_episodes=0,physical_smoke_episodes=0,
        success_rate=None,ordinary_repairs_allowed=True,formal_budget_frozen=False))
    write(ROOT/'session_resume.json',dict(phase='FLOW_MATCHING_OFFICIAL_AUDIT',active_method=METHOD,
        round_only=METHOD,base_commit=BASE,utc=now(),
        next='Inspect official HRI-EU core/Robomimic, real update and max2 smokes, freeze one run; no other baseline/Method'))
    from baselines.strong.matrix_current import main as matrix
    matrix();print('PRESERVED_ALL_PRIOR_EVIDENCE',len(files))

if __name__=='__main__':main()
