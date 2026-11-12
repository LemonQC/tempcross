"""Project durable server progress into PARTIAL metadata; never mark completion."""
from baselines.strong.common import ROOT,out,read,write,now

def main():
    folder=out('act');repair=read(folder/'technical_repair.json');assert str(ROOT)==repair['corrected_server_root']
    status=read(folder/'status.json')
    if status['status'].startswith('COMPLETE'):return
    total=target=source=0;groups={}
    for group in ['target_only','source_target']:
        p=ROOT/'act/training'/group/'status.json'
        if not p.exists():continue
        s=read(p);total+=s['actual_optimizer_updates'];target+=s['target_anchor_exposures'];source+=s['source_anchor_exposures']
        groups[group]=dict(status=s['status'],actual_optimizer_updates=s['actual_optimizer_updates'],best_validation=s['best'],server_status=str(p))
    physical=ROOT/'act/physical/formal';closed=list(physical.glob('*/*/condition-*/episode.json'));accepted=[]
    for p in closed:
        check=p.parent/'acceptance.json'
        if check.exists() and read(check)['status']=='PASS':accepted.append(p)
    status.update(actual_training_updates=total,corrected_actual_training_updates=total,
        corrected_target_anchor_exposures=target,corrected_source_anchor_exposures=source,
        recorded_formal_episodes=len(closed),physical_evaluation_episodes=len(accepted),
        success_rate=None,training_groups=groups,last_verified_utc=now(),
        prior_technical_attempt_updates=24000,total_executed_training_updates=24000+total,
        total_probe_optimizer_updates=4,technical_repairs=1,active_server_root=str(ROOT),
        completion_is_not_inferred_from_progress=True)
    write(folder/'status.json',status)
    from baselines.strong.matrix_details import main as matrix
    matrix()
    print(__import__('json').dumps({k:status[k] for k in ['status','actual_training_updates','recorded_formal_episodes','physical_evaluation_episodes','total_executed_training_updates','last_verified_utc']}))

if __name__=='__main__':main()
