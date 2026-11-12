"""Expose accepted cells honestly while the full ACT baseline remains PARTIAL."""
from collections import Counter
from baselines.strong.common import out,read,write,now

def main():
    folder=out('act');cells=[]
    for path in sorted(folder.glob('evaluation_*.json')):
        result=read(path)
        assert result['status']=='PASS' and result['role']=='EXPOSED_REFERENCE' and result['episodes']==40
        rows=result['rows'];assert len(rows)==40
        assert {r['episode']['official_condition_index'] for r in rows}==set(range(60,100))
        group=result['group'];selection=read(folder/(group+'_selection.json'))
        assert not selection['used_physical_success']
        assert all(r['episode']['checkpoint_sha256']==selection['sha256'] for r in rows)
        cells.append(dict(group=group,task=result['task'],episodes=40,successes=result['successes'],success_rate=result['successes']/40,
            role='EXPOSED_REFERENCE',checkpoint_sha256=selection['sha256'],selected_step=selection['selected']['step'],
            failure_counts=dict(Counter(r['failure_stage'] for r in rows)),all_trajectories_and_videos_accepted=True))
    write(folder/'partial_results.json',dict(status='PARTIAL',scope='Accepted cells only; neither remaining cells nor the complete method are claimed completed.',
        utc=now(),cells=cells,accepted_formal_episodes=sum(c['episodes'] for c in cells),independent_unseen_test=False))
    from baselines.strong.matrix_details import main as matrix
    matrix()
    print(__import__('json').dumps(cells,indent=2))

if __name__=='__main__':main()
