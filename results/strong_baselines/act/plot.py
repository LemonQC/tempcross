"""Plots only accepted complete ACT results; no policy or simulation calls."""
from baselines.strong.common import ROOT,out,read,write,digest

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out('act');results=read(folder/'results.json')
    assert results['status']=='COMPLETE_FAITHFUL_PORT' and results['formal_episodes']==160
    dest=folder/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=False)
    for ax,group in zip(axes,['target_only','source_target']):
        state=read(ROOT/'act/training'/group/'status.json');assert state['status']=='COMPLETE'
        points=state['validation_history'];assert len(points)==24
        ax.plot([p['step'] for p in points],[p['loss'] for p in points],label='L1 + 10 KL',marker='.',lw=1.4)
        ax.plot([p['step'] for p in points],[p['l1'] for p in points],label='Masked L1',lw=1)
        ax.axvline(state['best']['step'],ls=':',color='black',label='Selected checkpoint')
        ax.set(title=group.replace('_',' '),xlabel='Actual optimizer updates',ylabel='Held-out target validation loss')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('ACT: fixed validation; each setting uses its own training-only action normalizer',fontsize=10)
    fig.tight_layout();fig.savefig(dest/'validation.svg',metadata={'Date':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    labels=[c['group'].replace('_',' ')+'\n'+c['task'].replace('pnp_','').replace('_',' ') for c in results['cells']]
    values=[100*c['successes']/c['episodes'] for c in results['cells']]
    bars=ax.bar(range(4),values,color=['#3973ac','#3973ac','#d18b32','#d18b32'])
    for bar,c in zip(bars,results['cells']):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{c['successes']}/{c['episodes']}",ha='center')
    ax.set(xticks=range(4),xticklabels=labels,ylim=(0,105),ylabel='Official complete-task success (%)',title='ACT faithful port: EXPOSED_REFERENCE conditions 60–99')
    ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(dest/'reference_success.svg',metadata={'Date':None});plt.close(fig)
    write(folder/'figure_index.json',dict(status='COMPLETE',independent_unseen_test=False,
        source_results_sha256=digest(folder/'results.json'),files=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(dest.glob('*.svg'))]))

if __name__=='__main__':main()
