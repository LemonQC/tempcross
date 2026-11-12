"""Plots from completed training/accepted physics only; no model execution."""
import csv
from baselines.strong.common import ROOT,out,read,write,digest

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out('vqbet');cfg=read(folder/'protocol.json')
    assert str(ROOT)==read(folder/'technical_repair.json')['corrected_server_root']
    directory=folder/'figures';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),constrained_layout=True);rows=[];inputs=[]
    for group,color in [('target_only','#3269a8'),('source_target','#cc6b35')]:
        for ax,stage in zip(axes,['vqvae','bet']):
            path=folder/('training_'+group+'_'+stage+'.json')
            if not path.exists():continue
            state=read(path)
            if state['status']!='COMPLETE':continue
            inputs.append(dict(path=str(path),sha256=digest(path)))
            history=state['validation_history'];x=[r['step'] for r in history];y=[r['loss'] for r in history]
            ax.plot(x,y,label=group,color=color,lw=1.5)
            ax.scatter([state['best']['step']],[state['best']['loss']],marker='*',s=100,color=color,zorder=5)
            rows.extend(dict(group=group,stage=stage,update=r['step'],validation_loss=r['loss'],selected=r['step']==state['best']['step']) for r in history)
    axes[0].set(title='Residual VQ: held-out target reconstruction',ylabel='L1 in canonical action units',xlabel='Actual RVQ optimizer updates')
    axes[1].set(title='BeT: held-out target native objective',ylabel='Weighted focal-code + offset L1',xlabel='Actual BeT stage updates')
    axes[1].axvline(cfg['bet_full_optimizer_updates'],color='gray',ls='--',lw=1,label='Offset-only begins')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    path=directory/'validation.svg';fig.savefig(path,metadata={'Date':None});plt.close(fig)
    with (folder/'validation_curve.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['group','stage','update','validation_loss','selected'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    figures=[path]
    final=folder/'results.json'
    if final.exists() and read(final)['status']=='COMPLETE_FAITHFUL_PORT':
        result=read(final);cells=result['cells'];assert len(cells)==4 and all(c['episodes']==40 for c in cells)
        fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True)
        labels=[c['group']+'\n'+('Counter to Sink' if c['task'].endswith('counter_to_sink') else 'Sink to Counter') for c in cells]
        values=[100*c['success_rate'] for c in cells]
        bars=ax.bar(range(4),values,color=['#3269a8' if c['group']=='target_only' else '#cc6b35' for c in cells])
        for bar,c in zip(bars,cells):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{c['successes']}/40",ha='center',fontsize=10)
        ax.set(xticks=range(4),xticklabels=labels,ylim=(0,100),ylabel='Official complete-task success (%)',title='VQ-BeT / Panda / EXPOSED_REFERENCE 60-99')
        ax.grid(axis='y',alpha=.2);path=directory/'reference_success.svg';fig.savefig(path,metadata={'Date':None});plt.close(fig)
        figures.append(path);inputs.append(dict(path=str(final),sha256=digest(final)))
    write(folder/'figure_index.json',dict(figures=[dict(path=str(p),sha256=digest(p)) for p in figures],inputs=inputs,
        only_completed_training_and_accepted_physics=True,missing_cells_not_plotted_as_zero=True,new_optimizer_updates=0,new_physics_steps=0))
    print('Wrote',len(figures),'figure(s),',len(rows),'validation points')

if __name__=='__main__':main()
