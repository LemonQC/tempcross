"""Read-only stage snapshot and plots; training and evaluation never launched."""
import csv
from baselines.strong.common import ROOT,TASKS,out,read,write,digest,now

def main():
    folder=out('mimicgen_v2');cfg=read(folder/'protocol.json');assert str(ROOT)==cfg['server_root']
    path=ROOT/'mimicgen_v2/training/status.json';state=read(path)
    cells=[]
    for task in TASKS:
        p=folder/f'evaluation_human_plus_generated_{task}.json'
        if p.exists():
            a=read(p);assert a['status']=='PASS' and a['episodes']==40
            cells.append(dict(group='human_plus_generated',task=task,episodes=40,successes=a['successes'],success_rate=a['successes']/40,role='EXPOSED_REFERENCE'))
    closed=len(list((ROOT/'mimicgen_v2/physical/formal').glob('*/*/condition-*/episode.json')))
    final=folder/'results.json';complete=final.exists() and read(final).get('component_status')=='BOUNDED_DOWNSTREAM_CONTROL_COMPLETE'
    if not complete:
        status=read(folder/'status.json');status.update(physical_evaluation_episodes=sum(c['episodes'] for c in cells),closed_formal_episodes=closed)
        write(folder/'status.json',status)
        result=dict(status='PARTIAL',scope=cfg['method'],snapshot_utc=now(),cells=cells,actual_saved_training_updates=state['actual_optimizer_updates'],
            completed_training=state['status']=='COMPLETE',closed_formal_episodes=closed,accepted_formal_episodes=sum(c['episodes'] for c in cells),
            human_anchor_exposures=state['human_anchor_exposures'],generated_anchor_exposures=state['generated_anchor_exposures'],
            role='EXPOSED_REFERENCE',independent_unseen_test=False,full_original_mimicgen_reproduced=False)
        write(folder/'partial_results.json',result)
        text=['# MimicGen v2阶段快照','','总体PARTIAL；仅有界目标增强组件。未完成单元不填0%。','',
            f"保存的训练更新：{state['actual_optimizer_updates']}/48000；闭合正式episode：{closed}；完整验收：{result['accepted_formal_episodes']}条。",'',
            '| 已验收任务 | 官方成功/实际次数 |','|---|---:|']
        for c in cells:text.append(f"| {c['task']} | {c['successes']}/40 |")
        (folder/'PARTIAL_RESULTS.md').write_text('\n'.join(text)+'\n')
    if state['status']!='COMPLETE':
        print('Saved updates',state['actual_optimizer_updates'],'closed episodes',closed);return
    selected=read(folder/'selection.json');assert digest(selected['checkpoint'])==selected['sha256']
    control=read(folder/'reused_control.json')
    write(folder/'training_artifact_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted((ROOT/'mimicgen_v2/training').iterdir()) if p.is_file()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=read('/home/users/lemon/baseline-benchmark/run-20261109/training/B_bc_transformer/selected.json')
    series=[('Human-only (historical, reused)',old['validation_history'],'optimizer_step','#3269a8'),('Human + generated',state['validation_history'],'step','#cc6b35')]
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),constrained_layout=True);rows=[]
    for label,history,key,color in series:
        x=[r[key] for r in history]
        for ax,metric in zip(axes,['value','nll']):
            y=[r[metric] for r in history];ax.plot(x,y,lw=1.5,color=color,label=label);ax.scatter([x[-1]],[y[-1]],marker='*',s=90,color=color,zorder=4)
        rows.extend(dict(series=label,optimizer_update=r[key],expected_action_mse=r['value'],native_nll=r['nll']) for r in history)
    axes[0].set(title='Same held-out human anchors',ylabel='Expected-action MSE',xlabel='Actual optimizer updates')
    axes[1].set(title='Fixed final checkpoint: 48,000',ylabel='Native GMM negative log likelihood',xlabel='Actual optimizer updates')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    directory=folder/'figures';directory.mkdir(exist_ok=True);p=directory/'validation.svg';fig.savefig(p,metadata={'Date':None});plt.close(fig);figures=[p]
    with (folder/'validation_curve.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['series','optimizer_update','expected_action_mse','native_nll'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    if complete:
        final_result=read(final);bars=[]
        for task in TASKS:
            old_cell=next(c for c in control['cells'] if c['task']==task);new_cell=next(c for c in final_result['cells'] if c['task']==task)
            bars.extend([(task,'Historical human-only',old_cell['successes']),(task,'Human + generated',new_cell['successes'])])
        fig,ax=plt.subplots(figsize=(8.5,3.8),constrained_layout=True)
        labels=[('Counter to Sink' if t==TASKS[0] else 'Sink to Counter')+'\n'+name for t,name,k in bars]
        drawn=ax.bar(range(4),[100*k/40 for t,name,k in bars],color=['#3269a8','#cc6b35']*2)
        for bar,(_,_,k) in zip(drawn,bars):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{k}/40',ha='center')
        ax.set(xticks=range(4),xticklabels=labels,ylim=(0,100),ylabel='Official full-task success (%)',title='Bounded Panda augmentation / EXPOSED_REFERENCE')
        ax.grid(axis='y',alpha=.2);p=directory/'reference_comparison.svg';fig.savefig(p,metadata={'Date':None});plt.close(fig);figures.append(p)
    write(folder/'figure_index.json',dict(figures=[dict(path=str(p),sha256=digest(p)) for p in figures],new_physics_steps=0,new_training_updates=0,
        historical_control_episodes_reused=80,missing_cells_not_plotted_as_zero=True,full_original_mimicgen_claim=False))
    from baselines.strong.matrix_details import main as matrix
    matrix();print('Completed-training snapshot and',len(figures),'plot(s),no new experiments')

if __name__=='__main__':main()
