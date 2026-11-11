"""Standalone scientific figures from recorded results; zero model/physics calls."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from common import OUT,ROOT,digest,read,write


def main():
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'barx-target-adaptation-fixed',
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    recorded=list(csv.DictReader((ROOT/'training/updates.csv').open()))
    x=np.array([int(r['update']) for r in recorded]);loss=np.array([float(r['loss']) for r in recorded])
    curve=read(OUT/'validation_curve.json');selected=read(OUT/'selection.json')['step']
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),constrained_layout=True)
    axes[0].plot(x,loss,color='#718096',alpha=.30,lw=.5,label='Recorded update loss')
    if len(x)>=50:
        axes[0].plot(x[49:],np.convolve(loss,np.ones(50)/50,mode='valid'),color='#2459a6',lw=1.5,label='50-update mean')
    axes[0].set(title='Mixed-domain Joint training objective',xlabel='Optimizer update',ylabel='Weighted token loss')
    axes[0].legend(fontsize=8)
    for ax,tag,title,color in [(axes[1],'validation_target','Panda validation (10 held-out demos)','#bd3c37'),
                               (axes[2],'validation_source_ur5e','UR5e diagnostic (prior may have seen demos)','#168267')]:
        rows=sorted([r for r in curve if r['tag']==tag],key=lambda r:r['step'])
        xx=[r['step'] for r in rows];yy=[r['macro_task_response_nll'] for r in rows]
        ax.plot(xx,yy,'o-',color=color,lw=1.5,ms=4)
        if selected in xx:
            ax.scatter([selected],[yy[xx.index(selected)]],s=90,facecolors='none',edgecolors='black',zorder=5,label=f'Selected update {selected}')
        ax.set(title=title,xlabel='Optimizer update',ylabel='Macro task action-response NLL')
        ax.axvline(selected,color='#333333',ls=':',lw=.8,alpha=.6)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed budget: 3,000 updates / 768,000 sample positions. Validation NLL is not physical success.',fontsize=11)
    path=dest/'training_and_validation.svg';fig.savefig(path,metadata={'Date':None});fig.savefig(ROOT/'training_and_validation_preview.png',dpi=110);plt.close(fig)
    files=[path]
    results=OUT/'results.json'
    if results.exists():
        cells=read(results)['cells']
        settings=[('pnp_counter_to_sink','panda','Counter to Sink / Panda'),('pnp_sink_to_counter','panda','Sink to Counter / Panda'),
                  ('pnp_counter_to_sink','ur5e','Counter to Sink / UR5e'),('pnp_sink_to_counter','ur5e','Sink to Counter / UR5e')]
        models=[('A_barx_joint','Original Joint','#718096'),('C_barx_joint_96000','Legacy 96k LoRA (Panda only)','#d19a31'),
                ('new_joint_adaptation_768000','New validation-selected LoRA','#2459a6')]
        fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
        for j,(label,name,color) in enumerate(models):
            for i,(task,robot,_) in enumerate(settings):
                rr=[r for r in cells if r['label']==label and r['task']==task and r['robot']==robot]
                if not rr:continue
                r=rr[0];n=r['episodes'];k=r['successes'];p=k/n;z=1.959963984540054
                denom=1+z*z/n;center=(p+z*z/(2*n))/denom
                half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
                xx=i+(j-1)*.24
                ax.bar(xx,100*p,width=.22,color=color,label=name if i==0 else None)
                ax.errorbar(xx,100*p,yerr=np.array([[100*(p-center+half)],[100*(center+half-p)]]),color='black',capsize=3,lw=.8)
                ax.text(xx,100*(center+half)+2,f'{k}/{n}',ha='center',fontsize=8)
        ax.set(xticks=np.arange(4),xticklabels=[s[2] for s in settings],ylabel='Official complete-task success (%)',ylim=(0,100))
        ax.legend(fontsize=9,loc='upper right')
        ax.set_title('Previously exposed frozen reference conditions 60-99; no old physical experiment rerun')
        fig.supxlabel('40 episodes/cell. Error bars: descriptive per-cell Wilson 95% intervals, not causal effect tests.',fontsize=9)
        path=dest/'exposed_reference_success.svg';fig.savefig(path,metadata={'Date':None});fig.savefig(ROOT/'exposed_reference_success_preview.png',dpi=110);plt.close(fig);files.append(path)
    for path in files:
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    inputs=[str(ROOT/'training/updates.csv'),str(OUT/'validation_curve.json'),str(OUT/'selection.json')]
    if results.exists():inputs.append(str(results))
    write(OUT/'figure_index.json',[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=digest(p),
          inputs=inputs,
          new_model_or_physics_calls=0) for p in files])


if __name__=='__main__':main()
