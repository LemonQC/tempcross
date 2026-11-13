"""Plot the complete native training/validation record, without new inference."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines.strong.common import out,read

def main():
    directory=out('octo_small');records=read(directory/'training_metrics.json')['records']
    training=[r for r in records if 'metrics' in r];validation=[r for r in records if 'validation_ddpm_loss' in r]
    result=read(directory/'results.json');selected=result['selected_update']
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    ax.plot([r['update']/1000 for r in training],[r['metrics']['loss'] for r in training],
        color='#777777',alpha=.4,lw=.8,label='Augmented source+target training batches')
    ax.plot([r['update']/1000 for r in validation],[r['validation_ddpm_loss'] for r in validation],
        color='#0072B2',marker='o',ms=3,label='Fixed target validation (256 anchors)')
    ax.scatter([selected/1000],[result['selected_validation_ddpm_loss']],s=75,facecolors='none',
        edgecolors='#D55E00',lw=1.5,label=f'Selected update {selected:,}')
    ax.set(xlabel='Optimizer updates (thousands)',ylabel='Native DDPM loss',xlim=(0,24),
        title='Octo-Small: frozen source+target finetuning')
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    path=directory/'training_curve.svg';fig.savefig(path);plt.close(fig)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    text=['update,target_validation_ddpm_loss,selected']
    text += [f"{r['update']},{r['validation_ddpm_loss']},{int(r['update']==selected)}" for r in validation]
    (directory/'validation.csv').write_text('\n'.join(text)+'\n')

if __name__=='__main__':main()
