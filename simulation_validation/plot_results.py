"""Plot completed official physical evaluation cells and Wilson intervals."""
import argparse,json
from pathlib import Path


def plot(analysis,out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='barx-p2-fixed-evaluation'
    import matplotlib.pyplot as plt
    data=json.loads(Path(analysis).read_text());cells=data['cells']
    if len(cells)!=4 or any(c['status']!='COMPLETE' for c in cells):
        raise RuntimeError('All four physical cells must complete before plotting final results')
    labels=[c['method'].replace('_',' ')+'\n'+c['robot'] for c in cells]
    rates=[100*c['success_rate'] for c in cells]
    errors=[[100*(c['success_rate']-c['wilson95'][0]) for c in cells],
            [100*(c['wilson95'][1]-c['success_rate']) for c in cells]]
    fig,ax=plt.subplots(figsize=(7,4));bars=ax.bar(labels,rates,color=['#356f9c','#db8d35']*2)
    ax.errorbar(range(4),rates,yerr=errors,fmt='none',ecolor='black',capsize=5)
    for bar,c in zip(bars,cells):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{c['successes']}/{c['completed']}",ha='center')
    ax.set(ylabel='Official physical task success (%)',ylim=(0,105),title='BARX source-prior direct transfer: first 20 fixed conditions')
    ax.spines[['top','right']].set_visible(False);ax.text(.01,-.25,'Wilson 95% intervals; small sample. No target fine-tuning or Eye-in-Hand change.',transform=ax.transAxes,fontsize=8)
    fig.tight_layout();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        path=out/('physical_success.'+suffix)
        fig.savefig(path,dpi=180,bbox_inches='tight',metadata={'Date':None} if suffix=='svg' else None)
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--out',required=True);a=p.parse_args();plot(a.analysis,a.out)
