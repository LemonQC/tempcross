"""Standalone scientific summary from saved data only; no simulator imports."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from mechanism_validation.common import *


def main():
    data=json.loads((OUT/'analysis.json').read_text())
    grid=np.zeros((8,3))
    for y,i in enumerate(CASES):
        for x,m in enumerate(['M0','M1','M2']):
            r=next(r for r in data['episodes'] if r['condition']==i and r['method']==m)
            grid[y,x]=2 if r['task_success'] else 1 if r['stable_lift'] else 0
    matplotlib.rcParams['svg.hashsalt']='mechanism-20261110'
    fig,(left,right)=plt.subplots(1,2,figsize=(10,5.6),gridspec_kw={'width_ratios':[1,1.2]})
    colors=['#c9cdd2','#f0b35e','#4caa78']
    left.imshow(grid,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),aspect='auto')
    left.set_xticks(range(3),['M0 fixed pose','M1 BARX','M2 object feedback'],rotation=15,ha='right')
    left.set_yticks(range(8),[str(i) for i in CASES]);left.set_ylabel('Original official condition index')
    for y in range(8):
        for x in range(3):left.text(x,y,['No stable lift','Lift only','Task success'][int(grid[y,x])],ha='center',va='center',fontsize=8)
    left.set_title('All 24 first attempts')
    summary=data['summary'];x=np.arange(3)
    right.bar(x-.18,[r['stable_lift'] for r in summary],.34,color='#7790ac',label='Stable lift diagnostic')
    right.bar(x+.18,[r['official_success'] for r in summary],.34,color=colors[2],label='Official task success')
    right.set_xticks(x,['M0','M1','M2']);right.set_ylim(0,8.7);right.set_yticks(range(9));right.set_ylabel('Episodes / 8')
    for k,r in enumerate(summary):
        right.text(k-.18,r['stable_lift']+.1,str(r['stable_lift']),ha='center')
        right.text(k+.18,r['official_success']+.1,str(r['official_success']),ha='center')
    right.legend(loc='upper right',fontsize=8);right.grid(axis='y',alpha=.2);right.set_axisbelow(True)
    right.set_title('Proxy and official success kept separate')
    fig.suptitle('Final mechanism diagnostics: UR5e source to Panda target',fontsize=12)
    fig.text(.5,.025,'Outcome-selected hard development cases; not an independent test or algorithm ranking.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(OUT/'outcomes.svg',metadata={'Date':None})


if __name__=='__main__':main()
