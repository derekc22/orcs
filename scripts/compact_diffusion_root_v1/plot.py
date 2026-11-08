"""Export source, reference and physical XY attribution without fitted alignment."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ACTIVE={'pickup':(540,1601),'carry':(3449,4722),'placement':(6701,7761)}


def plot(out):
    fig,axes=plt.subplots(4,3,figsize=(14,10),constrained_layout=True)
    for col,(phase,(s,e)) in enumerate(ACTIVE.items()):
        time=np.arange(s,e)*.005
        for row,(seed,axis) in enumerate([(42,0),(42,1),(43,0),(43,1)]):
            ax=axes[row,col];z=np.load(out/f'proposal_{seed}_diagnostics.npz')
            new=np.load(out/f'execution_{seed}/rollout.npz')['qpos'];old=np.load(ROOT/f'logs/compact_diffusion_body_v1/execution_{seed}/rollout.npz')['qpos']
            requested=np.r_[z['requested_xy'][1:],z['requested_xy'][-1:]];refined=np.r_[z['refined_xy'][1:],z['refined_xy'][-1:]]
            ax.plot(time,1000*requested[s:e,axis],color='.65',label='Frozen source after support mask')
            ax.plot(time,1000*refined[s:e,axis],color='tab:orange',label='Constrained reference')
            ax.plot(time,1000*(new[s:e,axis]-old[s:e,axis]),color='tab:blue',label='Physical change vs same-seed Goal 5')
            ax.axhline(0,color='.8',lw=.5);ax.set_title(f'{phase}: sample {seed}, root {"XY"[axis]}');ax.set_ylabel('Added motion (mm)');ax.set_xlabel('Task time (s)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.savefig(out/'executed_root_attribution.png',dpi=150);fig.savefig(out/'executed_root_attribution.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();plot(a.out.resolve())
