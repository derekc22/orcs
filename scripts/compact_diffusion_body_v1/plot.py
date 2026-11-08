"""Executed Goal 4/5 comparisons; standalone scientific artifacts."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from report import ACTIVE,G4


def plot(out):
    pairs=[]
    for root,label,color in [(G4,'Goal 4','#6d7c8a'),(out,'Goal 5','#b12e44')]:
        a,b=[np.load(root/f'execution_{s}/rollout.npz')['qpos'] for s in [42,43]]
        d=a-b
        angle=(R.from_quat(a[:,3:7],scalar_first=True)*R.from_quat(b[:,3:7],scalar_first=True).inv()).as_rotvec()
        pairs.append((label,color,[1000*np.linalg.norm(d[:,:3],axis=1),np.degrees(np.linalg.norm(angle,axis=1)),
            np.degrees(d[:,19]),np.degrees(np.sqrt(np.mean(d[:,np.r_[22:29,33:40]]**2,axis=1))),1000*np.linalg.norm(d[:,44:47],axis=1)]))
    fig,axes=plt.subplots(5,3,figsize=(13,12),layout='constrained')
    for col,(phase,(s,e)) in enumerate(ACTIVE.items()):
        t=np.arange(s,e)*.005
        for row in range(5):
            ax=axes[row,col]
            for label,color,signals in pairs:
                ax.plot(t,signals[row][s:e],color=color,label=label,lw=1.6,ls='-' if label=='Goal 5' else '--')
            ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
            if row!=2:ax.set_ylim(bottom=0)
            if row==0:ax.set_title(phase.capitalize())
            if row==4:ax.set_xlabel('Physical task time (s)')
    labels=['Root position\ndifference (mm)','Root orientation\ndifference (degrees)','Waist yaw\ndifference (degrees)','Arm joint RMS\ndifference (degrees)','Object position\ndifference (mm)']
    for row,label in enumerate(labels):axes[row,0].set_ylabel(label)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Actual ORCS sample 42 vs 43: expanded body authority with retained arm/object motion',fontsize=13)
    for suffix in ['png','pdf']:fig.savefig(out/f'executed_body_authority.{suffix}',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();plot(a.out)
