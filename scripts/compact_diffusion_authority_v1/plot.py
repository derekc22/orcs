"""Standalone scientific figure of executed sample-dependent motion."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from refine import PHASES


def plot(out):
    a,b,base=[np.load(out/p/'rollout.npz') for p in ['execution_42','execution_43','disabled']]
    arms=np.r_[22:29,33:40]
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for col,(phase,(start,end,_,_)) in enumerate(PHASES.items()):
        first,last=round(start/.005),round(end/.005)
        t=np.arange(first,last+1)*.005
        for x,y,label,color,style in [(a,base,'42 vs disabled','#2773ae','--'),(b,base,'43 vs disabled','#269568','--'),(a,b,'42 vs 43','#b63440','-')]:
            d=x['qpos'][first:last+1]-y['qpos'][first:last+1]
            axes[0,col].plot(t,1000*np.linalg.norm(d[:,44:47],axis=1),style,color=color,label=label,lw=1.6)
            axes[1,col].plot(t,np.degrees(np.sqrt(np.mean(d[:,arms]**2,axis=1))),style,color=color,lw=1.6)
        axes[0,col].set_title(phase.capitalize())
        axes[0,col].axhline(10,color='#999999',lw=.7,ls=':')
        for row in range(2):
            axes[row,col].grid(alpha=.18);axes[row,col].set_ylim(bottom=0)
            axes[row,col].spines[['right','top']].set_visible(False)
        axes[1,col].set_xlabel('Physical task time (s)')
    axes[0,0].set_ylabel('Object position\ndifference (mm)')
    axes[1,0].set_ylabel('Arm joint RMS\ndifference (degrees)')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Frozen diffusion authority in actual ORCS execution',fontsize=15)
    for suffix in ['png','pdf']:
        fig.savefig(out/f'executed_authority.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();plot(a.out)
