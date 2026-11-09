"""Source, constrained reference and executed tilt, against same-seed Goal 6."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[2]
ACTIVE={'pickup':(540,1601),'carry':(3449,4722),'placement':(6701,7761)}


def plot(out):
    fig,axes=plt.subplots(4,3,figsize=(14,10),constrained_layout=True)
    for seed,color in [(42,'tab:orange'),(43,'tab:blue')]:
        z=np.load(out/f'safe_{seed}_diagnostics.npz');new=np.load(out/f'execution_{seed}/rollout.npz')['qpos'];old=np.load(ROOT/f'logs/compact_diffusion_root_v1/execution_{seed}/rollout.npz')['qpos']
        rotation=(R.from_quat(new[:,3:7],scalar_first=True)*R.from_quat(old[:,3:7],scalar_first=True).inv()).as_rotvec()[:,:2]
        actual=np.c_[rotation,new[:,20:22]-old[:,20:22]];p=z['parameters'];refined=p[:,np.r_[0,1,z['waist_variable_indices']]]
        target=np.r_[refined[1:],refined[-1:]];requested=np.r_[z['requested_modes'][1:],z['requested_modes'][-1:]]
        for col,(phase,(s,e)) in enumerate(ACTIVE.items()):
            time=np.arange(s,e)*.005
            for row,name in enumerate(['Root world-horizontal X','Root world-horizontal Y','Waist roll','Waist pitch']):
                ax=axes[row,col]
                ax.plot(time,np.rad2deg(requested[s:e,row]),':',color=color,alpha=.45,label=f'{seed} frozen innovation')
                ax.plot(time,np.rad2deg(target[s:e,row]),'--',color=color,alpha=.8,label=f'{seed} constrained reference')
                ax.plot(time,np.rad2deg(actual[s:e,row]),color=color,label=f'{seed} physical change')
                ax.set_title(f'{phase}: {name}');ax.set_xlabel('Task time (s)');ax.set_ylabel('Added rotation (degrees)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6)
    fig.savefig(out/'executed_tilt_attribution.png',dpi=150);fig.savefig(out/'executed_tilt_attribution.pdf');plt.close(fig)
