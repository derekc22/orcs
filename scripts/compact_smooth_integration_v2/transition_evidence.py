"""Reference-frame comparison and timing evidence from recorded physical states."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'logs/compact_smooth_integration_v2'
fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
for label,path,start,end,col in [('Accepted support',ROOT/'logs/compact_integration_v2/corner_recovered/rollout.npz',5901,6501,'#777777'),('Fade/blend without COM shift',OUT/'fade_only_control/rollout.npz',5901,None,'#b96635'),('Corrected COM + foot yaw',OUT/'canonical_entry/rollout.npz',5901,None,'#167e96')]:
 a=np.load(path);end=len(a['qpos']) if end is None else end;q=a['qpos'][start:end];v=a['qvel'][start:end];t=np.arange(len(q))*.005;pitch=R.from_quat(np.roll(q[:,3:7],-1,axis=1)).as_euler('xyz',degrees=True)[:,1];ax[0].plot(t,pitch,label=label,color=col);ax[1].plot(t,np.linalg.norm(v[:,3:6],axis=1),label=label,color=col)
ax[0].set_ylabel('Root pitch [deg]');ax[1].set_ylabel('Root angular speed [rad/s]');ax[1].set_xlabel('Time after braking [s]');ax[0].legend();[x.grid(alpha=.2) for x in ax];fig.tight_layout();fig.savefig(OUT/'support_transition_comparison.png',dpi=150)
if not (OUT/'complete_execution/rollout_verification_trace.npz').exists():sys.exit()
fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
for label,name,col in [('Fixed 700 ms wait','canonical_placement_probe','#b96635'),('Physical support gate only','complete_execution','#167e96')]:
 p=OUT/name;a=np.load(p/'rollout.npz');trace=np.load(p/'rollout_verification_trace.npz')['trace'];s=int(a['placement_start_physics_step']);b=s+round(5.3/.005);e=min(len(trace),s+round(8/.005));t=np.arange(b-s,e-s)*.005;x=trace[b:e]
 ax[0].plot(t,x[:,42]/(.05*9.81),label=label,color=col);ax[1].plot(t,np.linalg.norm(x[:,23:26],axis=1),color=col);ax[2].plot(t,np.linalg.norm(x[:,39:42],axis=1),color=col)
ax[0].axhline(.7,color='gray',ls='--',lw=1);ax[0].set_ylim(-.2,6);ax[0].set_ylabel('Terrain load / box weight');ax[1].set_ylabel('Box angular speed [rad/s]');ax[2].set_ylabel('Root angular speed [rad/s]');ax[2].set_xlabel('Time since placement entry [s]');ax[0].legend()
for x in ax:x.axvline(6.1,color='#167e96',ls='--');x.axvline(6.695,color='#b96635',ls=':');x.grid(alpha=.2)
fig.suptitle('Opening at the measured load gate avoids the later terrain strike');fig.tight_layout();fig.savefig(OUT/'release_timing_comparison.png',dpi=150)
