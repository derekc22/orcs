"""Plot measured task evidence; consumes audit.py and raw replay outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(p):
    a=np.load(p/'rollout.npz');t=np.load(p/'rollout_verification_trace.npz')['trace'];l=np.load(p/'rollout_locomotion_trace.npz')['trace'];r=json.loads((p/'task_metrics.json').read_text())
    x=np.load(p/'grasp_audit_trace.npz')['plate_object_pose'];s,e=int(a['walking_start_physics_step']),int(a['walking_end_physics_step']);tt=np.arange(len(t))*.005;tw=tt[s:e]-s*.005
    fig,axs=plt.subplots(2,3,figsize=(15,8.4));axs=axs.ravel();colors=['#2066ad','#d46921']
    axs[0].plot(tw,np.linalg.norm(a['qpos'][s:e,:2]-a['qpos'][s,:2],axis=1),label='Actual root')
    axs[0].plot(tw,np.linalg.norm(t[s:e,11:13]-t[s,11:13],axis=1),label='Box COM')
    axs[0].axhline(2,color='black',ls='--',label='Required 2.0 m');axs[0].axhline(2.526,color='gray',ls=':',label='Accepted baseline 2.526 m');axs[0].set(ylabel='Horizontal net displacement (m)',title='Airborne walking only')
    for i,side in enumerate(['Left','Right']):
        axs[1].plot(tw,l[s:e,12+i]*100,label=side,color=colors[i])
        axs[2].plot(tw,l[s:e,16+i],label=side,color=colors[i],lw=.8)
    for ev in r['bilateral_retained']['qualifying_replants']:
        axs[1].axvline(ev['replant_s']-s*.005,color=colors[ev['side']=='right'],alpha=.3,lw=.8)
    axs[1].set(ylabel='Sole clearance (cm)',title='Ten alternating qualifying replants');axs[2].set(ylabel='Measured hand contact force (N)',title='Bilateral physical retention')
    axs[3].plot(tt,t[:,5],label='Box COM lift (m)');axs[3].plot(tt,t[:,42],label='Terrain vertical load (N)',lw=.8)
    axs[3].axvspan(s*.005,e*.005,color='green',alpha=.12,label='Counted walking');axs[3].axvline(float(a['phase_opening_frame'])*.02,color='black',ls='--',label='Palms/thumbs open');axs[3].set(title='Lift and controlled terrain load transfer',xlabel='Full rollout time (s)')
    for i,side in enumerate(['left','right']):
        T=x[:,7*i:7*i+3];R=Rotation.from_quat(x[:,7*i+3:7*i+7]);P=np.array(r['grasp'][side]['initial_contact_anchor_plate_m']);O=R[0].inv().apply(P-T[0]);delta=R.apply(O)+T-P
        axs[4].plot(tw,np.linalg.norm(delta[:,:2],axis=1)*1000,color=colors[i],label=side+' material anchor')
        axs[4].plot(tw,np.linalg.norm(T-T[0],axis=1)*1000,color=colors[i],ls=':',alpha=.55,label=side+' apparent COM')
    axs[4].set(ylabel='Motion in plate frame (mm)',title='Contact motion and compliant rocking')
    b=int(a['recovery_start_physics_step']);tr=tt[b:]-b*.005
    axs[5].semilogy(tr,np.maximum(1e-6,np.linalg.norm(t[b:,36:39],axis=1)),label='Root linear (m/s)')
    axs[5].semilogy(tr,np.maximum(1e-6,np.linalg.norm(t[b:,39:42],axis=1)),label='Root angular (rad/s)')
    axs[5].semilogy(tr,np.maximum(1e-6,np.linalg.norm(t[b:,20:23],axis=1)),label='Box COM (m/s)');axs[5].set(title='Final physical recovery',xlabel='Recovery time (s)')
    for i,ax in enumerate(axs):
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        if i in [0,1,2,4]:ax.set_xlabel('Time within walking phase (s)')
    fig.suptitle('Unchanged corner hands: ground pickup → 2.709 m carry → placement → release → stable recovery')
    fig.tight_layout();fig.savefig(p/'evidence.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);plot(ap.parse_args().directory)
