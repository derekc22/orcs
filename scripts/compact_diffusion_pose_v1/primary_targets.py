"""Event-aligned full frozen proposals, with expert contact/end-point anchors.

This is a target constructor, not execution or a simulator. Unlike the initial
Goal 7 residual candidate, the interior root/waist/arm/object curves come from
the complete checkpoint samples. Accepted poses provide contact endpoints and
initial guesses. Foot trajectories and gripper schedules are never sampled.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R,Slerp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_root_v1'))
from root_refine import DT,sha,targets
from supported_refine import smoothstep,support_envelope

G6=ROOT/'logs/compact_diffusion_root_v1'
G7=ROOT/'logs/compact_diffusion_pose_v1'


def prestates(path):
    a=np.load(path/'rollout.npz')
    return dict(qref=np.r_[a['initial_qpos'][None],a['qpos'][:-1]],vref=np.r_[a['initial_qvel'][None],a['qvel'][:-1]],uref=a['ctrl_physics'])


def construct(seed,m):
    base={k:v.astype(float) for k,v in dict(np.load(G7/f'feasible_{seed}.npz')).items()}
    expert=prestates(G6/'disabled');expert={k:v.astype(float) for k,v in expert.items()}
    _,_,supported,meta=targets(seed,.008,20.)
    for s in [42,43]:
        l=np.load(G6/f'execution_{s}/rollout_locomotion_trace.npz')['trace'];supported &= l[:,14:16].sum(1)>30.
    support=support_envelope(supported)
    folder=ROOT/'logs/compact_diffusion_v1';a=np.load(folder/f'proposals/sample_{seed}.npz');cal=np.load(folder/'calibration/sample_0.npz')
    names=json.loads((folder/'proposals/provenance.json').read_text())['native_joint_names']
    allj=[m.joint(n).id for n in names];jq=m.jnt_qposadr[allj]
    # Source lift/contact events are determined once from the fixed calibration,
    # not fitted to either executed sample. The extra 100 ms is a grasp guard.
    ground_z=float(np.median(cal['object_body_position'][10:40,2]))
    native_lift=int(np.flatnonzero(cal['object_body_position'][:105,2]>ground_z+.05)[0])
    contacts=[np.load(G6/f'{run}/rollout_verification_trace.npz')['trace'][:,0] for run in ['execution_42','execution_43','disabled']]
    physical_lift=max(int(np.flatnonzero(c==0)[0]) for c in contacts)+20
    native_place_end=225+int(np.argmin(gaussian_filter1d(cal['object_body_position'][225:,2],4.)))
    phases=dict(pickup=(physical_lift,1600,native_lift,105),carry=(3449,4721,105,225),placement=(6701,7760,225,native_place_end))
    q=base['qref'].copy();warm=np.arange(len(q),dtype=float);mask=np.zeros(len(q));details={}
    euler=R.from_quat(a['root_quaternion'],scalar_first=True).as_euler('xyz')
    er=R.from_quat(expert['qref'][:,3:7],scalar_first=True).as_euler('xyz');br=R.from_quat(base['qref'][:,3:7],scalar_first=True).as_euler('xyz')
    rawj=a['raw'][:,9:38];root=a['raw'][:,:3];obj=a['object_body_position']
    native_heading=R.from_euler('z',euler[:,2,None]);relative_obj=native_heading.inv().apply(obj-root)
    for phase,(s,e,ns,ne) in phases.items():
        t=np.arange(e-s+1)*DT;u=np.linspace(0,1,e-s+1)[:,None];x=ns+(ne-ns)*u[:,0]
        gate=smoothstep(t/.25)*smoothstep(t[::-1]/.25)
        if phase=='carry':gate*=support[s:e+1]
        mask[s:e+1]=gate
        def interp(v):
            v=gaussian_filter1d(v,4.,axis=0,mode='nearest')
            return np.stack([np.interp(x,np.arange(300),v1) for v1 in v.T],axis=1)
        def anchored(v,start,end):
            v=interp(v);return v+(1-u)*(start-v[0])+u*(end-v[-1])
        def scaled_height(v,start,end):
            v=interp(v[:,None])[:,0];ratio=(v-v[0])/(v[-1]-v[0]);ratio=np.clip(ratio,0.,1.)
            return start+(end-start)*ratio
        desired=expert['qref'][s:e+1].copy();residual=base['qref'][s:e+1]-expert['qref'][s:e+1]
        desired[:,jq]=anchored(rawj,expert['qref'][s,jq],expert['qref'][e,jq])+residual[:,jq]
        if phase!='carry':
            # Intrinsic roll/pitch avoids mixing a native change in heading
            # with the task's authoritative foot-facing direction.
            rp=anchored(euler[:,:2],er[s,:2],er[e,:2])+(br[s:e+1,:2]-er[s:e+1,:2])
            desired[:,3:7]=R.from_euler('xyz',np.c_[rp,br[s:e+1,2]]).as_quat(scalar_first=True)
            desired[:,2]=scaled_height(root[:,2],expert['qref'][s,2],expert['qref'][e,2])+residual[:,2]
            heading=R.from_euler('z',er[s,2]-euler[ns,2])
            desired[:,:2]=anchored(heading.apply(root),expert['qref'][s,:3],expert['qref'][e,:3])[:,:2]+residual[:,:2]
            desired[:,44:47]=anchored(heading.apply(obj),expert['qref'][s,44:47],expert['qref'][e,44:47])+residual[:,44:47]
            desired[:,46]=scaled_height(obj[:,2],expert['qref'][s,46],expert['qref'][e,46])+residual[:,46]
        else:
            # Carry progress cannot retime the gait. Only upper-body pose and
            # the box path relative to the current pelvis are sourced here.
            desired[:,:22]=base['qref'][s:e+1,:22]
            eh=R.from_euler('z',er[s:e+1,2,None]);bh=R.from_euler('z',br[s:e+1,2,None])
            e_rel=eh.inv().apply(expert['qref'][s:e+1,44:47]-expert['qref'][s:e+1,:3])
            b_rel=bh.inv().apply(base['qref'][s:e+1,44:47]-base['qref'][s:e+1,:3])
            rel=anchored(relative_obj,e_rel[0],e_rel[-1])+(b_rel-e_rel)
            desired[:,44:47]=base['qref'][s:e+1,:3]+bh.apply(rel)
        # Object rotations retain task contact endpoints; full native interior
        # rotation is preferred and can subsequently be projected by grasp IK.
        nr=R.from_quat(a['object_quaternion'],scalar_first=True)
        rv=(nr*nr[ns].inv()).as_rotvec();rv=interp(rv);rv-=u*rv[-1]
        align=R.from_euler('z',er[s,2]-euler[ns,2]);rv=align.apply(rv)
        endpoint=Slerp([0.,1.],R.from_quat(expert['qref'][[s,e],47:51],scalar_first=True))(u[:,0])
        legacy=R.from_quat(base['qref'][s:e+1,47:51],scalar_first=True)*R.from_quat(expert['qref'][s:e+1,47:51],scalar_first=True).inv()
        desired[:,47:51]=(legacy*R.from_rotvec(rv)*endpoint).as_quat(scalar_first=True)
        # Expert waist yaw and all eight gripper state/control coordinates stay.
        desired[:,19]=base['qref'][s:e+1,19];desired[:,29:33]=base['qref'][s:e+1,29:33];desired[:,40:44]=base['qref'][s:e+1,40:44]
        delta=desired-base['qref'][s:e+1];delta[:,3:7]=0.;delta[:,47:51]=0.
        q[s:e+1]=base['qref'][s:e+1]+gate[:,None]*delta
        for lo,hi in [(3,7),(47,51)]:
            rot=R.from_quat(desired[:,lo:hi],scalar_first=True)*R.from_quat(base['qref'][s:e+1,lo:hi],scalar_first=True).inv()
            q[s:e+1,lo:hi]=(R.from_rotvec(rot.as_rotvec()*gate[:,None])*R.from_quat(base['qref'][s:e+1,lo:hi],scalar_first=True)).as_quat(scalar_first=True)
        if phase!='carry':
            prog=interp(obj[:,2:3])[:,0];prog=(prog-prog[0])/(prog[-1]-prog[0]);prog=np.maximum.accumulate(np.clip(prog,0,1))
            warm[s:e+1]=(1-gate)*np.arange(s,e+1)+gate*(s+(e-s)*prog)
        details[phase]=dict(physical_ticks=[s,e],native_frames=[ns,ne],root_height_peak_change_m=float(abs(q[s:e+1,2]-base['qref'][s:e+1,2]).max()),object_position_peak_change_m=float(np.linalg.norm(q[s:e+1,44:47]-base['qref'][s:e+1,44:47],axis=1).max()))
    # The targets do not touch any reference beyond their fixed phase windows.
    q[mask==0]=base['qref'][mask==0]
    provenance=dict(checkpoint_sha256=meta['checkpoint_sha256'],sample_sha256=meta['source_sha256'],calibration_sha256=meta['calibration_sha256'],
        native_lift_frame=native_lift,native_placement_end_frame=native_place_end,physical_lift_guard_tick=physical_lift,
        phases=details,method='Full frozen native curves aligned to shared contact events and task pose endpoints, plus retained prior sample-dependent motion. Source root/waist/object/arm interior preferences replace the expert curves; expert feet, waist yaw and gripper mechanics remain authoritative.')
    return base,expert,q,warm,mask,phases,provenance
