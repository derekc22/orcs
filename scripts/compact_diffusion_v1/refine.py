"""Deterministic contact-compatible projection of frozen diffusion innovations.

Stage 1 projects the generated-minus-conditioning-motion innovation into the
accepted compact contact topology. Stage 2 is closed-loop ORCS simulation; its
actual states/controls become a feasible reference only if all task gates pass.
There is no fitted model, gain update, synthetic action label, or random search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import mjlab
import mujoco
from orcs.tasks.uolm.compact.model import source_model,CHECKPOINT
from proposals import decode,centroid_offset

ROOT=Path(__file__).resolve().parents[2]
PHASES={'pickup':(2.5,8.,0,105),'carry':(17.245,23.605,105,225),'placement':(33.505,39.11,225,299)}


def bounded(a,limit):
    return a*np.minimum(1.,limit/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12))


def phase_curve(a,s,e,count):
    x=np.linspace(s,e,count);out=np.stack([np.interp(x,np.arange(len(a)),a[:,j]) for j in range(a.shape[1])],axis=1)
    out=gaussian_filter1d(out,3.,axis=0,mode='nearest')
    u=np.linspace(0,1,count)[:,None]
    return out*(np.sin(np.pi*u)**2)


def propose(source,out,gain,retime_max=0.,timing_calibration=None,sample_gain=None):
    out.parent.mkdir(parents=True,exist_ok=True)
    with np.load(CHECKPOINT) as z:base={k:z[k].copy() for k in z.files}
    a=np.load(source);raw=a['raw'];demo=np.load(source.parent/'conditioning.npz')['demonstration']
    _,rd=decode(demo);offset=centroid_offset()
    demo_body=demo[:,38:41]-Rotation.from_quat(rd,scalar_first=True).apply(offset)
    calibration=np.load(timing_calibration) if timing_calibration else None
    m=source_model();fk=mujoco.MjData(m)
    joint_ids=[j for j in range(m.njnt) if m.jnt_type[j]==3 and not m.joint(j).name.startswith('hand_')]
    jq=m.jnt_qposadr[joint_ids];jv=m.jnt_dofadr[joint_ids]
    allowed=np.r_[np.arange(6),jv]
    # A single source->task yaw transform, identical for both samples.
    source_heading=np.arctan2(demo[-1,39]-demo[0,39],demo[-1,38]-demo[0,38])
    target_heading=np.arctan2(base['qref'][-1,45]-base['qref'][0,45],base['qref'][-1,44]-base['qref'][0,44])
    align=Rotation.from_euler('z',target_heading-source_heading)
    objinnov=a['object_body_position']-demo_body
    rootinnov=raw[:,:3]-demo[:,:3]
    poseinnov=raw[:,9:38]-demo[:,9:38]
    rotinnov=(Rotation.from_quat(a['object_quaternion'],scalar_first=True)*Rotation.from_quat(rd,scalar_first=True).inv()).as_rotvec()
    if sample_gain is not None:
        if calibration is None:raise ValueError('Fixed diffusion calibration is required for sample spread retargeting')
        # Mean shrinkage and conditional spread use common, fixed gains for
        # every phase and sample. All variation is taken from genuine model
        # outputs BEFORE contact projection, never added to executed states.
        c=calibration['raw']
        objinnov=gain*(calibration['object_body_position']-demo_body)+sample_gain*(a['object_body_position']-calibration['object_body_position'])
        rootinnov=gain*(c[:,:3]-demo[:,:3])+sample_gain*(raw[:,:3]-c[:,:3])
        poseinnov=gain*(c[:,9:38]-demo[:,9:38])+sample_gain*(raw[:,9:38]-c[:,9:38])
        meanrot=(Rotation.from_quat(calibration['object_quaternion'],scalar_first=True)*Rotation.from_quat(rd,scalar_first=True).inv()).as_rotvec()
        sample_rot=(Rotation.from_quat(a['object_quaternion'],scalar_first=True)*Rotation.from_quat(calibration['object_quaternion'],scalar_first=True).inv()).as_rotvec()
        rotinnov=gain*meanrot+sample_gain*sample_rot
    else:
        objinnov*=gain;rootinnov*=gain;poseinnov*=gain;rotinnov*=gain
    objinnov=align.apply(objinnov);rootinnov=align.apply(rootinnov);rotinnov=align.apply(rotinnov)
    delta=np.zeros((len(base['qref']),m.nv));timing=np.zeros(len(delta));details={}
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    pads=[[m.geom(f'hand_{s}_pad{k}').id for k in range(3)] for s in ['left','right']]
    hands=[m.body('hand_'+s).id for s in ['left','right']]
    jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));maximum_foot=0.;maximum_palm=0.
    for phase,(start,end,ds,de) in PHASES.items():
        ticks=np.arange(round(start/.005),round(end/.005)+1,4)
        dc=bounded(phase_curve(objinnov,ds,de,len(ticks)),.006)
        dr=bounded(phase_curve(rootinnov,ds,de,len(ticks)),.004)
        rot=bounded(phase_curve(rotinnov,ds,de,len(ticks)),.012)
        dj=np.clip(phase_curve(poseinnov,ds,de,len(ticks)),-.025,.025)
        if retime_max:
            # Tangential projection of endpoint-matched generated motion onto
            # the conditioning primitive determines its progress in time.
            # Metres/radians are weighted consistently; no sample-specific
            # parameter, random optimizer, or post-refinement noise is used.
            clock=calibration['raw'] if calibration is not None else demo
            clock_body=calibration['object_body_position'] if calibration is not None else demo_body
            features=np.c_[clock[:,:3]*.5,clock_body,clock[:,9:38]*.10]
            innovation=np.c_[(raw[:,:3]-clock[:,:3])*.5,a['object_body_position']-clock_body,(raw[:,9:38]-clock[:,9:38])*.10]
            segment=innovation[ds:de+1].copy();u=np.linspace(0,1,len(segment))[:,None]
            segment-=segment[0]*(1-u)+segment[-1]*u
            velocity=np.gradient(features,.02,axis=0)[ds:de+1]
            shift=(segment*velocity).sum(-1)/((velocity*velocity).sum(-1)+.05**2)
            shift=gaussian_filter1d(shift,4.,mode='nearest')
            shift*=((end-start)/((de-ds)*.02))
            shift=np.clip(shift,-retime_max,retime_max)*np.sin(np.pi*u[:,0])**2
            time_full=np.arange(round(start/.005),round(end/.005)+1)
            timing[time_full]=np.interp(np.linspace(0,len(shift)-1,len(time_full)),np.arange(len(shift)),shift)
        solved=[]
        for i,tick in enumerate(ticks):
            q=base['qref'][tick];fk.qpos[:]=q;mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
            A=[];b=[]
            for f in feet:
                mujoco.mj_jacBody(m,fk,jp,jr,f);A.extend([150*jp[:,allowed],80*jr[:,allowed]]);b.extend([np.zeros(3),np.zeros(3)])
            for side,gs in enumerate(pads):
                center=fk.geom_xpos[gs].mean(0);mujoco.mj_jac(m,fk,jp,jr,center,hands[side])
                shift=dc[i]+np.cross(rot[i],center-q[44:47])
                A.extend([150*jp[:,allowed],40*jr[:,allowed]]);b.extend([150*shift,40*rot[i]])
            prior=np.r_[dr[i],np.zeros(3),dj[i]]
            A.append(np.eye(len(allowed)));b.append(prior)
            sol=np.linalg.lstsq(np.concatenate(A),np.concatenate(b),rcond=None)[0]
            sol[:3]=bounded(sol[None,:3],.008)[0];sol[3:6]=np.clip(sol[3:6],-.012,.012);sol[6:]=np.clip(sol[6:],-.035,.035)
            v=np.zeros(m.nv);v[allowed]=sol;v[43:46]=dc[i]
            v[46:49]=Rotation.from_quat(q[47:51],scalar_first=True).inv().apply(rot[i])
            trial=q.copy();mujoco.mj_integratePos(m,trial,v,1.)
            feet0=fk.xpos[feet].copy();palm0=np.array([fk.geom_xpos[g].mean(0) for g in pads])
            fk.qpos[:]=trial;mujoco.mj_kinematics(m,fk)
            maximum_foot=max(maximum_foot,float(abs(fk.xpos[feet]-feet0).max()))
            maximum_palm=max(maximum_palm,float(abs(np.array([fk.geom_xpos[g].mean(0) for g in pads])-palm0-dc[i]).max()))
            solved.append(v)
        solved=np.asarray(solved)
        full=np.arange(round(start/.005),round(end/.005)+1)
        for j in range(m.nv):delta[full,j]=np.interp(full,ticks,solved[:,j])
        details[phase]=dict(physical_interval_s=[start,end],diffusion_interval_frames=[ds,de],object_target_peak_m=float(np.linalg.norm(dc,axis=1).max()),
                           native_target_peak_rad=float(abs(solved[:,jv]).max()),timing_shift_peak_s=float(abs(timing[full]).max()))
    qref=base['qref'].copy()
    hands_q=m.jnt_qposadr[[j for j in range(m.njnt) if m.joint(j).name.startswith('hand_')]]
    for i,v in enumerate(delta):
        if timing[i]:
            frame=np.clip(i+timing[i]/.005,0,len(qref)-1);j=min(int(frame),len(qref)-2)
            tangent=np.empty(m.nv);mujoco.mj_differentiatePos(m,tangent,1.,base['qref'][j],base['qref'][j+1])
            qref[i]=base['qref'][j];mujoco.mj_integratePos(m,qref[i],tangent,frame-j)
        if v.any():mujoco.mj_integratePos(m,qref[i],v,1.)
        qref[i,hands_q]=base['qref'][i,hands_q]
    vref=base['vref'].copy();dq=np.zeros(m.nv);db=np.zeros(m.nv)
    for i in range(1,len(qref)-1):
        mujoco.mj_differentiatePos(m,dq,.01,qref[i-1],qref[i+1]);mujoco.mj_differentiatePos(m,db,.01,base['qref'][i-1],base['qref'][i+1])
        vref[i]+=dq-db
    np.savez_compressed(out,qref=qref,vref=vref,uref=base['uref'],goal_position=base['qref'][-1,44:47],goal_quaternion=base['qref'][-1,47:51],diffusion_delta=delta,diffusion_timing=timing)
    report=dict(source_sample=str(source.resolve()),mean_gain=gain,sample_gain=sample_gain,retime_max_s=retime_max,timing_calibration=str(timing_calibration.resolve()) if timing_calibration else None,phases=details,maximum_foot_kinematic_residual_m=maximum_foot,maximum_palm_translation_residual_m=maximum_palm,
        missing_hand_dofs='All eight use the accepted physical mechanism schedule; diffusion generates none.',
        method='Source-calibrated diffusion innovation; deterministic bounded differential IK preserving both sole trajectories and compact palm/object relative geometry; frozen controller-coupled ORCS projection required next.')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def realize(run,proposal,out):
    from evidence import physical
    screen=physical(run)
    if not screen.get('pass_physical_screen',False):raise ValueError(f'Physical refinement failed: {screen}')
    a=np.load(run/'rollout.npz');p=np.load(proposal)
    if len(a['qpos'])!=9629:raise ValueError('Cannot promote an incomplete physical rollout')
    q=np.concatenate([a['initial_qpos'][None],a['qpos'][:-1]])
    v=np.concatenate([a['initial_qvel'][None],a['qvel'][:-1]])
    np.savez_compressed(out,qref=q,vref=v,uref=a['ctrl_physics'],goal_position=p['goal_position'],goal_quaternion=p['goal_quaternion'])
    provenance=dict(method='Deterministic frozen-feedback closed-loop physical projection through registered ORCS UOLM; no policy fitting or physical state playback.',
        projection_run=str(run.resolve()),proposal=str(proposal.resolve()),
        proposal_sha256=hashlib.sha256(proposal.read_bytes()).hexdigest(),
        realized_reference_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),physical_screen=screen)
    out.with_suffix('.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('propose');a.add_argument('--sample',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--gain',type=float,default=.3);a.add_argument('--retime-max',type=float,default=0.);a.add_argument('--timing-calibration',type=Path);a.add_argument('--sample-gain',type=float)
    a=sub.add_parser('realize');a.add_argument('--run',type=Path,required=True);a.add_argument('--proposal',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=p.parse_args();propose(a.sample,a.out,a.gain,a.retime_max,a.timing_calibration,a.sample_gain) if a.mode=='propose' else realize(a.run,a.proposal,a.out)
