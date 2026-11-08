"""Supported-phase XY extension of Goal 5, without adding root tilt.

Root motion is a smooth source-derived trajectory. Bounded arm/leg IK retains
the Goal 5 soles and hands; the accepted height, orientation, waist, object and
gripper trajectories remain unchanged. This is offline analysis, never rollout.
"""
import argparse
import json
from pathlib import Path

import mjlab
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

from root_refine import ROOT,G5,PHASES,DT,sha,targets,wrench,contact_wrench_feasible
from orcs.tasks.uolm.compact.model import source_model


def smoothstep(x):
    x=np.clip(x,0.,1.)
    return x**3*(10-15*x+6*x*x)


def support_envelope(supported):
    # Product of entry/exit gates, rather than min(distance): the product has
    # no derivative cusp at the midpoint of a short supported interval.
    envelope=np.zeros(len(supported));edges=np.diff(np.r_[False,supported,False].astype(int))
    for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        # Two 300 ms ramps and two 50 ms guards must fit without overlap.
        # The short 21.46--21.975 s support bout has inadequate wrench
        # reserve in both accepted baselines; do not inject acceleration there.
        if (e-s-1)*DT<.70:continue
        elapsed=np.arange(e-s)*DT;remaining=elapsed[::-1]
        envelope[s:e]=smoothstep((elapsed-.05)/.30)*smoothstep((remaining-.05)/.30)
    return envelope


def generate(seed,out,limit=.008,gain=20.,disabled=False):
    out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    base_path=G5/(f'feasible_{seed}.npz' if not disabled else 'disabled.npz')
    raw=dict(np.load(base_path));base={k:v.astype(float) for k,v in raw.items()}
    if disabled:
        np.savez_compressed(out,**raw);out.with_suffix('.json').write_text(json.dumps(dict(disabled=True,source=str(base_path),source_sha256=sha(base_path)),indent=2));return
    source_xy,gate,supported,provenance=targets(seed,limit,gain)
    envelope=support_envelope(supported);xy=source_xy*envelope[:,None]
    m=source_model();joints=[j for j in range(m.njnt) if any(t in m.joint(j).name for t in ['hip_','knee_','ankle_','shoulder_','elbow_','wrist_'])]
    jq,jv=m.jnt_qposadr[joints],m.jnt_dofadr[joints];assert len(joints)==26
    links=[m.body(s).id for s in ['left_ankle_roll_link','right_ankle_roll_link','hand_left','hand_right']]
    old,fk=mujoco.MjData(m),mujoco.MjData(m);jp=np.zeros((3,m.nv));jr=jp.copy()
    delta=np.zeros((len(xy),26));refined_xy=np.zeros_like(xy);qref=base['qref'].copy()
    position_error=0.;angle_error=0.;com_shift=0.
    for phase,(start,end,_,_) in PHASES.items():
        s,e=round(start/DT),round(end/DT)
        phase_q=base['qref'][s:e+1,jq]
        lower_room=np.maximum(np.minimum(phase_q-m.jnt_range[joints,0]-.0005,.12),0.)
        upper_room=np.maximum(np.minimum(m.jnt_range[joints,1]-phase_q-.0005,.12),0.)
        # Anticipate a closing soft-stop margin, while retaining available
        # workspace elsewhere in the phase. No physical limit is enlarged.
        for room in [lower_room,upper_room]:
            for k in range(1,len(room)):room[k]=np.minimum(room[k],room[k-1]+.3*DT)
            for k in range(len(room)-2,-1,-1):room[k]=np.minimum(room[k],room[k+1]+.3*DT)
        previous=np.zeros(28)
        for tick in range(s,e+1):
            if not xy[tick].any():previous[:]=0.;continue
            q0=base['qref'][tick];old.qpos[:]=q0;mujoco.mj_kinematics(m,old);mujoco.mj_comPos(m,old)
            positions=old.xpos[links].copy();rotations=R.from_matrix(old.xmat[links].reshape(4,3,3))
            low=np.r_[np.repeat(-.014*envelope[tick],2),-lower_room[tick-s]]
            high=np.r_[np.repeat(.014*envelope[tick],2),upper_room[tick-s]]
            p=np.clip(previous,low,high)
            target=np.r_[xy[tick],np.zeros(26)];weights=np.r_[20.,20.,np.repeat(.02,26)]
            def cost(trial):
                fk.qpos[:]=q0;fk.qpos[:2]+=trial[:2];fk.qpos[jq]+=trial[2:]
                mujoco.mj_kinematics(m,fk)
                pe=250*(positions-fk.xpos[links])
                re=70*(rotations*R.from_matrix(fk.xmat[links].reshape(4,3,3)).inv()).as_rotvec()
                return float(np.sum(pe**2)+np.sum(re**2)+np.sum((weights*(target-trial))**2))
            for _ in range(16):
                fk.qpos[:]=q0;fk.qpos[:2]+=p[:2];fk.qpos[jq]+=p[2:]
                mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                rows=[];errors=[]
                for k,link in enumerate(links):
                    mujoco.mj_jacBody(m,fk,jp,jr,link)
                    rows.extend([250*np.c_[jp[:,:2],jp[:,jv]],70*np.c_[jr[:,:2],jr[:,jv]]])
                    errors.extend([250*(positions[k]-fk.xpos[link]),70*(rotations[k]*R.from_matrix(fk.xmat[link].reshape(3,3)).inv()).as_rotvec()])
                rows.append(np.diag(weights));errors.append(weights*(target-p))
                step_bound=np.r_[.001,.001,np.repeat(.025,26)]
                step=lsq_linear(np.concatenate(rows),np.concatenate(errors),bounds=(np.maximum(low-p,-step_bound)-1e-12,np.minimum(high-p,step_bound)+1e-12),method='bvls',tol=1e-10).x
                old_cost=float(np.concatenate(errors)@np.concatenate(errors))
                accepted=False
                for rate in [1.,.5,.25,.125,.0625,.03125]:
                    trial=p+rate*step
                    if cost(trial)<=old_cost+1e-12:p=trial;accepted=True;break
                if not accepted:break
                if np.linalg.norm(step)<1e-8:break
            qref[tick,:2]+=p[:2];qref[tick,jq]+=p[2:];refined_xy[tick]=p[:2];delta[tick]=p[2:];previous=p
            fk.qpos[:]=qref[tick];mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
            position_error=max(position_error,float(np.linalg.norm(fk.xpos[links]-positions,axis=1).max()))
            angle_error=max(angle_error,float((R.from_matrix(fk.xmat[links].reshape(4,3,3))*rotations.inv()).magnitude().max()))
            com_shift=max(com_shift,float(np.linalg.norm(fk.subtree_com[0,:2]-old.subtree_com[0,:2])))
        print('SUPPORTED_XY',phase,'peak',np.linalg.norm(refined_xy[s:e+1],axis=1).max(),'added joint peak',abs(delta[s:e+1]).max(),flush=True)
    # Goal 4's successful pattern: smooth only the additive motion, then restore
    # contact geometry at every physics tick. Never filter baseline impact data.
    initial_xy=refined_xy.copy();initial_delta=delta.copy()
    phase_mask=np.zeros(len(xy))
    for start,end,_,_ in PHASES.values():
        s,e=round(start/DT),round(end/DT);t=np.arange(e-s+1)*DT
        phase_mask[s:e+1]=smoothstep(t/.2)*smoothstep(t[::-1]/.2)
    target_xy=gaussian_filter1d(refined_xy,20.,axis=0,mode='constant')*envelope[:,None]*phase_mask[:,None]
    prior=gaussian_filter1d(delta,20.,axis=0,mode='constant')*envelope[:,None]*phase_mask[:,None]
    qref=base['qref'].copy();refined_xy[:]=0.;delta[:]=0.
    position_error=0.;angle_error=0.;com_shift=0.
    for phase,(start,end,_,_) in PHASES.items():
        s,e=round(start/DT),round(end/DT);previous=np.zeros(28)
        phase_q=base['qref'][s:e+1,jq]
        lower_room=np.maximum(np.minimum(phase_q-m.jnt_range[joints,0]-.0005,.12),0.)
        upper_room=np.maximum(np.minimum(m.jnt_range[joints,1]-phase_q-.0005,.12),0.)
        for room in [lower_room,upper_room]:
            for k in range(1,len(room)):room[k]=np.minimum(room[k],room[k-1]+.3*DT)
            for k in range(len(room)-2,-1,-1):room[k]=np.minimum(room[k],room[k+1]+.3*DT)
        for tick in range(s,e+1):
            if not target_xy[tick].any():previous[:]=0.;continue
            q0=base['qref'][tick];old.qpos[:]=q0;mujoco.mj_kinematics(m,old);mujoco.mj_comPos(m,old)
            positions=old.xpos[links].copy();rotations=R.from_matrix(old.xmat[links].reshape(4,3,3))
            target=np.r_[target_xy[tick],prior[tick]]
            low=np.r_[np.repeat(-.014*envelope[tick],2),-lower_room[tick-s]]
            high=np.r_[np.repeat(.014*envelope[tick],2),upper_room[tick-s]]
            velocity_box=np.r_[.00025,.00025,np.repeat(.0035,26)]
            low=np.maximum(low,previous-velocity_box);high=np.minimum(high,previous+velocity_box);low=np.minimum(low,high-1e-12)
            p=np.clip(target,low,high)
            for _ in range(7):
                fk.qpos[:]=q0;fk.qpos[:2]+=p[:2];fk.qpos[jq]+=p[2:]
                mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                rows=[];errors=[]
                for k,link in enumerate(links):
                    mujoco.mj_jacBody(m,fk,jp,jr,link)
                    rows.extend([250*np.c_[jp[:,:2],jp[:,jv]],70*np.c_[jr[:,:2],jr[:,jv]]])
                    errors.extend([250*(positions[k]-fk.xpos[link]),70*(rotations[k]*R.from_matrix(fk.xmat[link].reshape(3,3)).inv()).as_rotvec()])
                weights=np.r_[2.,2.,np.repeat(.04,26)]
                rows.append(np.diag(weights));errors.append(weights*(target-p))
                step=lsq_linear(np.concatenate(rows),np.concatenate(errors),bounds=(low-p-1e-12,high-p+1e-12),method='bvls',tol=1e-10).x
                p+=step
                if np.linalg.norm(step)<1e-8:break
            qref[tick,:2]+=p[:2];qref[tick,jq]+=p[2:];refined_xy[tick]=p[:2];delta[tick]=p[2:];previous=p
            fk.qpos[:]=qref[tick];mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
            position_error=max(position_error,float(np.linalg.norm(fk.xpos[links]-positions,axis=1).max()))
            angle_error=max(angle_error,float((R.from_matrix(fk.xmat[links].reshape(4,3,3))*rotations.inv()).magnitude().max()))
            com_shift=max(com_shift,float(np.linalg.norm(fk.subtree_com[0,:2]-old.subtree_com[0,:2])))
        print('POLISHED_XY',phase,'peak',np.linalg.norm(refined_xy[s:e+1],axis=1).max(),flush=True)
    velocity=np.gradient(delta,DT,axis=0);xyv=np.gradient(refined_xy,DT,axis=0)
    vref=base['vref'].copy();vref[:,:2]+=xyv;vref[:,jv]+=velocity
    a0=np.gradient(base['vref'],DT,axis=0);a1=np.gradient(vref,DT,axis=0)
    uref=base['uref'].copy();increments=np.zeros_like(uref);effort=np.zeros((len(xy),m.nv))
    nq=m.jnt_qposadr[m.actuator_trnid[:29,0]];nv=m.jnt_dofadr[m.actuator_trnid[:29,0]]
    old_tau=np.zeros(m.nv);new_tau=old_tau.copy();force_checks=[];H=np.zeros((len(xy),3));H0=H.copy()
    loads=np.load(G5/f'execution_{seed}/rollout_locomotion_trace.npz')['trace'][:,14:16]
    active=np.any(refined_xy!=0.,axis=1)|np.any(xyv!=0.,axis=1)|np.any(velocity!=0.,axis=1)
    for tick in np.flatnonzero(active):
        for data,q,v,a,tau,h in [(old,base['qref'][tick],base['vref'][tick],a0[tick],old_tau,H0),(fk,qref[tick],vref[tick],a1[tick],new_tau,H)]:
            data.qpos[:]=q;data.qvel[:]=v;data.qacc[:]=a
            mujoco.mj_fwdPosition(m,data);mujoco.mj_fwdVelocity(m,data);mujoco.mj_rne(m,data,1,tau);mujoco.mj_subtreeVel(m,data);h[tick]=data.subtree_angmom[0]
        effort[tick]=new_tau-old_tau
        kp=m.actuator_gainprm[:29,0];kd=-m.actuator_biasprm[:29,2]
        increments[tick,:29]=qref[tick,nq]-base['qref'][tick,nq]+(kd*(vref[tick,nv]-base['vref'][tick,nv])+effort[tick,nv])/kp
        if tick%20==0 and gate[tick]>.99:
            f0=contact_wrench_feasible(m,old,wrench(m,old,old_tau),loads[max(0,tick-1)])
            f1=contact_wrench_feasible(m,fk,wrench(m,fk,new_tau),loads[max(0,tick-1)])
            force_checks.append([int(tick),f0,f1])
    uref+=increments
    old_excess=np.maximum(base['qref'][:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-base['qref'][:,jq]);new_excess=np.maximum(qref[:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-qref[:,jq])
    excess=float((new_excess-np.maximum(old_excess,0.)).max())
    no_support=~supported
    # The centered velocity stencil is fully inside the 50 ms zero-offset guard.
    flight_position=float(abs(refined_xy[no_support]).max()) if no_support.any() else 0.
    flight_velocity=float(abs(xyv[no_support]).max()) if no_support.any() else 0.
    report=dict(seed=seed,base_reference=str(base_path),base_sha256=sha(base_path),compiler_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'scripts/compact_diffusion_root_v1/root_refine.py'),diffusion=provenance,
        additive_smoothing_sigma_s=.1,contact_cone='Unchanged accepted model; exact L1 facets for its pyramidal friction cone.',
        independent_coordinates=['root world X','root world Y'],root_roll_pitch_policy='Goal 5 reference retained; counterlean prototypes were rejected before physical rollout.',
        support_envelope=dict(guard_s=.05,entry_exit_blend_s=.30,minimum_supported_duration_s=.70,method='Product of quintic entry/exit gates in intervals containing both complete ramps.'),
        maximum_foot_hand_position_error_m=position_error,maximum_foot_hand_angle_error_rad=angle_error,maximum_horizontal_COM_displacement_m=com_shift,
        maximum_added_joint_limit_excursion_rad=excess,maximum_added_joint_velocity_radps=float(abs(velocity).max()),maximum_added_XY_velocity_mps=float(abs(xyv).max()),
        maximum_added_XY_acceleration_mps2=float(abs(np.gradient(xyv,DT,axis=0)).max()),
        maximum_feedforward_increment_rad=float(abs(increments).max()),maximum_incremental_native_effort_Nm=float(abs(effort[:,nv]).max()),
        flight_added_XY_position_m=flight_position,flight_added_XY_velocity_mps=flight_velocity,
        stable_support_wrench_checks=force_checks,baseline_feasible_checks=sum(x[1] for x in force_checks),introduced_wrench_infeasibilities=sum(x[1] and not x[2] for x in force_checks),
        retained_G5_height_orientation_waist_object=all(np.array_equal(qref[:,s:e],base['qref'][:,s:e]) for s,e in [(2,7),(19,22),(44,51)]),
        retained_gripper_feedforward=bool(np.array_equal(uref[:,29:],base['uref'][:,29:])),simulation_stepped=False,physical_model_changed=False)
    report['pass_cheap_gate']=bool(position_error<.0003 and angle_error<.003 and com_shift<.009 and excess<1e-8 and abs(velocity).max()<1.
        and abs(xyv).max()<.05 and abs(increments).max()<.18 and abs(effort[:,nv]).max()<25.
        and report['baseline_feasible_checks']>=20 and report['introduced_wrench_infeasibilities']==0 and flight_position==0 and flight_velocity==0)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out.with_name(out.stem+'_diagnostics.npz'),source_xy=source_xy,requested_xy=xy,initial_projected_xy=initial_xy,initial_joint_delta=initial_delta,smoothed_xy=target_xy,refined_xy=qref[:,:2]-base['qref'][:,:2],joint_delta=delta,support_envelope=envelope,support_gate=gate,incremental_effort=effort,feedforward_increment=increments)
    if not report['pass_cheap_gate']:raise RuntimeError({k:v for k,v in report.items() if k!='stable_support_wrench_checks'})
    np.savez_compressed(out,qref=qref,vref=vref,uref=uref,goal_position=raw['goal_position'],goal_quaternion=raw['goal_quaternion'])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=float,default=.008);p.add_argument('--gain',type=float,default=20.);p.add_argument('--disabled',action='store_true')
    a=p.parse_args();generate(a.seed,a.out,a.limit,a.gain,a.disabled)
