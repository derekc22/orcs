"""Goal 7: supported sampled root tilt and coordinated waist posture.

Extends the accepted Goal 6 reference, never its rejected full-body prototype.
Both soles and hands, root XYZ, waist yaw, object and gripper mechanics survive.
The only independent additions are four coherent frozen-checkpoint coordinates.
Offline FK/RNE screens never step a simulator. Actual ORCS determines success.
"""
import argparse
import json
from pathlib import Path
import sys

import mjlab
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_root_v1'))
from root_refine import DT,PHASES,sha,targets,left_jacobian,wrench,contact_wrench_feasible
from supported_refine import smoothstep,support_envelope
from orcs.tasks.uolm.compact.model import source_model

G6=ROOT/'logs/compact_diffusion_root_v1'


def source(seed,scale):
    _,gate,supported,meta=targets(seed,.008,20.)
    for s in [42,43]:
        l=np.load(G6/f'execution_{s}/rollout_locomotion_trace.npz')['trace']
        supported &= l[:,14:16].sum(1)>30.
    envelope=support_envelope(supported)
    folder=ROOT/'logs/compact_diffusion_v1';a=np.load(folder/f'proposals/sample_{seed}.npz');c=np.load(folder/'calibration/sample_0.npz')
    names=json.loads((folder/'proposals/provenance.json').read_text())['native_joint_names']
    orientation=(R.from_quat(a['root_quaternion'],scalar_first=True)*R.from_quat(c['root_quaternion'],scalar_first=True).inv()).as_rotvec()
    aligned=R.from_euler('z',meta['alignment_yaw_rad']).apply(orientation)[:,:2]
    waist=np.stack([a['raw'][:,9+names.index('waist_'+n+'_joint')]-c['raw'][:,9+names.index('waist_'+n+'_joint')] for n in ['roll','pitch']],axis=1)
    native=np.c_[aligned,waist];filtered=gaussian_filter1d(native,8.,axis=0,mode='nearest')
    request=np.zeros((len(envelope),4));mapped=request.copy();phase_details={}
    gains=np.array([60.,60.,30.,30.])*scale;limits=np.array([.035,.035,.04,.04])*scale
    for phase,(start,end,ds,de) in PHASES.items():
        s,e=round(start/DT),round(end/DT);rate=gate[s:e+1]
        progress=np.r_[0.,np.cumsum((rate[:-1]+rate[1:])*.5)];progress/=progress[-1]
        x=ds+(de-ds)*progress
        v=np.stack([np.interp(x,np.arange(300),column) for column in filtered.T],axis=1)
        mapped[s:e+1]=v
        desired=v*gains*np.sin(np.pi*progress[:,None])**4
        desired/=np.sqrt(1+(desired/limits)**2)
        request[s:e+1]=desired*envelope[s:e+1,None];request[[s,e]]=0.
        phase_details[phase]=dict(ticks=[s,e],source_frames=[ds,de],requested_peak_rad=abs(request[s:e+1]).max(0).tolist())
    meta.update(mode_gains=gains.tolist(),mode_limits_rad=limits.tolist(),phase_details=phase_details,
        coordinate_contract='Root world-horizontal relative rotation vector, plus native waist roll/pitch, relative to the same seed-0 calibration. No fitted execution alignment.')
    return request,envelope,supported,dict(meta),native,mapped


def generate(seed,out,scale=1.):
    assert scale>0 and scale<=1.;out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    base_path=G6/f'feasible_{seed}.npz';raw=dict(np.load(base_path));base={k:v.astype(float) for k,v in raw.items()}
    requested,envelope,supported,provenance,native_source,mapped_source=source(seed,scale)
    m=source_model();joints=[j for j in range(m.njnt) if any(t in m.joint(j).name for t in ['hip_','knee_','ankle_','waist_roll','waist_pitch','shoulder_','elbow_','wrist_'])]
    jq,jv=m.jnt_qposadr[joints],m.jnt_dofadr[joints];assert len(joints)==28
    wi=np.array([joints.index(m.joint('waist_'+n+'_joint').id) for n in ['roll','pitch']])
    links=[m.body(s).id for s in ['left_ankle_roll_link','right_ankle_roll_link','hand_left','hand_right']]
    old,fk=mujoco.MjData(m),mujoco.MjData(m);jp=np.zeros((3,m.nv));jr=jp.copy()
    B=np.zeros((m.nv,30));B[np.ix_(jv,np.arange(2,30))]=np.eye(28)
    parameters=np.zeros((len(requested),30));qref=base['qref'].copy();metrics={}
    def pose(q0,p):
        q=q0.copy();q[jq]+=p[2:]
        if p[:2].any():q[3:7]=(R.from_rotvec(np.r_[p[:2],0.])*R.from_quat(q0[3:7],scalar_first=True)).as_quat(scalar_first=True)
        return q
    for stage in [0,1]:
        if stage:
            initial=parameters.copy();prior=gaussian_filter1d(initial,20.,axis=0,mode='constant')*envelope[:,None]
            for start,end,_,_ in PHASES.values():
                s,e=round(start/DT),round(end/DT);t=np.arange(e-s+1)*DT;prior[s:e+1]*=(smoothstep(t/.2)*smoothstep(t[::-1]/.2))[:,None]
        parameters[:]=0.;qref=base['qref'].copy();max_pos=0.;max_angle=0.;max_com=0.
        for phase,(start,end,_,_) in PHASES.items():
            s,e=round(start/DT),round(end/DT);phase_q=base['qref'][s:e+1,jq]
            lower=np.maximum(np.minimum(phase_q-m.jnt_range[joints,0]-.0005,.12),0.)
            upper=np.maximum(np.minimum(m.jnt_range[joints,1]-phase_q-.0005,.12),0.)
            for room in [lower,upper]:
                for k in range(1,len(room)):room[k]=np.minimum(room[k],room[k-1]+.3*DT)
                for k in range(len(room)-2,-1,-1):room[k]=np.minimum(room[k],room[k+1]+.3*DT)
            previous=np.zeros(30);last_root_step=np.zeros(2)
            for tick in range(s,e+1):
                if not envelope[tick] or (not stage and not requested[tick].any()) or (stage and not prior[tick].any()):previous[:]=0.;last_root_step[:]=0.;continue
                q0=base['qref'][tick];old.qpos[:]=q0;mujoco.mj_kinematics(m,old);mujoco.mj_comPos(m,old)
                positions=old.xpos[links].copy();rotations=R.from_matrix(old.xmat[links].reshape(4,3,3))
                target=prior[tick].copy() if stage else np.zeros(30)
                if not stage:target[:2]=requested[tick,:2];target[2+wi]=requested[tick,2:]
                low=np.r_[np.repeat(-.05*envelope[tick],2),-lower[tick-s]];high=np.r_[np.repeat(.05*envelope[tick],2),upper[tick-s]]
                if stage:
                    velocity_box=np.r_[.00075,.00075,np.repeat(.0035,28)]
                    low=np.maximum(low,previous-velocity_box);high=np.minimum(high,previous+velocity_box);low=np.minimum(low,high-1e-12)
                    # Projection at a moving joint-workspace boundary can
                    # otherwise create a velocity corner despite smoothing.
                    # Bound tilt acceleration before any physical rollout.
                    low[:2]=np.maximum(low[:2],previous[:2]+last_root_step-1.5*DT**2)
                    high[:2]=np.minimum(high[:2],previous[:2]+last_root_step+1.5*DT**2)
                    low=np.minimum(low,high-1e-12)
                p=np.clip(target if stage else previous,low,high)
                weights=np.r_[3.,3.,np.repeat(.025 if not stage else .04,28)];weights[2+wi]=1. if not stage else .3
                def residual(trial,jacobian=False):
                    fk.qpos[:]=pose(q0,trial);mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                    B[3:6,:2]=R.from_quat(fk.qpos[3:7],scalar_first=True).inv().as_matrix()@left_jacobian(np.r_[trial[:2],0.])[:,:2]
                    errors=[];rows=[]
                    for k,link in enumerate(links):
                        errors.extend([250*(positions[k]-fk.xpos[link]),70*(rotations[k]*R.from_matrix(fk.xmat[link].reshape(3,3)).inv()).as_rotvec()])
                        if jacobian:
                            mujoco.mj_jacBody(m,fk,jp,jr,link);rows.extend([250*jp@B,70*jr@B])
                    errors.append(300*(old.subtree_com[0,:2]-fk.subtree_com[0,:2]))
                    if jacobian:mujoco.mj_jacSubtreeCom(m,fk,jp,0);rows.append(300*jp[:2]@B)
                    errors.append(weights*(target-trial))
                    if jacobian:rows.append(np.diag(weights))
                    return np.concatenate(errors),np.concatenate(rows) if jacobian else None
                for _ in range(16 if not stage else 8):
                    errors,rows=residual(p,True);bound=np.r_[.005,.005,np.repeat(.025,28)]
                    step=lsq_linear(rows,errors,bounds=(np.maximum(low-p,-bound)-1e-12,np.minimum(high-p,bound)+1e-12),method='bvls',tol=1e-10).x
                    cost=float(errors@errors);accepted=False
                    for rate in [1.,.5,.25,.125,.0625,.03125]:
                        trial=p+rate*step;rr,_=residual(trial)
                        if rr@rr<=cost+1e-12:p=trial;accepted=True;break
                    if not accepted or np.linalg.norm(step)<1e-8:break
                parameters[tick]=p;qref[tick]=pose(q0,p);last_root_step=p[:2]-previous[:2];previous=p
                fk.qpos[:]=qref[tick];mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                max_pos=max(max_pos,float(np.linalg.norm(fk.xpos[links]-positions,axis=1).max()))
                max_angle=max(max_angle,float((R.from_matrix(fk.xmat[links].reshape(4,3,3))*rotations.inv()).magnitude().max()))
                max_com=max(max_com,float(np.linalg.norm(fk.subtree_com[0,:2]-old.subtree_com[0,:2])))
            print('POSE',stage,phase,'root and waist peak',abs(parameters[s:e+1][:,np.r_[0,1,2+wi]]).max(0),flush=True)
        metrics[f'stage_{stage}']=dict(maximum_foot_hand_position_error_m=max_pos,maximum_foot_hand_angle_error_rad=max_angle,maximum_horizontal_COM_shift_m=max_com)
    pv=np.gradient(parameters,DT,axis=0);vref=base['vref'].copy();vref[:,jv]+=pv[:,2:]
    for tick in np.flatnonzero(np.any(parameters!=0.,axis=1)|np.any(pv!=0.,axis=1)):
        world_delta=left_jacobian(np.r_[parameters[tick,:2],0.])@np.r_[pv[tick,:2],0.]
        vref[tick,3:6]+=R.from_quat(qref[tick,3:7],scalar_first=True).inv().apply(world_delta)
    a0=np.gradient(base['vref'],DT,axis=0);a1=np.gradient(vref,DT,axis=0)
    uref=base['uref'].copy();increments=np.zeros_like(uref);effort=np.zeros((len(qref),m.nv));old_tau=np.zeros(m.nv);new_tau=old_tau.copy()
    nq=m.jnt_qposadr[m.actuator_trnid[:29,0]];nv=m.jnt_dofadr[m.actuator_trnid[:29,0]]
    loads=np.load(G6/f'execution_{seed}/rollout_locomotion_trace.npz')['trace'][:,14:16];force_checks=[]
    active=np.any(parameters!=0.,axis=1)|np.any(pv!=0.,axis=1)
    for tick in np.flatnonzero(active):
        for data,q,v,a,tau in [(old,base['qref'][tick],base['vref'][tick],a0[tick],old_tau),(fk,qref[tick],vref[tick],a1[tick],new_tau)]:
            data.qpos[:]=q;data.qvel[:]=v;data.qacc[:]=a;mujoco.mj_fwdPosition(m,data);mujoco.mj_fwdVelocity(m,data);mujoco.mj_rne(m,data,1,tau)
        effort[tick]=new_tau-old_tau;kp=m.actuator_gainprm[:29,0];kd=-m.actuator_biasprm[:29,2]
        increments[tick,:29]=qref[tick,nq]-base['qref'][tick,nq]+(kd*(vref[tick,nv]-base['vref'][tick,nv])+effort[tick,nv])/kp
        if tick%20==0 and envelope[tick]>.01:
            force_checks.append([int(tick),contact_wrench_feasible(m,old,wrench(m,old,old_tau),loads[max(0,tick-1)]),contact_wrench_feasible(m,fk,wrench(m,fk,new_tau),loads[max(0,tick-1)])])
    uref+=increments
    old_excess=np.maximum(base['qref'][:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-base['qref'][:,jq]);new_excess=np.maximum(qref[:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-qref[:,jq])
    excess=float((new_excess-np.maximum(old_excess,0.)).max());flight=~supported
    report=dict(seed=seed,scale=scale,base_reference=str(base_path),base_sha256=sha(base_path),compiler_sha256=sha(Path(__file__)),diffusion=provenance,
        strategy='Supported frozen root tilt plus waist tilt, with fixed Goal 6 root XYZ, waist yaw, hands, soles, object and gripper references; COM-preserving bounded IK, additive smoothing and differential RNE feedforward.',
        independently_sampled_coordinates=['root world-horizontal rotation X','root world-horizontal rotation Y','waist roll','waist pitch'],
        support_envelope=dict(minimum_supported_duration_s=.70,guard_s=.05,blend_s=.30),smoothing_sigma_s=.1,added_root_acceleration_limit_radps2=1.5,metrics=metrics,
        maximum_added_joint_limit_excursion_rad=excess,maximum_added_joint_velocity_radps=float(abs(pv[:,2:]).max()),maximum_added_root_rotation_velocity_radps=float(abs(pv[:,:2]).max()),
        maximum_added_root_rotation_acceleration_radps2=float(abs(np.gradient(pv[:,:2],DT,axis=0)).max()),maximum_feedforward_increment_rad=float(abs(increments).max()),maximum_incremental_native_effort_Nm=float(abs(effort[:,nv]).max()),
        stable_support_wrench_checks=force_checks,baseline_feasible_checks=sum(x[1] for x in force_checks),introduced_wrench_infeasibilities=sum(x[1] and not x[2] for x in force_checks),
        flight_added_pose=float(abs(parameters[flight]).max()),flight_added_velocity=float(abs(pv[flight]).max()),
        retained_root_XYZ=bool(np.array_equal(qref[:,:3],base['qref'][:,:3])),retained_waist_yaw=bool(np.array_equal(qref[:,19],base['qref'][:,19])),retained_object=bool(np.array_equal(qref[:,44:51],base['qref'][:,44:51])),retained_gripper_controls=bool(np.array_equal(uref[:,29:],base['uref'][:,29:])),
        physical_model_changed=False,simulation_stepped=False)
    report['pass_cheap_gate']=bool(max_pos<.0003 and max_angle<.003 and max_com<.003 and excess<1e-8 and abs(pv[:,2:]).max()<1.
        and abs(pv[:,:2]).max()<.25 and abs(np.gradient(pv[:,:2],DT,axis=0)).max()<3. and abs(increments).max()<.18 and abs(effort[:,nv]).max()<15.
        and report['baseline_feasible_checks']>=20 and report['introduced_wrench_infeasibilities']==0 and report['flight_added_pose']==0 and report['flight_added_velocity']==0)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out.with_name(out.stem+'_diagnostics.npz'),parameters=parameters,initial_parameters=initial,requested_modes=requested,native_source_modes=native_source,mapped_source_modes=mapped_source,support_envelope=envelope,incremental_effort=effort,feedforward_increment=increments,joint_qpos_columns=jq,joint_qvel_columns=jv,waist_variable_indices=2+wi)
    if not report['pass_cheap_gate']:raise RuntimeError({k:v for k,v in report.items() if k!='stable_support_wrench_checks'})
    np.savez_compressed(out,qref=qref,vref=vref,uref=uref,goal_position=raw['goal_position'],goal_quaternion=raw['goal_quaternion'])
    print(json.dumps({k:v for k,v in report.items() if k!='stable_support_wrench_checks'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--scale',type=float,default=1.)
    a=p.parse_args();generate(a.seed,a.out,a.scale)
