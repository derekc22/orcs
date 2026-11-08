"""Goal 5 extension: independent sampled XY with bounded COM counterlean.

All accepted 200 Hz samples survive. Goal 5 height/yaw, waist, object and
gripper references are retained. Feet and hands are constrained in world space.
This module never steps a simulator; final feasibility is actual ORCS execution.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mjlab
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter1d
from scipy.optimize import lsq_linear,linprog
from scipy.spatial.transform import Rotation as R
from orcs.tasks.uolm.compact.model import source_model

ROOT=Path(__file__).resolve().parents[2]
G5=ROOT/'logs/compact_diffusion_body_v1'
spec=importlib.util.spec_from_file_location('_goal5_refine',ROOT/'scripts/compact_diffusion_body_v1/refine.py')
body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)
PHASES,DT=body.PHASES,body.DT


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def skew(x):
    a,b,c=x;return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])


def left_jacobian(phi):
    t=np.linalg.norm(phi);S=skew(phi)
    a=(1-np.cos(t))/t**2 if t>1e-5 else .5-t*t/24
    b=(t-np.sin(t))/t**3 if t>1e-5 else 1/6-t*t/120
    return np.eye(3)+a*S+b*S@S


def euler_body_jacobian(roll,pitch):
    return np.array([[1.,0.,-np.sin(pitch)],[0.,np.cos(roll),np.sin(roll)*np.cos(pitch)],[0.,-np.sin(roll),np.cos(roll)*np.cos(pitch)]])


def clock():
    _,supported=body.support_clock()
    for seed in [42,43]:
        l=np.load(G5/f'execution_{seed}/rollout_locomotion_trace.npz')['trace']
        supported &= l[:,14:16].sum(1)>30.
    distance=distance_transform_edt(supported)*DT
    x=np.clip((distance-.025)/.075,0.,1.)
    return x**3*(10-15*x+6*x*x),supported


def targets(seed,limit,gain):
    folder=ROOT/'logs/compact_diffusion_v1'
    source=folder/f'proposals/sample_{seed}.npz';calibration=folder/'calibration/sample_0.npz'
    a,c=np.load(source),np.load(calibration)
    meta=json.loads((folder/'proposals/provenance.json').read_text())
    assert sha(body.DIFFUSION_CHECKPOINT)==meta['checkpoint_sha256']
    assert np.array_equal(a['condition'],c['condition'])
    demo=np.load(folder/'proposals/conditioning.npz')['demonstration']
    baseline=np.load(G5/'feasible_42.npz')
    h0=np.arctan2(*(demo[-1,38:40]-demo[0,38:40])[::-1])
    h1=np.arctan2(*(baseline['goal_position'][:2]-baseline['qref'][0,44:46])[::-1])
    align=R.from_euler('z',h1-h0)
    innovation=align.apply(a['raw'][:,:3]-c['raw'][:,:3])[:,:2]
    innovation=gaussian_filter1d(innovation,8.,axis=0,mode='nearest')
    gate,supported=clock();xy=np.zeros((len(gate),2));details={}
    for phase,(start,end,ds,de) in PHASES.items():
        s,e=round(start/DT),round(end/DT);rate=gate[s:e+1]
        progress=np.r_[0.,np.cumsum((rate[:-1]+rate[1:])*.5)];progress/=progress[-1]
        x=ds+(de-ds)*progress
        curve=np.stack([np.interp(x,np.arange(300),v) for v in innovation.T],axis=1)
        curve*=gain*np.sin(np.pi*progress[:,None])**4
        curve/=np.sqrt(1+np.sum(curve**2,axis=1,keepdims=True)/limit**2)
        xy[s:e+1]=curve;xy[[s,e]]=0.
        details[phase]=dict(physical_interval_s=[start,end],native_frames=[ds,de],
            requested_xy_rms_m=float(np.sqrt(np.mean(np.sum(curve**2,axis=1)))),requested_xy_peak_m=float(np.linalg.norm(curve,axis=1).max()),
            paused_clock_fraction=float(np.mean(rate==0.)))
    return xy,gate,supported,dict(source=str(source),source_sha256=sha(source),calibration_sha256=sha(calibration),
        checkpoint_sha256=meta['checkpoint_sha256'],alignment_yaw_rad=float(h1-h0),gain=gain,limit_m=limit,phases=details)


def wrench(m,d,tau):
    c=d.subtree_com[0]
    f=tau[:3]+tau[43:46]
    moment=R.from_quat(d.qpos[3:7],scalar_first=True).apply(tau[3:6])+np.cross(d.qpos[:3]-c,tau[:3])
    moment+=R.from_quat(d.qpos[47:51],scalar_first=True).apply(tau[46:49])+np.cross(d.qpos[44:47]-c,tau[43:46])
    return np.r_[f,moment]


def contact_wrench_feasible(m,d,w,loads):
    """Friction/COP feasibility at stable supported states, without stepping.

    Baseline and candidate use the same conservative inscribed friction cones.
    Existing baseline discretization failures are reported, never counted as
    evidence that a new candidate is feasible.
    """
    points=[];mu=[]
    for side,load in zip(['left','right'],loads):
        if load<30.:continue
        candidates=[]
        for g in range(m.ngeom):
            if not (m.geom(g).name or '').startswith(side+'_foot') or not m.geom_contype[g]:continue
            pos=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2]
            for sign in [-1,1]:
                p=pos+sign*m.geom_size[g,1]*axis-np.array([0.,0.,m.geom_size[g,0]])
                candidates.append((p.copy(),float(m.geom_friction[g,0])))
        if not candidates:continue
        lowest=min(p[0][2] for p in candidates)
        for p,friction in candidates:
            if p[2]<=lowest+.003:
                p[2]=0.;points.append(p);mu.append(friction)
    if not points:return False
    A=np.concatenate([np.r_[np.eye(3),skew(p-d.subtree_com[0])] for p in points],axis=1)
    inequalities=[]
    for i,friction in enumerate(mu):
        if int(m.opt.cone)==0:
            # Exact L1 facets of the accepted pyramidal contact model.
            normals=[(x,y,1.) for x in [-1.,1.] for y in [-1.,1.]]
        else:
            # Conservative 16-sided approximation of an elliptic cone.
            normals=[(np.cos(a),np.sin(a),np.cos(np.pi/16)) for a in np.arange(16)*2*np.pi/16]
        for x,y,radius in normals:
            row=np.zeros(3*len(points));row[3*i]=x;row[3*i+1]=y;row[3*i+2]=-friction*radius;inequalities.append(row)
    bounds=[v for _ in points for v in [(None,None),(None,None),(0,None)]]
    result=linprog(np.zeros(A.shape[1]),A_ub=np.asarray(inequalities),b_ub=np.zeros(len(inequalities)),A_eq=A,b_eq=w,bounds=bounds,method='highs')
    return bool(result.success)


def generate(seed,out,limit=.008,gain=20.,disabled=False):
    out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    base_path=G5/(f'feasible_{seed}.npz' if not disabled else 'disabled.npz')
    raw=dict(np.load(base_path));base={k:v.astype(float) for k,v in raw.items()}
    if disabled:
        np.savez_compressed(out,**raw);out.with_suffix('.json').write_text(json.dumps(dict(disabled=True,source=str(base_path),sha256=sha(base_path)),indent=2));return
    raw_requested,gate,supported,provenance=targets(seed,limit,gain)
    distance=distance_transform_edt(supported)*DT
    blend=np.clip((distance-.04)/.14,0.,1.)
    support_envelope=blend**3*(10-15*blend+6*blend*blend)
    requested=raw_requested*support_envelope[:,None]
    m=source_model()
    joints=[j for j in range(m.njnt) if any(t in m.joint(j).name for t in ['hip_','knee_','ankle_','shoulder_','elbow_','wrist_'])]
    assert len(joints)==26
    jq,jv=m.jnt_qposadr[joints],m.jnt_dofadr[joints]
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];hands=[m.body('hand_'+s).id for s in ['left','right']];bodies=feet+hands
    old,fk=mujoco.MjData(m),mujoco.MjData(m);jp=np.zeros((3,m.nv));jr=jp.copy();C=jp.copy()
    B=np.zeros((m.nv,30));B[:2,:2]=np.eye(2);B[np.ix_(jv,np.arange(4,30))]=np.eye(26)
    guard=distance_transform_edt(supported)<=1.01
    parameters=np.zeros((len(requested),30));qref=base['qref'].copy()
    base_euler=R.from_quat(base['qref'][:,3:7],scalar_first=True).as_euler('ZYX')
    base_rates=np.asarray([np.linalg.solve(euler_body_jacobian(e[2],e[1]),v[3:6]) for e,v in zip(base_euler,base['vref'])])
    maximum_position=0.;maximum_angle=0.;maximum_com=0.

    def pose(q0,p):
        q=q0.copy();q[:2]+=p[:2];q[jq]+=p[4:]
        if p[2:4].any():
            yaw,pitch,roll=R.from_quat(q0[3:7],scalar_first=True).as_euler('ZYX')
            q[3:7]=R.from_euler('ZYX',[yaw,pitch+p[3],roll+p[2]]).as_quat(scalar_first=True)
        return q

    def velocity_base(tick,p):
        v=base['vref'][tick].copy()
        v[3:6]=euler_body_jacobian(base_euler[tick,2]+p[2],base_euler[tick,1]+p[3])@base_rates[tick]
        return v

    for phase,(start,end,_,_) in PHASES.items():
        s,e=round(start/DT),round(end/DT);previous=np.zeros(30);last_H=np.zeros(3);last_com_z=0.;was_guard=False
        held_H=last_H.copy();held_z=0.;last_com_xy=np.zeros(2);held_xy=last_com_xy.copy()
        last_difference=np.zeros(30)
        for tick in range(s,e+1):
            q0=base['qref'][tick];old.qpos[:]=q0;old.qvel[:]=base['vref'][tick]
            mujoco.mj_kinematics(m,old);mujoco.mj_comPos(m,old);mujoco.mj_comVel(m,old);mujoco.mj_subtreeVel(m,old)
            H0=old.subtree_angmom[0].copy();positions=old.xpos[bodies].copy();rotations=R.from_matrix(old.xmat[bodies].reshape(4,3,3))
            paused=bool(guard[tick])
            if paused and not was_guard:held_H=last_H.copy();held_z=last_com_z;held_xy=last_com_xy.copy()
            tube=.006 if phase=='carry' else .008
            u=(tick-s)/(e-s);envelope=np.sin(np.pi*u)**2*support_envelope[tick]
            low=-np.r_[.014,.014,0.,0.,np.repeat(.12,26)]*envelope;high=-low
            low[4:]=np.maximum(low[4:],np.minimum(m.jnt_range[joints,0]-q0[jq]+.0002,0.));high[4:]=np.minimum(high[4:],np.maximum(m.jnt_range[joints,1]-q0[jq]-.0002,0.))
            speed=np.r_[.00025,.00025,.00075,.00075,np.repeat(.0035,26)]
            low=np.maximum(low,previous-speed);high=np.minimum(high,previous+speed);low=np.minimum(low,high-1e-12)
            # Counterlean has whole-body inertia. Bound its acceleration, rather
            # than relying only on a velocity limit that permits sudden flips.
            second_order=np.r_[.15,.15,3.,3.]*DT**2
            low[:4]=np.maximum(low[:4],previous[:4]+last_difference[:4]-second_order)
            high[:4]=np.minimum(high[:4],previous[:4]+last_difference[:4]+second_order)
            low=np.minimum(low,high-1e-12)
            p=np.clip(previous,low,high);p[2:4]=0.;target=np.r_[requested[tick],np.zeros(28)]
            for _ in range(8):
                fk.qpos[:]=pose(q0,p);mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                B[3:6,2:4]=0.
                rows=[];errors=[]
                for k,link in enumerate(bodies):
                    mujoco.mj_jacBody(m,fk,jp,jr,link)
                    rows.extend([250*jp@B,70*jr@B]);errors.extend([250*(positions[k]-fk.xpos[link]),70*(rotations[k]*R.from_matrix(fk.xmat[link].reshape(3,3)).inv()).as_rotvec()])
                mujoco.mj_jacSubtreeCom(m,fk,jp,0)
                offset=fk.subtree_com[0,:2]-old.subtree_com[0,:2];norm=np.linalg.norm(offset)
                if norm>tube:
                    normal=offset/norm;rows.append(500*normal[None]@jp[:2]@B);errors.append(np.array([500*(tube-norm)]))
                if paused:
                    rows.append(300*jp[2:3]@B);errors.append(np.array([300*(old.subtree_com[0,2]+held_z-fk.subtree_com[0,2])]))
                    mujoco.mj_angmomMat(m,fk,C,0)
                    velocity=velocity_base(tick,p)+B@(p-previous)/DT
                    rows.append(10*C@B/DT);errors.append(10*(held_H-(C@velocity-H0)))
                weights=np.r_[20.,20.,.08,.08,np.repeat(.015,26)]
                rows.append(np.diag(weights));errors.append(weights*(target-p))
                step=lsq_linear(np.concatenate(rows),np.concatenate(errors),bounds=(low-p-1e-12,high-p+1e-12),method='bvls',tol=1e-9).x
                p+=step;p[2:4]=0.
                if np.linalg.norm(step)<1e-8:break
            if tick in [s,e] or envelope<1e-14:p[:]=0.
            qref[tick]=pose(q0,p);parameters[tick]=p
            fk.qpos[:]=qref[tick];mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
            maximum_position=max(maximum_position,float(np.linalg.norm(fk.xpos[bodies]-positions,axis=1).max()))
            maximum_angle=max(maximum_angle,float((R.from_matrix(fk.xmat[bodies].reshape(4,3,3))*rotations.inv()).magnitude().max()))
            maximum_com=max(maximum_com,float(max(0.,np.linalg.norm(fk.subtree_com[0,:2]-old.subtree_com[0,:2])-tube)))
            B[3:6,2:4]=0.
            mujoco.mj_angmomMat(m,fk,C,0);last_H=C@(velocity_base(tick,p)+B@(p-previous)/DT)-H0
            last_com_z=float(fk.subtree_com[0,2]-old.subtree_com[0,2]);last_com_xy=fk.subtree_com[0,:2]-old.subtree_com[0,:2]
            last_difference=p-previous;previous=p;was_guard=paused
        print('XY_PHASE',phase,'xy/tilt peak',np.linalg.norm(parameters[s:e+1,:2],axis=1).max(),np.linalg.norm(parameters[s:e+1,2:4],axis=1).max(),flush=True)

    pv=np.gradient(parameters,DT,axis=0);vref=base['vref'].copy();vref[:,:2]+=pv[:,:2];vref[:,jv]+=pv[:,4:]
    for tick in np.flatnonzero(np.any(parameters!=0.,axis=1)|np.any(pv!=0.,axis=1)):
        vref[tick,3:6]=base['vref'][tick,3:6]
    a0=np.gradient(base['vref'],DT,axis=0);a1=np.gradient(vref,DT,axis=0)
    uref=base['uref'].copy();increments=np.zeros_like(uref);tau_delta=np.zeros((len(qref),m.nv));H=np.zeros((len(qref),3));H0=H.copy()
    old_tau=np.zeros(m.nv);new_tau=old_tau.copy();native=np.arange(29);nq=m.jnt_qposadr[m.actuator_trnid[:29,0]];nv=m.jnt_dofadr[m.actuator_trnid[:29,0]]
    loads=np.load(G5/f'execution_{seed}/rollout_locomotion_trace.npz')['trace'][:,14:16]
    force_checks=[];active=np.any(parameters!=0.,axis=1)|np.any(pv!=0.,axis=1)
    for tick in np.flatnonzero(active):
        for data,q,v,acc,tau,h in [(old,base['qref'][tick],base['vref'][tick],a0[tick],old_tau,H0),(fk,qref[tick],vref[tick],a1[tick],new_tau,H)]:
            data.qpos[:]=q;data.qvel[:]=v;data.qacc[:]=acc
            mujoco.mj_fwdPosition(m,data);mujoco.mj_fwdVelocity(m,data);mujoco.mj_rne(m,data,1,tau);mujoco.mj_subtreeVel(m,data);h[tick]=data.subtree_angmom[0]
        tau_delta[tick]=new_tau-old_tau
        kp=m.actuator_gainprm[:29,0];kd=-m.actuator_biasprm[:29,2]
        increments[tick,:29]=(qref[tick,nq]-base['qref'][tick,nq])+(kd*(vref[tick,nv]-base['vref'][tick,nv])+tau_delta[tick,nv])/kp
        if tick%20==0 and gate[tick]>.99:
            f0=contact_wrench_feasible(m,old,wrench(m,old,old_tau),loads[max(0,tick-1)])
            f1=contact_wrench_feasible(m,fk,wrench(m,fk,new_tau),loads[max(0,tick-1)])
            force_checks.append([int(tick),f0,f1])
    uref+=increments
    edges=np.diff(np.r_[False,~supported,False].astype(int));flights=[]
    for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if e-s<2 or not active[s:e].any():continue
        dh=H[s:e]-H0[s:e]
        flights.append(dict(ticks=[int(s),int(e)],extra_momentum_change=float(np.linalg.norm(dh-dh[0],axis=1).max()),extra_mean_external_moment_Nm=float(np.linalg.norm(dh[-1]-dh[0])/((e-s-1)*DT))))
    max_moment=max([f['extra_mean_external_moment_Nm'] for f in flights] or [0.])
    old_excess=np.maximum(base['qref'][:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-base['qref'][:,jq]);new_excess=np.maximum(qref[:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-qref[:,jq])
    excess=float((new_excess-np.maximum(old_excess,0.)).max())
    report=dict(seed=seed,base_reference=str(base_path),base_sha256=sha(base_path),compiler_sha256=sha(Path(__file__)),diffusion=provenance,
        independently_sampled_coordinates=['root world X','root world Y'],balance_coordinates='Supported-phase XY with bounded COM displacement and leg/arm compensation. Goal 5 root roll/pitch remains the reference; no added tilt is claimed.',
        maximum_foot_hand_position_error_m=maximum_position,maximum_foot_hand_angle_error_rad=maximum_angle,maximum_COM_tube_tracking_error_m=maximum_com,
        supported_COM_tube_m=dict(pickup=.008,carry=.006,placement=.008),COM_target='COM displacement is bounded inside the support tube; friction/COP feasibility is screened against the accepted baseline. All added pose changes return to zero with a C2 envelope near flight.',
        maximum_added_joint_limit_excursion_rad=excess,maximum_added_joint_velocity_radps=float(abs(pv[:,4:]).max()),maximum_added_XY_velocity_mps=float(abs(pv[:,:2]).max()),
        maximum_feedforward_increment_rad=float(abs(increments).max()),maximum_incremental_native_effort_Nm=float(abs(tau_delta[:,nv]).max()),
        maximum_flight_additional_moment_Nm=max_moment,flight_momentum=flights,
        stable_support_wrench_checks=force_checks,baseline_feasible_checks=sum(r[1] for r in force_checks),introduced_wrench_infeasibilities=sum(r[1] and not r[2] for r in force_checks),
        retained_G5_root_height=bool(np.array_equal(qref[:,2],base['qref'][:,2])),retained_G5_waist=bool(np.array_equal(qref[:,19:22],base['qref'][:,19:22])),
        retained_G5_object=bool(np.array_equal(qref[:,44:51],base['qref'][:,44:51])),retained_gripper_feedforward=bool(np.array_equal(uref[:,29:],base['uref'][:,29:])),
        physical_model_changed=False,simulation_stepped=False,physics_rate_reference_retained=True)
    report['pass_cheap_gate']=bool(maximum_position<.0003 and maximum_angle<.003 and maximum_com<.0001 and excess<1e-8
        and abs(pv[:,4:]).max()<1. and abs(increments).max()<.18 and abs(tau_delta[:,nv]).max()<25. and max_moment<.5
        and report['introduced_wrench_infeasibilities']==0)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out.with_name(out.stem+'_diagnostics.npz'),parameters=parameters,requested_xy=requested,raw_source_xy=raw_requested,support_envelope=support_envelope,support_gate=gate,incremental_effort=tau_delta,feedforward_increment=increments)
    if not report['pass_cheap_gate']:raise RuntimeError({k:v for k,v in report.items() if k not in ['stable_support_wrench_checks','flight_momentum']})
    np.savez_compressed(out,qref=qref,vref=vref,uref=uref,goal_position=raw['goal_position'],goal_quaternion=raw['goal_quaternion'])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=float,default=.008);p.add_argument('--gain',type=float,default=20.);p.add_argument('--disabled',action='store_true')
    a=p.parse_args();generate(a.seed,a.out,a.limit,a.gain,a.disabled)
