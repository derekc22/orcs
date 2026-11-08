"""Goal 4 extension: frozen-sample pelvis height/yaw and waist yaw.

Only additive changes to the successful physics-rate reference are constructed.
Feet, hands and object retain their Goal 4 trajectories. No simulation stepping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mjlab
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R
from orcs.tasks.uolm.compact.model import source_model

ROOT=Path(__file__).resolve().parents[2]
G4=ROOT/'logs/compact_diffusion_authority_v1'
sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_v1'))
from proposals import CHECKPOINT as DIFFUSION_CHECKPOINT

PHASES={'pickup':(2.7,8.,0,105),'carry':(17.245,23.605,105,225),'placement':(33.505,38.8,225,299)}
DT=.005


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(m):
    legs=[j for j in range(m.njnt) if any(s in m.joint(j).name for s in ['hip_','knee_','ankle_'])]
    arms=[j for j in range(m.njnt) if any(s in m.joint(j).name for s in ['shoulder_','elbow_','wrist_'])]
    waist=m.joint('waist_yaw_joint').id
    joints=legs+[waist]+arms
    return joints,m.jnt_qposadr[joints],m.jnt_dofadr[joints]


def support_clock():
    """Common expert guard, independent of selected sample or candidate state."""
    support=[]
    for name in ['execution_42','execution_43','disabled']:
        parent=G4 if name!='disabled' else ROOT/'logs/compact_orcs_v1'
        folder=parent/(name if name!='disabled' else 'complete_execution')
        l=np.load(folder/'rollout_locomotion_trace.npz')['trace']
        support.append(l[:,14:16].sum(1)>30.)
    supported=np.logical_and.reduce(support)
    distance=distance_transform_edt(supported)*DT
    x=np.clip((distance-.025)/.075,0.,1.)
    gate=x**3*(10.-15.*x+6.*x*x)
    return gate,supported


def targets(seed,scale):
    p=ROOT/'logs/compact_diffusion_v1'
    source=p/f'proposals/sample_{seed}.npz';calibration=p/'calibration/sample_0.npz'
    a,c=np.load(source),np.load(calibration)
    meta=json.loads((p/'proposals/provenance.json').read_text())
    assert sha(DIFFUSION_CHECKPOINT)==meta['checkpoint_sha256']
    assert np.array_equal(a['condition'],c['condition'])
    j=meta['native_joint_names'].index('waist_yaw_joint')+9
    z=a['raw'][:,2]-c['raw'][:,2]
    yaw=(R.from_quat(a['root_quaternion'],scalar_first=True)*R.from_quat(c['root_quaternion'],scalar_first=True).inv()).as_rotvec()[:,2]
    waist=a['raw'][:,j]-c['raw'][:,j]
    native=gaussian_filter1d(np.c_[z,yaw,waist],8.,axis=0,mode='nearest')
    gate,supported=support_clock()
    modes=np.zeros((len(gate),3));details={}
    for phase,(start,end,ds,de) in PHASES.items():
        s,e=round(start/DT),round(end/DT)
        rate=gate[s:e+1]
        clock=np.r_[0.,np.cumsum((rate[:-1]+rate[1:])*.5)]
        assert clock[-1]>0
        clock/=clock[-1]
        x=ds+(de-ds)*clock
        values=np.stack([np.interp(x,np.arange(300),v) for v in native.T],axis=1)
        values*=np.sin(np.pi*clock[:,None])**4
        limits=np.array([.008,.025,.06])
        gains=np.array([100.,60.,50.])
        modes[s:e+1]=scale*limits*np.tanh(values*gains/limits)
        modes[[s,e]]=0.
        details[phase]=dict(physical_interval_s=[start,end],native_interval_frames=[ds,de],
            requested_peak=np.max(abs(modes[s:e+1]),axis=0).tolist(),
            clock_paused_fraction=float(np.mean(rate==0.)),maximum_clock_speed=float(rate.max()/rate.mean()))
    return modes,gate,supported,dict(sample=str(source),sample_sha256=sha(source),calibration_sha256=sha(calibration),frozen_checkpoint_sha256=meta['checkpoint_sha256'],mode_gains=[100.,60.,50.],mode_limits=[.008,.025,.06],phases=details)


def generate(seed,out,scale=.6,disabled=False):
    out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    base_path=G4/(f'feasible_{seed}.npz' if not disabled else 'disabled.npz')
    raw=dict(np.load(base_path));base={k:v.astype(np.float64) for k,v in raw.items()}
    if disabled:
        np.savez_compressed(out,**raw)
        out.with_suffix('.json').write_text(json.dumps(dict(disabled=True,source=str(base_path),source_sha256=sha(base_path)),indent=2));return
    modes,gate,supported,provenance=targets(seed,scale)
    # The velocity correction uses a centered physics-rate derivative. Include
    # its one-tick stencil at flight edges, without constraining the subsequent
    # supported landing interval.
    momentum_guard=distance_transform_edt(supported)<=1.01
    m=source_model();joints,jq,jv=groups(m);assert len(joints)==27
    # Variables: root xyz, world yaw, twelve leg joints, waist yaw, fourteen arms.
    n=31;waist_column=16
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    hands=[m.body('hand_'+s).id for s in ['left','right']]
    bodies=feet+hands
    old,fk=mujoco.MjData(m),mujoco.MjData(m)
    jp=np.zeros((3,m.nv));jr=jp.copy();B=np.zeros((m.nv,n))
    angular_matrix=np.zeros((3,m.nv))
    B[:3,:3]=np.eye(3);B[np.ix_(jv,np.arange(4,n))]=np.eye(27)
    parameters=np.zeros((len(modes),n));maximum_geom=0.;maximum_angle=0.;maximum_com=0.
    qref=base['qref'].copy()

    def pose(q0,p):
        q=q0.copy();q[:3]+=p[:3]
        if p[3]:q[3:7]=(R.from_euler('z',p[3])*R.from_quat(q0[3:7],scalar_first=True)).as_quat(scalar_first=True)
        q[jq]+=p[4:]
        return q

    for phase,(start,end,_,_) in PHASES.items():
        s,e=round(start/DT),round(end/DT);previous=np.zeros(n)
        previous_h=np.zeros(3);held_h=np.zeros(3);held_z=0.;previous_com_z=0.;was_paused=False
        for tick in range(s,e+1):
            q0=base['qref'][tick];old.qpos[:]=q0
            mujoco.mj_kinematics(m,old);mujoco.mj_comPos(m,old)
            old.qvel[:]=base['vref'][tick];mujoco.mj_comVel(m,old);mujoco.mj_subtreeVel(m,old)
            baseline_h=old.subtree_angmom[0].copy()
            # The motion clock has a wider guard, but H is conserved only
            # without usable support. Once a foot lands, real ground moments
            # are available and the expert impact details must remain free.
            paused=bool(momentum_guard[tick])
            if paused and not was_paused:held_h=previous_h.copy();held_z=previous_com_z
            positions=old.xpos[bodies].copy();rotations=R.from_matrix(old.xmat[bodies].reshape(4,3,3))
            target=np.zeros(n);target[[2,3,waist_column]]=modes[tick]
            radius=np.sin(np.pi*(tick-s)/(e-s))**2
            low=np.r_[-.015,-.015,-.008,-.025,np.repeat(-.10,12),-.06,np.repeat(-.10,14)]*radius
            high=-low
            # Baseline soft-stop excursions remain allowed, never enlarged.
            low[4:]=np.maximum(low[4:],np.minimum(m.jnt_range[joints,0]-q0[jq]+.0002,0.))
            high[4:]=np.minimum(high[4:],np.maximum(m.jnt_range[joints,1]-q0[jq]-.0002,0.))
            speed=np.r_[.0002,.0002,.0002,.001,np.repeat(.0035,12),.0015,np.repeat(.0035,14)]
            low=np.maximum(low,previous-speed);high=np.minimum(high,previous+speed)
            low=np.minimum(low,high-1e-12)
            p=np.clip(previous,low,high)
            for iteration in range(7):
                fk.qpos[:]=pose(q0,p)
                mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                B[3:6,3]=R.from_quat(fk.qpos[3:7],scalar_first=True).inv().apply([0.,0.,1.])
                rows=[];errors=[]
                for k,body in enumerate(bodies):
                    mujoco.mj_jacBody(m,fk,jp,jr,body)
                    rows.extend([250*jp@B,70*jr@B])
                    errors.extend([250*(positions[k]-fk.xpos[body]),70*(rotations[k]*R.from_matrix(fk.xmat[body].reshape(3,3)).inv()).as_rotvec()])
                mujoco.mj_jacSubtreeCom(m,fk,jp,0)
                rows.append(500*jp[:2]@B);errors.append(500*(old.subtree_com[0,:2]-fk.subtree_com[0,:2]))
                if paused:
                    rows.append(300*jp[2:3]@B);errors.append(np.array([300*(old.subtree_com[0,2]+held_z-fk.subtree_com[0,2])]))
                    # Differential centroidal constraint, not a fictitious
                    # external moment: keep the added whole-system H constant
                    # while the expert trajectory has no usable foot support.
                    velocity=base['vref'][tick].copy()
                    velocity[3:6]=(R.from_quat(fk.qpos[3:7],scalar_first=True).inv()*R.from_quat(q0[3:7],scalar_first=True)).apply(velocity[3:6])
                    velocity+=B@(p-previous)/DT
                    mujoco.mj_angmomMat(m,fk,angular_matrix,0)
                    dh=angular_matrix@velocity-baseline_h
                    rows.append(10*angular_matrix@B/DT);errors.append(10*(held_h-dh))
                weights=np.r_[.5,.5,12.,2.,np.repeat(.01,12),2.,np.repeat(.01,14)]
                rows.append(np.diag(weights));errors.append(weights*(target-p))
                A,b=np.concatenate(rows),np.concatenate(errors)
                step=lsq_linear(A,b,bounds=(low-p-1e-12,high-p+1e-12),method='bvls',tol=1e-9).x
                p+=step
                if np.linalg.norm(step)<1e-8:break
            if tick in [s,e]:p[:]=0.
            qref[tick]=pose(q0,p);parameters[tick]=p;previous=p
            fk.qpos[:]=qref[tick];mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
            maximum_geom=max(maximum_geom,float(np.linalg.norm(fk.xpos[bodies]-positions,axis=1).max()))
            maximum_angle=max(maximum_angle,float((R.from_matrix(fk.xmat[bodies].reshape(4,3,3))*rotations.inv()).magnitude().max()))
            maximum_com=max(maximum_com,float(np.linalg.norm(fk.subtree_com[0,:2]-old.subtree_com[0,:2])))
            velocity=base['vref'][tick].copy()
            velocity[3:6]=(R.from_quat(fk.qpos[3:7],scalar_first=True).inv()*R.from_quat(q0[3:7],scalar_first=True)).apply(velocity[3:6])
            # previous already equals p above; obtain the prior sample from
            # stored parameters for the causal backward velocity estimate.
            prior=parameters[tick-1] if tick>s else np.zeros(n)
            velocity+=B@(p-prior)/DT
            mujoco.mj_angmomMat(m,fk,angular_matrix,0)
            previous_h=angular_matrix@velocity-baseline_h
            previous_com_z=float(fk.subtree_com[0,2]-old.subtree_com[0,2]);was_paused=paused
        print('BODY_PHASE',phase,'achieved root_z/yaw/waist_yaw',np.max(abs(parameters[s:e+1][:,[2,3,16]]),axis=0),flush=True)

    pv=np.gradient(parameters,DT,axis=0)
    vref=base['vref'].copy();vref[:,:3]+=pv[:,:3]
    rn=R.from_quat(qref[:,3:7],scalar_first=True);rb=R.from_quat(base['qref'][:,3:7],scalar_first=True)
    vref[:,3:6]=(rn.inv()*rb).apply(base['vref'][:,3:6])+rn.inv().apply(np.c_[np.zeros((len(pv),2)),pv[:,3]])
    vref[:,jv]+=pv[:,4:]
    active=np.any(parameters!=0.,axis=1)
    vref[~active & ~np.any(pv!=0.,axis=1)]=base['vref'][~active & ~np.any(pv!=0.,axis=1)]
    actuators=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in joints])
    uref=base['uref'].copy();dq=parameters[:,4:]
    base_acc=np.gradient(base['vref'],DT,axis=0);new_acc=np.gradient(vref,DT,axis=0)
    old_tau=np.zeros(m.nv);new_tau=old_tau.copy();tau_delta=np.zeros((len(qref),m.nv))
    increments=np.zeros_like(uref);H=np.zeros((len(qref),3));H0=H.copy();com=np.zeros((len(qref),3));com0=com.copy()
    for tick in np.flatnonzero(active | np.any(pv!=0.,axis=1)):
        for data,q,v,acc,tau,h,c in [(old,base['qref'][tick],base['vref'][tick],base_acc[tick],old_tau,H0,com0),
                                    (fk,qref[tick],vref[tick],new_acc[tick],new_tau,H,com)]:
            data.qpos[:]=q;data.qvel[:]=v;data.qacc[:]=acc
            mujoco.mj_fwdPosition(m,data);mujoco.mj_fwdVelocity(m,data);mujoco.mj_rne(m,data,1,tau)
            mujoco.mj_subtreeVel(m,data);h[tick]=data.subtree_angmom[0];c[tick]=data.subtree_com[0]
        tau_delta[tick]=new_tau-old_tau
        kp=m.actuator_gainprm[actuators,0];kd=-m.actuator_biasprm[actuators,2]
        increments[tick,actuators]=dq[tick]+(kd*pv[tick,4:]+tau_delta[tick,jv])/kp
    uref+=increments
    # Goal 3 diagnostic: inspect only incremental whole-system momentum during flight.
    flight=~supported
    flight_ranges=[];edges=np.diff(np.r_[False,flight,False].astype(int))
    for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if e-s<2 or not active[s:e].any():continue
        dh=H[s:e]-H0[s:e];dc=com[s:e]-com0[s:e]
        flight_ranges.append(dict(ticks=[int(s),int(e)],extra_angular_momentum_change=float(np.linalg.norm(dh-dh[0],axis=1).max()),
            extra_mean_external_moment_Nm=float(np.linalg.norm(dh[-1]-dh[0])/((e-s-1)*DT)),
            extra_com_displacement_change_m=float(np.linalg.norm(dc-dc[0],axis=1).max())))
    max_moment=max([r['extra_mean_external_moment_Nm'] for r in flight_ranges] or [0.])
    position_excess=np.maximum(qref[:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-qref[:,jq])
    old_excess=np.maximum(base['qref'][:,jq]-m.jnt_range[joints,1],m.jnt_range[joints,0]-base['qref'][:,jq])
    added_excess=float((position_excess-np.maximum(old_excess,0.)).max())
    grippers=[m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith('hand_')]
    report=dict(seed=seed,scale=scale,centroidal_guard_ticks=1,compiler_sha256=sha(Path(__file__)),base_reference=str(base_path),base_sha256=sha(base_path),diffusion=provenance,
        coordinates=['pelvis height','pelvis world yaw','waist yaw'],dependent_coordinates='Root xy preserves whole-system horizontal COM; leg and arm changes preserve both sole and hand transforms. These are constraint responses, not independent diffusion commands.',
        maximum_foot_or_hand_position_residual_m=maximum_geom,maximum_foot_or_hand_angle_residual_rad=maximum_angle,
        maximum_com_xy_change_m=maximum_com,maximum_added_joint_limit_excursion_rad=added_excess,
        maximum_added_joint_velocity_radps=float(abs(pv[:,4:]).max()),maximum_added_root_velocity_mps=float(abs(pv[:,:3]).max()),
        maximum_feedforward_increment_rad=float(abs(increments).max()),maximum_incremental_joint_torque_Nm=float(abs(tau_delta[:,jv]).max()),
        maximum_flight_additional_external_moment_Nm=max_moment,flight_diagnostics=flight_ranges,
        object_reference_bit_identical=all(np.array_equal(qref[:,44:51],base['qref'][:,44:51]) for _ in [0]),
        gripper_reference_bit_identical=bool(np.array_equal(qref[:,grippers],base['qref'][:,grippers])),
        gripper_feedforward_bit_identical=bool(np.array_equal(uref[:,29:],base['uref'][:,29:])),
        physics_rate_samples_preserved=True,actual_ORCS_required=True)
    report['pass_cheap_gate']=bool(maximum_geom<.0003 and maximum_angle<.003 and maximum_com<.0001 and added_excess<1e-8
        and abs(pv[:,4:]).max()<1.0 and abs(increments).max()<.18 and abs(tau_delta[:,jv]).max()<25. and max_moment<.5)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out.with_name(out.stem+'_diagnostics.npz'),parameters=parameters,requested_modes=modes,support_gate=gate,incremental_torque=tau_delta,feedforward_increment=increments)
    if not report['pass_cheap_gate']:raise RuntimeError(report)
    np.savez_compressed(out,qref=qref,vref=vref,uref=uref,goal_position=raw['goal_position'],goal_quaternion=raw['goal_quaternion'])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scale',type=float,default=.6);p.add_argument('--disabled',action='store_true')
    a=p.parse_args();generate(a.seed,a.out,a.scale,a.disabled)
