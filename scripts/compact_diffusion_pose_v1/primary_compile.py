"""Physics-rate, contact-loaded compilation of full frozen proposal targets.

The accepted source of footsteps, gripper mechanics and 200 Hz contact details
is retained. Only source-derived offsets are smoothed/projected. All dynamics
here are offline RNE/Jacobian calculations; genuine ORCS is the execution test.
"""
import argparse,json
from pathlib import Path

import mjlab,mujoco,numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from primary_targets import construct,ROOT,G7,DT
from primary_kinematics import Projector
from primary_dynamics import loaded
from root_refine import sha
from orcs.tasks.uolm.compact.model import source_model


def interpolate(base,frame):
    j=min(int(frame),len(base)-2);u=frame-j;q=(1-u)*base[j]+u*base[j+1]
    for s,e in [(3,7),(47,51)]:q[s:e]=R.from_quat(q[s:e],scalar_first=True).as_quat(scalar_first=True)
    return q


def transport(new,old,current):
    q=current+(new-old)
    for s,e in [(3,7),(47,51)]:q[s:e]=(R.from_quat(new[s:e],scalar_first=True)*R.from_quat(old[s:e],scalar_first=True).inv()*R.from_quat(current[s:e],scalar_first=True)).as_quat(scalar_first=True)
    return q


def compile_reference(seed,out,selected=None,reuse=None,continuous=False,fixed_motion=False,fixed_angles=False):
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    m=source_model();base,expert,target,warm,mask,phases,meta=construct(seed,m);project=Projector(m,base)
    meta['projection_options']=dict(continuous=continuous,fixed_motion=fixed_motion,fixed_angles=fixed_angles)
    if selected:
        phases={name:phases[name] for name in selected};meta['selected_phases']=selected
        included=np.zeros(len(mask),dtype=bool)
        for s,e,_,_ in phases.values():included[s:e+1]=True
        mask[~included]=0.;target[~included]=base['qref'][~included]
    qcoarse=base['qref'].copy();coarse_metrics=[]
    if reuse:
        saved=np.load(reuse);qcoarse[mask>0]=saved['qcoarse'][mask>0]
        meta['coarse_reuse']=dict(path=str(reuse),sha256=sha(reuse))
    for phase,(s,e,_,_) in ([] if reuse else phases.items()):
        ticks=np.unique(np.r_[np.arange(s,e+1,4),e]);qs=[];previous=None;last=None
        for tick in ticks:
            if mask[tick]<=1e-14:q=base['qref'][tick].copy();answer=None
            else:
                guess=transport(previous,base['qref'][last],base['qref'][tick]) if previous is not None else None
                answer=project.project(tick,target[tick],interpolate(base['qref'],warm[tick]),previous=guess,max_iterations=35,com_tube=.001 if phase=='carry' else .02)
                q=answer['q'];coarse_metrics.append(dict(tick=int(tick),phase=phase,**{k:v for k,v in answer.items() if k not in ['q','parameters']}))
            qs.append(q);previous=q;last=tick
        qs=np.asarray(qs);linear=qs-base['qref'][ticks];linear[:,3:7]=0.;linear[:,47:51]=0.
        rotations=[(R.from_quat(qs[:,a:b],scalar_first=True)*R.from_quat(base['qref'][ticks,a:b],scalar_first=True).inv()).as_rotvec() for a,b in [(3,7),(47,51)]]
        full=np.arange(s,e+1);delta=CubicSpline(ticks,linear,bc_type=((1,np.zeros(m.nq)),(1,np.zeros(m.nq))))(full)
        delta=gaussian_filter1d(delta,20.,axis=0,mode='constant')*mask[full,None]
        qcoarse[full]=base['qref'][full]+delta
        for (a,b),rot in zip([(3,7),(47,51)],rotations):
            curve=CubicSpline(ticks,rot,bc_type=((1,np.zeros(3)),(1,np.zeros(3))))(full)
            curve=gaussian_filter1d(curve,20.,axis=0,mode='constant')*mask[full,None]
            qcoarse[full,a:b]=(R.from_rotvec(curve)*R.from_quat(base['qref'][full,a:b],scalar_first=True)).as_quat(scalar_first=True)
        print('PRIMARY_COARSE',seed,phase,'max position',max([v['position_error_m'] for v in coarse_metrics if v['phase']==phase] or [0.]),flush=True)
    qcoarse[mask==0]=base['qref'][mask==0]
    # Reject unavailable geometry before the dense pass. Never count it as
    # executed influence or silently substitute the expert result.
    coarse_ok=all(v['position_error_m']<.001 and v['angle_error_rad']<.01 and not v['forbidden_penetrations'] for v in coarse_metrics)
    if not coarse_ok:
        out.with_suffix('.json').write_text(json.dumps(dict(seed=seed,stage='coarse',pass_cheap_gate=False,source=meta,coarse_metrics=coarse_metrics),indent=2))
        np.savez_compressed(out.with_name(out.stem+'_rejected.npz'),qcoarse=qcoarse,qtarget=target)
        raise RuntimeError('Coarse source-primary contact projection failed; evidence retained')
    qref=base['qref'].copy();dense_metrics=[]
    for phase,(s,e,_,_) in phases.items():
        previous=None;last=None;older=None;older_tick=None
        for tick in range(s,e+1):
            if mask[tick]<=1e-14:q=base['qref'][tick].copy()
            else:
                guess=transport(previous,base['qref'][last],base['qref'][tick]) if previous is not None else qcoarse[tick]
                continuity=None
                if continuous and previous is not None:
                    # Regularize only the projection correction. The smoothed
                    # source trajectory itself must keep advancing on time.
                    last_correction=transport(previous,qcoarse[last],qcoarse[tick])
                    oldguess=transport(older,qcoarse[older_tick],qcoarse[tick]) if older is not None else last_correction
                    continuity=(last_correction,oldguess)
                answer=project.project(tick,qcoarse[tick],qcoarse[tick],previous=guess,max_iterations=20 if continuous or fixed_motion or fixed_angles else 10,com_tube=.001 if phase=='carry' else .02,continuity=continuity,fixed_motion=fixed_motion,fixed_angles=fixed_angles)
                q=answer['q'];dense_metrics.append(dict(tick=int(tick),phase=phase,**{k:v for k,v in answer.items() if k not in ['q','parameters']}))
            qref[tick]=q;older,older_tick=previous,last;previous=q;last=tick
        print('PRIMARY_DENSE',seed,phase,'max position',max([v['position_error_m'] for v in dense_metrics if v['phase']==phase] or [0.]),flush=True)
    vref=base['vref'].copy();delta_v=np.zeros_like(vref);dq=np.zeros(m.nv);db=dq.copy()
    for tick in range(1,len(qref)):
        if not(np.array_equal(qref[tick],base['qref'][tick]) and np.array_equal(qref[tick-1],base['qref'][tick-1])):
            mujoco.mj_differentiatePos(m,dq,DT,qref[tick-1],qref[tick]);mujoco.mj_differentiatePos(m,db,DT,base['qref'][tick-1],base['qref'][tick-1+1]);delta_v[tick]=dq-db;vref[tick]+=delta_v[tick]
    # Forward velocity difference corresponds to the actuator command applied
    # at pre-step state t. The same baseline discretization is subtracted.
    a0=np.r_[np.diff(base['vref'],axis=0)/DT,np.zeros((1,m.nv))];a1=a0+np.r_[np.diff(delta_v,axis=0)/DT,np.zeros((1,m.nv))]
    native=m.actuator_trnid[:29,0];nq=m.jnt_qposadr[native];nv=m.jnt_dofadr[native];kp=m.actuator_gainprm[:29,0];kd=-m.actuator_biasprm[:29,2]
    uref=base['uref'].copy();increments=np.zeros_like(uref);tau_delta=np.zeros((len(qref),29));checks=[];required=[]
    old,new=mujoco.MjData(m),mujoco.MjData(m);loads=np.load(G7/f'execution_{seed}/rollout_locomotion_trace.npz')['trace'][:,14:16]
    active=np.flatnonzero(np.any(abs(qref-base['qref'])>1e-10,axis=1)|np.any(abs(delta_v)>1e-9,axis=1)|np.any(abs(a1-a0)>1e-8,axis=1))
    for tick in active:
        actual=kp*(base['uref'][tick,:29]-base['qref'][tick,nq])-kd*base['vref'][tick,nv]
        actual=np.clip(actual,m.actuator_forcerange[:29,0],m.actuator_forcerange[:29,1])
        records=[]
        for d,q,v,a,effort in [(old,base['qref'][tick],base['vref'][tick],a0[tick],actual),(new,qref[tick],vref[tick],a1[tick],None)]:
            d.qpos[:]=q;d.qvel[:]=v;d.qacc[:]=a;mujoco.mj_fwdPosition(m,d);mujoco.mj_fwdVelocity(m,d);records.append(loaded(m,d,loads[min(tick,len(loads)-1)],effort))
        d0,d1=records
        if 'effort' not in d0 or 'effort' not in d1:
            checks.append(dict(tick=int(tick),baseline_feasible=d0['feasible'],candidate_feasible=d1['feasible'],reason='Changed dynamics without an authoritative loaded foot'));continue
        tau_delta[tick]=d1['effort']-d0['effort']
        increments[tick,:29]=qref[tick,nq]-base['qref'][tick,nq]+(kd*delta_v[tick,nv]+tau_delta[tick])/kp
        predicted=actual+tau_delta[tick];required.append(float((abs(predicted)-np.max(abs(m.actuator_forcerange[:29]),axis=1)).max()))
        if tick%10==0 or (d0['feasible'] and not d1['feasible']):checks.append(dict(tick=int(tick),baseline_feasible=d0['feasible'],candidate_feasible=d1['feasible'],wrench_error=float(abs(d1['wrench_error']).max()),maximum_loaded_effort_excess_Nm=d1['maximum_effort_excess_Nm']))
        if tick%250==0:print('PRIMARY_DYNAMICS',seed,int(tick),flush=True)
    uref+=increments
    limits=m.actuator_ctrlrange.astype(np.float32).astype(float);excess=np.maximum(uref-limits[:,1],limits[:,0]-uref)
    bad=[v for v in dense_metrics if v['position_error_m']>=.0003 or v['angle_error_rad']>=.003 or v['forbidden_penetrations']]
    introduced=[v for v in checks if v['baseline_feasible'] and not v['candidate_feasible']]
    report=dict(seed=seed,source=meta,compiler_sha256=sha(Path(__file__)),helper_sha256={p:sha(ROOT/'scripts/compact_diffusion_pose_v1'/p) for p in ['primary_targets.py','primary_kinematics.py','primary_dynamics.py']},base_sha256=sha(G7/f'feasible_{seed}.npz'),
        maximum_contact_position_error_m=max(v['position_error_m'] for v in dense_metrics),maximum_contact_angle_error_rad=max(v['angle_error_rad'] for v in dense_metrics),
        geometry_failure_count=len(bad),introduced_dynamic_infeasibilities=introduced,loaded_dynamics_checks=checks,
        maximum_predicted_actuator_force_excess_Nm=max(required or [0.]),maximum_control_range_excess=float(np.maximum(excess,0).max()),
        maximum_native_velocity_radps=float(abs(vref[:,nv]).max()),maximum_added_native_velocity_radps=float(abs(delta_v[:,nv]).max()),maximum_control_increment_rad=float(abs(increments).max()),
        preserved_gripper_controls=bool(np.array_equal(uref[:,29:],base['uref'][:,29:])),physical_model_changed=False,simulator_stepped=False,
        contract='Full frozen-proposal interior preferences, fixed expert sole poses and gripper mechanics, loaded contact inverse-dynamics delta preserving all baseline 200 Hz feedforward details. Necessary screens only; actual ORCS execution and unchanged gates remain mandatory.')
    report['pass_cheap_gate']=bool(not bad and not introduced and report['maximum_predicted_actuator_force_excess_Nm']<1. and report['maximum_control_range_excess']<.01 and report['maximum_added_native_velocity_radps']<3. and report['preserved_gripper_controls'])
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out.with_name(out.stem+'_diagnostics.npz'),qtarget=target,qcoarse=qcoarse,qref=qref,vref=vref,control_increment=increments,loaded_effort_increment=tau_delta,gate=mask,warm_frame=warm)
    if not report['pass_cheap_gate']:raise RuntimeError({k:v for k,v in report.items() if k not in ['loaded_dynamics_checks','introduced_dynamic_infeasibilities','source']})
    np.savez_compressed(out,qref=qref,vref=vref,uref=uref,goal_position=base['goal_position'],goal_quaternion=base['goal_quaternion']);print('PRIMARY_REFERENCE_ACCEPTED',seed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',action='append',choices=['pickup','carry','placement']);p.add_argument('--reuse-coarse',type=Path);p.add_argument('--continuous',action='store_true');p.add_argument('--fixed-motion',action='store_true')
    p.add_argument('--fixed-angles',action='store_true')
    a=p.parse_args();compile_reference(a.seed,a.out,a.phase,a.reuse_coarse,a.continuous,a.fixed_motion,a.fixed_angles)
