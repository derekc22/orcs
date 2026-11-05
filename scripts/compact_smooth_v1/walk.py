"""Corner-hand walking with stance and upper-body reference compatibility.

The native SONIC policy remains lower-body feedback authority. Its reference is
recomputed by native FK with the verified loaded waist and arm posture. All real
motion, including object motion, is exclusively finite-actuator MuJoCo stepping.
"""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts/compact_integration_v1'),str(ROOT/'scripts')]
sys.path.append(str(ROOT/'scripts/compact_integration_v2'))
from mechanism import indices
from physical import Arms,contacts,smooth,thumb_control,ObjectServo
from loaded_arms import LoadedArms


def controller(m,d,out,args):
    import frozen
    import build_engineered_ground_large_model as builder
    import sonic_cpu as sonic
    import active_largebox_motion as oldmotion
    if args.repetitions:
        import long_motion as motion
    else:
        import sustained_carry_motion as motion
    from run_engineered_locomomanipulation import _quat_delta_omega
    builder.large_ground_gripper_spec=frozen.native_spec
    sonic.large_ground_gripper_spec=frozen.native_spec
    sonic.original_largebox_spec=frozen.object_spec
    motion.OUT=out
    kwargs=dict(speed=args.speed)
    if args.repetitions:kwargs['cycles']=args.repetitions
    path,meta=motion.build_reference(m,d.qpos.copy(),**kwargs)
    native=frozen.native_spec().compile();poses=np.load(out/'walking_reference_qpos.npy')
    joints=np.array(sorted([j for j in range(native.njnt) if native.jnt_type[j]==3],key=lambda j:native.jnt_qposadr[j]));jq=native.jnt_qposadr[joints]
    for j in joints:
        name=native.joint(int(j)).name
        freeze=args.freeze_waist and ('waist_' in name if args.waist_mode=='all' else name=='waist_yaw_joint')
        if freeze:poses[:,native.jnt_qposadr[j]]=d.qpos[m.jnt_qposadr[m.joint(name).id]]
    ramp=np.array([smooth((f-10)/30) for f in range(len(poses))])
    for side,sign in [('left',1),('right',-1)]:
        for suffix,direction in [('hip_roll',sign),('ankle_roll',-sign)]:
            j=native.joint(side+'_'+suffix+'_joint').id
            poses[:,native.jnt_qposadr[j]]=np.clip(poses[:,native.jnt_qposadr[j]]+args.abduction*direction*ramp,native.jnt_range[j,0]+.005,native.jnt_range[j,1]-.005)
    fk=mujoco.MjData(native);pos=[];quat=[]
    for q in poses:
        fk.qpos[:]=q;mujoco.mj_kinematics(native,fk);pos.append(fk.xpos[1:].copy());quat.append(fk.xquat[1:].copy())
    pos=np.array(pos);quat=np.array(quat)
    np.savez_compressed(path,joint_pos=poses[:,jq].astype('float32'),joint_vel=np.gradient(poses[:,jq],.02,axis=0).astype('float32'),body_pos_w=pos.astype('float32'),body_quat_w=quat.astype('float32'),body_lin_vel_w=np.gradient(pos,.02,axis=0).astype('float32'),body_ang_vel_w=_quat_delta_omega(quat,.02).astype('float32'),fps=np.array(50.))
    np.save(out/'walking_reference_qpos.npy',poses)
    meta['waist_frozen_in_reference']=args.freeze_waist
    meta['waist_mode']=args.waist_mode
    meta['hip_abduction_offset_rad']=args.abduction
    meta['added_native_cycles']=args.repetitions
    (out/'walking_reference_manifest.json').write_text(json.dumps(meta,indent=2))
    oldmotion.write=lambda arm_reference=None:path
    return sonic.Controller(m),meta


def run(args):
    source=ROOT/args.source;out=ROOT/'logs/compact_smooth_v1'/args.tag;out.mkdir(parents=True,exist_ok=False)
    a=np.load(source);m=mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')))
    d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
    for u in a['ctrl_physics']:d.ctrl[:]=u;mujoco.mj_step(m,d)
    aj,aq,av,arm,pads=indices(m);loaded=d.ctrl.copy();rest=d.qpos.copy();bias0=d.qfrc_bias.copy();origin=rest[:3].copy();waist=np.array([i for i,j in enumerate(aj[:29]) if ('waist_' in m.joint(int(j)).name if args.waist_mode=='all' else m.joint(int(j)).name=='waist_yaw_joint')])
    arms=LoadedArms(m,d,max_speed=5.);ob=m.body('object/object').id;servo=ObjectServo(m,d,pads,aq)
    fk=mujoco.MjData(m);fk.qpos[:]=rest;fk.qpos[aq[29:31]]=0.;mujoco.mj_kinematics(m,fk);anchor=fk.geom_xpos[pads].copy();normal=fk.xmat[arms.hands].reshape(2,3,3)[:,:,2].copy()
    yaw0=R.from_quat(np.roll(rest[3:7],-1)).as_euler('xyz')[2]
    ctrl,meta=controller(m,d,out,args)
    cs=list(a['ctrl_physics']);qs=list(a['qpos']);vs=list(a['qvel']);cts=list(a['contacts']);rows=[];sums=np.zeros(6);reason='completed';M=np.zeros((m.nv,m.nv));filtered_lower=loaded[:29].copy();last_acc=d.qacc.copy();constraint=d.qfrc_constraint.copy();target=anchor.copy()
    seconds=args.seconds if args.seconds is not None else (meta['braking_end_frame']+35)*.02
    for step in range(round(seconds/.005)):
        yaw=R.from_quat(np.roll(d.qpos[3:7],-1)).as_euler('xyz')[2]-yaw0;rz=R.from_euler('z',yaw)
        if step%4==0:
            lower=ctrl.control(d)
            if args.hands=='inverse':
                target=rz.apply(anchor-origin)+d.qpos[:3]
                arms.solve(d,target,rest)
        filtered_lower+=(1.-np.exp(-.005/args.control_tau))*(lower-filtered_lower) if args.control_tau>0 else lower-filtered_lower
        u=np.r_[filtered_lower,np.zeros(m.nu-29)]
        if args.blend>0:
            alpha=smooth(step*.005/args.blend);u[:29]=loaded[:29]*(1-alpha)+u[:29]*alpha
        if args.hands=='loaded':
            u[arm]=loaded[arm]+(d.qfrc_bias[av[arm]]-bias0[av[arm]])/m.actuator_gainprm[arm,0]
        else:
            mujoco.mj_fullM(m,M,d.qM)
            last_acc=.8*last_acc+.2*d.qacc
            acc=last_acc.copy();acc[av[arm]]=np.clip(180*(arms.seed-d.qpos[aq[arm]])+28*(arms.vel-d.qvel[av[arm]]),-100,100)
            constraint=.5*constraint+.5*d.qfrc_constraint
            tau=(M@acc+d.qfrc_bias-constraint)[av[arm]]
            u[arm]=d.qpos[aq[arm]]+(tau-m.actuator_biasprm[arm,2]*d.qvel[av[arm]])/m.actuator_gainprm[arm,0]
        if args.freeze_waist:u[waist]=loaded[waist]+(d.qfrc_bias[av[waist]]-bias0[av[waist]])/m.actuator_gainprm[waist,0]
        u[29:31]=arms.grip(d,2.);u[33:35]=loaded[33:35];u[35:37]=thumb_control(m,d,torque=args.thumb_torque,max_angle=1.1+(args.thumb_limit-1.1)*smooth(step*.005/.8))
        u[arm]-=smooth(step*.005/.4)*args.arm_damping*(-m.actuator_biasprm[arm,2])*d.qvel[av[arm]]/m.actuator_gainprm[arm,0]
        u[29:31]-=args.press_damping*d.qvel[av[29:31]]/800.
        u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
        touch,N,imp,v,si,invalid=contacts(m,d);sums+=np.r_[imp,si,invalid]
        cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch)
        desired=d.qpos[:3]+rz.apply(servo.C0-servo.root0);slip=R.from_matrix(d.xmat[ob].reshape(3,3))*servo.R0.inv()
        rows.append(np.r_[step*.005,d.qpos[:3],d.xipos[ob],N,sums,arms.cost,desired-d.xipos[ob],slip.magnitude(),np.abs(d.actuator_force[arm])])
        if step%100==99:print(args.tag,round(step*.005,3),'distance',round(np.linalg.norm(d.qpos[:2]-origin[:2]),3),'rootZ',round(d.qpos[2],3),'N',np.round(N,2),'position_error',np.round(desired-d.xipos[ob],3),'bad',sums[3:],flush=True)
        if sums[3]>1e-5:reason='forbidden_object'
        elif sums[4]>.001:reason='self_contact'
        elif sums[5]>.001:reason='invalid_terrain'
        elif touch[0]:reason='object_ground'
        elif d.qpos[2]<.4:reason='fall'
        if reason!='completed':print('EARLY_REJECT',reason,flush=True);break
    prefix=len(a['ctrl_physics'])
    np.savez_compressed(out/'rollout.npz',initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,walking_diagnostics=rows,walking_start_physics_step=prefix+meta['walk_start_frame']*4,walking_end_physics_step=prefix+meta['walk_end_frame']*4,phase_opening_frame=-1,phase_withdrawal_end_frame=-1,thumb_max_angle=args.thumb_limit)
    mujoco.mj_saveModel(m,str(out/'rollout_model.mjb'),None)
    result=dict(arguments=vars(args),reason=reason,prefix=prefix,continuation_seconds=len(rows)*.005,impulses=sums.tolist(),reference=meta,root_displacement_continuation_m=float(np.linalg.norm(d.qpos[:2]-origin[:2])))
    (out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--tag',required=True);p.add_argument('--freeze-waist',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--waist-mode',choices=['all','yaw'],default='all');p.add_argument('--hands',choices=['loaded','inverse'],default='loaded');p.add_argument('--seconds',type=float);p.add_argument('--blend',type=float,default=.2);p.add_argument('--speed',type=float,default=.8);p.add_argument('--abduction',type=float,default=0.);p.add_argument('--repetitions',type=int,default=0)
    p.add_argument('--thumb-limit',type=float,default=1.1)
    p.add_argument('--thumb-torque',type=float,default=.1)
    p.add_argument('--control-tau',type=float,default=0.);p.add_argument('--arm-damping',type=float,default=0.);p.add_argument('--press-damping',type=float,default=0.)
    run(p.parse_args())
