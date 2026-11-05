"""Actual-state SONIC braking into the verified parallel-foot carry posture."""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
from mechanism import indices
from physical import contacts,Arms,thumb_control,smooth

def run(args):
    source=ROOT/args.source;out=ROOT/'logs/compact_integration_v2'/args.tag;out.mkdir(parents=True,exist_ok=False)
    a=np.load(source);m=mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
    for u in a['ctrl_physics']:d.ctrl[:]=u;mujoco.mj_step(m,d)
    loaded=d.ctrl.copy();rest=d.qpos.copy();bias0=d.qfrc_bias.copy();aj,aq,av,arm,pads=indices(m);arms=Arms(m,rest);waist=[i for i,j in enumerate(aj[:29]) if m.joint(int(j)).name=='waist_yaw_joint']
    import frozen,build_engineered_ground_large_model as builder,active_largebox_sonic as sonic,active_largebox_motion as motion
    from run_engineered_locomomanipulation import _quat_delta_omega
    builder.large_ground_gripper_spec=frozen.native_spec;sonic.large_ground_gripper_spec=frozen.native_spec;sonic.original_largebox_spec=frozen.object_spec
    native=frozen.native_spec().compile();joints=np.array(sorted([j for j in range(native.njnt) if native.jnt_type[j]==3],key=lambda j:native.jnt_qposadr[j]));jq=native.jnt_qposadr[joints]
    start=native.qpos0.copy();start[:7]=rest[:7]
    for j in joints:start[native.jnt_qposadr[j]]=rest[m.jnt_qposadr[m.joint(native.joint(int(j)).name).id]]
    stand=np.load(ROOT/'logs/compact_integration_v2/corner_parallel/rollout.npz')['qpos'][-1];goal=start.copy()
    for j in joints:
        name=native.joint(int(j)).name
        if any(k in name for k in ['hip_','knee_','ankle_']):goal[native.jnt_qposadr[j]]=stand[m.jnt_qposadr[m.joint(name).id]]
    yaw=R.from_quat(np.roll(rest[3:7],-1)).as_euler('xyz')[2];goal[3:7]=np.roll(R.from_euler('z',yaw).as_quat(),1);goal[2]=stand[2]
    velocity=np.clip(d.qvel[:2],-.3,.3);goal[:2]+=velocity*.4
    delta=np.zeros(native.nv);mujoco.mj_differentiatePos(native,delta,1.,start,goal)
    poses=[]
    for frame in range(round(args.seconds/.02)+40):
        sec=frame*.02;alpha=smooth(sec/.8);q=start.copy();mujoco.mj_integratePos(native,q,delta,alpha);t=min(1.,sec/.8);q[:2]=start[:2]+velocity*.8*(t-.5*t*t);poses.append(q)
    poses=np.array(poses);fk=mujoco.MjData(native);pos=[];quat=[]
    for q in poses:fk.qpos[:]=q;mujoco.mj_kinematics(native,fk);pos.append(fk.xpos[1:].copy());quat.append(fk.xquat[1:].copy())
    pos=np.array(pos);quat=np.array(quat);path=out/'braking_reference.npz'
    np.savez_compressed(path,joint_pos=poses[:,jq].astype('float32'),joint_vel=np.gradient(poses[:,jq],.02,axis=0).astype('float32'),body_pos_w=pos.astype('float32'),body_quat_w=quat.astype('float32'),body_lin_vel_w=np.gradient(pos,.02,axis=0).astype('float32'),body_ang_vel_w=_quat_delta_omega(quat,.02).astype('float32'),fps=np.array(50.))
    motion.write=lambda arm_reference=None:path;ctrl=sonic.Controller(m)
    cs=list(a['ctrl_physics']);qs=list(a['qpos']);vs=list(a['qvel']);cts=list(a['contacts']);sums=np.zeros(6);rows=[];reason='completed'
    for step in range(round(args.seconds/.005)):
        if step%4==0:lower=ctrl.control(d)
        alpha=smooth(step*.005/.2);u=np.r_[loaded[:29]*(1-alpha)+lower*alpha,np.zeros(m.nu-29)]
        u[arm]=loaded[arm]+(d.qfrc_bias[av[arm]]-bias0[av[arm]])/m.actuator_gainprm[arm,0]
        u[waist]=loaded[waist]+(d.qfrc_bias[av[waist]]-bias0[av[waist]])/m.actuator_gainprm[waist,0]
        u[29:31]=arms.grip(d,2.);u[33:35]=loaded[33:35];u[35:37]=thumb_control(m,d);u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
        touch,N,imp,v,si,invalid=contacts(m,d);sums+=np.r_[imp,si,invalid];cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);rows.append(np.r_[step*.005,d.qpos[:3],d.qvel[:6],N,sums])
        if step%100==99:print(args.tag,round(step*.005,3),'root',np.round(d.qpos[:3],3),'rootv',np.round(d.qvel[:6],3),'N',np.round(N,2),'bad',sums[3:],flush=True)
        if sums[3]>1e-5:reason='forbidden_object'
        elif sums[4]>.001:reason='self_contact'
        elif sums[5]>.001:reason='invalid_terrain'
        elif touch[0]:reason='object_ground'
        elif d.qpos[2]<.4:reason='fall'
        if reason!='completed':print('EARLY_REJECT',reason,flush=True);break
    data=dict(initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,braking_diagnostics=rows,phase_opening_frame=-1,phase_withdrawal_end_frame=-1)
    for key in ['walking_start_physics_step','walking_end_physics_step']:data[key]=a[key]
    np.savez_compressed(out/'rollout.npz',**data);mujoco.mj_saveModel(m,str(out/'rollout_model.mjb'),None)
    result=dict(arguments=vars(args),reason=reason,prefix=len(a['ctrl_physics']),continuation_seconds=len(rows)*.005,impulses=sums.tolist());(out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--tag',required=True);p.add_argument('--seconds',type=float,default=4.);run(p.parse_args())
