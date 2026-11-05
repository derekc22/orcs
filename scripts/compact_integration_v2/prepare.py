"""Load-preserving physical foot alignment before the corner-hand walking handoff."""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/compact_integration_v1'),str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
from gait import LegIK
from mechanism import indices
from physical import contacts,smooth,Arms,thumb_control
from compact_gripper_balance import Balance

def run(args):
    source=ROOT/args.source;out=ROOT/'logs/compact_integration_v2'/args.tag;out.mkdir(parents=True,exist_ok=False)
    a=np.load(source);m=mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')))
    d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
    for u in a['ctrl_physics']:d.ctrl[:]=u;mujoco.mj_step(m,d)
    aj,aq,av,arm,pads=indices(m);rest=d.qpos.copy();loaded=d.ctrl.copy();balance=Balance(m,d);ik=LegIK(m,d);arms=Arms(m,rest)
    if args.support_controller=='bounded':
        from bounded_balance import BoundedBalance
        balance=BoundedBalance(m,d)
    waist=np.array([i for i,j in enumerate(aj[:29]) if 'waist_' in m.joint(int(j)).name]);C=d.subtree_com[1,:2].copy();P=balance.pos.copy();R0=balance.rot.copy()
    C0=C.copy();C1=P[:,:2].mean(0) if args.center_com else C.copy()
    yaw=R.from_quat(np.roll(rest[3:7],-1)).as_euler('xyz')[2]
    R1=np.array([R.from_euler('z',yaw+sign*np.radians(args.toe_out)).as_matrix() for sign in [1,-1]])
    drot=np.array([R.from_matrix(R1[i]@R0[i].T).as_rotvec() for i in range(2)])
    end=ik.solve(P,R1,C1);endpoint=dict(max_foot_error_m=ik.error,ankle_roll_rad=[float(end[m.jnt_qposadr[m.joint(s+'_ankle_roll_joint').id]]) for s in ['left','right']])
    (out/'endpoint.json').write_text(json.dumps(endpoint,indent=2));print('ENDPOINT',endpoint,flush=True)
    if ik.error>.005:print('GEOMETRIC_GATE_FAILED',flush=True);return
    ik.x=np.r_[rest[:3],rest[ik.qids]]
    cs=list(a['ctrl_physics']);qs=list(a['qpos']);vs=list(a['qvel']);cts=list(a['contacts']);rows=[];sums=np.zeros(6);prev=rest.copy();vel=np.zeros(m.nv);reason='completed';fc=d.qfrc_constraint.copy()
    for step in range(round((args.duration+1.)/.005)):
        sec=step*.005;alpha=smooth(sec/args.duration)
        balance.rot=np.array([R.from_rotvec(alpha*drot[i]).as_matrix()@R0[i] for i in range(2)])
        C=C0*(1-alpha)+C1*alpha
        if step%4==0:
            qref=ik.solve(P,balance.rot,C);mujoco.mj_differentiatePos(m,vel,.02,prev,qref);prev=qref.copy()
        if args.support_controller in ['balance','bounded']:
            tau,_=balance.torques(d,qref,vel)
            u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0]
        else:
            fc=.8*fc+.2*d.qfrc_constraint
            u=qref[aq]+(d.qfrc_bias[av]-fc[av])/m.actuator_gainprm[:,0]+(-m.actuator_biasprm[:,2]/m.actuator_gainprm[:,0])*vel[av]
        u[arm]=loaded[arm];u[waist]=loaded[waist];u[29:31]=arms.grip(d,2.);u[33:35]=loaded[33:35];u[35:37]=thumb_control(m,d)
        u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
        touch,N,imp,v,si,invalid=contacts(m,d);sums+=np.r_[imp,si,invalid]
        cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);rows.append(np.r_[sec,d.qpos[:3],N,sums,ik.error])
        if step%100==99:print(args.tag,round(sec,3),'root',np.round(d.qpos[:3],3),'force',np.round(N,2),'bad',sums[3:],flush=True)
        if sums[3]>1e-5:reason='forbidden_object'
        elif sums[4]>.001:reason='self_contact'
        elif sums[5]>.001:reason='invalid_terrain'
        elif touch[0]:reason='object_ground'
        elif d.qpos[2]<.45:reason='fall'
        if reason!='completed':print('EARLY_REJECT',reason,flush=True);break
    data=dict(initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,preparation_diagnostics=rows,phase_opening_frame=-1,phase_withdrawal_end_frame=-1)
    for key in ['walking_start_physics_step','walking_end_physics_step']:
        if key in a:data[key]=a[key]
    np.savez_compressed(out/'rollout.npz',**data);mujoco.mj_saveModel(m,str(out/'rollout_model.mjb'),None)
    result=dict(arguments=vars(args),reason=reason,prefix=len(a['ctrl_physics']),continuation_seconds=len(rows)*.005,impulses=sums.tolist(),endpoint=endpoint)
    (out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='logs/compact_integration_v1/corner_loaded_hold/rollout.npz');p.add_argument('--tag',required=True);p.add_argument('--toe-out',type=float,default=0.);p.add_argument('--duration',type=float,default=3.);p.add_argument('--support-controller',choices=['balance','contact','bounded'],default='balance');p.add_argument('--center-com',action='store_true')
    run(p.parse_args())
