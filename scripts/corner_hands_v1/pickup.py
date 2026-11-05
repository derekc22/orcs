"""Physical reaching, six rotary fingers, and balance-controlled standing lift."""
import argparse
import numpy as np
import mujoco
from mechanism import OUT,indices
from physical import Arms,contacts,smooth,squeeze_load,PitchServo,thumb_control,thumb_normal_loads
from compact_gripper_balance import Balance

def run(tag,geometry='reach4',torque=.016,squeeze=.0,forward=.0,force_target=0.,hold_fingers=False,forward_return=False,second_rise=1.5,body_lift_delay=0.,pitch_servo=False,second_body_delay=0.):
    m=mujoco.MjModel.from_binary_path(str(OUT/'palms_model.mjb'))
    a=np.load(OUT/(geometry+'.npz'));poses=a['qpos'].copy();poses[:,indices(m)[1][29:31]] = -.025
    oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id]
    poses[:,oq:oq+7]=poses[1,oq:oq+7].copy()
    d=mujoco.MjData(m);d.qpos[:]=poses[0];d.qvel[:]=0;mujoco.mj_forward(m,d)
    q0=d.qpos.copy();v0=d.qvel.copy();ob=m.body('object/object').id;obj0=d.xipos[ob].copy()
    aj,aq,av,arm,pads=indices(m);balance=Balance(m,d);arms=Arms(m,q0)
    attitude=PitchServo(m,d,a['rot'][0,:,2])
    anchor=a['anchors'] if 'anchors' in a else [a['target']+[0,0,h] for h in (a['heights'] if 'heights' in a else [.08,0.,.18,.38])]
    cs=[];qs=[];vs=[];cts=[];diag=[];sums=np.zeros(6);closure=np.zeros(2);filtered_force=np.zeros(2);finger_hold=None
    for step in range(int((6.1+second_rise+body_lift_delay+second_body_delay)/.005)):
        sec=step*.005
        if sec<1.2:i=0;alpha=smooth(sec/1.2)
        elif sec<3.0:i=1;alpha=0.
        elif sec<4.5+body_lift_delay:i=1;alpha=smooth((sec-3.0-body_lift_delay)/1.5)
        elif sec<4.5+body_lift_delay+second_body_delay+second_rise:i=2;alpha=smooth((sec-4.5-body_lift_delay-second_body_delay)/second_rise)
        else:i=2;alpha=1.
        qref=poses[i]*(1-alpha)+poses[i+1]*alpha;target=anchor[i]*(1-alpha)+anchor[i+1]*alpha
        if i==1 and sec>=3:
            armalpha=smooth((sec-3.)/1.5);target=anchor[1]*(1-armalpha)+anchor[2]*armalpha
        if i==2:
            armalpha=smooth((sec-4.5-body_lift_delay)/second_rise);target=anchor[2]*(1-armalpha)+anchor[3]*armalpha
        if sec>2.0:closure=np.clip(closure+.8*(d.qpos[aq[29:31]]-.012)*.005,-.02,.035)
        for side in range(2):target[side*3:side*3+3]-=(squeeze*smooth((sec-2.1)/.7)+closure[side])*a['rot'][side,:,2]
        fwd=np.cross(a['rot'][0,:,2],np.array([0.,0.,1.]));fwd/=np.linalg.norm(fwd)
        return_start=4.5+body_lift_delay+(second_body_delay+.5 if second_body_delay else 0)
        return_duration=second_rise-(.5 if second_body_delay else 0)
        target+=forward*smooth((sec-3.)/1.)*(1-smooth((sec-return_start)/return_duration) if forward_return else 1)*fwd
        if pitch_servo and sec>=3:target=attitude.correct(d,target,smooth((sec-3)/.4))
        if step%4==0:arms.solve(d,target,qref)
        qbalance=qref.copy();qbalance[arms.armq]=arms.seed;vbalance=np.zeros(m.nv);vbalance[av[arm]]=arms.vel
        tau,wrench=balance.torques(d,qbalance,vbalance)
        u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0]
        u=arms.apply(d,u)
        if sec<1.3:u[29:31]=-.025
        else:
            u[29:31]=arms.grip(d,.2*smooth((sec-1.3)/.5)+(torque-.2)*smooth((sec-2.2)/.6))
            if hold_fingers and sec>=2.95:
                if finger_hold is None:finger_hold=u[29:31].copy()
                u[29:31]=finger_hold
        u[33:35]=1.57*smooth((sec-3.85)/.65)
        u[35:37]=thumb_control(m,d,max_angle=1.10*smooth((sec-1.5)/.7),torque=.010+.090*smooth((sec-2.2)/.6))
        u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
        touch,force,imp,vertical,selfimp,invalid=contacts(m,d);sums+=np.r_[imp,selfimp,invalid]
        thumb_load=thumb_normal_loads(m,d,fwd)
        net_load=squeeze_load(m,d,a['rot'][:,:,2])
        filtered_force+=.15*(net_load-filtered_force)
        cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);diag.append(np.r_[sec,d.xipos[ob,2]-obj0[2],force,arms.cost,sums,closure,net_load])
        if step%100==99:print(tag,round(sec,3),'root',round(d.qpos[2],3),'lift',round(d.xipos[ob,2]-obj0[2],3),'touch',touch,'N',np.round(force,3),'cost',round(arms.cost,3),'bad',sums[3:], 'flex',np.round(d.qpos[aq[29:31]],2),'thumbN',np.round(thumb_load,2),'thumbq',np.round(d.qpos[aq[35:37]],2),flush=True)
        if 2.94<sec<2.951 and thumb_load.min()<.01:
            print('FAILED_FORE_AFT_ACQUISITION',thumb_load,flush=True);break
        if d.qpos[2]<.32 or sums[3]>.00001 or sums[4]>.001 or sums[5]>.001:print('EARLY_REJECT',flush=True);break
    np.savez_compressed(OUT/(tag+'.npz'),initial_qpos=q0,initial_qvel=v0,ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,diagnostics=diag,phase_opening_frame=np.array(-1),phase_withdrawal_end_frame=np.array(-1),finger_hold_control=np.array([]) if finger_hold is None else finger_hold,pitch_servo_enabled=np.array(pitch_servo))
    mujoco.mj_saveModel(m,str(OUT/(tag+'_model.mjb')),None)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='pickup1');p.add_argument('--geometry',default='reach4');p.add_argument('--torque',type=float,default=5.);p.add_argument('--squeeze',type=float,default=0.);p.add_argument('--forward',type=float,default=0.);p.add_argument('--force-target',type=float,default=0.);p.add_argument('--hold-fingers',action='store_true');p.add_argument('--forward-return',action='store_true');p.add_argument('--second-rise',type=float,default=1.5);p.add_argument('--body-lift-delay',type=float,default=0.);p.add_argument('--pitch-servo',action='store_true');p.add_argument('--second-body-delay',type=float,default=0.);a=p.parse_args();run(a.tag,a.geometry,a.torque,a.squeeze,a.forward,a.force_target,a.hold_fingers,a.forward_return,a.second_rise,a.body_lift_delay,a.pitch_servo,a.second_body_delay)
