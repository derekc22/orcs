"""Slow alternating steps using the existing finite-actuator support controller.

Keeps the verified loaded arm posture, plans native leg IK, and uses real foot
contact for support. No real floating-base/object states or forces are edited.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/corner_hands_v1'),
                str(ROOT/'scripts/articulated_gripper_v1'), str(ROOT/'scripts')]
from mechanism import indices
from physical import contacts, smooth, thumb_control, Arms
from support_balance import SupportBalance


class LegIK:
    def __init__(self, m, d):
        self.m = m
        self.fk = mujoco.MjData(m)
        self.rest = d.qpos.copy()
        self.feet = [m.body(s+'_ankle_roll_link').id for s in ['left', 'right']]
        self.aj, self.aq, self.av, _, _ = indices(m)
        self.acts = np.array([i for i,j in enumerate(self.aj[:29])
                              if any(s in m.joint(int(j)).name for s in ['hip_', 'knee_', 'ankle_'])])
        self.qids = self.aq[self.acts]
        self.x = np.r_[d.qpos[:3], d.qpos[self.qids]]
        self.lo = np.r_[[-100, -100, .52], m.jnt_range[self.aj[self.acts], 0]+.005]
        self.hi = np.r_[[100, 100, .81], m.jnt_range[self.aj[self.acts], 1]-.005]
        self.error = 0.

    def solve(self, pos, rot, com):
        m = self.m
        fk = self.fk
        def residual(x):
            fk.qpos[:] = self.rest
            fk.qpos[:3] = x[:3]
            fk.qpos[self.qids] = x[3:]
            mujoco.mj_kinematics(m, fk)
            mujoco.mj_comPos(m, fk)
            er = np.concatenate([Rotation.from_matrix(rot[k]@fk.xmat[b].reshape(3,3).T).as_rotvec()
                                 for k,b in enumerate(self.feet)])
            return np.r_[150*(fk.xpos[self.feet]-pos).ravel(), 35*er,
                         120*(fk.subtree_com[1,:2]-com),
                         25*(x[2]-self.rest[2]), .02*(x[3:]-self.rest[self.qids])]
        fit = least_squares(residual, np.clip(self.x, self.lo+1e-8, self.hi-1e-8),
                            bounds=(self.lo,self.hi), max_nfev=12, ftol=1e-6)
        self.x = fit.x
        residual(self.x)
        self.error = float(np.max(np.linalg.norm(fk.xpos[self.feet]-pos,axis=1)))
        return fk.qpos.copy()


def run(args):
    source = ROOT/args.source
    out = ROOT/'logs/compact_integration_v1'/args.tag
    out.mkdir(parents=True, exist_ok=False)
    a = np.load(source)
    m = mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')))
    d = mujoco.MjData(m)
    d.qpos[:] = a['initial_qpos']; d.qvel[:] = a['initial_qvel']
    mujoco.mj_forward(m,d)
    prefix = len(a['ctrl_physics'])
    for u in a['ctrl_physics']:
        d.ctrl[:] = u; mujoco.mj_step(m,d)
    aj, aq, av, arm, pads = indices(m)
    ik = LegIK(m,d)
    balance = SupportBalance(m,d)
    rest = d.qpos.copy()
    loaded = d.ctrl.copy()
    arms = Arms(m,rest)
    fp = d.xpos[balance.feet].copy()
    fr = d.xmat[balance.feet].reshape(2,3,3).copy()
    yaw = Rotation.from_quat(np.roll(rest[3:7],-1)).as_euler('xyz')[2]
    forward = np.array([np.cos(yaw),np.sin(yaw),0.])
    com0 = d.subtree_com[1,:2].copy()
    c0 = com0.copy()
    phases = []
    # Put the measured robot COM above the left support, then alternate.
    stance = 0
    goal_com = fp[stance,:2] + args.com_forward*forward[:2]
    phases.append(dict(kind='shift', duration=2., c0=c0.copy(), c1=goal_com.copy(),
                       p0=fp.copy(), p1=fp.copy(), support=np.array([1,1],bool)))
    c0 = goal_com.copy()
    for n in range(args.steps):
        swing = 1-stance
        end = fp.copy()
        # Consecutive alternating replants advance by the requested step length.
        desired_x = fp[stance]@forward+args.length
        end[swing] += (desired_x-fp[swing]@forward)*forward
        end[swing,2] = fp[swing,2]
        support = np.array([i==stance for i in range(2)])
        phases.append(dict(kind='swing', duration=args.swing, c0=c0.copy(), c1=c0.copy(),
                           p0=fp.copy(), p1=end.copy(), support=support, swing=swing))
        fp = end
        next_com = (fp.mean(0)[:2] if n==args.steps-1 else fp[swing,:2]) + args.com_forward*forward[:2]
        phases.append(dict(kind='shift', duration=args.transfer, c0=c0.copy(), c1=next_com.copy(),
                           p0=fp.copy(), p1=fp.copy(), support=np.array([1,1],bool)))
        c0 = next_com.copy()
        stance = swing
    phases.append(dict(kind='settle',duration=1.2,c0=c0.copy(),c1=c0.copy(),p0=fp.copy(),p1=fp.copy(),support=np.array([1,1],bool)))
    cs=list(a['ctrl_physics']); qs=list(a['qpos']); vs=list(a['qvel']); cts=list(a['contacts'])
    rows=[]; sums=np.zeros(6); step=0; reason='completed'; prevq=rest.copy(); qref=rest.copy(); vref=np.zeros(m.nv)
    qvel_filter=np.zeros(m.nv)
    origin=rest[:3].copy(); ob=m.body('object/object').id
    walking_start=prefix+round(2/.005)
    for phase,p in enumerate(phases):
        duration=p['duration']
        for j in range(round(duration/.005)):
            t=min(1.,(j+1)*.005/duration); alpha=smooth(t); speed=6*t*(1-t)/duration
            P=p['p0']*(1-alpha)+p['p1']*alpha
            vp=(p['p1']-p['p0'])*speed
            C=p['c0']*(1-alpha)+p['c1']*alpha
            if p['kind']=='swing':
                P[p['swing'],2]+=args.clearance*np.sin(np.pi*t)**2
                vp[p['swing'],2]+=args.clearance*np.pi/duration*np.sin(2*np.pi*t)
            if step%4==0:
                qref=ik.solve(P,fr,C)
                mujoco.mj_differentiatePos(m,vref,.02,prevq,qref)
                qvel_filter=.5*qvel_filter+.5*np.clip(vref,-4.,4.)
                vref=qvel_filter.copy();prevq=qref.copy()
            tau,wrench=balance.torques(d,qref,vref,P,fr,vp,np.zeros((2,3)),p['support'])
            u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0]
            u[arm]=loaded[arm]
            waist=np.array([i for i,jj in enumerate(aj[:29]) if 'waist_' in m.joint(int(jj)).name])
            u[waist]=loaded[waist]
            u[29:31]=arms.grip(d,args.squeeze)
            u[33:35]=loaded[33:35]
            if m.nu==37:u[35:37]=thumb_control(m,d)
            u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
            d.ctrl[:]=u;mujoco.mj_step(m,d)
            touch,N,imp,vertical,si,invalid=contacts(m,d);sums+=np.r_[imp,si,invalid]
            cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch)
            rows.append(np.r_[step*.005,phase,d.qpos[:3],d.xipos[ob],d.subtree_com[1,:2],C,N,sums,ik.error])
            if step%100==99:
                print(args.tag,round(step*.005,3),'phase',phase,p['kind'],'root',np.round(d.qpos[:3],3),
                      'comerror',np.round(d.subtree_com[1,:2]-C,3),'N',np.round(N,2),'IK',round(ik.error,4),'bad',sums[3:],flush=True)
            step+=1
            if sums[3]>1e-5:reason='forbidden_object_contact'
            elif sums[4]>.001:reason='robot_self_contact'
            elif sums[5]>.001:reason='invalid_terrain_contact'
            elif touch[0]:reason='object_terrain_contact'
            elif d.qpos[2]<.4:reason='robot_fall'
            elif ik.error>.015:reason='leg_ik_failure'
            if reason!='completed':
                print('EARLY_REJECT',reason,step,'impulses',sums,flush=True);break
        if reason!='completed':break
    walking_end=prefix+step-round(1.2/.005) if reason=='completed' else prefix+step
    np.savez_compressed(out/'rollout.npz',initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],
                        ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,gait_diagnostics=rows,
                        walking_start_physics_step=walking_start,walking_end_physics_step=walking_end,
                        phase_opening_frame=-1,phase_withdrawal_end_frame=-1)
    mujoco.mj_saveModel(m,str(out/'rollout_model.mjb'),None)
    result=dict(arguments=vars(args),reason=reason,continuation_seconds=step*.005,impulses=sums.tolist(),
                root_net_displacement_m=float(np.linalg.norm(d.qpos[:2]-origin[:2])))
    (out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--length',type=float,default=.22)
    p.add_argument('--swing',type=float,default=1.2);p.add_argument('--transfer',type=float,default=1.0)
    p.add_argument('--clearance',type=float,default=.055);p.add_argument('--com-forward',type=float,default=0.)
    p.add_argument('--squeeze',type=float,default=2.)
    run(p.parse_args())
