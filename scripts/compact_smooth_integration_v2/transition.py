"""Continuous measured support transition from the preserved smoothed brake.

Reference-only LegIK moves COM forward to the accepted COM/ankle relationship.
Feet and their yaw are retained. Real state is only advanced by native controls.
"""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/compact_integration_v2'),str(ROOT/'scripts/compact_integration_v1'),str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
from mechanism import indices
from physical import Arms,contacts,thumb_control,smooth
from bounded_balance import BoundedBalance
from gait import LegIK

def run(args):
 source=ROOT/args.source;out=ROOT/'logs/compact_smooth_integration_v2'/args.tag;out.mkdir(parents=True,exist_ok=False)
 a=np.load(source);m=mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
 for u in a['ctrl_physics']:d.ctrl[:]=u;mujoco.mj_step(m,d)
 aj,aq,av,arm,pads=indices(m);rest=d.qpos.copy();oldctrl=d.ctrl.copy();bias0=d.qfrc_bias.copy();kd=-m.actuator_biasprm[:,2];kp=m.actuator_gainprm[:,0]
 loaded=oldctrl.copy();loaded[arm]+=.7*kd[arm]*d.qvel[av[arm]]/kp[arm]
 balance=BoundedBalance(m,d);ik=LegIK(m,d);arms=Arms(m,rest);C0=d.subtree_com[1,:2].copy();yaw=R.from_quat(np.roll(rest[3:7],-1)).as_euler('xyz')[2];forward=np.array([np.cos(yaw),np.sin(yaw)])
 shift=args.forward
 if shift is None:
  report=json.loads((ROOT/'logs/compact_smooth_integration_v2/entry_comparison.json').read_text());b=report['accepted_brake'];offset=b['robot_com_heading'][0]-np.array(b['feet_pos_heading'])[:,0].mean();shift=float((balance.pos[:,:2].mean(0)-C0)@forward+offset)
 goal=C0+shift*forward+args.lateral*np.array([-np.sin(yaw),np.cos(yaw)]);rot0=balance.rot.copy();rot1=rot0.copy()
 if args.align_yaw:
  er=json.loads((ROOT/'logs/compact_smooth_integration_v2/entry_comparison.json').read_text())['accepted_brake']
  for i in range(2):
   now=R.from_matrix(rot0[i]).as_euler('xyz')[2];target=yaw+np.radians(er['feet_rpy_heading'][i][2]);rot1[i]=R.from_euler('z',target-now).as_matrix()@rot0[i]
 prev=rest.copy();vel=np.zeros(m.nv);sums=np.zeros(6);rows=[];reason='completed';cs=list(a['ctrl_physics']);qs=list(a['qpos']);vs=list(a['qvel']);cts=list(a['contacts'])
 for step in range(round(args.seconds/.005)):
  sec=step*.005;alpha=smooth(sec/args.shift_time);C=C0*(1-alpha)+goal*alpha
  for i in range(2):balance.rot[i]=R.from_rotvec(smooth((sec-1.)/1.5)*R.from_matrix(rot1[i]@rot0[i].T).as_rotvec()).as_matrix()@rot0[i]
  if step%4==0:
   qref=ik.solve(balance.pos,balance.rot,C);mujoco.mj_differentiatePos(m,vel,.02,prev,qref);vel=np.clip(vel,-2.,2.);prev=qref.copy()
  tau,_=balance.torques(d,qref,vel);u=d.qpos[aq]+(tau[av]+kd*d.qvel[av])/kp
  damping=.7*(1-smooth(sec/args.fade_time));pressD=6*(1-smooth(sec/args.fade_time))
  u[arm]=loaded[arm]+(d.qfrc_bias[av[arm]]-bias0[av[arm]])/kp[arm]-damping*kd[arm]*d.qvel[av[arm]]/kp[arm]
  u[29:31]=arms.grip(d,2.)-pressD*d.qvel[av[29:31]]/800.;u[33:35]=oldctrl[33:35];u[35:37]=thumb_control(m,d)
  blend=smooth(sec/.35);u=oldctrl*(1-blend)+u*blend;u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
  touch,F,imp,V,si,invalid=contacts(m,d);sums+=np.r_[imp,si,invalid];cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);rows.append(np.r_[sec,d.qpos[:3],d.qvel[:6],d.subtree_com[1,:2]-C,F,sums,ik.error,balance.last_result['root_wrench_error']])
  if step%100==99:print(args.tag,round(sec,3),'root',np.round(d.qpos[:3],3),'v',np.round(d.qvel[:6],3),'COMerror',np.round(d.subtree_com[1,:2]-C,3),'N',np.round(F,2),'IK',ik.error,'bad',sums[3:],flush=True)
  if sums[3]>1e-5:reason='forbidden_object'
  elif sums[4]>.001:reason='self_contact'
  elif sums[5]>.001:reason='invalid_terrain'
  elif touch[0]:reason='object_ground'
  elif d.qpos[2]<.4:reason='fall'
  elif ik.error>.005:reason='IK_error'
  if reason!='completed':print('REJECT',reason,flush=True);break
 data={k:a[k] for k in a.files if a[k].ndim==0 or k in ['initial_qpos','initial_qvel']};data.update(ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,transition_diagnostics=rows,transition_start_physics_step=len(a['ctrl_physics']))
 np.savez_compressed(out/'rollout.npz',**data);mujoco.mj_saveModel(m,str(out/'rollout_model.mjb'),None)
 result=dict(arguments=vars(args),COM_forward_shift_m=shift,reason=reason,prefix=len(a['ctrl_physics']),continuation_seconds=len(rows)*.005,impulses=sums.tolist());(out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',default='logs/compact_smooth_v1/best_partial/rollout.npz');p.add_argument('--tag',required=True);p.add_argument('--seconds',type=float,default=3.);p.add_argument('--forward',type=float);p.add_argument('--align-yaw',action='store_true');p.add_argument('--lateral',type=float,default=0.);p.add_argument('--shift-time',type=float,default=1.);p.add_argument('--fade-time',type=float,default=1.);run(p.parse_args())
