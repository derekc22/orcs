"""Continuous controlled placement after compact-gripper transport."""
import argparse,json
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from compact_gripper_gate import OUT
from compact_gripper_balance import Balance

def smooth(t):
 t=np.clip(t,0,1);return t*t*(3-2*t)

def plan(m,start,target,feet_pos,feet_rot,seed=None):
 """Geometric planning only: predicted object pose lives in this separate FK data."""
 d=mujoco.MjData(m);q0=start.copy();aj=m.actuator_trnid[:,0];acts=np.r_[np.arange(29),35];j=aj[acts];aq=m.jnt_qposadr[j];pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];ob=m.body('object/object').id;oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id];tor=m.body('torso_link').id;R0=Rotation.from_quat(np.roll(q0[3:7],-1));forward=q0[oq:oq+3]-q0[:3];forward[2]=0;forward/=np.linalg.norm(forward);d.qpos[:]=q0;mujoco.mj_forward(m,d);com=d.subtree_com[1,:2].copy();left=[k for k,jj in enumerate(j) if 'left' in m.joint(int(jj)).name and any(t in m.joint(int(jj)).name for t in ['shoulder','elbow','wrist'])]
 def pose(x):
  d.qpos[:]=q0;d.qpos[:3]=x[:3];d.qpos[3:7]=np.roll((Rotation.from_rotvec(x[3:6])*R0).as_quat(),1);d.qpos[aq]=x[6:];d.qpos[m.jnt_qposadr[aj[29:35]]]=0;mujoco.mj_forward(m,d)
 def residual(x):
  pose(x);orientation=np.concatenate([Rotation.from_matrix(d.xmat[b].reshape(3,3)@feet_rot[i].T).as_rotvec() for i,b in enumerate(feet)]);pen=np.zeros(m.ngeom)
  for c in d.contact:
   b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
   if ob in [b1,b2]:
    other=int(c.geom2 if b1==ob else c.geom1)
    if m.geom(other).name.startswith('clamp_') or other==m.geom('terrain').id:continue
    pen[other]=min(pen[other],c.dist-.006)
   elif b1>0 and b2>0:pen[c.geom1]=min(pen[c.geom1],c.dist-.008)
  clear=[]
  for n1,n2,gap in [('right_shoulder_yaw_collision','torso_collision',.018),('right_wrist_collision','right_hip_collision',.03),('left_wrist_collision','torso_collision',.035),('left_wrist_collision','left_hip_collision',.035)]:clear.append(100*max(0,gap-mujoco.mj_geomDistance(m,d,m.geom(n1).id,m.geom(n2).id,.2,None)))
  return np.r_[80*(d.geom_xpos[pads]-target).ravel(),150*(d.xpos[feet]-feet_pos).ravel(),40*orientation,80*(d.subtree_com[1,:2]-com),.3*d.xmat[tor].reshape(3,3)[:2,2],3*(x[2]-q0[2]),.5*(x[6:]-q0[aq]),2*(x[6+np.array(left)]-q0[aq[left]]),150*pen,clear]
 lo=np.r_[q0[:2]-.22,.43,[-.85,-.85,-.35],m.jnt_range[j,0]+.008];hi=np.r_[q0[:2]+.22,.80,[.85,.85,.35],m.jnt_range[j,1]-.008];x=np.r_[q0[:3],np.zeros(3),q0[aq]] if seed is None else seed
 fit=least_squares(residual,np.clip(x,lo+.00001,hi-.00001),bounds=(lo,hi),max_nfev=130);pose(fit.x)
 return d.qpos.copy(),fit.x,dict(cost=fit.cost,error_m=float(np.linalg.norm(d.geom_xpos[pads]-target,axis=1).max()),root=float(d.qpos[2]))

def run(tag,source):
 a=np.load(OUT/(source+'.npz'));m=mujoco.MjModel.from_binary_path(str(OUT/(source+'_model.mjb')));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d);rqs=[];rvs=[]
 for u0 in a['ctrl_physics']:d.ctrl[:]=u0;mujoco.mj_step(m,d);rqs.append(d.qpos.copy());rvs.append(d.qvel.copy())
 mujoco.mj_saveModel(m,str(OUT/(tag+'_model.mjb')),None)
 aj=m.actuator_trnid[:,0];aq=m.jnt_qposadr[aj];av=m.jnt_dofadr[aj];arm=np.r_[[i for i,j in enumerate(aj) if 'right' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])],35];left=np.array([i for i,j in enumerate(aj) if 'left' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])]);armq=aq[arm];gq=aq[29:35];ob=m.body('object/object').id;oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];od=m.jnt_dofadr[oj];pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];fk=mujoco.MjData(m);fk.qpos[:]=d.qpos;fk.qpos[gq]=0;mujoco.mj_forward(m,fk);anchor=fk.geom_xpos[pads].copy();start=d.qpos.copy();initial_height=a['initial_qpos'][oq+2];lift=d.qpos[oq+2]-initial_height;balance=Balance(m,d);feet_pos=balance.pos.copy();feet_rot=balance.rot.copy();poses=[];report=[];plan_seed=None
 for lower in np.linspace(0,lift+.025,13):
  predicted=start.copy();predicted[oq+2]-=lower;q,plan_seed,r=plan(m,predicted,anchor-[0,0,lower],feet_pos,feet_rot,plan_seed);poses.append(q);report.append(r);print('PLACE_GEOMETRY',lower,r,flush=True)
 (OUT/(tag+'_placement_geometry.json')).write_text(json.dumps(report,indent=2));np.savez(OUT/(tag+'_placement_plan.npz'),qpos=poses)
 cs=list(a['ctrl_physics']);qs=rqs;vs=rvs;cts=list(a['contacts']);seed=d.qpos[armq].copy();vel=np.zeros(len(arm));load_anchor=None;grip_hold=None;load_run=0;release_step=None;release_jaw=None;withdrawal_end=-1;z_correction=0.;withdraw_q=None;prefix=len(cs);phase_rows=[]
 for step in range(2900):
  sec=step*.005;alpha=smooth(sec/4.);t=alpha*12;idx=min(int(t),11);frac=t-idx;qref=poses[idx]*(1-frac)+poses[idx+1]*frac
  if sec<.7:qref=start*(1-smooth(sec/.7))+qref*smooth(sec/.7)
  target=anchor-[0,0,(lift+.025)*alpha]
  if load_anchor is None and sec>1 and d.qpos[oq+2]-initial_height<.07:
   z_correction=float(np.clip(z_correction+(-.028-d.qvel[od+2])*.005,0,.04));target[:,2]+=z_correction
  if load_anchor is not None:target=load_anchor.copy();qref=held_q.copy()
  if load_anchor is not None and sec>5.0 and release_step is None:release_step=step;release_jaw=d.qpos[gq].copy();print('OPEN',step,flush=True)
  if release_step is not None:
   elapsed=(step-release_step)*.005;up=smooth((elapsed-5.6)/2.2);target=load_anchor+[0,0,.32*up];withdrawal_end=prefix+release_step+1560
   if elapsed>=5.6 and withdraw_q is None:
    withdraw_q,_,r=plan(m,d.qpos.copy(),load_anchor+[0,0,.32],feet_pos,feet_rot);print('WITHDRAW_POSE',r,flush=True)
   if withdraw_q is not None:qref=held_q*(1-up)+withdraw_q*up
  if step%4==0:
   previous=seed.copy();base=d.qpos.copy()
   def residual(x):
    fk.qpos[:]=base;fk.qpos[gq]=0;fk.qpos[armq]=x;mujoco.mj_forward(m,fk);return np.r_[60*(fk.geom_xpos[pads]-target).ravel(),.9*(x-qref[armq]),.1*(x-seed),100*max(0,.015-mujoco.mj_geomDistance(m,fk,m.geom('right_shoulder_yaw_collision').id,m.geom('torso_collision').id,.1,None)),60*max(0,.025-mujoco.mj_geomDistance(m,fk,m.geom('right_wrist_collision').id,m.geom('right_hip_collision').id,.2,None))]
   fit=least_squares(residual,seed,bounds=(m.jnt_range[aj[arm],0]+.005,m.jnt_range[aj[arm],1]-.005),max_nfev=35);seed=fit.x;vel=.65*vel+.35*np.clip((seed-previous)/.02,-3,3)
  tau,wrench=balance.torques(d,qref);u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0];u[left]=qref[aq[left]]+d.qfrc_bias[av[left]]/m.actuator_gainprm[left,0];u[arm]=seed+d.qfrc_bias[av[arm]]/m.actuator_gainprm[arm,0]+(-m.actuator_biasprm[arm,2]/m.actuator_gainprm[arm,0])*vel+.5*np.clip(seed-d.qpos[armq],-.24,.24)
  if release_step is None:u[29:35]=np.clip(d.qpos[gq]+(1.+d.qvel[av[29:35]])/180,-.05,.22) if grip_hold is None else grip_hold
  else:u[29:35]=np.maximum(release_jaw-.05*(step-release_step)*.005,-.04)
  u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d);touch=np.zeros(3,bool);vertical=0.;forbid=0.
  for ci,c in enumerate(d.contact):
   b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
   if ob not in [b1,b2]:continue
   other=int(c.geom2 if b1==ob else c.geom1);n=m.geom(other).name;force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
   if n=='terrain':touch[0]=True;world=c.frame.reshape(3,3).T@force[:3];vertical+=(world[2] if b2==ob else -world[2])
   elif n.startswith('clamp_'):touch[1 if n.startswith('clamp_pad') and int(n[9:])<3 else 2]=True
   else:forbid+=np.linalg.norm(force[:3])*.005
  gentle=np.linalg.norm(d.qvel[od:od+3])<.08 and np.linalg.norm(d.qvel[od+3:od+6])<.3
  load_run=load_run+1 if touch[0] and vertical>.7*.05*9.81 and gentle and d.qpos[oq+2]-initial_height<.004 else 0
  if load_run>=30 and load_anchor is None:
   fk.qpos[:]=d.qpos;fk.qpos[gq]=0;mujoco.mj_forward(m,fk);load_anchor=fk.geom_xpos[pads].copy();grip_hold=d.qpos[gq].copy();held_q=qref.copy();seed=d.qpos[armq].copy();vel[:]=0;print('TERRAIN_WEIGHT_TRANSFER',sec,flush=True)
  cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);phase_rows.append([sec,d.qpos[oq+2]-initial_height,vertical,forbid])
  if step%100==99:print(tag,sec,'root',round(d.qpos[2],3),'lift',round(d.qpos[oq+2]-initial_height,3),'contact',touch,'load',round(vertical,3),'cost',round(fit.cost,3),'forbid',forbid,flush=True)
  if d.qpos[2]<.43:print('EARLY_STOP_FALL',flush=True);break
 np.savez_compressed(OUT/(tag+'.npz'),initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,phase_opening_frame=np.array(-1 if release_step is None else (prefix+release_step)/4),phase_withdrawal_end_frame=np.array(withdrawal_end/4),placement_trace=phase_rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',default='complete1');p.add_argument('--source',default='carry2');a=p.parse_args();run(a.tag,a.source)
