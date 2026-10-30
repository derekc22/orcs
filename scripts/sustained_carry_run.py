"""A continuous proven pickup followed by extended native walking, with finite grasp."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sustained_carry_motion import OUT,controller
ROOT=Path(__file__).resolve().parents[1]
def run(tag,source='logs/compact_gripper_v1/static6.npz',prefix=1200):
 source=ROOT/source;a=np.load(source);m=mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
 for u0 in a['ctrl_physics'][:prefix]:d.ctrl[:]=u0;mujoco.mj_step(m,d)
 ctrl,meta=controller(m,d.qpos.copy());mujoco.mj_saveModel(m,str(OUT/(tag+'_model.mjb')),None)
 aj=m.actuator_trnid[:,0];aq=m.jnt_qposadr[aj];av=m.jnt_dofadr[aj];arm=np.r_[[i for i,j in enumerate(aj) if 'right' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])],35];left=np.array([i for i,j in enumerate(aj) if 'left' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])]);left_rest=d.qpos[aq[left]].copy();gq=aq[29:35];pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];armq=aq[arm];fk=mujoco.MjData(m);fk.qpos[:]=d.qpos;fk.qpos[gq]=0;mujoco.mj_forward(m,fk);anchor=fk.geom_xpos[pads].copy();origin=d.qpos[:3].copy();yaw0=Rotation.from_quat(np.roll(d.qpos[3:7],-1)).as_euler('xyz')[2];yaw=yaw0;seed=d.qpos[armq].copy();arm_ref=seed.copy();vel=np.zeros(len(arm));cs=list(a['ctrl_physics'][:prefix]);qs=list(a['qpos'][:prefix]);vs=list(a['qvel'][:prefix]);cts=list(a['contacts'][:prefix]);ob=m.body('object/object').id;oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id];forbidden=0.;lost=False;records=[]
 for step in range((meta['braking_end_frame']+110)*4):
  if step%4==0:lower=ctrl.control(d)
  actual_yaw=Rotation.from_quat(np.roll(d.qpos[3:7],-1)).as_euler('xyz')[2];yaw+=.04*np.arctan2(np.sin(actual_yaw-yaw),np.cos(actual_yaw-yaw));rot=Rotation.from_euler('z',yaw-yaw0);target=rot.apply(anchor-origin)+d.qpos[:3]
  if step%4==0:
   previous=seed.copy();base=d.qpos.copy()
   def residual(x):
    fk.qpos[:]=base;fk.qpos[gq]=0;fk.qpos[armq]=x;mujoco.mj_forward(m,fk);return np.r_[60*(fk.geom_xpos[pads]-target).ravel(),1.0*(x-arm_ref),.1*(x-seed),100*max(0,.01-mujoco.mj_geomDistance(m,fk,m.geom('right_shoulder_yaw_collision').id,m.geom('torso_collision').id,.1,None)),60*max(0,.02-mujoco.mj_geomDistance(m,fk,m.geom('right_wrist_collision').id,m.geom('right_hip_collision').id,.2,None))]
   fit=least_squares(residual,seed,bounds=(m.jnt_range[aj[arm],0]+.005,m.jnt_range[aj[arm],1]-.005),max_nfev=35);seed=fit.x;vel=.65*vel+.35*np.clip((seed-previous)/.02,-3,3)
  jaw=np.clip(d.qpos[gq]+(.9+d.qvel[av[29:35]])/180,-.05,.22);u=np.r_[lower,jaw,0.];u[left]=left_rest+d.qfrc_bias[av[left]]/m.actuator_gainprm[left,0];u[arm]=seed+d.qfrc_bias[av[arm]]/m.actuator_gainprm[arm,0]+(-m.actuator_biasprm[arm,2]/m.actuator_gainprm[arm,0])*vel+.5*np.clip(seed-d.qpos[armq],-.24,.24);u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d);touch=np.zeros(3,bool)
  for ci,c in enumerate(d.contact):
   b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
   if ob not in [b1,b2]:continue
   other=int(c.geom2 if b1==ob else c.geom1);n=m.geom(other).name
   if n=='terrain':touch[0]=True
   elif n.startswith('clamp_'):touch[1 if n.startswith('clamp_pad') and int(n[9:])<3 else 2]=True
   else:
    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);forbidden+=np.linalg.norm(f[:3])*.005
  cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);distance=np.linalg.norm(d.qpos[:2]-origin[:2]);records.append([step*.005,distance,d.qpos[2],d.qpos[oq+2],fit.cost,forbidden])
  if step%100==99:print(tag,round(step*.005,3),'ROOT_DISTANCE',round(distance,3),'rootZ',round(d.qpos[2],3),'lift',round(d.qpos[oq+2]-a['initial_qpos'][oq+2],3),'touch',touch,'cost',round(fit.cost,3),'forbid',round(forbidden,5),flush=True)
  if d.qpos[2]<.43 or forbidden>1e-4 or touch[0]:print('EARLY_PHYSICAL_REJECTION',flush=True);lost=True;break
 np.savez_compressed(OUT/(tag+'.npz'),initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,carry_diagnostics=records,phase_opening_frame=np.array(-1),phase_withdrawal_end_frame=np.array(-1),walking_start_physics_step=np.array(prefix+meta['walk_start_frame']*4),walking_end_physics_step=np.array(prefix+meta['walk_end_frame']*4))
 (OUT/(tag+'_generation.json')).write_text(json.dumps(dict(physical_rejected=lost,root_final_distance_from_carry_handoff_m=float(distance),reference=meta,strict_forbidden_impulse=forbidden,prefix_physics_steps=prefix),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',default='walk1');p.add_argument('--source',default='logs/compact_gripper_v1/static6.npz');p.add_argument('--prefix',type=int,default=1200);a=p.parse_args();run(a.tag,a.source,a.prefix)
