"""Short-horizon physics gate with contact-consistent gravity compensation."""
import argparse,json
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from compact_gripper_gate import OUT as INPUT,BASE
from sustained_carry_motion import OUT,ROOT
from compact_gripper_common import build
from compact_gripper_balance import Balance

def smooth(t):
 t=np.clip(t,0,1);return t*t*(3-2*t)
def run(tag,shorten=.30):
 m=mujoco.MjModel.from_binary_path(str(ROOT/'logs/compact_gripper_v1/static6_model.mjb'));mujoco.mj_saveModel(m,str(OUT/(tag+'_model.mjb')),None);key=f'{shorten:.2f}'
 poses=[np.load(INPUT/f'posture_v5_{key}_{h:.2f}.npz')['qpos'] for h in [.08,0.,.16,.30]];poses[3]=np.load(INPUT/'carry_pose_0.30.npz')['qpos']
 # Model differs only in the connector geometry/inertia from the geometric gate.
 aj=m.actuator_trnid[:,0];aq=m.jnt_qposadr[aj];av=m.jnt_dofadr[aj];feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];tor=m.body('torso_link').id;ob=m.body('object/object').id;oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];od=m.jnt_dofadr[oj];pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];gq=aq[29:35];arm=np.r_[[i for i,j in enumerate(aj) if 'right' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])],35];armq=aq[arm]
 d=mujoco.MjData(m);fk=mujoco.MjData(m);q0=poses[0].copy();d.qpos[:]=q0;d.qvel[:]=0;mujoco.mj_forward(m,d);v0=d.qvel.copy();qstart=q0.copy();obj0=d.xipos[ob].copy();seed=q0[armq].copy();lastseed=seed.copy();vel=np.zeros(len(arm));anchor=[]
 for p in poses:fk.qpos[:]=p;mujoco.mj_forward(m,fk);anchor.append(fk.geom_xpos[pads].copy())
 balance=Balance(m,d)
 cs=[];qs=[];vs=[];cts=[];sums=np.zeros(4);jacs=[np.zeros((3,m.nv)) for _ in feet];jr=[np.zeros((3,m.nv)) for _ in feet];J=np.zeros((12,m.nv));opening=None
 for step in range(880):
  sec=step*.005
  if sec<1.:targetq=poses[0]*(1-smooth(sec))+poses[1]*smooth(sec);target=anchor[0]*(1-smooth(sec))+anchor[1]*smooth(sec)
  elif sec<2.:targetq=poses[1].copy();target=anchor[1].copy()
  elif sec<3.:alpha=smooth(sec-2.);targetq=poses[1]*(1-alpha)+poses[2]*alpha;target=anchor[1]*(1-alpha)+anchor[2]*alpha
  elif sec<4.:alpha=smooth(sec-3.);targetq=poses[2]*(1-alpha)+poses[3]*alpha;target=anchor[2]*(1-alpha)+anchor[3]*alpha
  else:targetq=poses[3].copy();target=anchor[3].copy()
  # Feet-held gravity load allocation. Diagnostic is free base, actual collisions only.
  for i,b in enumerate(feet):mujoco.mj_jacBody(m,d,jacs[i],jr[i],b);J[6*i:6*i+3]=jacs[i];J[6*i+3:6*i+6]=jr[i]
  tau,wrench=balance.torques(d,targetq)
  u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0]
  if step%4==0:
   base=d.qpos.copy();previous=seed.copy()
   def residual(x):
    fk.qpos[:]=base;fk.qpos[gq]=0;fk.qpos[armq]=x;mujoco.mj_forward(m,fk);return np.r_[60*(fk.geom_xpos[pads]-target).ravel(),1.2*(x-targetq[armq]),.1*(x-seed),100*max(0,.012-mujoco.mj_geomDistance(m,fk,m.geom('right_shoulder_yaw_collision').id,m.geom('torso_collision').id,.1,None)),60*max(0,.025-mujoco.mj_geomDistance(m,fk,m.geom('right_wrist_collision').id,m.geom('right_hip_collision').id,.2,None))]
   fit=least_squares(residual,seed,bounds=(m.jnt_range[aj[arm],0]+.005,m.jnt_range[aj[arm],1]-.005),max_nfev=30);seed=fit.x;vel=.65*vel+.35*np.clip((seed-previous)/.02,-3,3)
  u[arm]=seed+d.qfrc_bias[av[arm]]/m.actuator_gainprm[arm,0]+(-m.actuator_biasprm[arm,2]/m.actuator_gainprm[arm,0])*vel+.5*np.clip(seed-d.qpos[armq],-.24,.24)
  left=np.array([i for i,j in enumerate(aj) if 'left' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])]);u[left]=targetq[aq[left]]+d.qfrc_bias[av[left]]/m.actuator_gainprm[left,0]
  u[29:35]=.065*smooth((sec-.8)/.8) if sec<2 else np.clip(d.qpos[gq]+(.9+d.qvel[av[29:35]])/180,-.05,.22)
  u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d);touch=np.zeros(3,bool);imp=np.zeros(4)
  for ci,c in enumerate(d.contact):
   b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
   if ob not in [b1,b2]:continue
   other=int(c.geom2 if b1==ob else c.geom1);n=m.geom(other).name;force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);mag=np.linalg.norm(force[:3])*.005
   if n=='terrain':touch[0]=True;imp[0]+=mag
   elif n.startswith('clamp_'):i=0 if n.startswith('clamp_pad') and int(n[9:])<3 else 1;touch[i+1]=True;imp[i+1]+=mag
   else:imp[3]+=mag
  sums+=imp;cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch)
  if step%100==99:print(tag,sec,'root',round(d.qpos[2],3),'lift',round(d.xipos[ob,2]-obj0[2],3),'touch',touch,'cost',round(fit.cost,4),'forbid',round(sums[-1],5),flush=True)
  if d.qpos[2]<.4:print('EARLY_STOP_FALL',flush=True);break
 np.savez_compressed(OUT/(tag+'.npz'),initial_qpos=q0,initial_qvel=v0,ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,phase_opening_frame=np.array(-1),phase_withdrawal_end_frame=np.array(-1))
 print('SUMS',sums,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',default='static1');p.add_argument('--shorten',type=float,default=.30);a=p.parse_args();run(a.tag,a.shorten)
