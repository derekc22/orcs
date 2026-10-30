#!/usr/bin/env python3
"""Architecture 2: one wrist carries both opposed jaws on a rigid openable clamp.
Eliminates the relative-wrist degree of freedom from the proven six-pad test rig.
"""
import json,argparse
import mujoco,numpy as np
from scipy.optimize import least_squares
from active_largebox_config import ROOT, OUT, INPUTS, SOURCE, name, reference_model
from active_largebox_artifacts import materialize
from build_engineered_ground_large_model import large_ground_gripper_spec
from build_engineered_ground_original_largebox_model import original_largebox_spec
from orcs.assets.g1 import find_body

def build(tag):
 old=mujoco.MjModel.from_binary_path(str(reference_model()));source=np.load(SOURCE);d=mujoco.MjData(old);d.qpos[:]=source['initial_qpos'];mujoco.mj_forward(old,d)
 assembly=json.loads((INPUTS/'manifest_parallel.json').read_text());oldextended=mujoco.MjModel.from_binary_path(str(materialize(INPUTS/'parallel_robot1_model.mjb')));qnew=np.load(INPUTS/'right_wrist_outer_qpos.npy');new=mujoco.MjData(oldextended);new.qpos[:]=qnew;mujoco.mj_forward(oldextended,new);rb=oldextended.body('right_wrist_yaw_link').id;W=new.xmat[rb].reshape(3,3);origin=new.xpos[rb]
 s=large_ground_gripper_spec();s.option.timestep=.005;s.option.integrator=3;s.option.iterations=30
 for g in list(s.geoms):
  if 'guard_collision' in g.name:s.delete(g);continue
  try:og=old.geom('robot/'+g.name).id;g.contype=int(old.geom_contype[og]);g.conaffinity=int(old.geom_conaffinity[og])
  except KeyError:pass
 wrist_parent=find_body(s.worldbody,'right_wrist_yaw_link');wrist=wrist_parent.add_body(name='clamp_frame');wrist.add_joint(name='clamp_yaw',type=3,axis=[0,0,1],range=[-1.57,1.57],limited=True,damping=.5,armature=.002);config=[];points=[]
 forward=source['initial_qpos'][-7:-4]-source['initial_qpos'][:3];forward[2]=0;forward/=np.linalg.norm(forward);scene_shift=.05*forward
 for i,c in enumerate(assembly['fingers']):
  side=c['joint'].split('_')[0];b=old.body('robot/'+side+'_wrist_yaw_link').id;OW=d.xmat[b].reshape(3,3);world=d.xpos[b]+OW@c['mount_local']+scene_shift;axis=OW@c['axis_local'];local=W.T@(world-origin)
  f=wrist.add_body(name=f'clamp_finger{i}',pos=local);f.add_joint(name=f'clamp_grip{i}',type=2,axis=W.T@axis,range=[-.05,.22],limited=True,damping=.5,armature=.005)
  f.add_geom(name=f'clamp_pad{i}',type=2,size=[.025,0,0],mass=.017,friction=[2.5,.1,.003],condim=4,contype=4,conaffinity=13,solref=[.03,1],solimp=[.8,.95,.01,.5,2],rgba=[.95,.35,.1,1]);f.add_geom(name=f'clamp_piston{i}',type=3,fromto=np.r_[np.zeros(3),-.22*(W.T@axis)],size=[.003,0,0],mass=.008,friction=[2.5,.1,.003],condim=4,contype=4,conaffinity=13,solref=[.03,1],solimp=[.8,.95,.01,.5,2],rgba=[.45,.48,.5,1]);points.append(world)
  config.append(dict(joint=f'clamp_grip{i}',mount_local=local.tolist(),axis_local=(W.T@axis).tolist(),pad_world_initial=world.tolist()))
 top=np.mean(points,axis=0);top[2]=max(source['initial_qpos'][-5]+.34,.52);hub=W.T@(top-origin)
 def strut(n,p1,p2,mass=.008):
  wrist.add_geom(name=n,type=3,fromto=np.r_[p1,p2],size=[.008,0,0],mass=mass,contype=4,conaffinity=13,rgba=[.25,.35,.4,1])
 lateral=source['initial_qpos'][-7:-4]-source['initial_qpos'][:3];lateral[2]=0;lateral/=np.linalg.norm(lateral);outward=np.array([lateral[1],-lateral[0],0.]);bypass=W.T@(outward*.16+np.array([0,0,.04]));strut('clamp_mount',np.array([.08,0,0]),bypass,.015);strut('clamp_spine',bypass,hub,.04)
 for i,p in enumerate(points):
  upper=p.copy();upper[2]=top[2];up=W.T@(upper-origin);lo=W.T@(p-origin);strut(f'clamp_bridge{i}',hub,up);strut(f'clamp_rail{i}',up,lo)
 for i in range(old.nu):
  j=int(old.actuator_trnid[i,0]);n=name(old,mujoco.mjtObj.mjOBJ_JOINT,j)
  s.add_actuator(name='drive_'+n,target=n,trntype=0,gaintype=int(old.actuator_gaintype[i]),gainprm=old.actuator_gainprm[i],biastype=int(old.actuator_biastype[i]),biasprm=old.actuator_biasprm[i],ctrllimited=True,ctrlrange=old.actuator_ctrlrange[i],forcelimited=True,forcerange=old.actuator_forcerange[i],gear=old.actuator_gear[i])
 for i in range(6):s.add_actuator(name=f'drive_clamp{i}',target=f'clamp_grip{i}',trntype=0,gainprm=[180]+[0]*9,biastype=1,biasprm=[0,-180,-1]+[0]*7,ctrllimited=True,ctrlrange=[-.05,.22],forcelimited=True,forcerange=[-2,2])
 s.add_actuator(name='drive_clamp_yaw',target='clamp_yaw',trntype=0,gainprm=[80]+[0]*9,biastype=1,biasprm=[0,-80,-4]+[0]*7,ctrllimited=True,ctrlrange=[-1.57,1.57],forcelimited=True,forcerange=[-5,5])
 s.worldbody.add_geom(name='terrain',type=0,size=[0,0,.1],contype=9,conaffinity=1,friction=[1,.005,.0001],rgba=[.8,.8,.8,1]);s.attach(original_largebox_spec(),prefix='object/',frame=s.worldbody.add_frame());m=s.compile()
 for j in range(old.njnt):
  n=name(old,mujoco.mjtObj.mjOBJ_JOINT,j);nj=m.joint(('object/' if n=='floating_base_joint' and old.jnt_qposadr[j]>10 else '')+n).id;k=6 if old.jnt_type[j]==0 else 1;ad=old.jnt_dofadr[j];nd=m.jnt_dofadr[nj]
  for attr in ['dof_damping','dof_armature','dof_frictionloss']:getattr(m,attr)[nd:nd+k]=getattr(old,attr)[ad:ad+k]
 for i in range(29):
  if any(t in m.joint(int(m.actuator_trnid[i,0])).name for t in ['shoulder','elbow','wrist']):m.actuator_gainprm[i,0]*=5;m.actuator_biasprm[i,1]*=5;m.actuator_biasprm[i,2]*=np.sqrt(5)
 q=m.qpos0.copy();v=np.zeros(m.nv)
 for j in range(old.njnt):
  n=name(old,mujoco.mjtObj.mjOBJ_JOINT,j);nn=('object/' if n=='floating_base_joint' and old.jnt_qposadr[j]>10 else '')+n;nj=m.joint(nn).id;ej=oldextended.joint(nn).id;k=7 if old.jnt_type[j]==0 else 1;l=6 if k==7 else 1
  q[m.jnt_qposadr[nj]:m.jnt_qposadr[nj]+k]=qnew[oldextended.jnt_qposadr[ej]:oldextended.jnt_qposadr[ej]+k];v[m.jnt_dofadr[nj]:m.jnt_dofadr[nj]+l]=source['initial_qvel'][old.jnt_dofadr[j]:old.jnt_dofadr[j]+l]
 q[m.jnt_qposadr[m.joint('object/floating_base_joint').id]:m.jnt_qposadr[m.joint('object/floating_base_joint').id]+3]+=scene_shift
 mujoco.mj_saveModel(m,str(OUT/(tag+'_model.mjb')),None);(OUT/(tag+'_manifest.json')).write_text(json.dumps(dict(architecture='single-wrist rigid bilateral parallel clamp',fingers=config,pad_radius_m=.025,grip_actuator_force_limit_N=2,mass_kg=.05,box_dimensions_m=[.4735718,.46160235,.40955865],rig_evidence='rig_parallel_verified.npz',bridge_height_world_m=float(top[2])),indent=2));return m,q,v

def run(tag):
 m,q,v=build(tag);d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];ob=m.jnt_bodyid[oj];tor=m.body('torso_link').id;aj=m.actuator_trnid[:,0];aq=m.jnt_qposadr[aj];arm=np.array([i for i,j in enumerate(aj) if 'right' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])]);arm=np.r_[arm,35];armq=aq[arm];pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];gripq=m.jnt_qposadr[m.actuator_trnid[29:35,0]];anchor=d.geom_xpos[pads].copy();seed=d.qpos[armq].copy();obj0=d.qpos[oq:oq+3].copy();fk=mujoco.MjData(m)
 # Initialize the robot above the ground grasp, then physically reach down.
 desired=anchor+np.array([0,0,.08]);original=q.copy()
 def initial_residual(x):
  fk.qpos[:]=original;fk.qpos[armq]=x;mujoco.mj_forward(m,fk);return np.r_[60*(fk.geom_xpos[pads]-desired).ravel(),.02*(x-original[armq])]
 raised=least_squares(initial_residual,original[armq],bounds=(m.jnt_range[aj[arm],0]+.005,m.jnt_range[aj[arm],1]-.005),max_nfev=180);q[armq]=raised.x;v[m.jnt_dofadr[aj[arm]]]=0;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);seed=q[armq].copy();print('REACH_START_IK_COST',raised.cost,flush=True)
 # Fail early for penetration by a non-end-effector body at initialization.
 invalid=[]
 for c in d.contact:
  ns=[m.geom(int(g)).name for g in [c.geom1,c.geom2]]
  if c.dist<-.003 and any('clamp_' in n for n in ns) and not any(n.startswith('object/') or n=='' for n in ns):invalid.append([ns,float(c.dist)])
 (OUT/(tag+'_initial_contacts.json')).write_text(json.dumps(invalid,indent=2));print('INITIAL_INTERFERENCE',invalid,flush=True)
 if invalid:raise RuntimeError('Mechanical feasibility gate rejected initial assembly interference')
 from active_largebox_sonic import Controller
 left=np.array([i for i,j in enumerate(aj) if 'left' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])]);left_rest=q[aq[left]].copy()
 left_body=m.body('left_wrist_yaw_link').id;left_hip=m.geom('left_hip_collision').id;left_wrist=m.geom('left_wrist_collision').id
 lateral=np.array([-obj0[1]+q[1],obj0[0]-q[0],0.]);lateral/=np.linalg.norm(lateral);left_target=d.xpos[left_body]+.13*lateral+np.array([0,0,.04]);left_base=q.copy()
 def safe_left(x):
  fk.qpos[:]=left_base;fk.qpos[aq[left]]=x;mujoco.mj_forward(m,fk);distance=mujoco.mj_geomDistance(m,fk,left_hip,left_wrist,.2,None)
  return np.r_[80*(fk.xpos[left_body]-left_target),100*max(0,.04-distance),.10*(x-left_base[aq[left]])]
 left_fit=least_squares(safe_left,left_rest,bounds=(m.jnt_range[aj[left],0]+.006,m.jnt_range[aj[left],1]-.006),max_nfev=160);left_rest=left_fit.x;q[aq[left]]=left_rest;v[m.jnt_dofadr[aj[left]]]=0;d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);print('SAFE_LEFT_COST',left_fit.cost,flush=True)
 shoulder=m.geom('right_shoulder_yaw_collision').id;torso_geom=m.geom('torso_collision').id;right_wrist=m.geom('right_wrist_collision').id;right_hip=m.geom('right_hip_collision').id
 controller=Controller(m,arm_reference={'left':left_rest.copy()})
 cs=[];qs=[];vs=[];cts=[];summary=[];ips=[]
 seed_velocity=np.zeros(len(arm));release_anchor=None;release_jaw_start=None;load_transfer_anchor=None;grip_hold=None;support_run=0;earned_lift=False;release_start=None;height_correction=0.
 for step in range(4000):
  f=step/4
  if step%4==0:lower=controller.control(d)
  close=np.clip((f-40)/40,0,1)
  if f>=510 and load_transfer_anchor is not None and release_start is None:release_start=f
  opening=0. if release_start is None else np.clip((f-release_start)/20,0,1);opening=opening*opening*(3-2*opening)
  jaw_target=np.ones(6)*.065*close
  if f>=80 and release_start is None:
   near_ground=earned_lift and f>=270 and d.qpos[oq+2]-obj0[2]<.07
   force=(1.0 if near_ground else .6)*min((f-80)/20,1.)
   jaw_target=np.clip(d.qpos[gripq]+(force+d.qvel[m.jnt_dofadr[m.actuator_trnid[29:35,0]]])/180,-.05,.22)
  if grip_hold is not None and release_start is None:jaw_target=grip_hold.copy()
  if release_start is not None:
   if release_jaw_start is None:release_jaw_start=d.qpos[gripq].copy()
   jaw_target=np.maximum(release_jaw_start-.05*max(0,f-release_start)*.02,-.04)
  u=np.r_[lower,jaw_target,0.]
  u[left]=np.clip(left_rest+d.qfrc_bias[m.jnt_dofadr[aj[left]]]/m.actuator_gainprm[left,0],m.actuator_ctrlrange[left,0],m.actuator_ctrlrange[left,1])
  h=.16*np.clip((f-100)/65,0,1) if f<270 else .16*(1-np.clip((f-270)/230,0,1))
  if f<40:
   reach=np.clip(f/40,0,1);h+=.08*(1-reach*reach*(3-2*reach))

  if earned_lift and load_transfer_anchor is None and f>=270 and (d.qpos[oq+2]-obj0[2]<.07 or height_correction>0):
   vz=d.qvel[m.jnt_dofadr[oj]+2];height_correction=float(np.clip(height_correction+(-.035-vz)*.005,0,.04))
  h+=height_correction
  target=anchor+np.array([0,0,h]);early_forward=obj0[:2]-q[:2];early_forward/=np.linalg.norm(early_forward);target[:,:2]+=.06*np.clip((f-100)/40,0,1)*early_forward
  if step==500:carry_origin=d.qpos[:2].copy()
  if step>=500:
   if step==1200:stop_delta=d.qpos[:2]-carry_origin
   target[:,:2]+=d.qpos[:2]-carry_origin if step<1200 else stop_delta
   forward=obj0[:2]-q[:2];forward/=np.linalg.norm(forward);target[:,:2]+=.12*np.clip((f-190)/50,0,1)*forward
  if load_transfer_anchor is not None:target=load_transfer_anchor.copy()
  if release_start is not None and abs(f-release_start)<1e-6:
   fk.qpos[:]=d.qpos;fk.qpos[gripq]=0;mujoco.mj_forward(m,fk);release_anchor=fk.geom_xpos[pads].copy();seed=d.qpos[armq].copy();seed_velocity[:]=0
  if release_anchor is not None:target=release_anchor+np.array([0,0,.08*np.clip((f-release_start-280)/60,0,1)])
  if step%4==0:
   previous_seed=seed.copy();base=d.qpos.copy()
   def residual(x):
    fk.qpos[:]=base;fk.qpos[gripq]=0;fk.qpos[armq]=x;mujoco.mj_forward(m,fk);distance=mujoco.mj_geomDistance(m,fk,shoulder,torso_geom,.1,None);wrist_distance=mujoco.mj_geomDistance(m,fk,right_wrist,right_hip,.2,None);return np.r_[60*(fk.geom_xpos[pads]-target).ravel(),100*max(0,.007-distance),40*max(0,.003-wrist_distance),.05*(x-seed)]
   fit=least_squares(residual,np.clip(seed,m.jnt_range[aj[arm],0]+.006,m.jnt_range[aj[arm],1]-.006),bounds=(m.jnt_range[aj[arm],0]+.005,m.jnt_range[aj[arm],1]-.005),max_nfev=45);seed=fit.x;seed_velocity=.65*seed_velocity+.35*np.clip((seed-previous_seed)/.02,-3,3)
  gravity=d.qfrc_bias[m.jnt_dofadr[aj[arm]]]/m.actuator_gainprm[arm,0];u[arm]=np.clip(seed+gravity+(-m.actuator_biasprm[arm,2]/m.actuator_gainprm[arm,0])*seed_velocity+.5*np.clip(seed-d.qpos[armq],-.24,.24),m.actuator_ctrlrange[arm,0],m.actuator_ctrlrange[arm,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
  ground=False;hands=np.zeros(2,bool);imp=np.zeros(4);floor_vertical=0.
  for ci,c in enumerate(d.contact):
   if ob not in [m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]]:continue
   other=int(c.geom2 if m.geom_bodyid[c.geom1]==ob else c.geom1);n=m.geom(other).name;force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);mag=np.linalg.norm(force[:3])*.005
   if n=='terrain':
    ground=True;imp[0]+=mag;worldforce=c.frame.reshape(3,3).T@force[:3];floor_vertical+=(worldforce[2] if m.geom_bodyid[c.geom2]==ob else -worldforce[2])
   elif n.startswith('clamp_'):i=0 if n.startswith('clamp_pad') and int(n[9:])<3 else 1;hands[i]=True;imp[1+i]+=mag
   else:imp[3]+=mag
  if not ground and d.qpos[oq+2]-obj0[2]>.05:earned_lift=True
  od=m.jnt_dofadr[oj];gentle=np.linalg.norm(d.qvel[od:od+3])<.10 and np.linalg.norm(d.qvel[od+3:od+6])<.3
  support_run=support_run+1 if ground and floor_vertical>.7*.05*9.81 and gentle and earned_lift and f>290 else 0
  if support_run>=20 and load_transfer_anchor is None:
   fk.qpos[:]=d.qpos;fk.qpos[gripq]=0;mujoco.mj_forward(m,fk);load_transfer_anchor=fk.geom_xpos[pads].copy();grip_hold=d.qpos[gripq].copy();seed=d.qpos[armq].copy();seed_velocity[:]=0;print('TERRAIN_LOAD_TRANSFER',f,flush=True)
  tilt=np.degrees(np.arccos(np.clip(d.xmat[tor].reshape(3,3)[2,2],-1,1)));cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append([ground,*hands]);summary.append([d.qpos[oq+2],d.qpos[2],tilt]);ips.append(imp)
  if step%200==199:print(tag,step,np.round(summary[-1],3),'contact',cts[-1],'IK_cost',fit.cost,flush=True)
 np.savez_compressed(OUT/(tag+'.npz'),initial_qpos=q,initial_qvel=v,ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,summary=summary,impulses=ips,phase_opening_frame=np.array(-1 if release_start is None else release_start),phase_withdrawal_end_frame=np.array(-1 if release_start is None else release_start+340))
 print('IMPULSES',np.sum(ips,axis=0).tolist(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',default='clamp1');args=p.parse_args();run(args.tag)
