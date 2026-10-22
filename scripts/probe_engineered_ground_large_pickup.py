#!/usr/bin/env python3
"""Upright bilateral side-grasp and sustained-lift gate for the large box."""

from __future__ import annotations
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from build_engineered_ground_large_model import BOX_HALF,BOX_MASS,MODEL,build_model

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"logs/engineering_ground_large";UPRIGHT=ROOT/"logs/hand_manipulation_research/upright_balance_vertical_v4.npz";SUBSTEPS=4;DT=.005
def smoothstep(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)

def main():
 model=build_model();model.opt.timestep=DT;up=np.load(UPRIGHT);q0=up["qpos"][250].copy();u0=up["ctrl_physics"][1000].copy()
 rj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"robot/floating_base_joint");oj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"object/floating_base_joint");rq=int(model.jnt_qposadr[rj]);oq=int(model.jnt_qposadr[oj]);od=int(model.jnt_dofadr[oj]);aj=model.actuator_trnid[:,0].astype(int);aq=model.jnt_qposadr[aj].astype(int);names=[(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(j))or"").split("/")[-1] for j in aj];arm=np.asarray([i for i,n in enumerate(names) if any(k in n for k in ("shoulder","elbow","wrist"))]);armq=aq[arm];armj=aj[arm];lo=model.jnt_range[armj,0]+.006;hi=model.jnt_range[armj,1]-.006
 guards=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f"robot/{s}_hand_front_guard_collision") for s in ("left","right")];hand_geoms=[{mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f"robot/{s}_{x}") for x in ("hand_collision","hand_front_guard_collision","hand_trailing_guard_collision")} for s in ("left","right")];ob=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"object/object");ogs={i for i in range(model.ngeom) if int(model.geom_bodyid[i])==ob};torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"robot/torso_link");feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"robot/{s}_ankle_roll_link") for s in ("left","right")]
 rot=Rotation.from_quat(np.r_[q0[rq+4:rq+7],q0[rq+3]]);fwd=rot.apply([1.,0,0]);fwd[2]=0;fwd/=np.linalg.norm(fwd);lat=rot.apply([0,1.,0]);lat[2]=0;lat/=np.linalg.norm(lat);root=q0[rq:rq+3].copy();offset=u0[arm]-q0[armq];yawq=Rotation.from_euler("z",np.arctan2(fwd[1],fwd[0])).as_quat();object_quat=np.r_[yawq[3],yawq[:3]]
 fk=mujoco.MjData(model)
 def solve(base,obj,spread,height,seed):
  # Use the near side/corners rather than the middle of the deep side faces.
  # This retains opposing lateral normals without requiring the arms to wrap
  # an unnecessary full half-depth around the box.
  near=obj-(BOX_HALF[0]-.055)*fwd
  target=np.asarray([near+spread*lat,near-spread*lat]);target[:,2]=height
  def residual(v):
   fk.qpos[:]=base;fk.qvel[:]=0;fk.qpos[armq]=v;mujoco.mj_forward(model,fk);return np.r_[80*(fk.geom_xpos[guards]-target).ravel(),.025*(v-seed)]
  fit=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=240,ftol=1e-9,xtol=1e-9,gtol=1e-9);fk.qpos[:]=base;fk.qpos[armq]=fit.x;mujoco.mj_forward(model,fk);return fit.x,float(np.max(np.linalg.norm(fk.geom_xpos[guards]-target,axis=1)))
 candidates=[]
 for distance,height in ((.64,.55),):
  obj=root+distance*fwd;obj[2]=BOX_HALF[2]+.001;closed=BOX_HALF[1]+.050;arms,error=solve(q0,obj,closed,height,q0[armq]);pose=q0.copy();pose[armq]=arms;pose[oq:oq+3]=obj;pose[oq+3:oq+7]=object_quat;fk.qpos[:]=pose;mujoco.mj_forward(model,fk);clearance=float(np.linalg.norm(fk.xpos[torso,:2]-obj[:2])-BOX_HALF[0]);candidates.append((error,-clearance,distance,height,obj,arms,clearance));print(json.dumps({"distance_m":distance,"contact_height_m":height,"hand_ik_error_m":error,"horizontal_torso_clearance_proxy_m":clearance}),flush=True)
 candidates.sort(key=lambda x:(x[0]>.025,x[0],x[1]));_,_,distance,height,obj,closed_arms,clearance=candidates[0];closed=BOX_HALF[1]+.020;closed_arms,_=solve(q0,obj,closed,height,closed_arms);opened=closed+.040

 # Rebalance the whole body around the reachable closed-contact posture while
 # keeping the proven feet fixed.  Extended arms move enough link mass forward
 # that simply reusing the upright lower-body targets eventually tips.
 balance_data=mujoco.MjData(model);balance_data.qpos[:]=q0;balance_data.qvel[:]=0;mujoco.mj_forward(model,balance_data)
 foot_pos0=balance_data.xpos[feet].copy();foot_rot0=[balance_data.xmat[b].reshape(3,3).copy() for b in feet]
 root_body=int(model.jnt_bodyid[rj]);com0=balance_data.subtree_com[root_body,:2].copy();torso_up0=balance_data.xmat[torso].reshape(3,3)[:,2].copy();joint0=q0[aq].copy();joint_lo=model.jnt_range[aj,0]+.006;joint_hi=model.jnt_range[aj,1]-.006
 near=obj-(BOX_HALF[0]-.055)*fwd;closed_targets=np.asarray([near+closed*lat,near-closed*lat]);closed_targets[:,2]=height
 def set_balanced(v,object_position):
  balance_data.qpos[:]=q0;balance_data.qvel[:]=0;balance_data.qpos[rq:rq+3]=v[:3];rr=Rotation.from_rotvec(v[3:6])*rot;qq=rr.as_quat();balance_data.qpos[rq+3:rq+7]=np.r_[qq[3],qq[:3]];balance_data.qpos[aq]=v[6:];balance_data.qpos[oq:oq+3]=object_position;balance_data.qpos[oq+3:oq+7]=object_quat;mujoco.mj_forward(model,balance_data)
 def balance_residual(v,object_position,hand_targets):
  set_balanced(v,object_position);foot_orientation=np.concatenate([Rotation.from_matrix(balance_data.xmat[b].reshape(3,3)@foot_rot0[i].T).as_rotvec() for i,b in enumerate(feet)])
  return np.r_[130*(balance_data.xpos[feet]-foot_pos0).ravel(),28*foot_orientation,90*(balance_data.geom_xpos[guards]-hand_targets).ravel(),45*(balance_data.subtree_com[root_body,:2]-com0),24*(balance_data.xmat[torso].reshape(3,3)[:2,2]-torso_up0[:2]),12*(balance_data.qpos[rq+2]-q0[rq+2]),.018*(v[6:]-joint0)]
 seed=np.r_[q0[rq:rq+3],np.zeros(3),joint0];seed[6+arm]=closed_arms
 lower=np.r_[q0[rq:rq+2]-.18,.56,[-.30,-.30,-.25],joint_lo];upper=np.r_[q0[rq:rq+2]+.18,.72,[.30,.30,.25],joint_hi]
 fit=least_squares(lambda v:balance_residual(v,obj,closed_targets),np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=500,ftol=2e-9,xtol=2e-9,gtol=2e-9);set_balanced(fit.x,obj);closed_pose=balance_data.qpos.copy();closed_arms=closed_pose[armq].copy();balanced_hand_error=float(np.max(np.linalg.norm(balance_data.geom_xpos[guards]-closed_targets,axis=1)));balanced_foot_error=float(np.max(np.linalg.norm(balance_data.xpos[feet]-foot_pos0,axis=1)));balanced_com_error=float(np.linalg.norm(balance_data.subtree_com[root_body,:2]-com0))
 rd=int(model.jnt_dofadr[rj]);ad=model.jnt_dofadr[aj].astype(int);seed_control=np.clip(closed_pose[aq]+u0-q0[aq],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
 def equilibrium_control(pose,seed_u):
  def acceleration(control):
   balance_data.qpos[:]=pose;balance_data.qvel[:]=0;balance_data.ctrl[:]=control;mujoco.mj_forward(model,balance_data);return np.r_[balance_data.qacc[rd:rd+6],.10*balance_data.qacc[ad],.001*(control-seed_u)]
  equilibrium=least_squares(acceleration,seed_u,bounds=(model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]),max_nfev=220,ftol=1e-8,xtol=1e-8,gtol=1e-8)
  return equilibrium.x.copy(),float(np.linalg.norm(acceleration(equilibrium.x)[:-model.nu]))
 base_control,equilibrium_residual=equilibrium_control(closed_pose,seed_control);offset=base_control[arm]-closed_pose[armq]

 # Rebalance again at the carried height.  This plans the arm-mass/support
 # transition before lift, rather than holding the ground-contact equilibrium
 # until it tips.  The object remains fully dynamic in the rollout.
 lift_height=.18;lifted_obj=obj.copy();lifted_obj[2]+=lift_height;lifted_targets=closed_targets.copy();lifted_targets[:,2]+=lift_height
 lifted_fit=least_squares(lambda v:balance_residual(v,lifted_obj,lifted_targets),fit.x,bounds=(lower,upper),max_nfev=500,ftol=2e-9,xtol=2e-9,gtol=2e-9);set_balanced(lifted_fit.x,lifted_obj);lifted_pose=balance_data.qpos.copy();lifted_hand_error=float(np.max(np.linalg.norm(balance_data.geom_xpos[guards]-lifted_targets,axis=1)));lifted_foot_error=float(np.max(np.linalg.norm(balance_data.xpos[feet]-foot_pos0,axis=1)));lifted_com_error=float(np.linalg.norm(balance_data.subtree_com[root_body,:2]-com0));lifted_seed=np.clip(lifted_pose[aq]+base_control-closed_pose[aq],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]);lifted_control,lifted_equilibrium_residual=equilibrium_control(lifted_pose,lifted_seed)
 # Execute from the solved whole-body equilibrium.  At the legal stand-off
 # selected above its acceleration residual is small; replacing it with the
 # old upright control discards the support compensation for the extended arms.
 open_arms,open_error=solve(closed_pose,obj,opened,height,closed_arms);initial=closed_pose.copy();initial[armq]=open_arms;initial[oq:oq+3]=obj;initial[oq+3:oq+7]=object_quat
 controls=[];seed=open_arms.copy()
 for _ in range(6):
  u=base_control.copy();u[arm]=seed+offset;controls.extend([np.clip(u,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])]*SUBSTEPS)
 for frame in range(15):
  a=smoothstep((frame+1)/15);spread=(1-a)*opened+a*closed;seed,_=solve(closed_pose,obj,spread,height,seed);u=base_control.copy();u[arm]=seed+offset;controls.extend([np.clip(u,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])]*SUBSTEPS)
 lifted_offset=lifted_control[arm]-lifted_pose[armq]
 for frame in range(35):
  a=smoothstep((frame+1)/35);reference=closed_pose.copy();reference[rq:rq+3]=(1-a)*closed_pose[rq:rq+3]+a*lifted_pose[rq:rq+3];reference[aq]=(1-a)*closed_pose[aq]+a*lifted_pose[aq];reference[rq+3:rq+7]=(1-a)*closed_pose[rq+3:rq+7]+a*lifted_pose[rq+3:rq+7];reference[rq+3:rq+7]/=np.linalg.norm(reference[rq+3:rq+7]);desired=obj.copy();desired[2]+=a*lift_height;seed,_=solve(reference,desired,closed,height+a*lift_height,seed);u=(1-a)*base_control+a*lifted_control;u[arm]=seed+(1-a)*offset+a*lifted_offset;controls.extend([np.clip(u,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])]*SUBSTEPS)
 controls.extend([controls[-1].copy()]*(45*SUBSTEPS));controls=np.asarray(controls)
 d=mujoco.MjData(model);d.qpos[:]=initial;d.qvel[:]=0;d.qvel[od:od+6]=0;mujoco.mj_forward(model,d);initial_obj=d.qpos[oq:oq+3].copy();initial_feet=d.xpos[feet].copy();nonhand=0.;maxlift=0.;run=longest=0;bilateral=0;first_invalid=None;tr={k:[] for k in ("qpos","qvel","ctrl","object_pos","object_quat","hand_contact","foot_pos","root_height","torso_tilt","nonhand_impulse")}
 for step,u in enumerate(controls):
  d.ctrl[:]=u;mujoco.mj_step(model,d);hand=np.zeros(2,dtype=bool)
  for ci in range(d.ncon):
   pair={int(d.contact[ci].geom1),int(d.contact[ci].geom2)}
   if not pair&ogs:continue
   other=next((g for g in pair if g not in ogs),None);matched=False
   for s,gg in enumerate(hand_geoms):
    if other in gg:hand[s]=True;matched=True
   if not matched and other is not None:
    body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[other]))or""
    if body.startswith("robot/"):force=np.zeros(6);mujoco.mj_contactForce(model,d,ci,force);nonhand+=float(np.linalg.norm(force[:3])*DT)
  if step%SUBSTEPS!=SUBSTEPS-1:continue
  frame=step//SUBSTEPS;lift=float(d.qpos[oq+2]-initial_obj[2]);maxlift=max(maxlift,lift);bilateral+=int(hand.all());supported=bool(hand.all() and lift>.05);run=run+1 if supported else 0;longest=max(longest,run);upv=d.xmat[torso].reshape(3,3)[:,2];tilt=float(np.degrees(np.arccos(np.clip(upv[2],-1,1))));rootz=float(d.qpos[rq+2]);
  if first_invalid is None and (rootz<.50 or tilt>45):first_invalid=[frame,"root_height" if rootz<.50 else "torso_tilt"]
  tr["qpos"].append(d.qpos.copy());tr["qvel"].append(d.qvel.copy());tr["ctrl"].append(u.copy());tr["object_pos"].append(d.qpos[oq:oq+3].copy());tr["object_quat"].append(d.qpos[oq+3:oq+7].copy());tr["hand_contact"].append(hand.copy());tr["foot_pos"].append(d.xpos[feet].copy());tr["root_height"].append(rootz);tr["torso_tilt"].append(tilt);tr["nonhand_impulse"].append(nonhand)
 metrics={"geometry_full_size_m":[2*x for x in BOX_HALF],"mass_kg":BOX_MASS,"selected_distance_m":distance,"contact_height_m":height,"initial_object_position":initial_obj.tolist(),"initial_object_quaternion_wxyz":object_quat.tolist(),"object_initially_on_terrain":bool(abs(initial_obj[2]-BOX_HALF[2]-.001)<.003),"closed_hand_ik_error_m":float(candidates[0][0]),"balanced_hand_error_m":balanced_hand_error,"balanced_foot_error_m":balanced_foot_error,"balanced_com_error_m":balanced_com_error,"equilibrium_acceleration_residual":equilibrium_residual,"lifted_balanced_hand_error_m":lifted_hand_error,"lifted_balanced_foot_error_m":lifted_foot_error,"lifted_balanced_com_error_m":lifted_com_error,"lifted_equilibrium_acceleration_residual":lifted_equilibrium_residual,"open_hand_ik_error_m":open_error,"horizontal_torso_clearance_proxy_m":clearance,"full_valid":first_invalid is None,"first_invalid":first_invalid,"maximum_lift_m":maxlift,"longest_bilateral_supported_lift_frames":longest,"bilateral_contact_frames":bilateral,"nonhand_object_impulse_ns":nonhand,"minimum_root_height_m":float(np.min(tr["root_height"])),"maximum_torso_tilt_deg":float(np.max(tr["torso_tilt"])),"foot_translation_m":np.linalg.norm(d.xpos[feet,:2]-initial_feet[:,:2],axis=1).tolist(),"final_object_position":d.qpos[oq:oq+3].tolist()};np.savez_compressed(OUT/"ground_large_pickup_gate_v1.npz",initial_qpos=initial,initial_qvel=np.zeros(model.nv),ctrl_physics=controls,goal_position=np.asarray(metrics["final_object_position"]),goal_quaternion=object_quat,**{k:np.asarray(v) for k,v in tr.items()});(OUT/"ground_large_pickup_gate_v1.json").write_text(json.dumps(metrics,indent=2)+"\n");print("RESULT",json.dumps(metrics),flush=True)
if __name__=="__main__":main()
