#!/usr/bin/env python3
"""Independent physics-rate replay, disjoint contact attribution, video and task gates."""
import argparse,json,collections
from pathlib import Path
import mujoco,numpy as np
from active_largebox_artifacts import materialize

def longest(mask):
 edges=np.diff(np.r_[False,mask,False].astype(int));return int(np.max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1),initial=0))
def verify_physics(path,video=False):
 a=np.load(path);m=mujoco.MjModel.from_binary_path(str(materialize(path.with_name(path.stem+'_model.mjb'))));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
 oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];od=m.jnt_dofadr[oj];ob=m.jnt_bodyid[oj];tor=m.body('torso_link').id;feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];initial_obj=d.xipos[ob].copy();feet0=d.xpos[feet].copy();initial_quat=d.qpos[oq+3:oq+7].copy()
 sums=collections.defaultdict(float);first={};trace=[];extra=[];fk=mujoco.MjData(m);foot_geoms=[[g for g in range(m.ngeom) if m.geom(g).name.startswith(side+'_foot') and m.geom_contype[g]] for side in ['left','right']];qp=0.;qv=0.;renderer=None;max_actuator_force=np.zeros(m.nu);max_external_force=0.
 if video:
  import imageio.v2 as imageio
  from PIL import Image,ImageDraw,ImageFont
  m.vis.global_.offwidth=960;m.vis.global_.offheight=720;renderer=mujoco.Renderer(m,height=720,width=960);font=ImageFont.load_default(size=18);camera=mujoco.MjvCamera();camera.distance=3;camera.azimuth=130;camera.elevation=-20;writer=imageio.get_writer(str(path.with_suffix('.mp4')),fps=50,codec='libx264',quality=7,macro_block_size=None);montage=[]
 for step,u in enumerate(a['ctrl_physics']):
  foot_load=np.zeros(2);jaw_force=np.zeros(2)
  d.ctrl[:]=u;mujoco.mj_step(m,d);max_actuator_force=np.maximum(max_actuator_force,np.abs(d.actuator_force));max_external_force=max(max_external_force,float(np.max(np.abs(d.xfrc_applied))),float(np.max(np.abs(d.qfrc_applied))));qp=max(qp,float(np.max(np.abs(d.qpos-a['qpos'][step]))));qv=max(qv,float(np.max(np.abs(d.qvel-a['qvel'][step]))));ground=False;hands=np.zeros(2,bool);support=np.zeros(2,bool);invalid_terrain=0.;forbidden=0.;vertical=0.;terrain_vertical=0.;ee_any=False
  for ci,c in enumerate(d.contact):
   g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2]);n1,n2=m.geom(g1).name,m.geom(g2).name;force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);imp=float(np.linalg.norm(force[:3])*m.opt.timestep)
   if b1==ob or b2==ob:
    other=g2 if b1==ob else g1;n=m.geom(other).name;sums[n]+=imp
    if imp>1e-7 and n not in first:first[n]=step
    if n.startswith('clamp_'):
     token=n.rsplit('clamp_',1)[1];digits=''.join(c for c in token if c.isdigit())
     if digits:jaw_force[0 if int(digits)<3 else 1]+=np.linalg.norm(force[:3])
    if n=='terrain':
     ground=True;worldforce=c.frame.reshape(3,3).T@force[:3];terrain_vertical+=(worldforce[2] if b2==ob else -worldforce[2])
    elif n.startswith('clamp_') or any(t in n for t in ['_pad','_strut','_mount','hand_collision']):
     ee_any=True
     if n.startswith('clamp_pad'):hands[0 if int(n[9:])<3 else 1]=True
     elif not n.startswith('clamp_'):hands[0 if n.startswith('left') else 1]=True
     worldforce=c.frame.reshape(3,3).T@force[:3];vertical+=(worldforce[2] if b2==ob else -worldforce[2])
    else:forbidden+=imp
   if b1>0 and b2>0 and b1!=ob and b2!=ob and imp>1e-6:
    sums['ROBOT_SELF:'+':'.join(sorted([n1,n2]))]+=imp
   if n1=='terrain' or n2=='terrain':
    other=g2 if n1=='terrain' else g1;n=m.geom(other).name
    if m.geom_bodyid[other]==ob:continue
    if 'foot' in n:
     side=0 if n.startswith('left') else 1;support[side]=True;foot_load[side]+=abs((c.frame.reshape(3,3).T@force[:3])[2])
    elif imp>1e-6:invalid_terrain+=imp;sums['INVALID_TERRAIN:'+n]+=imp
  rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,d.qpos[oq+3:oq+7]);rotation=rotation.reshape(3,3);offset=rotation@m.body_ipos[ob];com=d.qpos[oq:oq+3]+offset;omega=rotation@d.qvel[od+3:od+6];com_velocity=d.qvel[od:od+3]+np.cross(omega,offset)
  tilt=float(np.degrees(np.arccos(np.clip(d.xmat[tor].reshape(3,3)[2,2],-1,1))))
  trace.append([ground,*hands,*support,com[2]-initial_obj[2],d.qpos[2],tilt,forbidden,invalid_terrain,vertical,*com,*d.xpos[feet].ravel(),*com_velocity,*omega,*d.qpos[m.jnt_qposadr[m.actuator_trnid[29:35,0]]],*d.qpos[oq+3:oq+7],*d.qvel[:6],terrain_vertical,ee_any])
  fk.qpos[:]=d.qpos;mujoco.mj_kinematics(m,fk);sole=np.zeros(2)
  for side,gs in enumerate(foot_geoms):
   lows=[]
   for g in gs:
    typ=int(m.geom_type[g]);size=m.geom_size[g];R=fk.geom_xmat[g].reshape(3,3);p=fk.geom_xpos[g]
    dz=size[0] if typ==2 else size[0]+size[1]*abs(R[2,2]) if typ==3 else np.abs(R[2])@size
    lows.append(p[2]-dz)
   sole[side]=min(lows)
  extra.append(np.r_[d.qpos[:3],d.qvel[:3],fk.xpos[feet].ravel(),sole,foot_load,jaw_force])
  if video and step%4==3:
   camera.lookat[:]=.5*d.qpos[:3]+.5*d.qpos[oq:oq+3];camera.lookat[2]=.6;renderer.update_scene(d,camera=camera)
   # Ground grid is a rendering decoration, never a physics geom or support.
   for axis in [0,1]:
    for coordinate in np.arange(-3.0,4.51,.25) if axis==0 else np.arange(1.0,5.01,.25):
     g=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
     mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.35,.35,.35,1.]))
     p1=np.array([coordinate,1.,.001]) if axis==0 else np.array([-3.,coordinate,.001]);p2=np.array([coordinate,5.,.001]) if axis==0 else np.array([4.5,coordinate,.001]);mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.,p1,p2)
   im=Image.fromarray(renderer.render().copy());dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,62],fill='black');dr.text((10,8),f'{path.stem} frame {step//4}  terrain {ground}  L/R {hands.tolist()}  lift {com[2]-initial_obj[2]:+.3f} m',fill='white',font=font);dr.text((10,34),f'root {d.qpos[2]:.3f} m tilt {tilt:.1f} deg | finite physical servos, exact replay',fill='white',font=font);writer.append_data(np.array(im))
   if step//4 in np.linspace(0,len(a['ctrl_physics'])//4-1,16).astype(int):montage.append(im.resize((400,300)))
 if video:
  writer.close();renderer.close();sheet=Image.new('RGB',(1600,1200))
  for i,im in enumerate(montage):sheet.paste(im,((i%4)*400,(i//4)*300))
  sheet.save(path.with_name(path.stem+'_montage.png'))
 np.savez_compressed(path.with_name(path.stem+'_locomotion_trace.npz'),trace=np.array(extra),schema=np.array(['root_xyz:0:3','root_vxyz:3:6','feet_xyz:6:12','sole_clearance_m:12:14','foot_normal_force_N:14:16','jaw_force_norm_N:16:18']))
 tr=np.array(trace);N=len(tr)//4;task=tr[:N*4].reshape(N,4,-1);air=~task[:,:,0].any(1);ee=task[:,:,1:3].any(1).all(1);lift=task[:,:,5].min(1);valid=(task[:,:,6].min(1)>.45)&(task[:,:,7].max(1)<45)&(task[:,:,9].sum(1)<1e-3)
 supported=air&ee&(lift>.05);steps=np.flatnonzero(supported);transport=float(np.max(np.linalg.norm(tr[steps*4,11:13]-tr[steps[0]*4,11:13],axis=1),initial=0)) if len(steps) else 0.
 result={'artifact':str(path),'exact_qpos_error':qp,'exact_qvel_error':qv,'maximum_lift_m':float(tr[:,5].max()),'terrain_free_task_frames':int(air.sum()),'longest_free_bilateral_lift_task_frames':longest(supported),'terrain_free_supported_transport_m':transport,'forbidden_object_impulse_Ns':float(tr[:,8].sum()),'invalid_robot_terrain_impulse_Ns':float(tr[:,9].sum()),'minimum_root_m':float(tr[:,6].min()),'maximum_torso_tilt_deg':float(tr[:,7].max()),'full_valid':bool(valid.all()),'final_object_linear_speed_mps':float(np.linalg.norm(d.qvel[od:od+3])),'final_object_angular_speed_radps':float(np.linalg.norm(d.qvel[od+3:od+6])),'final_robot_root_speed':float(np.linalg.norm(d.qvel[:6])),'object_angle_from_initial_rad':float(2*np.arccos(np.clip(abs(np.dot(initial_quat,d.qpos[oq+3:oq+7])),0,1))),'foot_relocation_m':np.linalg.norm(d.xpos[feet,:2]-feet0[:,:2],axis=1).tolist(),'contact_impulses_Ns':dict(sums),'first_contact_physics_step':first,'maximum_abs_actuator_force':max_actuator_force.tolist(),'maximum_applied_external_force':max_external_force,'equality_constraints':int(m.neq),'physics_timestep_s':float(m.opt.timestep),'disabled_physics_flags':int(m.opt.disableflags),'success':False}
 np.savez_compressed(path.with_name(path.stem+'_verification_trace.npz'),trace=tr,task_supported=supported,task_air=air,task_valid=valid);path.with_name(path.stem+'_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);args=p.parse_args();verify_physics(args.artifact)
 from audit_active_largebox import run as audit
 audit(args.artifact)
 from sustained_carry_analysis import analyze
 result=analyze(args.artifact)
 raise SystemExit(0 if result['all_machine_gates_pass'] else 1)
