"""Extended native walking reference; no new robot morphology or policy training."""
from pathlib import Path
import json
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from active_largebox_config import ROOT
OUT=ROOT/'logs/compact_integration_v2'

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def build_reference(physical,qref,*,start=5485,stop=5630,speed=.80,cycles=2):
 from mocke import default_motion_file
 from build_engineered_ground_large_model import large_ground_gripper_spec
 from run_engineered_locomomanipulation import _quat_delta_omega
 a=np.load(default_motion_file());native=large_ground_gripper_spec().compile();j=np.array(sorted([i for i in range(native.njnt) if native.jnt_type[i]==3],key=lambda i:native.jnt_qposadr[i]));jq=native.jnt_qposadr[j];qstart=native.qpos0.copy();qstart[:7]=qref[:7]
 for jj in j:qstart[native.jnt_qposadr[jj]]=qref[physical.jnt_qposadr[physical.joint(native.joint(int(jj)).name).id]]
 ids=np.arange(start,stop+1);raw=np.zeros((len(ids),native.nq));raw[:,:3]=a['body_pos_w'][ids,0];raw[:,3:7]=a['body_quat_w'][ids,0];raw[:,jq]=a['joint_pos'][ids]
 source_yaw=Rotation.from_quat(np.roll(raw[0,3:7],-1)).as_euler('xyz')[2];target_yaw=Rotation.from_quat(np.roll(qstart[3:7],-1)).as_euler('xyz')[2];align=Rotation.from_euler('z',target_yaw-source_yaw);origin=raw[0,:3].copy();raw[:,:3]=align.apply(raw[:,:3]-origin)+qstart[:3];raw[:,2]+=origin[2]-qstart[2]
 for q in raw:q[3:7]=np.roll((align*Rotation.from_quat(np.roll(q[3:7],-1))).as_quat(),1)
 # The same mechanically matched native cycle used by the existing hook/corner
 # motion builder. Repeat it only to meet the measured 2 m carry requirement.
 ia=5535-start;ib=5589-start;A=raw[ia].copy();B=raw[ib].copy()
 ya=Rotation.from_quat(np.roll(A[3:7],-1)).as_euler('xyz')[2];yb=Rotation.from_quat(np.roll(B[3:7],-1)).as_euler('xyz')[2];cycleR=Rotation.from_euler('z',yb-ya)
 def advance(q):
  q=q.copy();q[:3]=cycleR.apply(q[:3]-A[:3])+B[:3];q[3:7]=np.roll((cycleR*Rotation.from_quat(np.roll(q[3:7],-1))).as_quat(),1);return q
 before=raw[:ib+1];chunk=raw[ia+1:ib+1].copy();tail=raw[ib+1:].copy();parts=[before]
 for _ in range(cycles):
  chunk=np.array([advance(q) for q in chunk]);tail=np.array([advance(q) for q in tail]);parts.append(chunk.copy())
 raw=np.concatenate(parts+[tail])
 # Constant carried-arm references; all joints in the real model remain actuated.
 armj=[jj for jj in j if any(t in native.joint(int(jj)).name for t in ['shoulder','elbow','wrist'])];armq=native.jnt_qposadr[armj];raw[:,armq]=qstart[armq]
 # Slow the native walking clip through proper position/quaternion interpolation.
 count=int(np.ceil((len(raw)-1)/speed))+1;walk=[]
 for t in np.linspace(0,len(raw)-1,count):
  i=min(int(t),len(raw)-2);alpha=t-i;dv=np.zeros(native.nv);mujoco.mj_differentiatePos(native,dv,1.,raw[i],raw[i+1]);q=raw[i].copy();mujoco.mj_integratePos(native,q,dv,alpha);walk.append(q)
 walk=np.array(walk);poses=[qstart.copy() for _ in range(10)];tangent=np.zeros(native.nv);mujoco.mj_differentiatePos(native,tangent,1.,qstart,walk[0])
 for f in range(30):
  q=qstart.copy();mujoco.mj_integratePos(native,q,tangent,smooth((f+1)/30));poses.append(q)
 walk_start=len(poses);poses.extend(walk);walk_end=len(poses);end=walk[-1].copy();stand=qstart.copy();er=Rotation.from_quat(np.roll(end[3:7],-1));sr=Rotation.from_quat(np.roll(stand[3:7],-1));delta_yaw=er.as_euler('xyz')[2]-sr.as_euler('xyz')[2];stand[3:7]=np.roll((Rotation.from_euler('z',delta_yaw)*sr).as_quat(),1);velocity=(walk[-1,:2]-walk[-5,:2])/.08;stand[:2]=end[:2]+velocity*.6;mujoco.mj_differentiatePos(native,tangent,1.,end,stand)
 for f in range(60):
  q=end.copy();u=(f+1)/60;mujoco.mj_integratePos(native,q,tangent,smooth(u));q[:2]=end[:2]+velocity*1.2*(u-.5*u*u);poses.append(q)
 braking_end=len(poses);poses.extend([stand.copy() for _ in range(150)]);poses=np.array(poses);d=mujoco.MjData(native);pos=[];quat=[]
 for q in poses:d.qpos[:]=q;mujoco.mj_forward(native,d);pos.append(d.xpos[1:].copy());quat.append(d.xquat[1:].copy())
 pos=np.array(pos);quat=np.array(quat);arrays=dict(joint_pos=poses[:,jq].astype('float32'),joint_vel=np.gradient(poses[:,jq],.02,axis=0).astype('float32'),body_pos_w=pos.astype('float32'),body_quat_w=quat.astype('float32'),body_lin_vel_w=np.gradient(pos,.02,axis=0).astype('float32'),body_ang_vel_w=_quat_delta_omega(quat,.02).astype('float32'),fps=np.array(50.))
 path=OUT/'walking_reference.npz';np.savez_compressed(path,**arrays);np.save(OUT/'walking_reference_qpos.npy',poses);manifest=dict(source='existing native SONIC demo',source_frames=[start,stop],speed=speed,walk_start_frame=walk_start,walk_end_frame=walk_end,braking_end_frame=braking_end,total_frames=len(poses),reference_root_walk_displacement_m=float(np.linalg.norm(walk[-1,:2]-walk[0,:2])),reference_root_final_displacement_m=float(np.linalg.norm(stand[:2]-qstart[:2])));(OUT/'walking_reference_manifest.json').write_text(json.dumps(manifest,indent=2));print('WALK_REFERENCE',manifest,flush=True);return path,manifest

def controller(m,qref,**kwargs):
 import active_largebox_motion as motion
 import active_largebox_sonic as sonic
 path,meta=build_reference(m,qref,**kwargs)
 motion.write=lambda arm_reference=None:path
 aj=m.actuator_trnid[:,0];left=[i for i,j in enumerate(aj) if 'left' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])]
 return sonic.Controller(m,arm_reference={'left':qref[m.jnt_qposadr[aj[left]]].copy()}),meta
