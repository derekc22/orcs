"""Demonstrated gait followed by a smooth return to its two-foot standing pose."""
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from active_largebox_config import OUT, ROOT, INPUTS
from active_largebox_artifacts import materialize
from build_engineered_ground_large_model import large_ground_gripper_spec
from run_engineered_locomomanipulation import _quat_delta_omega

def write(arm_reference=None):
 source=np.load(INPUTS/'original_largebox_vertical_rail_motion_v1.npz');m=large_ground_gripper_spec().compile();j=np.array(sorted([j for j in range(m.njnt) if m.jnt_type[j]==3],key=lambda j:m.jnt_qposadr[j]));jq=m.jnt_qposadr[j]
 q=np.zeros((220,m.nq));q[:,:3]=source['body_pos_w'][:,0];q[:,3:7]=source['body_quat_w'][:,0];q[:,jq]=source['joint_pos'];end=q[-1].copy();stand=q[100].copy();velocity=source['body_lin_vel_w'][-1,0,:2].astype(float);stand[:2]=end[:2]+velocity*.6;sr=Rotation.from_quat(np.roll(stand[3:7],-1));er=Rotation.from_quat(np.roll(end[3:7],-1));yaw=er.as_euler('xyz')[2]-sr.as_euler('xyz')[2];stand[3:7]=np.roll((Rotation.from_euler('z',yaw)*sr).as_quat(),1);tangent=np.zeros(m.nv);mujoco.mj_differentiatePos(m,tangent,1,end,stand);poses=list(q)
 for f in range(60):
  alpha=(f+1)/60;alpha=alpha*alpha*(3-2*alpha);p=end.copy();mujoco.mj_integratePos(m,p,tangent,alpha);u=(f+1)/60;p[:2]=end[:2]+velocity*1.2*(u-.5*u*u);poses.append(p)
 poses.extend([stand.copy() for _ in range(800)]);poses=np.array(poses)
 if arm_reference is not None:
  left=[int(m.jnt_qposadr[j]) for j in range(m.njnt) if 'left' in m.joint(j).name and any(k in m.joint(j).name for k in ['shoulder','elbow','wrist'])]
  # Match actuator order rather than tree order (the wrist pitch/yaw actuators occur last).
  joint_names=[('left_'+k+'_joint') for k in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
  left=[int(m.jnt_qposadr[m.joint(n).id]) for n in joint_names];poses[:,left]=arm_reference['left']
  realized=np.load(INPUTS/'controlled_release1.npz');physical=mujoco.MjModel.from_binary_path(str(materialize(INPUTS/'controlled_release1_model.mjb')))
  for j in range(m.njnt):
   n=m.joint(j).name
   if 'right' not in n or not any(k in n for k in ['shoulder','elbow','wrist']):continue
   source_j=physical.joint(n).id;index=int(physical.jnt_qposadr[source_j]);target=int(m.jnt_qposadr[j])
   for f in range(len(poses)):poses[f,target]=realized['initial_qpos'][index] if f==0 else realized['qpos'][min(f*4-1,len(realized['qpos'])-1),index]
 d=mujoco.MjData(m);pos=[];quat=[]
 for p in poses:d.qpos[:]=p;mujoco.mj_forward(m,d);pos.append(d.xpos[1:].copy());quat.append(d.xquat[1:].copy())
 pos=np.array(pos);quat=np.array(quat);arrays=dict(joint_pos=poses[:,jq].astype('float32'),joint_vel=np.gradient(poses[:,jq],.02,axis=0).astype('float32'),body_pos_w=pos.astype('float32'),body_quat_w=quat.astype('float32'),body_lin_vel_w=np.gradient(pos,.02,axis=0).astype('float32'),body_ang_vel_w=_quat_delta_omega(quat,.02).astype('float32'),fps=np.array(50.))
 for key in arrays:
  if key!='fps' and arm_reference is None:arrays[key][:220]=source[key]
 path=OUT/'motion.npz';np.savez_compressed(path,**arrays);np.save(OUT/'reference_qpos.npy',poses);return path
