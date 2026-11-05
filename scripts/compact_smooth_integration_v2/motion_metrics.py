"""Physical motion diagnostics; filtering is diagnostic only, never state/video edits."""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.signal import butter,sosfiltfilt,welch
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[2]
DT=.005

def rms(x):return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
def stats(x):
 x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
 hp=sosfiltfilt(butter(3,6,fs=200,btype='highpass',output='sos'),x,axis=0)
 f,p=welch(x,fs=200,nperseg=min(512,len(x)),axis=0);p=p.sum(1)
 return dict(rms=rms(x),peak=float(np.linalg.norm(x,axis=1).max()),above_6Hz_rms=rms(hp),dominant_Hz=float(f[1:][p[1:].argmax()]))
def measure(path,out):
 a=np.load(path);m=mujoco.MjModel.from_binary_path(str(path.with_name(path.stem+'_model.mjb')));q=a['qpos'];v=a['qvel'];u=a['ctrl_physics'];N=len(q)
 oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];od=m.jnt_dofadr[oj];ob=m.jnt_bodyid[oj];aq=m.jnt_qposadr[m.actuator_trnid[:,0]];av=m.jnt_dofadr[m.actuator_trnid[:,0]]
 arm=np.array([i for i in range(29) if any(t in m.actuator(i).name for t in ['shoulder','elbow','wrist'])]+[31,32]);lower=np.array([i for i in range(29) if i not in arm])
 objrot=R.from_quat(np.roll(q[:,oq+3:oq+7],-1,axis=1));omega=objrot.apply(v[:,od+3:od+6]);rootw=R.from_quat(np.roll(q[:,3:7],-1,axis=1)).apply(v[:,3:6]);tau=m.actuator_gainprm[:,0]*u+m.actuator_biasprm[:,1]*q[:,aq]+m.actuator_biasprm[:,2]*v[:,av];tau=np.clip(tau,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
 d=mujoco.MjData(m);rel=[]
 plates=[m.body('hand_'+s+'_plate').id for s in ['left','right']]
 for qi in q:
  d.qpos[:]=qi;mujoco.mj_kinematics(m,d);row=[]
  for b in plates:
   H=d.xmat[b].reshape(3,3);row.extend(H.T@(d.xpos[ob]-d.xpos[b]));row.extend(R.from_matrix(H.T@d.xmat[ob].reshape(3,3)).as_quat())
  rel.append(row)
 rel=np.array(rel);relw=[]
 for i in range(2):
  rr=R.from_quat(rel[:,7*i+3:7*i+7]);ww=(rr[1:]*rr[:-1].inv()).as_rotvec()/DT;relw.append(np.vstack([ww[0],ww]))
 arrays=dict(object_omega=omega,object_alpha=np.gradient(omega,DT,axis=0),root_omega=rootw,root_alpha=np.gradient(rootw,DT,axis=0),plate_object_pose=rel,relative_omega=np.hstack(relw),actuator_torque=tau)
 s=int(a['walking_start_physics_step']);e=min(N,int(a['walking_end_physics_step']));ps=int(a['placement_start_physics_step']) if 'placement_start_physics_step' in a else 6501;rec=int(a['recovery_start_physics_step']) if 'recovery_start_physics_step' in a else N
 phases={'pickup':[0,624],'loaded_preparation':[624,min(3289,N)],'walking':[s,e],'braking':[e,min(5901,N)],'entry_transition':[5901,min(ps,N)],'placement':[ps,min(rec,N)],'recovery':[rec,N],'complete_execution':[0,N]}
 result=dict(artifact=str(path),duration_s=N*DT,diagnostic_method='200 Hz raw physical states; third-order zero-phase 6 Hz high-pass used only to quantify vibration above natural gait. Raw RMS/peak retained; controls, states and videos unfiltered.',phases={})
 tracepath=path.with_name(path.stem+'_locomotion_trace.npz');trace=np.load(tracepath)['trace'] if tracepath.exists() else None
 for phase,(b,c) in phases.items():
  if c-b<50:continue
  z={k:stats(x[b:c]) for k,x in arrays.items() if k not in ['plate_object_pose','actuator_torque']}
  z['arm_joint_speed']=stats(v[b:c,av[arm]]);z['lower_joint_speed']=stats(v[b:c,av[lower]]);z['arm_torque_rate']=stats(np.gradient(tau[b:c,arm],DT,axis=0));z['lower_command_rate']=stats(np.gradient(u[b:c,lower],DT,axis=0));z['press_torque']=stats(tau[b:c,29:31]-tau[b:c,29:31].mean(0));z['press_torque_mean_N']=tau[b:c,29:31].mean(0).tolist()
  if trace is not None:
   force=trace[b:c,16:18];z['hand_force_variation']=stats(force-force.mean(0));z['hand_force_mean_N']=force.mean(0).tolist();z['hand_force_min_N']=force.min(0).tolist()
  result['phases'][phase]=z
 out.mkdir(exist_ok=True,parents=True);(out/'motion_metrics.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'motion_trace.npz',**arrays)
 print(json.dumps({k:{x:round(y['rms'],4) for x,y in z.items() if isinstance(y,dict)} for k,z in result['phases'].items()},indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();measure(a.artifact,a.out)
