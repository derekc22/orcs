"""State-feedback SONIC lower-body controller using a native-joint observation adapter.
The shadow simulation supplies observations only; the custom model is physics authority.
"""
import numpy as np
import mujoco,torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import SonicBaseModel
from mocke import PRETRAINED_DIR
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from run_engineered_locomomanipulation import engineering_robot_cfg
from build_engineered_ground_large_model import large_ground_gripper_spec
from build_engineered_ground_original_largebox_model import original_largebox_spec
from active_largebox_config import INPUTS
from active_largebox_artifacts import materialize
class Controller:
 def __init__(self,m,arm_reference=None):
  from active_largebox_motion import write
  cfg=sonic_tracking_env_cfg(str(write(arm_reference)),play=True,il_ordered=False);cfg.scene.num_envs=1
  cfg.scene.entities['robot']=engineering_robot_cfg();cfg.scene.entities['robot'].spec_fn=large_ground_gripper_spec
  cfg.scene.entities['object']=EntityCfg(spec_fn=original_largebox_spec)
  cfg.sim.nconmax=256;cfg.sim.njmax=1000;cfg.auto_reset=False;cfg.terminations={};cfg.events={n:t for n,t in cfg.events.items() if t.mode!='startup'}
  self.raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');self.env=RslRlVecEnvWrapper(self.raw,clip_actions=None)
  obs=self.env.get_observations();self.policy=SonicBaseModel(obs=obs,obs_groups={'actor':['policy']},obs_set='actor',output_dim=29,base_checkpoint=str(materialize(INPUTS/'sonic_last_ported.pt'))).to(self.raw.device);self.policy.eval()
  self.term=self.raw.action_manager.get_term('joint_pos');self.shadow=self.raw.sim.mj_model
  self.indices=[];self.mapping=[]
  names=tuple(self.raw.scene['robot'].joint_names)
  for j in range(self.shadow.njnt):
   n=self.shadow.joint(j).name; nn=n.split('/')[-1];nn=('object/' if n.startswith('object/') else '')+nn;nj=m.joint(nn).id
   self.mapping.append((int(self.shadow.jnt_qposadr[j]),int(m.jnt_qposadr[nj]),7 if m.jnt_type[nj]==0 else 1,int(self.shadow.jnt_dofadr[j]),int(m.jnt_dofadr[nj]),6 if m.jnt_type[nj]==0 else 1))
  self.overridden=np.array([i for i,j in enumerate(self.shadow.actuator_trnid[:,0]) if any(t in self.shadow.joint(int(j)).name for t in ['shoulder','elbow','wrist'])])
  self.indices=np.array([names.index(self.shadow.joint(int(j)).name.split('/')[-1]) for j in self.shadow.actuator_trnid[:,0]])
 def control(self,d):
  r=self.raw;qp=np.zeros(self.shadow.nq);qv=np.zeros(self.shadow.nv)
  for sq,cq,k,sv,cv,l in self.mapping:qp[sq:sq+k]=d.qpos[cq:cq+k];qv[sv:sv+l]=d.qvel[cv:cv+l]
  r.sim.data.qpos[0]=torch.as_tensor(qp,device=r.device);r.sim.data.qvel[0]=torch.as_tensor(qv,device=r.device)
  r.sim.forward();r.scene.update(dt=r.physics_dt);r.sim.sense();r.observation_manager._obs_buffer=None
  obs=self.env.get_observations()
  with torch.inference_mode():
   action=self.policy(obs);bias=self.term._entity.data.encoder_bias[:,self.term.target_ids]
   u=(action*self.term.scale+self.term.offset-bias)[0].cpu().numpy()[self.indices].copy()
   actual=torch.as_tensor(d.ctrl[:29],device=r.device,dtype=action.dtype)
   aidx=self.indices[self.overridden]
   actual_action=torch.zeros_like(action);actual_action[:,self.indices]=actual[None]
   rawtarget=(actual_action+bias-self.term.offset)/self.term.scale
   action[:,aidx]=rawtarget[:,aidx]
   self.env.step(action)
  return u
