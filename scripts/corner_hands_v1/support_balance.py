"""Finite-actuator inverse dynamics with explicit stance/swing foot selection.

Computed contact wrenches are used only to obtain motor torques. They are never
applied to the model; MuJoCo's actual foot contacts supply all support.
"""
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
class SupportBalance:
 def __init__(self,m,d):
  self.m=m;self.feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];self.J=np.zeros((12,m.nv));self.M=np.zeros((m.nv,m.nv));self.err=np.zeros(m.nv);self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.rows=np.r_[np.arange(6),m.jnt_dofadr[m.actuator_trnid[:29,0]]];self.I=np.eye(m.nv)[self.rows]
 def torques(self,d,qref,vref,pos,rot,vfeet,wfeet,support):
  m=self.m;acc=[]
  for i,b in enumerate(self.feet):
   mujoco.mj_jacBody(m,d,self.jp,self.jr,b);self.J[6*i:6*i+3]=self.jp;self.J[6*i+3:6*i+6]=self.jr;R=d.xmat[b].reshape(3,3);er=Rotation.from_matrix(rot[i]@R.T).as_rotvec();acc.extend(400*(pos[i]-d.xpos[b])+40*(vfeet[i]-self.jp@d.qvel));acc.extend(225*er+30*(wfeet[i]-self.jr@d.qvel))
  mujoco.mj_differentiatePos(m,self.err,1.,d.qpos,qref);des=100*self.err+20*(vref-d.qvel);des[:3]=80*self.err[:3]+18*(vref[:3]-d.qvel[:3]);des[3:6]=130*self.err[3:6]+24*(vref[3:6]-d.qvel[3:6]);weights=np.r_[np.ones(6)*2,np.ones(29)*.18];task=np.repeat(np.where(support,15.,7.),6)
  qacc=np.linalg.lstsq(np.r_[task[:,None]*self.J,weights[:,None]*self.I],np.r_[task*np.array(acc),weights*des[self.rows]],rcond=None)[0]
  mujoco.mj_fullM(m,self.M,d.qM);required=self.M@qacc+d.qfrc_bias;rows=np.concatenate([np.arange(6*i,6*i+6) for i in range(2) if support[i]]);J=self.J[rows];scale=np.tile([1,1,1,.04,.04,.04],int(sum(support)));wrench=scale*np.linalg.lstsq(J[:,:6].T*scale,required[:6],rcond=None)[0];return required-J.T@wrench,wrench
