"""Contact-consistent floating-base inverse dynamics; physical actuator torques only."""
import mujoco,numpy as np
class Balance:
 def __init__(self,m,d):
  self.m=m;self.feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];self.pos=d.xpos[self.feet].copy();self.rot=d.xmat[self.feet].reshape(2,3,3).copy();self.J=np.zeros((12,m.nv));self.M=np.zeros((m.nv,m.nv));self.err=np.zeros(m.nv);self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.rows=np.r_[np.arange(6),m.jnt_dofadr[m.actuator_trnid[:29,0]]];self.I=np.eye(m.nv)[self.rows];self.lastJ=None
 def torques(self,d,qref,vref=None):
  m=self.m;J=self.J;acc=[]
  for i,b in enumerate(self.feet):
   mujoco.mj_jacBody(m,d,self.jp,self.jr,b);J[6*i:6*i+3]=self.jp;J[6*i+3:6*i+6]=self.jr
   # Small-angle world rotation error.
   R=d.xmat[b].reshape(3,3);rot_err=.5*sum((np.cross(R[:,k],self.rot[i,:,k]) for k in range(3)),start=np.zeros(3))
   acc.extend(250*(self.pos[i]-d.xpos[b])-30*(self.jp@d.qvel));acc.extend(150*rot_err-24*(self.jr@d.qvel))
  mujoco.mj_differentiatePos(m,self.err,1.,d.qpos,qref);vref=np.zeros(m.nv) if vref is None else vref
  des=100*self.err+20*(vref-d.qvel);des[:3]=80*self.err[:3]+18*(vref[:3]-d.qvel[:3]);des[3:6]=130*self.err[3:6]+24*(vref[3:6]-d.qvel[3:6]);weights=np.r_[np.ones(6)*2,np.ones(29)*.18]
  # Foot acceleration is the primary constraint. Object and slider accelerations excluded.
  qacc=np.linalg.lstsq(np.r_[15*J,weights[:,None]*self.I],np.r_[15*np.array(acc),weights*des[self.rows]],rcond=None)[0]
  mujoco.mj_fullM(m,self.M,d.qM);required=self.M@qacc+d.qfrc_bias
  scale=np.tile([1,1,1,.04,.04,.04],2);A=J[:,:6].T*scale
  wrench=scale*np.linalg.lstsq(A,required[:6],rcond=None)[0];tau=required-J.T@wrench
  return tau,wrench
