"""Shared physical contact attribution and finite arm/finger control."""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mechanism import indices
from collision import ObjectHull

def contacts(m,d):
    ob=m.body('object/object').id
    touch=np.zeros(3,bool);forces=np.zeros(2);imp=np.zeros(4);vertical=0.;self_imp=0.;invalid=0.
    for ci,c in enumerate(d.contact):
        b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
        n1,n2=m.geom(c.geom1).name,m.geom(c.geom2).name
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);mag=np.linalg.norm(f[:3])
        world=c.frame.reshape(3,3).T@f[:3]
        if ob in [b1,b2]:
            other=c.geom2 if b1==ob else c.geom1;n=m.geom(other).name
            if n=='terrain':touch[0]=True;imp[0]+=mag*.005;vertical+=world[2] if b2==ob else -world[2]
            elif n.startswith('hand_'):
                side=0 if n.startswith('hand_left_') else 1;touch[1+side]=True;forces[side]+=mag;imp[1+side]+=mag*.005
            else:imp[3]+=mag*.005
        elif b1>0 and b2>0:self_imp+=mag*.005
        elif 'terrain' in [n1,n2] and not any('foot' in n for n in [n1,n2]):invalid+=mag*.005
    return touch,forces,imp,vertical,self_imp,invalid

def squeeze_load(m,d,normals):
    """Actual inward resultant on the object; opposing mesh normals cancel."""
    ob=m.body('object/object').id;load=np.zeros(2)
    for ci,c in enumerate(d.contact):
        b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
        if ob not in [b1,b2]:continue
        n=m.geom(c.geom2 if b1==ob else c.geom1).name
        if not n.startswith('hand_'):continue
        side=0 if n.startswith('hand_left_') else 1
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
        on_object=(1 if b2==ob else -1)*(c.frame.reshape(3,3).T@f[:3])
        load[side]+=-on_object@normals[side]
    return load

class Arms:
    def __init__(self,m,q,max_speed=3.):
        self.max_speed=max_speed
        self.m=m;self.aj,self.aq,self.av,self.arm,self.pads=indices(m);self.armq=self.aq[self.arm]
        self.fk=mujoco.MjData(m);self.seed=q[self.armq].copy();self.vel=np.zeros(len(self.arm));self.cost=0.
        self.hull=ObjectHull(m)
        self.hands=[m.body('hand_'+s).id for s in ['left','right']]
        ob=m.body('object/object').id;self.object_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob and m.geom_contype[g]]
    def solve(self,d,target,qref,rotations=None):
        m=self.m;fk=self.fk;aq=self.aq;aj=self.aj;arm=self.arm;base=d.qpos.copy();previous=self.seed.copy()
        def residual(x):
            fk.qpos[:]=base;fk.qpos[aq[29:31]]=0.;fk.qpos[self.armq]=x;mujoco.mj_kinematics(m,fk)
            clear=[]
            for side in ['left','right']:
                for n1,n2,gap in [(side+'_shoulder_yaw_collision','torso_collision',.013),(side+'_wrist_collision',side+'_hip_collision',.025),(side+'_wrist_collision',side+'_thigh_collision',.035),(side+'_elbow_yaw_collision',side+'_thigh_collision',.025),(side+'_elbow_yaw_collision',side+'_hip_collision',.02)]:
                    clear.append(100*max(0,gap-mujoco.mj_geomDistance(m,fk,m.geom(n1).id,m.geom(n2).id,.2,None)))
                for suffix,gap in [('_wrist_collision',.012),('_elbow_yaw_collision',.03)]:
                    g=m.geom(side+suffix).id
                    clear.append(150*max(0,gap-self.hull.gap(fk,g)))
                clear.extend([5*max(0,abs(fk.qpos[m.jnt_qposadr[m.joint(side+'_wrist_yaw_joint').id]])-1.15),3*max(0,fk.qpos[m.jnt_qposadr[m.joint(side+'_elbow_joint').id]]-1.8)])
                clear.append(150*max(0,.029-mujoco.mj_geomDistance(m,fk,m.geom('hand_'+side+'_palm').id,m.geom(side+'_wrist_collision').id,.1,None)))
            return np.r_[100*(fk.geom_xpos[self.pads]-target).ravel(),.35*(x-qref[self.armq]),.1*(x-self.seed),clear]
        lo=np.maximum(m.jnt_range[aj[arm],0]+.005,self.seed-self.max_speed*.02)
        hi=np.minimum(m.jnt_range[aj[arm],1]-.005,self.seed+self.max_speed*.02)
        fit=least_squares(residual,np.clip(self.seed,lo+1e-10,hi-1e-10),bounds=(lo,hi),max_nfev=30)
        self.seed=fit.x;self.vel=.65*self.vel+.35*np.clip((self.seed-previous)/.02,-self.max_speed,self.max_speed);self.cost=fit.cost
    def apply(self,d,u):
        m=self.m;i=self.arm
        u[i]=self.seed+d.qfrc_bias[self.av[i]]/m.actuator_gainprm[i,0]+(-m.actuator_biasprm[i,2]/m.actuator_gainprm[i,0])*self.vel+.5*np.clip(self.seed-d.qpos[self.armq],-.24,.24)
        return u
    def grip(self,d,torque=5.):
        return np.clip(d.qpos[self.aq[29:31]]+(torque+3*d.qvel[self.av[29:31]])/800,-.025,.035)

def thumb_control(m,d,max_angle=1.10,torque=.10):
    aq=m.jnt_qposadr[m.actuator_trnid[35:37,0]];av=m.jnt_dofadr[m.actuator_trnid[35:37,0]]
    return np.clip(d.qpos[aq]+(torque+.08*d.qvel[av])/4.,0,max_angle)

def smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)

class PitchServo:
    """Object attitude feedback through finite wrist and arm motors only."""
    def __init__(self,m,d,normal):
        self.m=m;self.oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id]
        self.reference=Rotation.from_quat(np.roll(d.qpos[self.oq+3:self.oq+7],-1))
        self.normal=np.asarray(normal).copy();self.normal/=np.linalg.norm(self.normal)
        self.filtered=0.;self.integral=0.;self.last_error=0.;self.last_angle=0.
    def correct(self,d,target,scale=1.,yaw=0.):
        rz=Rotation.from_euler('z',yaw);normal=rz.apply(self.normal)
        actual=Rotation.from_quat(np.roll(d.qpos[self.oq+3:self.oq+7],-1))
        error=float(((rz*self.reference)*actual.inv()).as_rotvec()@normal)
        self.filtered+=.1*(error-self.filtered)
        self.integral=float(np.clip(self.integral+.6*self.filtered*.005*scale,-.35,.35))
        angle=float(scale*np.clip(self.filtered+self.integral,-.4,.4));rot=Rotation.from_rotvec(angle*normal)
        result=target.copy()
        for i,side in enumerate(['left','right']):
            try:pivot=d.xanchor[self.m.joint('hand_'+side+'_swivel').id]
            except KeyError:pivot=target[i*3:i*3+3].mean(0)
            result[i*3:i*3+3]=rot.apply(target[i*3:i*3+3]-pivot)+pivot
        self.last_error=error;self.last_angle=angle
        return result

class ObjectServo:
    """Measured-object pose feedback implemented solely by finite arm motors.

The hands follow the real box with bounded position/orientation corrections
rather than forcing a rigid root-relative hand path after contact has slipped.
No object state, generalized force, constraint or attachment is applied.
"""
    def __init__(self,m,d,pads,aq):
        self.m=m;self.ob=m.body('object/object').id;self.oj=m.joint('object/floating_base_joint').id;self.od=m.jnt_dofadr[self.oj]
        self.C0=d.xipos[self.ob].copy();self.R0=Rotation.from_matrix(d.xmat[self.ob].reshape(3,3));self.root0=d.qpos[:3].copy()
        fk=mujoco.MjData(m);fk.qpos[:]=d.qpos;fk.qpos[aq[29:31]]=0.;mujoco.mj_forward(m,fk)
        self.local=self.R0.inv().apply(fk.geom_xpos[pads]-self.C0)
        normals=fk.xmat[[m.body('hand_'+s).id for s in ['left','right']]].reshape(2,3,3)[:,:,2]
        self.local_normals=self.R0.inv().apply(normals)
    def targets(self,d,yaw):
        C=d.xipos[self.ob];R=Rotation.from_matrix(d.xmat[self.ob].reshape(3,3));rz=Rotation.from_euler('z',yaw)
        desiredC=d.qpos[:3]+rz.apply(self.C0-self.root0);desiredR=rz*self.R0
        relative_velocity=d.qvel[self.od:self.od+3]-d.qvel[:3]
        delta=np.clip(.25*(desiredC-C)-.030*relative_velocity,-.012,.012)
        angle=(desiredR*R.inv()).as_rotvec();omega=R.apply(d.qvel[self.od+3:self.od+6])
        correction=.15*angle-.020*omega;norm=np.linalg.norm(correction)
        if norm>.04:correction*=.04/norm
        commandR=Rotation.from_rotvec(correction)*R
        return commandR.apply(self.local)+C+delta,commandR.apply(self.local_normals)

def thumb_normal_loads(m,d,forward):
    ob=m.body('object/object').id;out=np.zeros(2)
    for ci,c in enumerate(d.contact):
        b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
        if ob not in [b1,b2]:continue
        name=m.geom(c.geom2 if b1==ob else c.geom1).name
        if '_thumb_' not in name:continue
        side=0 if name.startswith('hand_left_') else 1
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
        normal=(1 if b2==ob else -1)*c.frame.reshape(3,3)[0]*f[0]
        out[side]+=normal@(forward if side==0 else -forward)
    return out
