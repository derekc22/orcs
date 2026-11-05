"""Load-continuous arm IK with physical clearance and unchanged effort caps."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from physical import Arms


class ClearancePlacementArms(Arms):
    def __init__(self,m,d,floor_clearance=0.,**kwargs):
        super().__init__(m,d.qpos,**kwargs)
        self.floor_clearance=floor_clearance
        self.terrain=m.geom('terrain').id
        self.hand_geoms=[g for g in range(m.ngeom) if m.geom(g).name.startswith('hand_') and m.geom_contype[g]]
        self.bias0=d.qfrc_bias[self.av[self.arm]].copy()
        self.offset=d.ctrl[self.arm]-d.qpos[self.armq]

    def solve(self,d,target,qref,rotations=None):
        m=self.m;fk=self.fk;previous=self.seed.copy();base=d.qpos.copy()
        def residual(x):
            fk.qpos[:]=base;fk.qpos[self.aq[29:31]]=0.;fk.qpos[self.armq]=x
            mujoco.mj_kinematics(m,fk)
            clear=[]
            for side in ['left','right']:
                for suffix,gap in [('_wrist_collision',.025),('_elbow_yaw_collision',.025)]:
                    clear.append(400*max(0,gap-self.hull.gap(fk,m.geom(side+suffix).id)))
                for n1,n2,gap in [(side+'_shoulder_yaw_collision','torso_collision',.008),
                                  (side+'_wrist_collision',side+'_hip_collision',.020),
                                  (side+'_wrist_collision',side+'_thigh_collision',.025),
                                  (side+'_elbow_yaw_collision',side+'_thigh_collision',.020)]:
                    clear.append(150*max(0,gap-mujoco.mj_geomDistance(m,fk,m.geom(n1).id,m.geom(n2).id,.2,None)))
            if self.floor_clearance>0:
                clear.extend(400*max(0,self.floor_clearance-mujoco.mj_geomDistance(m,fk,g,self.terrain,.1,None)) for g in self.hand_geoms)
            return np.r_[150*(fk.geom_xpos[self.pads]-target).ravel(),
                         .35*(x-qref[self.armq]),.05*(x-previous),clear]
        lo=np.maximum(m.jnt_range[self.aj[self.arm],0]+.005,previous-self.max_speed*.02)
        hi=np.minimum(m.jnt_range[self.aj[self.arm],1]-.005,previous+self.max_speed*.02)
        fit=least_squares(residual,np.clip(previous,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=25)
        self.seed=fit.x
        self.vel=.65*self.vel+.35*np.clip((self.seed-previous)/.02,-self.max_speed,self.max_speed)
        self.cost=fit.cost

    def apply(self,d,u):
        m=self.m;i=self.arm
        u[i]=self.seed+self.offset+(d.qfrc_bias[self.av[i]]-self.bias0)/m.actuator_gainprm[i,0]
        u[i]+=(-m.actuator_biasprm[i,2]/m.actuator_gainprm[i,0])*self.vel
        return u
