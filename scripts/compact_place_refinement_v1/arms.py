"""Placement-only continuous arm reference; the physical model is unchanged."""
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]/'compact_smooth_integration_v2'))
from arms_with_floor import ClearancePlacementArms

class PlacementArms(ClearancePlacementArms):
    def __init__(self,m,d,reference_ramp=False,arm_damping=0.,press_damping=0.,**kwargs):
        super().__init__(m,d,**kwargs)
        self.reference_ramp=reference_ramp
        self.arm_damping=arm_damping
        self.press_damping=press_damping
        self.command=self.seed.copy();self.ramp_start=self.command.copy();self.ramp_step=4
    def solve(self,d,target,qref,rotations=None):
        self.ramp_start=self.command.copy();self.ramp_step=0
        super().solve(d,target,qref,rotations)
    def apply(self,d,u):
        self.ramp_step+=1
        self.command=self.ramp_start+(self.seed-self.ramp_start)*min(self.ramp_step/4.,1.) if self.reference_ramp else self.seed.copy()
        m=self.m;i=self.arm
        u[i]=self.command+self.offset+(d.qfrc_bias[self.av[i]]-self.bias0)/m.actuator_gainprm[i,0]
        u[i]+=(-m.actuator_biasprm[i,2]/m.actuator_gainprm[i,0])*self.vel
        u[i]-=self.arm_damping*d.qvel[self.av[i]]/m.actuator_gainprm[i,0]
        return u
    def grip(self,d,torque=5.):
        return np.clip(d.qpos[self.aq[29:31]]+(torque+(3-self.press_damping)*d.qvel[self.av[29:31]])/800,-.025,.035)
