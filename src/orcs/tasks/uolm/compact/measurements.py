"""Physics-substep contact attribution, collected by the ORCS metrics manager.

    MJWarp data is copied OUT for measurement. This module never advances a
    simulator, modifies its state, or commands a force.
"""
from collections import defaultdict
import mujoco
import mujoco_warp
import numpy as np
import torch
from .model import canonical


class PhysicsMeasurements:
    def __init__(self, env):
        assert env.num_envs == 1, "Full disjoint audit currently validates one physical world"
        self.m = env.sim.mj_model
        self.d = mujoco.MjData(self.m)
        self.fk = mujoco.MjData(self.m)
        self.ob = self.m.body('object/object').id
        self.feet = [self.m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']]
        self.torso = self.m.body('robot/torso_link').id
        self.foot_geoms = [[g for g in range(self.m.ngeom) if canonical(self.m.geom(g).name or '').startswith(s+'_foot') and self.m.geom_contype[g]] for s in ['left','right']]
        self.trace=[];self.locomotion=[];self.contact_rows=[];self.sums=defaultdict(float)
        self.bad=np.zeros(3);self.max_force=np.zeros(self.m.nu);self.max_external=0.
        self.initial_com=None

    def record(self, env):
        m=self.m;d=self.d
        mujoco_warp.get_data_into(d,m,env.sim.wp_data,world_id=0)
        ground=False;hands=np.zeros(2,bool);support=np.zeros(2,bool)
        foot_load=np.zeros(2);jaw_force=np.zeros(2);hand_force=np.zeros((2,3));hand_torque=np.zeros((2,3))
        forbidden=0.;invalid=0.;selfimp=0.;vertical=0.;terrain_vertical=0.;ee=False
        for ci,c in enumerate(d.contact):
            # MJWarp's get_data_into fills the modern contact.geom pair, not
            # MuJoCo's separate legacy geom1/geom2 convenience fields.
            g1,g2=map(int,c.geom);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            n1,n2=[canonical(m.geom(g).name or '') for g in [g1,g2]]
            force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
            imp=float(np.linalg.norm(force[:3])*.005)
            world=c.frame.reshape(3,3).T@force[:3]
            if self.ob in [b1,b2]:
                n=n2 if b1==self.ob else n1
                self.sums[n]+=imp
                if n=='terrain':ground=True;terrain_vertical+=world[2] if b2==self.ob else -world[2]
                elif n.startswith('hand_'):
                    side=0 if n.startswith('hand_left_') else 1
                    ee=True;hands[side]=True;jaw_force[side]+=np.linalg.norm(force[:3])
                    on_object=(1 if b2==self.ob else -1)*world
                    spin=(1 if b2==self.ob else -1)*(c.frame.reshape(3,3).T@force[3:])
                    hand_force[side]+=on_object;hand_torque[side]+=spin+np.cross(c.pos-d.xipos[self.ob],on_object)
                    vertical+=on_object[2]
                else:forbidden+=imp
            elif b1>0 and b2>0 and imp>1e-6:
                selfimp+=imp;self.sums['ROBOT_SELF:'+':'.join(sorted([n1,n2]))]+=imp
            if 'terrain' in [n1,n2]:
                other=g2 if n1=='terrain' else g1;n=canonical(m.geom(other).name or '')
                if m.geom_bodyid[other]==self.ob:continue
                if 'foot' in n:
                    side=0 if n.startswith('left') else 1
                    support[side]=True;foot_load[side]+=abs(world[2])
                elif imp>1e-6:
                    invalid+=imp;self.sums['INVALID_TERRAIN:'+n]+=imp
            if imp>1e-9:self.contact_rows.append([len(self.trace),g1,g2,imp])
        self.bad+=np.array([forbidden,selfimp,invalid])
        self.max_force=np.maximum(self.max_force,abs(d.actuator_force))
        self.max_external=max(self.max_external,float(abs(d.xfrc_applied).max()),float(abs(d.qfrc_applied).max()))
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,d.qpos[47:51]);rotation=rotation.reshape(3,3)
        offset=rotation@m.body_ipos[self.ob];com=d.qpos[44:47]+offset
        omega=rotation@d.qvel[46:49];com_velocity=d.qvel[43:46]+np.cross(omega,offset)
        if self.initial_com is None:
            cmd=env.command_manager.get_term('motion');q=cmd.qref[0].cpu().numpy()
            R=np.empty(9);mujoco.mju_quat2Mat(R,q[47:51]);self.initial_com=q[44:47]+R.reshape(3,3)@m.body_ipos[self.ob]
        tilt=float(np.degrees(np.arccos(np.clip(d.xmat[self.torso].reshape(3,3)[2,2],-1,1))))
        press=m.jnt_qposadr[m.actuator_trnid[29:31,0]]
        self.trace.append([ground,*hands,*support,com[2]-self.initial_com[2],d.qpos[2],tilt,forbidden,invalid,vertical,*com,*d.xpos[self.feet].ravel(),*com_velocity,*omega,*np.repeat(d.qpos[press],3),*d.qpos[47:51],*d.qvel[:6],terrain_vertical,ee])
        fk=self.fk;fk.qpos[:]=d.qpos;mujoco.mj_kinematics(m,fk);sole=[]
        for gs in self.foot_geoms:
            lows=[]
            for g in gs:
                typ=int(m.geom_type[g]);size=m.geom_size[g];R=fk.geom_xmat[g].reshape(3,3);p=fk.geom_xpos[g]
                dz=size[0] if typ==2 else size[0]+size[1]*abs(R[2,2]) if typ==3 else np.abs(R[2])@size
                lows.append(p[2]-dz)
            sole.append(min(lows))
        inward=np.array([-hand_force[i]@d.xmat[m.body('robot/hand_'+s).id].reshape(3,3)[:,2] for i,s in enumerate(['left','right'])])
        self.locomotion.append(np.r_[d.qpos[:3],d.qvel[:3],fk.xpos[self.feet].ravel(),sole,foot_load,jaw_force,inward,hand_force.ravel(),hand_torque.ravel()])


def record_physics(env):
    if not hasattr(env,'compact_measurements'):
        env.compact_measurements=PhysicsMeasurements(env)
    env.compact_measurements.record(env)
    return torch.as_tensor([float(not env.compact_measurements.bad.any())],device=env.device)


def invalid_contacts(env):
    if not hasattr(env,'compact_measurements'):
        return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    b=env.compact_measurements.bad
    return torch.as_tensor([b[0]>1e-5 or b[1]>.001 or b[2]>.001],device=env.device)
