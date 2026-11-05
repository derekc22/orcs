"""Architecture family 2: two independent compact force-driven palm carriages.

Each wrist has one 60 mm linear palm jaw and a powered compact swivel. Three
visible solid rubber blocks form a broad contact triangle on each rigid palm.
There is no object-spanning frame and no remote finger rods.
"""
from pathlib import Path
import sys
import numpy as np,mujoco
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/articulated_gripper_v1'));sys.path.insert(0,str(HERE))
import frozen
OUT=ROOT/'logs/corner_hands_v1';BASE=ROOT/'logs/sustained_carry_v1'

def find_body(b,name):
    if b.name==name:return b
    for c in b.bodies:
        found=find_body(c,name)
        if found is not None:return found

def build():
    old=frozen.reference();s=frozen.spec()
    def geom(b,n,t,mass,**kw):
        return b.add_geom(name=n,type=t,mass=mass,contype=4,conaffinity=13,
                          condim=4,friction=[2.5,.1,.003],solref=[.03,1],
                          solimp=[.8,.95,.01,.5,2],**kw)
    for side in ['left','right']:
        w=find_body(s.worldbody,side+'_wrist_yaw_link')
        adapter=w.add_body(name='hand_'+side+'_adapter')
        geom(adapter,'hand_'+side+'_adapter_link',3,.015,fromto=[.012,0,0,.075,0,0],size=[.015,0,0],rgba=[.30,.35,.40,1])
        geom(adapter,'hand_'+side+'_servo_housing',5,.025,pos=[.075,0,0],size=[.022,.015,0],rgba=[.25,.30,.36,1])
        palm_forward=-.070 if side=='left' else .070
        p=adapter.add_body(name='hand_'+side,pos=[0,0,-.055])
        p.add_joint(name=f'hand_{side}_swivel',type=3,axis=[0,0,1],pos=[.075,0,.055],limited=True,range=[-.9,.9],damping=.08,armature=.0003)
        geom(p,'hand_'+side+'_mount',3,.018,fromto=[.075,0,.055,.12,palm_forward,.035],size=[.015,0,0],rgba=[.30,.35,.40,1])
        geom(p,'hand_'+side+'_linear_housing',6,.055,pos=[.12,palm_forward,.035],size=[.025,.028,.045],rgba=[.38,.44,.49,1])
        f=p.add_body(name='hand_'+side+'_plate',pos=[.12,palm_forward,-.025])
        f.add_joint(name=f'hand_{side}_press',type=2,axis=[0,0,-1],limited=True,range=[-.025,.035],damping=.4,armature=.004)
        geom(f,'hand_'+side+'_palm',6,.030,pos=[0,0,.015],size=[.060,.065,.005],rgba=[.48,.54,.59,1])
        geom(f,'hand_'+side+'_piston',3,.012,fromto=[0,0,0,0,0,.080],size=[.010,0,0],rgba=[.66,.69,.72,1])
        for k,(pos,size) in enumerate([
            ([-.04,0,0],[.020,.055,.020]),
            ([.025,-.030,0],[.030,.025,.020]),
            ([.025,.030,0],[.030,.025,.020]),
        ]):
            geom(f,f'hand_{side}_pad{k}',6,.018,pos=pos,size=size,rgba=[.10,.15,.18,1])
        hook=f.add_body(name='hand_'+side+'_hook',pos=[.055,0,.035])
        hook.add_joint(name='hand_'+side+'_curl',type=3,axis=[0,1,0],limited=True,range=[0,1.85],damping=.05,armature=.0004)
        # One rigid fork, driven by one hinge: two visible 150 mm fingers and
        # 80 mm inward toes. Open fingers point away from the box; after lift
        # they rotate downwards and beneath its lower edge.
        geom(hook,'hand_'+side+'_crossbar',3,.010,fromto=[0,-.055,0,0,.055,0],size=[.009,0,0],rgba=[.40,.46,.50,1])
        for k,y in enumerate([-.055,.055]):
            geom(hook,f'hand_{side}_finger{k}',3,.018,fromto=[0,y,0,0,y,.150],size=[.009,0,0],rgba=[.40,.46,.50,1])
            geom(hook,f'hand_{side}_toe{k}',3,.014,fromto=[0,y,.150,.080,y,.150],size=[.011,0,0],rgba=[.10,.15,.18,1])
        geom(f,'hand_'+side+'_thumb_mount',3,.006,fromto=[.045,.025,.005,.080,.025,0],size=[.008,0,0],rgba=[.40,.46,.50,1])
        geom(f,'hand_'+side+'_thumb_motor',5,.018,pos=[.080,.025,0],quat=[.7071067811865476,0,.7071067811865476,0],size=[.012,.012,0],rgba=[.25,.30,.36,1])
        thumb=f.add_body(name='hand_'+side+'_thumb',pos=[.080,.025,0],quat=[.962425197628238, -.271546936956113,0,0])
        thumb.add_joint(name='hand_'+side+'_opposition',type=3,axis=[-1,0,0],limited=True,range=[0,1.30],damping=.06,armature=.0003)
        geom(thumb,'hand_'+side+'_thumb_stem',3,.015,fromto=[0,0,0,0,0,.050],size=[.0085,0,0],rgba=[.40,.46,.50,1])
        geom(thumb,'hand_'+side+'_thumb_contact',3,.015,fromto=[0,0,.050,0,.070,.050],size=[.012,0,0],rgba=[.10,.15,.18,1])
        geom(thumb,'hand_'+side+'_thumb_tip',2,.004,pos=[0,.070,.050],size=[.015,0,0],rgba=[.10,.15,.18,1])
    for side in ['left','right']:
        s.add_actuator(name=f'drive_hand_{side}_press',target=f'hand_{side}_press',trntype=0,gainprm=[800]+[0]*9,biastype=1,biasprm=[0,-800,-3]+[0]*7,ctrllimited=True,ctrlrange=[-.025,.035],forcelimited=True,forcerange=[-10,10])
    for side in ['left','right']:
        s.add_actuator(name=f'drive_hand_{side}_swivel',target=f'hand_{side}_swivel',trntype=0,gainprm=[40]+[0]*9,biastype=1,biasprm=[0,-40,-1.2]+[0]*7,ctrllimited=True,ctrlrange=[-.9,.9],forcelimited=True,forcerange=[-1.5,1.5])
    for side in ['left','right']:
        s.add_actuator(name=f'drive_hand_{side}_curl',target=f'hand_{side}_curl',trntype=0,gainprm=[4]+[0]*9,biastype=1,biasprm=[0,-4,-.10]+[0]*7,ctrllimited=True,ctrlrange=[0,1.85],forcelimited=True,forcerange=[-.30,.30])
    for side in ['left','right']:
        s.add_actuator(name=f'drive_hand_{side}_opposition',target=f'hand_{side}_opposition',trntype=0,gainprm=[4]+[0]*9,biastype=1,biasprm=[0,-4,-.08]+[0]*7,ctrllimited=True,ctrlrange=[0,1.30],forcelimited=True,forcerange=[-.40,.40])
    m=frozen.restore(s.compile())
    for j in range(m.njnt):
        if m.joint(j).name.startswith('hand_'):continue
        oj=old.joint(m.joint(j).name).id;k=6 if m.jnt_type[j]==0 else 1
        for field in ['dof_damping','dof_armature','dof_frictionloss']:
            getattr(m,field)[m.jnt_dofadr[j]:m.jnt_dofadr[j]+k]=getattr(old,field)[old.jnt_dofadr[oj]:old.jnt_dofadr[oj]+k]
    mujoco.mj_saveModel(m,str(OUT/'palms_model.mjb'),None);return m

def map_pose(m,qold,old=None):
    if old is None:old=frozen.reference()
    q=m.qpos0.copy()
    for j in range(m.njnt):
        try:oj=old.joint(m.joint(j).name).id
        except KeyError:continue
        k=7 if m.jnt_type[j]==0 else 1;q[m.jnt_qposadr[j]:m.jnt_qposadr[j]+k]=qold[old.jnt_qposadr[oj]:old.jnt_qposadr[oj]+k]
    return q

def indices(m):
    aj=m.actuator_trnid[:,0];aq=m.jnt_qposadr[aj];av=m.jnt_dofadr[aj]
    arm=np.r_[[i for i,j in enumerate(aj[:29]) if any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])],np.arange(31,33)].astype(int)
    pads=np.array([m.geom(f'hand_{side}_pad{k}').id for side in ['left','right'] for k in range(3)])
    return aj,aq,av,arm,pads

if __name__=='__main__':
    m=build();print('COMPILED',m.nq,m.nv,m.nu)
