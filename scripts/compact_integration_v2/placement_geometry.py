"""Cheap feet-fixed whole-body reach and collision feasibility checks."""
import argparse
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from mechanism import ROOT, OUT, BASE, build, map_pose, indices
from collision import ObjectHull


def face_axis(m):
    ob=m.body('object/object').id;points=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=ob or not m.geom_contype[g]:continue
        mi=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]
        points.append(Rotation.from_quat(np.roll(m.geom_quat[g],-1)).apply(v)+m.geom_pos[g])
    points=np.concatenate(points);hull=ConvexHull(points);best=None
    for i,tri in enumerate(hull.simplices):
        n=hull.equations[i,:3]
        if abs(n[2])>.3 or n[0]<.6:continue
        area=np.linalg.norm(np.cross(points[tri[1]]-points[tri[0]],points[tri[2]]-points[tri[0]]))/2
        if best is None or area>best[0]:best=(area,n)
    return best[1]

def targets(m,q,hand_x=0.,hand_z=.14,gap=0.,orientation='down',surface_frame=True,diagonal_offset=0.):
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);ob=m.body('object/object').id;R=d.xmat[ob].reshape(3,3)
    forward=R@face_axis(m);forward[2]=0;forward/=np.linalg.norm(forward);up=np.array([0.,0,1]);lateral=np.cross(up,forward)
    og=[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob and m.geom_contype[g]];pads=indices(m)[4];points=[];rotations=[]
    for side,sign in [('left',1),('right',-1)]:
        H=np.column_stack([-up,-sign*forward,sign*lateral]);rotations.append(H)
        origin=d.xpos[ob]+(hand_x-sign*diagonal_offset)*forward+sign*(.27+gap)*lateral+[0,0,hand_z]
        local=np.array([[.08,0,-.025],[.145,-.03,-.025],[.145,.03,-.025]])
        pp=local@H.T+origin
        hull=ObjectHull(m);sizes=[m.geom_size[p] for p in pads[:3]];shift=hull.contact_translation(d,pp,H,sizes,sign*lateral);pp+=shift
        ds=[float(x) for x in shift]
        print('PALM_SURFACE_PROBE',side,ds,flush=True);points.extend(pp)
    return np.array(points),np.array(rotations)


def plan(m, q0, target, rotations, feet_pos, feet_rot, seed=None, max_nfev=180, upright=False, com_target=None,com_forward=.055,min_root=.32,target_yaw_override=None,waist_yaw_reference=0.,root_height_target=None):
    aj, aq, av, arm, pads = indices(m)
    acts=np.r_[np.arange(29),np.arange(31,33)].astype(int)
    j = aj[acts]; jq = aq[acts]
    d = mujoco.MjData(m)
    feet = [m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    hands = [m.body('hand_'+s).id for s in ['left','right']]
    ob = m.body('object/object').id
    object_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob and m.geom_contype[g]]
    hull=ObjectHull(m)
    object_clearance=[]
    for side in ['left','right']:
        for suffix,gap in [('_hip_collision',.035),('_shin_collision',.025),('_thigh_collision',.018),('_wrist_collision',.015),('_elbow_yaw_collision',.03)]:
            object_clearance.append((m.geom(side+suffix).id,gap))
    object_clearance.extend([(m.geom('torso_collision').id,.025),(m.geom('head_collision').id,.025)])
    tor = m.body('torso_link').id
    pelvis=m.body('pelvis').id
    waistq=[m.jnt_qposadr[m.joint(n).id] for n in ['waist_roll_joint','waist_pitch_joint']]
    waistyaw=m.jnt_qposadr[m.joint('waist_yaw_joint').id]
    R0 = Rotation.from_quat(np.roll(q0[3:7], -1))
    com = feet_pos[:, :2].mean(0)
    heading = np.cross(rotations[0,:,2],[0,0,1]);heading/=np.linalg.norm(heading)
    targetyaw=np.arctan2(heading[1],heading[0])
    if target_yaw_override is not None:targetyaw=target_yaw_override
    com += com_forward * heading[:2]
    if com_target is not None:com=np.asarray(com_target).copy()
    pairs=[]
    for side in ['left','right']:
        pairs.append((m.geom(side+'_thigh_collision').id,m.geom('torso_collision').id,.025))
        for n1,n2,clear in [(side+'_wrist_collision',side+'_hip_collision',.025),(side+'_wrist_collision',side+'_thigh_collision',.035),(side+'_elbow_yaw_collision',side+'_thigh_collision',.025),(side+'_elbow_yaw_collision',side+'_hip_collision',.02),(side+'_shoulder_yaw_collision','torso_collision',.015)]:
            pairs.append((m.geom(n1).id,m.geom(n2).id,clear))
    def pose(x):
        d.qpos[:] = q0
        d.qpos[:3] = x[:3]
        d.qpos[3:7] = np.roll((Rotation.from_rotvec(x[3:6])*R0).as_quat(),1)
        d.qpos[jq] = x[6:]
        d.qpos[aq[29:31]] = 0.
        mujoco.mj_forward(m,d)
    def residual(x):
        pose(x)
        ori=np.concatenate([Rotation.from_matrix(d.xmat[b].reshape(3,3)@feet_rot[i].T).as_rotvec() for i,b in enumerate(feet)])
        handori=np.concatenate([Rotation.from_matrix(d.xmat[b].reshape(3,3)@rotations[i].T).as_rotvec() for i,b in enumerate(hands)])
        pen=np.zeros(m.ngeom)
        for c in d.contact:
            b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            n1,n2=m.geom(c.geom1).name,m.geom(c.geom2).name
            if ob in [b1,b2]:
                other=c.geom2 if b1==ob else c.geom1
                if m.geom(other).name.startswith('hand_') or m.geom(other).name=='terrain': continue
                pen[other]=min(pen[other],c.dist-.008)
            elif b1>0 and b2>0:
                pen[c.geom1]=min(pen[c.geom1],c.dist-.006)
        clear=[100*max(0,gap-mujoco.mj_geomDistance(m,d,g,h,.2,None)) for g,h,gap in pairs]
        clear.extend(130*max(0,gap-hull.gap(d,g)) for g,gap in object_clearance)
        margin=[]
        for side in ['left','right']:
            # The true joint bounds remain enforced. Legacy advisory margins
            # exclude the verified loaded corner pose and distort this solve.
            margin.extend([0.,0.])
        Rpelvis=d.xmat[pelvis].reshape(3,3);yawerr=np.arctan2(Rpelvis[1,0],Rpelvis[0,0])-targetyaw;yawerr=np.arctan2(np.sin(yawerr),np.cos(yawerr))
        return np.r_[100*(d.geom_xpos[pads]-target).ravel(),
                     150*(d.xpos[feet]-feet_pos).ravel(),40*ori,3*handori,
                     90*(d.subtree_com[1,:2]-com),
                     (8 if upright else 3)*d.xmat[tor].reshape(3,3)[:2,2],
                     (30 if upright else 0)*d.xmat[pelvis].reshape(3,3)[:2,2],
                     (4 if upright else 0)*d.qpos[waistq],
                     (35*(x[2]-root_height_target) if root_height_target is not None else (15 if upright else 1)*(x[2]-(.73 if upright else .36))),
                     .10*(x[6:]-q0[jq]),200*pen,clear,margin,15*yawerr,5*(d.qpos[waistyaw]-waist_yaw_reference)]
    lo=np.r_[q0[:2]-.28,min_root,[-.85,-.85,-.8],m.jnt_range[j,0]+.008]
    hi=np.r_[q0[:2]+.28,.82,[.85,.85,.8],m.jnt_range[j,1]-.008]
    x=np.r_[q0[:3],np.zeros(3),q0[jq]] if seed is None else seed
    fit=least_squares(residual,np.clip(x,lo+.00001,hi-.00001),bounds=(lo,hi),max_nfev=max_nfev)
    pose(fit.x)
    bad=[]
    for c in d.contact:
        ns=[m.geom(int(g)).name for g in [c.geom1,c.geom2]]
        bodies=[m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]]
        if c.dist<-.001 and not ('terrain' in ns or (ob in bodies and any(n.startswith('hand_') for n in ns))):bad.append([ns,float(c.dist)])
    r=dict(cost=float(fit.cost),max_tip_error_m=float(np.linalg.norm(d.geom_xpos[pads]-target,axis=1).max()),
           foot_error_m=float(np.linalg.norm(d.xpos[feet]-feet_pos,axis=1).max()),
           root_z=float(d.qpos[2]),com_xy_error=float(np.linalg.norm(d.subtree_com[1,:2]-com)),
           torso_tilt_deg=float(np.degrees(np.arccos(d.xmat[tor].reshape(3,3)[2,2]))),interference=bad)
    return d.qpos.copy(),fit.x,r


def run(tag='reach1', back=.12, hand_x=-.10, hand_z=.075,orientation='forward',surface_frame=False,seed_file=None,lift=.38,toe_out_deg=0.,stance_width=None,flat_feet=False,com_forward=.055,min_root=.32,diagonal_offset=0.):
    m=build();a=np.load(BASE/'complete2.npz');q0=map_pose(m,a['initial_qpos'])
    aj,aq,av,arm,pads=indices(m)
    d=mujoco.MjData(m);d.qpos[:]=q0;mujoco.mj_forward(m,d)
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    fp=d.xpos[feet].copy();fr=d.xmat[feet].reshape(2,3,3).copy()
    oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id]
    forward=q0[oq:oq+3]-q0[:3];forward[2]=0;forward/=np.linalg.norm(forward)
    q0[oq:oq+3]-=back*forward
    if stance_width is not None:
        lateral=np.cross([0.,0.,1.],forward);mid=fp.mean(0);fp[0]=mid+stance_width*.5*lateral;fp[1]=mid-stance_width*.5*lateral;fp[:,2]=.035
    if flat_feet:
        fr=np.array([Rotation.from_euler('z',np.arctan2(forward[1],forward[0])).as_matrix() for _ in range(2)])
    if toe_out_deg:
        yaw=np.arctan2(forward[1],forward[0])
        for i,sign in enumerate([1,-1]):
            oldyaw=np.arctan2(fr[i,1,0],fr[i,0,0]);fr[i]=Rotation.from_euler('z',yaw+sign*np.radians(toe_out_deg)-oldyaw).as_matrix()@fr[i]
    if surface_frame:
        R=Rotation.from_quat(np.roll(q0[oq+3:oq+7],-1));face=R.apply(face_axis(m));delta=np.arctan2(forward[1],forward[0])-np.arctan2(face[1],face[0])
        q0[oq+3:oq+7]=np.roll((Rotation.from_euler('z',delta)*R).as_quat(),1)
        print('FACE_FRAME',face_axis(m),'initial_world_yaw_change_deg',np.degrees(delta),flush=True)
    target,rot=targets(m,q0,hand_x,hand_z,orientation=orientation,surface_frame=surface_frame,diagonal_offset=diagonal_offset)
    rows=[];poses=[];seed=None
    if seed_file:
        sq=np.load(OUT/(seed_file+'.npz'))['qpos'][1].copy();sq[2]-=.06
        dr=(Rotation.from_quat(np.roll(sq[3:7],-1))*Rotation.from_quat(np.roll(q0[3:7],-1)).inv()).as_rotvec()
        seed=np.r_[sq[:3],dr,sq[aq[np.r_[np.arange(29),np.arange(31,33)].astype(int)]]]
    heights=[.08,0.,.18,lift]
    for index,h in enumerate(heights):
        predicted=q0.copy();predicted[oq+2]+=0. if index==0 else h
        q,seed,r=plan(m,predicted,target+[0,0,h],rot,fp,fr,seed,upright=h>.2,com_forward=com_forward,min_root=min_root)
        print(tag,h,r,flush=True);poses.append(q);rows.append(r)
    np.savez(OUT/(tag+'.npz'),qpos=poses,target=target,rot=rot,feet_pos=fp,feet_rot=fr,heights=heights)
    (OUT/(tag+'.json')).write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='reach1');p.add_argument('--back',type=float,default=.12);p.add_argument('--hand-x',type=float,default=-.10);p.add_argument('--hand-z',type=float,default=.075);p.add_argument('--orientation',choices=['forward','down'],default='forward');p.add_argument('--surface-frame',action='store_true');p.add_argument('--seed-file');p.add_argument('--lift',type=float,default=.38);p.add_argument('--toe-out-deg',type=float,default=0.);p.add_argument('--stance-width',type=float);p.add_argument('--flat-feet',action='store_true');p.add_argument('--com-forward',type=float,default=.055);p.add_argument('--min-root',type=float,default=.32);p.add_argument('--diagonal-offset',type=float,default=0.);a=p.parse_args();run(a.tag,a.back,a.hand_x,a.hand_z,a.orientation,a.surface_frame,a.seed_file,a.lift,a.toe_out_deg,a.stance_width,a.flat_feet,a.com_forward,a.min_root,a.diagonal_offset)
