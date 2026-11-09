"""Actual gait/grasp/placement quality from recorded physical states; no stepping."""
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d


def summary(x):
    x=np.asarray(x)
    return dict(rms=float(np.sqrt(np.mean(x*x))),peak=float(abs(x).max()),p95=float(np.percentile(abs(x),95)))


def quality(path,m):
    a=np.load(path/'rollout.npz');l=np.load(path/'rollout_locomotion_trace.npz')['trace'];t=np.load(path/'rollout_verification_trace.npz')['trace']
    torso=m.body('robot/torso_link').id;feet=[m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']]
    geoms=[[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('robot/'+side+'_foot') and m.geom_contype[g]] for side in ['left','right']]
    d=mujoco.MjData(m);jp=np.zeros((3,m.nv));jr=jp.copy();torso_omega=[];slips=[]
    s,e=3449,4721
    for tick in range(s,e):
        d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        mujoco.mj_jacBody(m,d,jp,jr,torso);torso_omega.append(jr@a['qvel'][tick])
        for side in [0,1]:
            if l[tick,14+side]<30.:continue
            points=[]
            for g in geoms[side]:
                for sign in [-1,1]:points.append(d.geom_xpos[g]+sign*m.geom_size[g,1]*d.geom_xmat[g].reshape(3,3)[:,2]-np.r_[0.,0.,m.geom_size[g,0]])
            point=min(points,key=lambda p:p[2]);mujoco.mj_jac(m,d,jp,jr,point,feet[side]);slips.append(np.linalg.norm((jp@a['qvel'][tick])[:2]))
    angular=np.array(torso_omega);hf=angular-gaussian_filter1d(angular,20.,axis=0,mode='nearest')
    root_acc=np.gradient(a['qvel'][s:e,:3],.005,axis=0);box_acc=np.gradient(t[s:e,20:23],.005,axis=0)
    place_v=t[6701:7880,20:23];place_a=np.gradient(place_v,.005,axis=0)
    return dict(walking_torso_angular_velocity_radps=summary(np.linalg.norm(angular,axis=1)),
        walking_torso_high_frequency_angular_velocity_radps=summary(np.linalg.norm(hf,axis=1)),
        walking_box_angular_velocity_radps=summary(np.linalg.norm(t[s:e,23:26],axis=1)),
        walking_box_linear_acceleration_mps2=summary(np.linalg.norm(box_acc,axis=1)),
        walking_root_linear_acceleration_mps2=summary(np.linalg.norm(root_acc,axis=1)),
        loaded_foot_lowest_surface_point_slip_mps=summary(slips),
        placement_box_linear_velocity_mps=summary(np.linalg.norm(place_v,axis=1)),
        placement_box_linear_acceleration_mps2=summary(np.linalg.norm(place_a,axis=1)),
        minimum_walk_hand_force_N=l[s:e,16:18].min(0).tolist(),
        method='Actual post-step states and velocities, native kinematic Jacobians. Lowest loaded collision-surface point includes baseline foot rolling; no stance/body retiming or simulator stepping. Torso HF subtracts a 100 ms Gaussian from angular velocity.')


def compare(new,old):
    names=['walking_torso_high_frequency_angular_velocity_radps','walking_box_angular_velocity_radps','walking_box_linear_acceleration_mps2','walking_root_linear_acceleration_mps2','placement_box_linear_velocity_mps','placement_box_linear_acceleration_mps2']
    ratios={k:{v:new[k][v]/old[k][v] for v in ['rms','peak','p95']} for k in names}
    gates={k:bool(new[k]['rms']<=1.10*old[k]['rms'] and new[k]['p95']<=1.15*old[k]['p95']) for k in names}
    k='loaded_foot_lowest_surface_point_slip_mps';ratios[k]={v:new[k][v]/old[k][v] for v in ['rms','peak','p95']}
    gates[k]=bool(new[k]['rms']<=1.15*old[k]['rms']+.002 and new[k]['p95']<=1.15*old[k]['p95']+.002)
    gates['bilateral_grasp_force_retained']=all(n>=.7*b for n,b in zip(new['minimum_walk_hand_force_N'],old['minimum_walk_hand_force_N']))
    return dict(goal7=new,goal6=old,ratios=ratios,gates=gates,pass_quality=all(gates.values()))
