"""Continuous online policy execution. No runtime state replacement or forces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
from controller import SOURCE, MODEL, OUT, Policy, error

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
from physical import contacts

PROFILES = {
    'small': dict(root_xy_m=.001,root_yaw_rad=np.deg2rad(.2),joint_rad=.001,
                  joint_velocity_radps=.005,root_velocity_mps=.002,root_angular_radps=.002,
                  box_xy_m=.001,box_yaw_rad=np.deg2rad(.2)),
    'modest': dict(root_xy_m=.005,root_yaw_rad=np.deg2rad(1.),joint_rad=.005,
                   joint_velocity_radps=.02,root_velocity_mps=.005,root_angular_radps=.005,
                   box_xy_m=.005,box_yaw_rad=np.deg2rad(1.)),
}
TRAIN_SEEDS = [1105101,1105102,1105103]
TEST_SEEDS = list(range(71105001,71105011))


def initial(m,a,seed,profile):
    q=a['initial_qpos'].copy();v=a['initial_qvel'].copy()
    if seed is None:return q,v,dict(seed=None,profile='nominal')
    rng=np.random.default_rng(seed);p=PROFILES[profile]
    q[:2]+=rng.uniform(-p['root_xy_m'],p['root_xy_m'],2)
    yaw=rng.uniform(-p['root_yaw_rad'],p['root_yaw_rad']);quat=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    orig=q[3:7].copy();mujoco.mju_mulQuat(q[3:7],quat,orig)
    aq=m.jnt_qposadr[m.actuator_trnid[:,0]];av=m.jnt_dofadr[m.actuator_trnid[:,0]]
    # Perturb all 29 native robot joints; compact hand state stays at its
    # prescribed physically open initial condition, preserving task semantics.
    q[aq[:29]]+=rng.uniform(-p['joint_rad'],p['joint_rad'],29)
    v[av[:29]]+=rng.uniform(-p['joint_velocity_radps'],p['joint_velocity_radps'],29)
    v[:3]+=rng.uniform(-p['root_velocity_mps'],p['root_velocity_mps'],3)
    v[3:6]+=rng.uniform(-p['root_angular_radps'],p['root_angular_radps'],3)
    oj=m.joint('object/floating_base_joint').id;oq=int(m.jnt_qposadr[oj])
    q[oq:oq+2]+=rng.uniform(-p['box_xy_m'],p['box_xy_m'],2)
    yaw=rng.uniform(-p['box_yaw_rad'],p['box_yaw_rad']);quat=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    orig=q[oq+3:oq+7].copy();mujoco.mju_mulQuat(q[oq+3:oq+7],quat,orig)
    return q,v,dict(seed=seed,profile=profile,distribution='independent bounded uniform per scalar',
                    magnitudes=p,delta_qpos=(q-a['initial_qpos']).tolist(),delta_qvel=(v-a['initial_qvel']).tolist())


def run(checkpoint,out,seed=None,profile='small',feedback=True,seconds=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    archive=np.load(SOURCE);a={key:archive[key] for key in archive.files}
    m=mujoco.MjModel.from_binary_path(str(MODEL));d=mujoco.MjData(m)
    q,v,perturb=initial(m,a,seed,profile);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
    init_touch,_,init_imp,_,init_self,init_invalid=contacts(m,d)
    initial_contacts=[dict(geom1=m.geom(int(c.geom1)).name,geom2=m.geom(int(c.geom2)).name,distance_m=float(c.dist)) for c in d.contact]
    valid=bool(init_touch[0] and init_imp[3]<1e-7 and init_self<1e-7 and init_invalid<1e-7)
    policy=Policy(m,checkpoint);cs=[];qs=[];vs=[];cts=[];rows=[];sums=np.zeros(3);reason='completed'
    total=len(a['ctrl_physics']) if seconds is None else min(len(a['ctrl_physics']),round(seconds/.005))
    begin=time.time();maxdelta=np.zeros(m.nu);step_errors=[]
    if valid:
        for step in range(total):
            u=policy.control(step,d,feedback=feedback);maxdelta=np.maximum(maxdelta,abs(u-a['ctrl_physics'][step]))
            d.ctrl[:]=u;mujoco.mj_step(m,d)
            touch,N,imp,vertical,selfimp,invalid=contacts(m,d);sums+=np.r_[imp[3],selfimp,invalid]
            cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch.copy())
            if step%policy.stride==0:
                step_errors.append(np.r_[step,policy.last_error,policy.delta])
            if step%200==199:
                rows.append(dict(time_s=(step+1)*.005,root=d.qpos[:3].tolist(),root_reference_error_m=float(np.linalg.norm(d.qpos[:3]-a['qpos'][step,:3])),
                                 object_reference_error_m=float(np.linalg.norm(d.qpos[44:47]-a['qpos'][step,44:47])),hand_force_N=N.tolist(),bad_impulses_Ns=sums.tolist()))
            if sums[0]>1e-5:reason='forbidden_object_contact'
            elif sums[1]>.001:reason='robot_self_contact'
            elif sums[2]>.001:reason='invalid_robot_terrain_contact'
            elif d.qpos[2]<.32:reason='fall'
            elif not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():reason='nonfinite'
            if reason!='completed':break
    else:reason='invalid_initial_state'
    data=dict(initial_qpos=q,initial_qvel=v,ctrl_physics=np.asarray(cs).reshape(-1,m.nu),
              qpos=np.asarray(qs).reshape(-1,m.nq),qvel=np.asarray(vs).reshape(-1,m.nv),contacts=np.asarray(cts).reshape(-1,3),
              policy_diagnostics=np.asarray(step_errors))
    for key in a:
        if 'step' in key or 'frame' in key:data[key]=a[key]
    np.savez_compressed(out/'rollout.npz',**data)
    (out/'rollout_model.mjb').symlink_to(os.path.relpath(MODEL,out))
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                feedback=feedback,perturbation=perturb,initial_state_valid=valid,initial_contacts=initial_contacts,
                reason=reason,physics_steps=len(cs),duration_s=len(cs)*.005,wall_seconds=time.time()-begin,
                forbidden_self_invalid_impulses_Ns=sums.tolist(),maximum_abs_action_correction=maxdelta.tolist(),diagnostics_1Hz=rows)
    (out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['diagnostics_1Hz','initial_contacts','maximum_abs_action_correction','perturbation']},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=OUT/'policy.npz');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int);p.add_argument('--profile',choices=PROFILES,default='small');p.add_argument('--feedback-off',action='store_true');p.add_argument('--seconds',type=float)
    args=p.parse_args();run(args.checkpoint,args.out,args.seed,args.profile,not args.feedback_off,args.seconds)
