"""Arm-only, contact-preserving frozen-sample retargeting of the accepted motion.

No whole-body reference compiler: every physics-rate baseline sample survives.
The only kinematic variables are the fourteen arm joints and object pose.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mjlab
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from orcs.tasks.uolm.compact.model import source_model, CHECKPOINT

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/compact_diffusion_v1'))
from proposals import CHECKPOINT as DIFFUSION_CHECKPOINT

PHASES = {'pickup': (2.7, 8., 0, 105),
          'carry': (17.245, 23.605, 105, 225),
          'placement': (33.505, 38.8, 225, 299)}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def curve(values, start, end, count):
    values = gaussian_filter1d(values, 8., axis=0, mode='nearest')
    x = np.linspace(start, end, count)
    y = np.stack([np.interp(x, np.arange(len(values)), a) for a in values.T], -1)
    # Zero position, velocity and acceleration at the phase boundaries.
    u = np.linspace(0, 1, count)[:, None]
    return y * np.sin(np.pi*u)**4


def dense_project(m, base, delta, rotations, aq, av, joints, hands):
    """Restore rigid grasp geometry after smoothing, with bounded arm velocity.

    The object pose remains a variable, so unavailable motion is projected out
    instead of stretching the two-hand grasp or moving a support joint.
    """
    d0,fk=mujoco.MjData(m),mujoco.MjData(m)
    jp=np.zeros((3,m.nv));jr=jp.copy()
    for phase,(start,end,_,_) in PHASES.items():
        first,last=round(start/.005),round(end/.005)
        phase_q=base['qref'][first:last+1,aq]
        low=np.minimum(np.maximum(m.jnt_range[joints,0]-phase_q.min(0)+.002,-.2),0.)
        high=np.maximum(np.minimum(m.jnt_range[joints,1]-phase_q.max(0)-.002,.2),0.)
        previous=np.zeros(14)
        for tick in range(first,last+1):
            q0=base['qref'][tick];d0.qpos[:]=q0;mujoco.mj_kinematics(m,d0)
            positions=d0.xpos[hands].copy();r0=Rotation.from_matrix(d0.xmat[hands].reshape(2,3,3))
            desired_delta=delta[tick,aq].copy();desired_object=np.r_[delta[tick,44:47],rotations[tick]]
            radius=np.sin(np.pi*(tick-first)/(last-first))**2
            lower=np.maximum(low*radius,previous-.0035)
            upper=np.minimum(high*radius,previous+.0035)
            # Endpoint envelope can become tighter than the velocity box by
            # floating-point roundoff only near zero motion.
            lower=np.minimum(lower,upper-1e-12)
            arm=np.clip(desired_delta,lower,upper);dc=desired_object.copy()
            for _ in range(6):
                fk.qpos[:]=q0;fk.qpos[aq]+=arm
                mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
                rot=Rotation.from_rotvec(dc[3:]);rel=rot.apply(positions-q0[44:47]);desired=rot*r0
                rows=[];errors=[]
                for side,body in enumerate(hands):
                    mujoco.mj_jacBody(m,fk,jp,jr,body)
                    x,y,z=rel[side];skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    rows.extend([np.c_[jp[:,av],-np.eye(3),skew],np.c_[.3*jr[:,av],np.zeros((3,3)),-.3*np.eye(3)]])
                    errors.extend([q0[44:47]+dc[:3]+rel[side]-fk.xpos[body],.3*(desired[side]*Rotation.from_matrix(fk.xmat[body].reshape(3,3)).inv()).as_rotvec()])
                w=np.r_[np.ones(3),np.ones(3)*.2]
                A=np.r_[100*np.concatenate(rows),np.c_[.04*np.eye(14),np.zeros((14,6))],np.c_[np.zeros((6,14)),np.diag(w)]]
                b=np.r_[100*np.concatenate(errors),.04*(desired_delta-arm),w*(desired_object-dc)]
                step=lsq_linear(A,b,bounds=(np.r_[lower-arm,np.r_[np.ones(3)*-.04,np.ones(3)*-.1]-dc]-1e-12,np.r_[upper-arm,np.r_[np.ones(3)*.04,np.ones(3)*.1]-dc]+1e-12),method='bvls',tol=1e-9).x
                arm+=step[:14];dc+=step[14:]
                if np.linalg.norm(step)<1e-8:break
            delta[tick,aq]=arm;delta[tick,44:47]=dc[:3];rotations[tick]=dc[3:];previous=arm
        delta[[first,last]]=0.;rotations[[first,last]]=0.
        print('DENSE_PHASE',phase,flush=True)


def generate(seed, out, object_limit=.03, spread=35., posture_gain=12., disabled=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), out
    base = dict(np.load(CHECKPOINT))
    m = source_model()
    if disabled:
        np.savez_compressed(out,**{k:base[k] for k in ['qref','vref','uref']},goal_position=base['qref'][-1,44:47],goal_quaternion=base['qref'][-1,47:51])
        out.with_suffix('.json').write_text(json.dumps(dict(diffusion_disabled=True,method='Exact identity branch of the additive reference path; accepted physics-rate arrays are copied without resampling.'),indent=2))
        return
    p = ROOT / 'logs/compact_diffusion_v1'
    sample_path = p / f'proposals/sample_{seed}.npz'
    calibration_path = p / 'calibration/sample_0.npz'
    sample, calibration = np.load(sample_path), np.load(calibration_path)
    provenance = json.loads((p/'proposals/provenance.json').read_text())
    assert sha(DIFFUSION_CHECKPOINT) == provenance['checkpoint_sha256']
    assert np.array_equal(sample['condition'], calibration['condition'])
    native = provenance['native_joint_names']
    joints = [j for j in range(m.njnt) if any(t in m.joint(j).name for t in ['shoulder_', 'elbow_', 'wrist_'])]
    aq, av = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    actuators = np.array([np.flatnonzero(m.actuator_trnid[:, 0] == j)[0] for j in joints])
    assert len(joints) == 14
    pose_columns = [9+native.index(m.joint(j).name) for j in joints]
    demo = np.load(p/'proposals/conditioning.npz')['demonstration']
    heading0 = np.arctan2(*(demo[-1, 38:40]-demo[0, 38:40])[::-1])
    heading1 = np.arctan2(*(base['qref'][-1, 44:46]-base['qref'][0, 44:46])[::-1])
    align = Rotation.from_euler('z', heading1-heading0)
    obj_innovation = align.apply(sample['object_body_position']-calibration['object_body_position'])
    rot_innovation=align.apply((Rotation.from_quat(sample['object_quaternion'],scalar_first=True)*Rotation.from_quat(calibration['object_quaternion'],scalar_first=True).inv()).as_rotvec())
    arm_innovation = sample['raw'][:, pose_columns]-calibration['raw'][:, pose_columns]
    delta = np.zeros_like(base['qref'])
    object_rotation=np.zeros((len(delta),3))
    d0, fk = mujoco.MjData(m), mujoco.MjData(m)
    hands = [m.body('hand_'+side).id for side in ['left', 'right']]
    jp = np.zeros((3, m.nv)); jr = jp.copy()
    maximum_residual = 0.
    detail = {}
    for phase, (start, end, ds, de) in PHASES.items():
        first, last = round(start/.005), round(end/.005)
        ticks = np.unique(np.r_[np.arange(first, last+1, 4), last])
        target = spread*curve(obj_innovation, ds, de, len(ticks))
        # Smooth norm bound, shared across samples and phases.
        target /= np.sqrt(1.+np.sum(target**2, axis=1, keepdims=True)/object_limit**2)
        target_rot=spread*curve(rot_innovation,ds,de,len(ticks))
        target_rot/=np.sqrt(1.+np.sum(target_rot**2,axis=1,keepdims=True)/.08**2)
        target=np.c_[target,target_rot]
        prior = .12*np.tanh(posture_gain*curve(arm_innovation, ds, de, len(ticks))/.12)
        solved = []; projected = []; previous = np.zeros(14)
        for k, tick in enumerate(ticks):
            q0 = base['qref'][tick]
            d0.qpos[:] = q0
            mujoco.mj_kinematics(m, d0)
            baseline_hand_position = d0.xpos[hands].copy()
            target_rotation = Rotation.from_matrix(d0.xmat[hands].reshape(2, 3, 3))
            q = q0.copy()
            radius = .2*np.sin(np.pi*k/(len(ticks)-1))**2
            lower = np.maximum(q0[aq]-radius, np.minimum(m.jnt_range[joints, 0]+.002, q0[aq]))
            upper = np.minimum(q0[aq]+radius, np.maximum(m.jnt_range[joints, 1]-.002, q0[aq]))
            dc = target[k].copy()
            for iteration in range(12):
                fk.qpos[:] = q
                mujoco.mj_kinematics(m, fk); mujoco.mj_comPos(m, fk)
                matrices, errors = [], []
                rot=Rotation.from_rotvec(dc[3:])
                rel=rot.apply(baseline_hand_position-q0[44:47])
                desired=rot*target_rotation
                for side, body in enumerate(hands):
                    mujoco.mj_jacBody(m, fk, jp, jr, body)
                    x,y,z=rel[side];skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    matrices.extend([np.c_[jp[:, av], -np.eye(3),skew], np.c_[.3*jr[:, av], np.zeros((3,3)),-.3*np.eye(3)]])
                    errors.extend([q0[44:47]+dc[:3]+rel[side]-fk.xpos[body],
                        .3*(desired[side]*Rotation.from_matrix(fk.xmat[body].reshape(3, 3)).inv()).as_rotvec()])
                A, b = np.concatenate(matrices), np.concatenate(errors)
                # Bounded damped IK avoids singular null-space jumps and never
                # enlarges the baseline's existing soft joint-limit excursions.
                weights=np.r_[np.ones(3),np.ones(3)*.2]
                A = np.r_[100*A, np.c_[.005*np.eye(14), np.zeros((14,6))], np.c_[.02*np.eye(14),np.zeros((14,6))],np.c_[np.zeros((6,14)), np.diag(weights)]]
                b = np.r_[100*b, .005*(prior[k]-(q[aq]-q0[aq])), .02*(previous-(q[aq]-q0[aq])), weights*(target[k]-dc)]
                lo = np.r_[np.maximum(-.04, lower-q[aq]), np.r_[np.ones(3)*-.04,np.ones(3)*-.10]-dc]
                hi = np.r_[np.minimum(.04, upper-q[aq]), np.r_[np.ones(3)*.04,np.ones(3)*.10]-dc]
                step = lsq_linear(A, b, bounds=(lo-1e-12, hi+1e-12), method='bvls', tol=1e-10).x
                q[aq] += step[:14]; dc += step[14:]
                if np.linalg.norm(step) < 1e-8:
                    break
            fk.qpos[:] = q; mujoco.mj_kinematics(m, fk)
            expected=q0[44:47]+dc[:3]+Rotation.from_rotvec(dc[3:]).apply(baseline_hand_position-q0[44:47])
            maximum_residual = max(maximum_residual, float(np.linalg.norm(fk.xpos[hands]-expected, axis=1).max()))
            solved.append(q[aq]-q0[aq])
            projected.append(dc)
            previous=q[aq]-q0[aq]
        solved = np.asarray(solved)
        solved[[0, -1]] = 0.
        full = np.arange(first, last+1)
        delta[np.ix_(full, aq)] = CubicSpline(ticks, solved, bc_type=((1, np.zeros(14)), (1, np.zeros(14))))(full)
        projected=np.asarray(projected);projected[[0,-1]]=0.
        motion=CubicSpline(ticks, projected, bc_type=((1, np.zeros(6)), (1, np.zeros(6))))(full)
        delta[full, 44:47] = motion[:,:3];object_rotation[full]=motion[:,3:]
        detail[phase] = dict(peak_requested_translation_m=float(np.linalg.norm(target[:,:3], axis=1).max()),
                            peak_projected_translation_m=float(np.linalg.norm(projected[:,:3],axis=1).max()),peak_projected_rotation_rad=float(np.linalg.norm(projected[:,3:],axis=1).max()),
                            peak_arm_delta_rad=float(abs(solved).max()))
    # Remove redundant-IK jitter without filtering any accepted reference data.
    for start,end,_,_ in PHASES.values():
        first,last=round(start/.005),round(end/.005)
        full=np.arange(first,last+1)
        window=np.sin(np.pi*np.linspace(0,1,len(full)))**2
        delta[full]=gaussian_filter1d(delta[full],20.,axis=0,mode='constant')*window[:,None]
        object_rotation[full]=gaussian_filter1d(object_rotation[full],20.,axis=0,mode='constant')*window[:,None]
    if not disabled:
        dense_project(m,base,delta,object_rotation,aq,av,joints,hands)
    qref = base['qref']+delta
    qref[:,47:51]=(Rotation.from_rotvec(object_rotation)*Rotation.from_quat(base['qref'][:,47:51],scalar_first=True)).as_quat(scalar_first=True)
    # Preserve exact untouched values, including quaternion signs/rounding.
    qref[~np.any(object_rotation,axis=1),47:51]=base['qref'][~np.any(object_rotation,axis=1),47:51]
    dv = np.gradient(delta[:, aq], .005, axis=0)
    da = np.gradient(dv, .005, axis=0)
    vref = base['vref'].copy()
    vref[:, av] += dv
    vref[:, 43:46] += np.gradient(delta[:, 44:47], .005, axis=0)
    vref[:,46:49]+=Rotation.from_quat(base['qref'][:,47:51],scalar_first=True).inv().apply(np.gradient(object_rotation,.005,axis=0))
    uref = base['uref'].copy()
    baseline_acc = np.gradient(base['vref'], .005, axis=0)
    increments = np.zeros_like(uref)
    baseline_rne, changed_rne = np.zeros(m.nv), np.zeros(m.nv)
    interpolated_residual=0.; interpolated_angle=0.
    if not disabled:
        for tick in np.flatnonzero(np.any(delta != 0., axis=1)):
            for d, q, v, acc, result in [
                (d0, base['qref'][tick], base['vref'][tick], baseline_acc[tick], baseline_rne),
                (fk, qref[tick], vref[tick], baseline_acc[tick].copy(), changed_rne)]:
                d.qpos[:] = q; d.qvel[:] = v; d.qacc[:] = acc
                if d is fk:
                    d.qacc[av] += da[tick]
                mujoco.mj_fwdPosition(m, d); mujoco.mj_fwdVelocity(m, d)
                mujoco.mj_rne(m, d, 1, result)
            rr=Rotation.from_rotvec(object_rotation[tick])
            expected=qref[tick,44:47]+rr.apply(d0.xpos[hands]-base['qref'][tick,44:47])
            interpolated_residual=max(interpolated_residual,float(np.linalg.norm(fk.xpos[hands]-expected,axis=1).max()))
            interpolated_angle=max(interpolated_angle,float(np.linalg.norm((Rotation.from_matrix(fk.xmat[hands].reshape(2,3,3))*(rr*Rotation.from_matrix(d0.xmat[hands].reshape(2,3,3))).inv()).as_rotvec(),axis=1).max()))
            kp = m.actuator_gainprm[actuators, 0]
            kd = -m.actuator_biasprm[actuators, 2]
            increments[tick, actuators] = delta[tick, aq]+(kd*dv[tick]+(changed_rne-baseline_rne)[av])/kp
        uref += increments
    unchanged = np.ones(m.nq, dtype=bool); unchanged[np.r_[aq, 44:51]] = False
    old_excess = np.maximum(base['qref'][:, aq]-m.jnt_range[joints, 1], m.jnt_range[joints, 0]-base['qref'][:, aq])
    new_excess = np.maximum(qref[:, aq]-m.jnt_range[joints, 1], m.jnt_range[joints, 0]-qref[:, aq])
    limit_excess = float((new_excess-np.maximum(old_excess,0.)).max())
    report = dict(seed=seed,object_limit_m=object_limit,spread_gain=spread,posture_gain=posture_gain,
        source_sample=str(sample_path),source_sample_sha256=sha(sample_path),calibration_sha256=sha(calibration_path),
        frozen_checkpoint_sha256=provenance['checkpoint_sha256'],phases=detail,
        maximum_hand_ik_residual_m=maximum_residual,maximum_arm_position_delta_rad=float(abs(delta[:, aq]).max()),
        maximum_physics_rate_hand_residual_m=interpolated_residual,maximum_physics_rate_hand_angle_residual_rad=interpolated_angle,
        maximum_added_arm_velocity_radps=float(abs(dv).max()),maximum_added_arm_acceleration_radps2=float(abs(da).max()),
        added_arm_position_limit_excess_rad=limit_excess,baseline_soft_limit_excursion_rad=float(old_excess.max()),maximum_servo_increment_rad=float(abs(increments).max()),
        root_legs_waist_gripper_bit_identical=bool(np.array_equal(qref[:, unchanged],base['qref'][:, unchanged])),
        nonarm_feedforward_bit_identical=bool(np.array_equal(uref[:, np.setdiff1d(np.arange(m.nu),actuators)],base['uref'][:, np.setdiff1d(np.arange(m.nu),actuators)])),
        method='Nonlinear arm-only rigid-grasp IK, frozen-sample object translation/rotation and arm posture; smoothing followed by physics-rate bounded grasp projection; additive baseline references and servo feedforward with RNE torque difference; no root/leg/waist/gripper changes.')
    report['pass_cheap_gate'] = bool(maximum_residual<.0002 and interpolated_residual<.001 and interpolated_angle<.01 and limit_excess<=1e-10 and abs(dv).max()<1.5 and abs(increments).max()<.35 and report['root_legs_waist_gripper_bit_identical'])
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    if not report['pass_cheap_gate']:
        np.savez_compressed(out.with_name(out.stem+'_rejected.npz'),delta=delta,dv=dv,da=da)
        raise RuntimeError(report)
    np.savez_compressed(out,qref=qref,vref=vref,uref=uref,goal_position=base['qref'][-1,44:47],goal_quaternion=base['qref'][-1,47:51])
    print(json.dumps(report, indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--object-limit',type=float,default=.03);p.add_argument('--spread',type=float,default=35.)
    p.add_argument('--posture-gain',type=float,default=12.);p.add_argument('--disabled',action='store_true')
    a=p.parse_args();generate(a.seed,a.out,a.object_limit,a.spread,a.posture_gain,a.disabled)
