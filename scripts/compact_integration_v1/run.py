"""Continue preserved compact-hand pickups using finite actuators only.

No model edits or physical state edits occur after initialization. Prefix replay
preserves MuJoCo solver history. Planning data and SONIC's observation simulator
do not control the real object's state or forces.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    source = ROOT / args.source
    branch = args.branch or source.parent.name
    sys.path[:0] = [str(ROOT / 'scripts' / branch),
                    str(ROOT / 'scripts/articulated_gripper_v1'), str(ROOT / 'scripts')]
    import mechanism
    import physical
    import frozen
    from compact_gripper_balance import Balance

    out = ROOT / 'logs/compact_integration_v1' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    a = np.load(source)
    m = mujoco.MjModel.from_binary_path(str(source.with_name(source.stem + '_model.mjb')))
    prefix = len(a['ctrl_physics']) if args.prefix is None else args.prefix
    assert 0 < prefix <= len(a['ctrl_physics'])
    d = mujoco.MjData(m)
    d.qpos[:] = a['initial_qpos']
    d.qvel[:] = a['initial_qvel']
    mujoco.mj_forward(m, d)
    for u in a['ctrl_physics'][:prefix]:
        d.ctrl[:] = u
        mujoco.mj_step(m, d)
    aj, aq, av, arm, pads = mechanism.indices(m)
    arms = physical.Arms(m, d.qpos, max_speed=5.)
    if args.hand_control == 'loaded_ik':
        from loaded_arms import LoadedArms
        arms = LoadedArms(m, d, max_speed=4.)
    servo = physical.ObjectServo(m, d, pads, aq)
    balance = Balance(m, d)
    qref = d.qpos.copy()
    yaw0 = Rotation.from_quat(np.roll(d.qpos[3:7], -1)).as_euler('xyz')[2]
    origin = d.qpos[:3].copy()
    loaded_hand_control = a['ctrl_physics'][prefix-1].copy()
    waist = np.array([i for i,j in enumerate(aj[:29]) if 'waist_' in m.joint(int(j)).name])
    press_rest = d.qpos[aq[29:31]].copy()
    fk = mujoco.MjData(m)
    fk.qpos[:]=d.qpos;fk.qpos[aq[29:31]]=0.;mujoco.mj_kinematics(m,fk)
    root_anchor=fk.geom_xpos[pads].copy()
    root_normals=fk.xmat[arms.hands].reshape(2,3,3)[:,:,2].copy()
    ob = m.body('object/object').id
    oq = m.jnt_qposadr[m.joint('object/floating_base_joint').id]
    meta = dict(walk_start_frame=-1, walk_end_frame=-1)
    ctrl = None
    if args.phase == 'walk':
        import sustained_carry_motion as motion
        import build_engineered_ground_large_model as native_builder
        import active_largebox_sonic as sonic
        native_builder.large_ground_gripper_spec = frozen.native_spec
        sonic.large_ground_gripper_spec = frozen.native_spec
        sonic.original_largebox_spec = frozen.object_spec
        motion.OUT = out
        ctrl, meta = motion.controller(m, d.qpos.copy())
    cs = list(a['ctrl_physics'][:prefix])
    qs = list(a['qpos'][:prefix])
    vs = list(a['qvel'][:prefix])
    cts = list(a['contacts'][:prefix])
    rows = []
    following = []
    closure = np.zeros(2)
    sums = np.zeros(6)
    reason = 'duration_reached'
    for step in range(int(args.seconds / m.opt.timestep)):
        yaw = Rotation.from_quat(np.roll(d.qpos[3:7], -1)).as_euler('xyz')[2] - yaw0
        rz = Rotation.from_euler('z', yaw)
        C = d.xipos[ob]
        R = Rotation.from_matrix(d.xmat[ob].reshape(3, 3))
        desired = d.qpos[:3] + rz.apply(servo.C0 - servo.root0)
        delta = np.clip(args.restore * (desired-C) - .03*(d.qvel[servo.od:servo.od+3]-d.qvel[:3]), -.025, .025)
        angle = ((rz*servo.R0)*R.inv()).as_rotvec()
        correction = .2*angle - .02*R.apply(d.qvel[servo.od+3:servo.od+6])
        size = np.linalg.norm(correction)
        if size > .04:
            correction *= .04/size
        command_R = Rotation.from_rotvec(correction)*R
        target = command_R.apply(servo.local) + C + delta
        normals = command_R.apply(servo.local_normals)
        if args.hand_control == 'loaded_ik':
            target = rz.apply(root_anchor-origin)+d.qpos[:3]
            target += physical.smooth(step*.005/2.)*(args.forward*np.cross(root_normals[0],[0.,0.,1.])+[0,0,-args.lower])
            normals = rz.apply(root_normals)
        closure = np.clip(closure+2.*(d.qpos[aq[29:31]]-press_rest)*.005, -.025, .035)
        for side in range(2):
            target[side*3:side*3+3] -= closure[side]*normals[side]
        if step % 4 == 0:
            if args.hand_control in ['object','loaded_ik']:
                arms.solve(d, target, qref)
            if ctrl is not None:
                lower = ctrl.control(d)
        if ctrl is None:
            qb = qref.copy()
            qb[arms.armq] = arms.seed
            vb = np.zeros(m.nv)
            vb[av[arm]] = arms.vel
            tau, _ = balance.torques(d, qb, vb)
            u = d.qpos[aq] + (tau[av]-m.actuator_biasprm[:, 2]*d.qvel[av])/m.actuator_gainprm[:, 0]
        else:
            u = np.r_[lower, np.zeros(m.nu-29)]
            if args.blend_seconds > 0:
                blend=physical.smooth(step*.005/args.blend_seconds)
                u[:29]=loaded_hand_control[:29]*(1-blend)+u[:29]*blend
        if args.waist_loaded:
            u[waist]=loaded_hand_control[waist]
        if args.hand_control in ['object','loaded_ik']:
            u = arms.apply(d, u)
        else:
            u[arm] = loaded_hand_control[arm]
        u[29:31] = arms.grip(d, args.squeeze)
        u[33:35] = 1.57
        if m.nu == 37:
            u[35:37] = physical.thumb_control(m, d)
        u = np.clip(u, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        d.ctrl[:] = u
        mujoco.mj_step(m, d)
        touch, force, imp, vertical, selfimp, invalid = physical.contacts(m, d)
        sums += np.r_[imp, selfimp, invalid]
        cs.append(u.copy()); qs.append(d.qpos.copy()); vs.append(d.qvel.copy()); cts.append(touch)
        following.append(np.r_[arms.seed, d.qpos[arms.armq], target.ravel(), d.geom_xpos[pads].ravel()])
        rows.append(np.r_[step*.005, d.qpos[:3], d.xipos[ob], force, arms.cost, sums, closure, delta, angle])
        if step % 100 == 99:
            print(args.tag, round(step*.005, 3), 'distance', round(np.linalg.norm(d.qpos[:2]-origin[:2]), 3),
                  'rootZ', round(d.qpos[2], 3), 'objroot', np.round(desired-C, 3),
                  'N', np.round(force, 2), 'cost', round(arms.cost, 3), 'bad', sums[3:], flush=True)
        if sums[3] > 1e-5: reason = 'forbidden_object_contact'
        elif sums[4] > .001: reason = 'robot_self_contact'
        elif sums[5] > .001: reason = 'invalid_terrain_contact'
        elif touch[0]: reason = 'object_terrain_contact'
        elif d.qpos[2] < .4: reason = 'robot_fall'
        if reason != 'duration_reached':
            print('EARLY_REJECT', reason, 'step', step, 'impulses', sums, flush=True)
            break
    np.savez_compressed(out/'rollout.npz', initial_qpos=a['initial_qpos'], initial_qvel=a['initial_qvel'],
                        ctrl_physics=cs, qpos=qs, qvel=vs, contacts=cts, integration_diagnostics=rows,
                        arm_following=following, phase_opening_frame=-1, phase_withdrawal_end_frame=-1,
                        walking_start_physics_step=prefix+meta['walk_start_frame']*4,
                        walking_end_physics_step=prefix+meta['walk_end_frame']*4)
    mujoco.mj_saveModel(m, str(out/'rollout_model.mjb'), None)
    result = dict(source=str(source.relative_to(ROOT)), prefix=prefix, arguments=vars(args),
                  reason=reason, continuation_seconds=len(rows)*.005, impulses=sums.tolist(), reference=meta,
                  root_displacement_continuation_m=float(np.linalg.norm(d.qpos[:2]-origin[:2])))
    (out/'generation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--branch', choices=['hook_palms_v1', 'corner_hands_v1'])
    p.add_argument('--tag', required=True)
    p.add_argument('--prefix', type=int)
    p.add_argument('--phase', choices=['hold', 'walk'], default='hold')
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--restore', type=float, default=1.)
    p.add_argument('--squeeze', type=float, default=2.)
    p.add_argument('--hand-control', choices=['object', 'loaded', 'loaded_ik'], default='object')
    p.add_argument('--lower',type=float,default=0.)
    p.add_argument('--forward',type=float,default=0.)
    p.add_argument('--waist-loaded',action='store_true')
    p.add_argument('--blend-seconds',type=float,default=0.)
    run(p.parse_args())
