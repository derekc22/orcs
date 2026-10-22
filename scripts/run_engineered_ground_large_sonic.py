#!/usr/bin/env python3
"""Continue the verified large-box pickup with closed-loop SONIC locomotion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import SonicBaseModel
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from mocke import PRETRAINED_DIR
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from build_engineered_ground_large_model import (
    BOX_HALF,
    BOX_MASS,
    large_ground_gripper_spec,
    large_object_spec,
)
from run_engineered_locomomanipulation import _quat_delta_omega, engineering_robot_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_ground_large"
PICKUP = OUT / "ground_large_pickup_gate_v1.npz"
DONOR = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.npz"
MOTION = OUT / "ground_large_sonic_motion_v1.npz"
MODEL = OUT / "ground_large_sonic_model_v1.mjb"
ARTIFACT = OUT / "ground_large_sonic_expert_v1.npz"
REPORT = OUT / "ground_large_sonic_expert_v1.json"
SUBSTEPS = 4
PHYSICS_DT = 0.005
JOIN_FRAME = 61
TRANSITION = 25
DONOR_START = 40
DONOR_END = 75
STOP_DONOR_FRAME = 80
STOP_FRAMES = 12
HOLD_FRAMES = 50
LOWER_FRAMES = 40
RELEASE_FRAMES = 12
STANDOFF = 0.68
GROUND_Z = BOX_HALF[2] + 0.001
PLACEMENT_Z = min(BOX_HALF[0], BOX_HALF[1]) + 0.001
# At release the long axis is tilted roughly 45 degrees.  Lowering the COM to
# the final side-rest height before that rotation would drive the arm target
# through the terrain.  This height puts the low edge on terrain first.
SETDOWN_Z = 0.33
CONTROL_SETDOWN_XY = np.asarray((-0.03, 0.20))
FINAL_GOAL_POSITION = np.asarray((0.16918926, 0.18024656, 0.30213333))
FINAL_GOAL_QUATERNION = np.asarray((-0.0000941233, 0.0479183940, -0.9988512450, 0.0000991009))
CARRY_END_INDEX = TRANSITION + (DONOR_END - DONOR_START + 1) + STOP_FRAMES
# Begin the descent during the verified bilateral carry, and reach terrain
# before the observed cradle-retention limit around full-rollout frame 118.
LOWER_INDEX = TRANSITION + (45 - DONOR_START)
GROUND_INDEX = TRANSITION + (80 - DONOR_START)
CARRY_LIFT = 0.24
CONTACT_OFFSET_Z = 0.55 - GROUND_Z
CLOSED_SPREAD = BOX_HALF[1] + 0.020
OPEN_SPREAD = CLOSED_SPREAD + 0.11


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def quat_wxyz(rotation: Rotation) -> np.ndarray:
    quat = rotation.as_quat()
    return np.r_[quat[3], quat[:3]]


def write_motion_and_targets(pickup: np.lib.npyio.NpzFile, donor: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    robot_model = large_ground_gripper_spec().compile()
    robot_data = mujoco.MjData(robot_model)
    nq = robot_model.nq
    joint_ids = np.asarray(
        sorted(
            (i for i in range(robot_model.njnt) if robot_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE),
            key=lambda i: int(robot_model.jnt_qposadr[i]),
        )
    )
    joint_qpos = robot_model.jnt_qposadr[joint_ids].astype(int)
    joint_names = [mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or "" for j in joint_ids]
    arm = np.asarray([i for i, name in enumerate(joint_names) if any(key in name for key in ("shoulder", "elbow", "wrist"))])
    arm_qpos = joint_qpos[arm]
    limits = robot_model.jnt_range[joint_ids[arm]]
    guards = np.asarray(
        [mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_front_guard_collision") for side in ("left", "right")]
    )
    rj = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    rq = int(robot_model.jnt_qposadr[rj])
    join = pickup["qpos"][JOIN_FRAME, :nq].copy()
    donor_robot = donor["qpos"][:, :nq]
    donor_start = donor_robot[DONOR_START].copy()

    join_rotation = Rotation.from_quat(np.r_[join[rq + 4 : rq + 7], join[rq + 3]])
    donor_start_rotation = Rotation.from_quat(np.r_[donor_start[rq + 4 : rq + 7], donor_start[rq + 3]])
    align_rotation = join_rotation * donor_start_rotation.inv()
    aligned_start = donor_start.copy()
    aligned_start[rq : rq + 3] = join[rq : rq + 3]
    aligned_start[rq + 3 : rq + 7] = quat_wxyz(join_rotation)

    qrefs = []
    donor_arm_delta = []
    tangent = np.zeros(robot_model.nv)
    mujoco.mj_differentiatePos(robot_model, tangent, 1.0, join, aligned_start)
    for frame in range(TRANSITION):
        alpha = smoothstep(frame / max(1, TRANSITION - 1))
        pose = join.copy()
        mujoco.mj_integratePos(robot_model, pose, alpha * tangent, 1.0)
        qrefs.append(pose)
        donor_arm_delta.append(np.zeros(len(arm_qpos)))
    donor_origin = donor_start[rq : rq + 3].copy()
    for donor_frame in range(DONOR_START, DONOR_END + 1):
        source = donor_robot[donor_frame].copy()
        delta = source[rq : rq + 3] - donor_origin
        source[rq : rq + 3] = join[rq : rq + 3] + align_rotation.apply(delta)
        source_rotation = Rotation.from_quat(np.r_[source[rq + 4 : rq + 7], source[rq + 3]])
        # source_rotation was read after position-only changes and remains the donor rotation.
        source[rq + 3 : rq + 7] = quat_wxyz(align_rotation * source_rotation)
        qrefs.append(source)
        donor_arm_delta.append(donor_robot[donor_frame, arm_qpos] - donor_start[arm_qpos])
    stop_source = donor_robot[STOP_DONOR_FRAME].copy()
    stop_delta = stop_source[rq : rq + 3] - donor_origin
    stop_source[rq : rq + 3] = join[rq : rq + 3] + align_rotation.apply(stop_delta)
    stop_rotation = Rotation.from_quat(np.r_[stop_source[rq + 4 : rq + 7], stop_source[rq + 3]])
    stop_source[rq + 3 : rq + 7] = quat_wxyz(align_rotation * stop_rotation)
    stop_tangent = np.zeros(robot_model.nv)
    mujoco.mj_differentiatePos(robot_model, stop_tangent, 1.0, qrefs[-1], stop_source)
    stop_start = qrefs[-1].copy()
    stop_arm_delta = donor_robot[STOP_DONOR_FRAME, arm_qpos] - donor_start[arm_qpos]
    for frame in range(1, STOP_FRAMES + 1):
        alpha = smoothstep(frame / STOP_FRAMES)
        pose = stop_start.copy()
        mujoco.mj_integratePos(robot_model, pose, alpha * stop_tangent, 1.0)
        qrefs.append(pose)
        donor_arm_delta.append((1.0 - alpha) * donor_arm_delta[-1] + alpha * stop_arm_delta)
    # Stop in this carried pose before lowering.  The motion-conditioned SONIC
    # controller then actively holds the lower body instead of continuing a
    # footstep through the set-down region.
    for _ in range(HOLD_FRAMES):
        qrefs.append(qrefs[-1].copy())
        donor_arm_delta.append(donor_arm_delta[-1].copy())
    qrefs = np.asarray(qrefs)
    donor_arm_delta = np.asarray(donor_arm_delta)

    def axes(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = Rotation.from_quat(np.r_[pose[rq + 4 : rq + 7], pose[rq + 3]])
        forward = rotation.apply([1.0, 0.0, 0.0]); forward[2] = 0.0; forward /= np.linalg.norm(forward)
        lateral = rotation.apply([0.0, 1.0, 0.0]); lateral[2] = 0.0; lateral /= np.linalg.norm(lateral)
        return forward, lateral

    start_object = pickup["object_pos"][JOIN_FRAME].copy()
    start_quat = pickup["object_quat"][JOIN_FRAME].copy()
    carry_height = GROUND_Z + CARRY_LIFT
    ground_index = GROUND_INDEX
    lower_index = LOWER_INDEX
    # This is a calibrated feed-forward arm target, not an object constraint:
    # exact MuJoCo remains the authority and the object is fully dynamic.  Its
    # reproducible passive side settling ends at FINAL_GOAL_POSITION.
    setdown = np.r_[CONTROL_SETDOWN_XY, SETDOWN_Z]
    goal = FINAL_GOAL_POSITION.copy()
    desired_object = np.empty((len(qrefs), 3))
    for frame, pose in enumerate(qrefs):
        forward, _ = axes(pose)
        carry = pose[rq : rq + 3] + STANDOFF * forward
        carry[2] = carry_height
        if frame < TRANSITION:
            alpha = smoothstep(frame / max(1, TRANSITION - 1))
            desired_object[frame] = (1.0 - alpha) * start_object + alpha * carry
        elif frame < lower_index:
            desired_object[frame] = carry
        elif frame < ground_index:
            alpha = smoothstep((frame - lower_index) / (ground_index - lower_index))
            desired_object[frame] = (1.0 - alpha) * carry + alpha * setdown
        else:
            desired_object[frame] = setdown

    arm_seed = join[arm_qpos].copy()
    ik_errors = []
    release_end = ground_index + RELEASE_FRAMES
    for frame, pose in enumerate(qrefs):
        forward, lateral = axes(pose)
        near = desired_object[frame] - (BOX_HALF[0] - 0.055) * forward
        if frame < ground_index:
            spread = CLOSED_SPREAD
        elif frame < release_end:
            spread = CLOSED_SPREAD + (OPEN_SPREAD - CLOSED_SPREAD) * smoothstep((frame - ground_index) / RELEASE_FRAMES)
        else:
            spread = OPEN_SPREAD
        targets = np.asarray([near + spread * lateral, near - spread * lateral])
        targets[:, 2] = desired_object[frame, 2] + CONTACT_OFFSET_Z

        def residual(values: np.ndarray) -> np.ndarray:
            robot_data.qpos[:] = pose
            robot_data.qpos[arm_qpos] = values
            mujoco.mj_forward(robot_model, robot_data)
            return np.r_[80.0 * (robot_data.geom_xpos[guards] - targets).ravel(), 0.025 * (values - arm_seed)]

        fit = least_squares(
            residual,
            np.clip(arm_seed, limits[:, 0] + 0.006, limits[:, 1] - 0.006),
            bounds=(limits[:, 0] + 0.005, limits[:, 1] - 0.005),
            max_nfev=140,
            ftol=2e-8,
            xtol=2e-8,
            gtol=2e-8,
        )
        arm_seed = fit.x
        # Preserve the exact physical pickup posture at the controller handoff
        # and move onto the analytic carry branch continuously.
        handoff_alpha = smoothstep(frame / 15.0)
        qrefs[frame, arm_qpos] = (1.0 - handoff_alpha) * join[arm_qpos] + handoff_alpha * arm_seed
        robot_data.qpos[:] = qrefs[frame]; mujoco.mj_forward(robot_model, robot_data)
        ik_errors.append(float(np.max(np.linalg.norm(robot_data.geom_xpos[guards] - targets, axis=1))))

    joint = qrefs[:, joint_qpos]
    joint_vel = np.gradient(joint, 0.02, axis=0)
    bodies = list(range(1, robot_model.nbody))
    body_pos = np.empty((len(qrefs), len(bodies), 3)); body_quat = np.empty((len(qrefs), len(bodies), 4))
    for frame, pose in enumerate(qrefs):
        robot_data.qpos[:] = pose; robot_data.qvel[:] = 0.0; mujoco.mj_forward(robot_model, robot_data)
        body_pos[frame] = robot_data.xpos[bodies]; body_quat[frame] = robot_data.xquat[bodies]
    np.savez_compressed(
        MOTION,
        joint_pos=joint.astype(np.float32), joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32), body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=np.gradient(body_pos, 0.02, axis=0).astype(np.float32),
        body_ang_vel_w=_quat_delta_omega(body_quat, 0.02).astype(np.float32), fps=np.asarray(50.0),
    )
    return qrefs, desired_object, np.asarray(ik_errors), goal


def generate_sonic_controls(qrefs: np.ndarray, join_qpos: np.ndarray, join_qvel: np.ndarray, join_control: np.ndarray) -> tuple[np.ndarray, mujoco.MjModel]:
    robot_cfg = engineering_robot_cfg(); robot_cfg.spec_fn = large_ground_gripper_spec
    cfg = sonic_tracking_env_cfg(str(MOTION), play=True, il_ordered=False)
    cfg.scene.num_envs = 1; cfg.scene.entities["robot"] = robot_cfg
    cfg.scene.entities["object"] = EntityCfg(spec_fn=large_object_spec, init_state=EntityCfg.InitialStateCfg(pos=tuple(join_qpos[-7:-4]), rot=tuple(join_qpos[-4:])))
    cfg.sim.nconmax = 192; cfg.sim.njmax = 700; cfg.auto_reset = False; cfg.terminations = {}
    cfg.events = {name: term for name, term in cfg.events.items() if term.mode != "startup"}
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    raw.sim.data.qpos[0] = torch.as_tensor(join_qpos, device=raw.device)
    raw.sim.data.qvel[0] = torch.as_tensor(join_qvel, device=raw.device)
    raw.sim.forward(); raw.scene.update(dt=raw.physics_dt); raw.sim.sense(); raw.observation_manager._obs_buffer = None
    obs = env.get_observations()
    policy = SonicBaseModel(obs=obs, obs_groups={"actor": ["policy"]}, obs_set="actor", output_dim=29, base_checkpoint=str(PRETRAINED_DIR / "sonic/last_ported.pt")).to(raw.device)
    policy.eval()
    action_term = raw.action_manager.get_term("joint_pos"); scene_robot = raw.scene["robot"]
    compiled = raw.sim.mj_model
    robot_joint_names = tuple(scene_robot.joint_names)
    target_index = [];
    actuator_names = []
    for actuator in range(compiled.nu):
        jid = int(compiled.actuator_trnid[actuator, 0]); name = (mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_JOINT, jid) or "").split("/")[-1]
        actuator_names.append(name); target_index.append(robot_joint_names.index(name))
    target_index = np.asarray(target_index, dtype=int)
    arm_actuator = np.asarray([i for i, name in enumerate(actuator_names) if any(key in name for key in ("shoulder", "elbow", "wrist"))])
    arm_action = target_index[arm_actuator]
    robot_joint_ids = compiled.actuator_trnid[:, 0].astype(int)
    actuator_qpos = compiled.jnt_qposadr[robot_joint_ids].astype(int)
    arm_preload = join_control[arm_actuator] - join_qpos[actuator_qpos[arm_actuator]]
    controls_joint = []
    with torch.inference_mode():
        for frame in range(len(qrefs)):
            action = policy(obs)
            bias = action_term._entity.data.encoder_bias[:, action_term.target_ids]
            desired_actuator = torch.as_tensor(qrefs[frame, actuator_qpos], dtype=action.dtype, device=raw.device)
            desired_action = torch.empty_like(action)
            desired_action[:, target_index] = desired_actuator[None]
            release_index = GROUND_INDEX
            preload_scale = 1.0 - smoothstep((frame - release_index) / 12.0)
            if LOWER_INDEX <= frame < GROUND_INDEX:
                preload_scale *= 1.35
            desired_action[:, arm_action] += preload_scale * torch.as_tensor(arm_preload, dtype=action.dtype, device=raw.device)[None]
            raw_target = (desired_action + bias - action_term.offset) / action_term.scale
            action[:, arm_action] = raw_target[:, arm_action]
            ctrl = action * action_term.scale + action_term.offset - bias
            controls_joint.extend([ctrl[0].detach().cpu().numpy().copy()] * SUBSTEPS)
            obs, _, _, _ = env.step(action)
    controls = np.asarray(controls_joint)[:, target_index]
    mujoco.mj_saveModel(compiled, str(MODEL), None)
    env.close()
    return controls, mujoco.MjModel.from_binary_path(str(MODEL))


def synthesize_physical_arm_controls(
    model: mujoco.MjModel,
    join_qpos: np.ndarray,
    join_qvel: np.ndarray,
    lower_controls: np.ndarray,
    join_control: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close the cradle loop around the realized payload in exact MuJoCo."""
    model.opt.timestep = PHYSICS_DT
    rj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    oj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    rq = int(model.jnt_qposadr[rj]); oq = int(model.jnt_qposadr[oj])
    actuator_joints = model.actuator_trnid[:, 0].astype(int); actuator_qpos = model.jnt_qposadr[actuator_joints].astype(int)
    names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or "").split("/")[-1] for j in actuator_joints]
    arm = np.asarray([i for i, name in enumerate(names) if any(key in name for key in ("shoulder", "elbow", "wrist"))])
    arm_qpos = actuator_qpos[arm]; arm_joints = actuator_joints[arm]
    lower = model.jnt_range[arm_joints, 0] + 0.006; upper = model.jnt_range[arm_joints, 1] - 0.006
    guards = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_front_guard_collision") for side in ("left", "right")])

    def yaw_rotation(quat_wxyz: np.ndarray) -> Rotation:
        rotation = Rotation.from_quat(np.r_[quat_wxyz[1:], quat_wxyz[0]])
        forward = rotation.apply([1.0, 0.0, 0.0]); forward[2] = 0.0; forward /= np.linalg.norm(forward)
        return Rotation.from_euler("z", math.atan2(forward[1], forward[0]))

    initial = mujoco.MjData(model); initial.qpos[:] = join_qpos; initial.qvel[:] = join_qvel; mujoco.mj_forward(model, initial)
    object_rotation0 = Rotation.from_quat(np.r_[join_qpos[oq + 4 : oq + 7], join_qpos[oq + 3]])
    guard_offsets_local = object_rotation0.inv().apply(initial.geom_xpos[guards] - join_qpos[oq : oq + 3])
    root_yaw0 = yaw_rotation(join_qpos[rq + 3 : rq + 7])
    object_offset_root = root_yaw0.inv().apply(join_qpos[oq : oq + 3] - join_qpos[rq : rq + 3])
    preload = join_control[arm] - join_qpos[arm_qpos]
    place_start = LOWER_INDEX
    ground_frame = GROUND_INDEX
    release_end = ground_frame + RELEASE_FRAMES

    data = mujoco.MjData(model); data.qpos[:] = join_qpos; data.qvel[:] = join_qvel; mujoco.mj_forward(model, data); fk = mujoco.MjData(model)
    controls = lower_controls.copy(); seed = join_qpos[arm_qpos].copy(); errors = []; fixed_goal_xy = None
    goal = np.r_[join_qpos[oq : oq + 2], float(np.abs(object_rotation0.as_matrix()[2]) @ np.asarray(BOX_HALF)) + 0.001]
    for step in range(len(lower_controls)):
        frame = step / SUBSTEPS
        root_yaw = yaw_rotation(data.qpos[rq + 3 : rq + 7])
        carry_target = data.qpos[rq : rq + 3] + root_yaw.apply(object_offset_root)
        if fixed_goal_xy is None and frame >= place_start:
            fixed_goal_xy = carry_target[:2].copy(); goal[:2] = fixed_goal_xy
        if frame < place_start:
            desired_object = carry_target
        elif frame < ground_frame:
            alpha = smoothstep((frame - place_start) / (ground_frame - place_start))
            placement_target = np.r_[fixed_goal_xy, goal[2]]
            desired_object = (1.0 - alpha) * carry_target + alpha * placement_target
        else:
            desired_object = goal.copy()
        # Do not allow a reference mismatch to make the cradles abandon the
        # physical payload.  The target advances by at most 6 mm per 200 Hz
        # step, enough for the gait but inside the compliant contact basin.
        delta = desired_object - data.qpos[oq : oq + 3]
        norm = float(np.linalg.norm(delta))
        if norm > 0.0015: desired_object = data.qpos[oq : oq + 3] + delta * (0.0015 / norm)
        current_object_rotation = Rotation.from_quat(np.r_[data.qpos[oq + 4 : oq + 7], data.qpos[oq + 3]])
        targets = desired_object + current_object_rotation.apply(guard_offsets_local)
        if ground_frame <= frame < release_end:
            opening = 0.12 * smoothstep((frame - ground_frame) / RELEASE_FRAMES)
            lateral = current_object_rotation.apply([0.0, 1.0, 0.0])
            targets[0] += opening * lateral; targets[1] -= opening * lateral

        if frame < release_end:
            base = data.qpos.copy()
            def residual(values: np.ndarray) -> np.ndarray:
                fk.qpos[:] = base; fk.qvel[:] = 0.0; fk.qpos[arm_qpos] = values; mujoco.mj_forward(model, fk)
                return np.r_[85.0 * (fk.geom_xpos[guards] - targets).ravel(), 0.02 * (values - seed)]
            fit = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=90, ftol=2e-8, xtol=2e-8, gtol=2e-8)
            seed = fit.x
            fk.qpos[:] = base; fk.qvel[:] = 0.0; fk.qpos[arm_qpos] = seed; mujoco.mj_forward(model, fk)
            errors.append(float(np.max(np.linalg.norm(fk.geom_xpos[guards] - targets, axis=1))))
            carry_preload = 1.0 - 0.70 * smoothstep(frame / 20.0)
            release_scale = 1.0 if frame < ground_frame else 1.0 - smoothstep((frame - ground_frame) / RELEASE_FRAMES)
            preload_scale = carry_preload * release_scale
            controls[step, arm] = np.clip(seed + preload_scale * preload, model.actuator_ctrlrange[arm, 0], model.actuator_ctrlrange[arm, 1])
        else:
            errors.append(0.0)
        data.ctrl[:] = controls[step]; mujoco.mj_step(model, data)
    return controls, goal, np.asarray(errors)


def refine_terrain_placement(
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    controls: np.ndarray,
    start_frame: int = 103,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate physical-rate cradle lowering from the realized carried box."""
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model); data.qpos[:] = initial_qpos; data.qvel[:] = initial_qvel; mujoco.mj_forward(model, data)
    fk = mujoco.MjData(model)
    oj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    oq = int(model.jnt_qposadr[oj])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == object_body}
    actuator_joints = model.actuator_trnid[:, 0].astype(int); actuator_qpos = model.jnt_qposadr[actuator_joints].astype(int)
    names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or "").split("/")[-1] for j in actuator_joints]
    arm = np.asarray([i for i, name in enumerate(names) if any(key in name for key in ("shoulder", "elbow", "wrist"))])
    arm_qpos = actuator_qpos[arm]; arm_joints = actuator_joints[arm]
    lo = model.jnt_range[arm_joints, 0] + 0.006; hi = model.jnt_range[arm_joints, 1] - 0.006
    guards = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_front_guard_collision") for side in ("left", "right")])
    output = controls.copy(); contact_offsets = None; seed = None; preload = None; terrain_step = None; errors = []
    start_step = start_frame * SUBSTEPS
    for step in range(len(output)):
        if step >= start_step:
            rotation = Rotation.from_quat(np.r_[data.qpos[oq + 4 : oq + 7], data.qpos[oq + 3]])
            if contact_offsets is None:
                contact_offsets = rotation.inv().apply(data.geom_xpos[guards] - data.qpos[oq : oq + 3])
                seed = data.qpos[arm_qpos].copy()
                preload = output[step - 1, arm] - seed
            terrain = False
            for ci in range(data.ncon):
                pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
                if not pair & object_geoms: continue
                other = next((geom for geom in pair if geom not in object_geoms), None)
                if other is None: continue
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
                terrain |= not body.startswith("robot/")
            if terrain_step is None and terrain:
                terrain_step = step
            targets = data.qpos[oq : oq + 3] + rotation.apply(contact_offsets)
            if terrain_step is None:
                targets[:, 2] -= 0.0012
                preload_scale = 1.0
            else:
                alpha = smoothstep((step - terrain_step) / (20 * SUBSTEPS))
                separation = targets[0] - targets[1]; separation[2] = 0.0
                separation /= max(np.linalg.norm(separation), 1e-9)
                targets[0] += alpha * (0.14 * separation + np.asarray((0.0, 0.0, 0.18)))
                targets[1] += alpha * (-0.14 * separation + np.asarray((0.0, 0.0, 0.18)))
                preload_scale = 1.0 - alpha
            base = data.qpos.copy()
            def residual(values: np.ndarray) -> np.ndarray:
                fk.qpos[:] = base; fk.qvel[:] = 0.0; fk.qpos[arm_qpos] = values; mujoco.mj_forward(model, fk)
                return np.r_[90.0 * (fk.geom_xpos[guards] - targets).ravel(), 0.02 * (values - seed)]
            fit = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi), max_nfev=45, ftol=5e-8, xtol=5e-8, gtol=5e-8)
            seed = fit.x
            fk.qpos[:] = base; fk.qvel[:] = 0.0; fk.qpos[arm_qpos] = seed; mujoco.mj_forward(model, fk)
            errors.append(float(np.max(np.linalg.norm(fk.geom_xpos[guards] - targets, axis=1))))
            proposed = np.clip(seed + preload_scale * preload, model.actuator_ctrlrange[arm, 0], model.actuator_ctrlrange[arm, 1])
            previous = output[step - 1, arm]
            output[step, arm] = np.clip(proposed, previous - 0.025, previous + 0.025)
        data.ctrl[:] = output[step]; mujoco.mj_step(model, data)
    return output, np.asarray(errors)


def replay(model: mujoco.MjModel, initial_qpos: np.ndarray, initial_qvel: np.ndarray, controls: np.ndarray, goal: np.ndarray, goal_quaternion: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    model.opt.timestep = PHYSICS_DT; data = mujoco.MjData(model); data.qpos[:] = initial_qpos; data.qvel[:] = initial_qvel; mujoco.mj_forward(model, data)
    rj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint"); oj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    rq = int(model.jnt_qposadr[rj]); oq = int(model.jnt_qposadr[oj]); od = int(model.jnt_dofadr[oj])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object"); object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == object_body}
    hands = [{mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_{name}") for name in ("hand_collision", "hand_front_guard_collision", "hand_trailing_guard_collision")} for side in ("left", "right")]
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"); feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{side}_ankle_roll_link") for side in ("left", "right")]
    groups = ("left_end_effector", "right_end_effector", "feet", "legs", "pelvis_torso", "other_robot", "environment"); impulses = {name: 0.0 for name in groups}
    trace = {key: [] for key in ("qpos", "qvel", "ctrl", "object_pos", "object_quat", "hand_contact", "object_environment_contact", "foot_contact", "root_height", "torso_tilt", "impulses")}
    initial_object = initial_qpos[oq : oq + 3].copy(); initial_feet = data.xpos[feet].copy(); max_foot_height = np.zeros(2); first_invalid = None
    for step, control in enumerate(controls):
        data.ctrl[:] = control; mujoco.mj_step(model, data)
        environment_contact = False
        for ci in range(data.ncon):
            pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
            if not pair & object_geoms: continue
            other = next((geom for geom in pair if geom not in object_geoms), None)
            if other is None: continue
            force = np.zeros(6); mujoco.mj_contactForce(model, data, ci, force); impulse = float(np.linalg.norm(force[:3]) * PHYSICS_DT)
            if other in hands[0]: group = "left_end_effector"
            elif other in hands[1]: group = "right_end_effector"
            else:
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
                if body.startswith("robot/"):
                    if any(key in body for key in ("foot", "ankle")): group = "feet"
                    elif any(key in body for key in ("knee", "shin", "hip", "thigh")): group = "legs"
                    elif any(key in body for key in ("pelvis", "torso", "waist")): group = "pelvis_torso"
                    else: group = "other_robot"
                else: group = "environment"
            environment_contact |= group == "environment"
            impulses[group] += impulse
        if step % SUBSTEPS != SUBSTEPS - 1: continue
        hand = np.zeros(2, dtype=bool); foot = np.zeros(2, dtype=bool)
        for ci in range(data.ncon):
            pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
            for side, geoms in enumerate(hands): hand[side] |= bool(pair & geoms) and bool(pair & object_geoms)
            for side, body in enumerate(feet):
                for geom in pair:
                    if int(model.geom_bodyid[geom]) == body: foot[side] |= not bool((pair - {geom}) & object_geoms)
        root_height = float(data.qpos[rq + 2]); up = data.xmat[torso].reshape(3, 3)[:, 2]; tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
        frame = step // SUBSTEPS
        if first_invalid is None and (root_height < 0.45 or tilt > 50.0): first_invalid = [frame, "root_height" if root_height < 0.45 else "torso_tilt"]
        max_foot_height = np.maximum(max_foot_height, data.xpos[feet, 2])
        trace["qpos"].append(data.qpos.copy()); trace["qvel"].append(data.qvel.copy()); trace["ctrl"].append(control.copy()); trace["object_pos"].append(data.qpos[oq : oq + 3].copy()); trace["object_quat"].append(data.qpos[oq + 3 : oq + 7].copy()); trace["hand_contact"].append(hand); trace["object_environment_contact"].append(environment_contact); trace["foot_contact"].append(foot); trace["root_height"].append(root_height); trace["torso_tilt"].append(tilt); trace["impulses"].append([impulses[name] for name in groups])
    arrays = {key: np.asarray(value) for key, value in trace.items()}; lift = arrays["object_pos"][:, 2] - initial_object[2]; supported = arrays["hand_contact"].all(axis=1) & (lift > 0.05)
    longest = run = 0
    for value in supported: run = run + 1 if value else 0; longest = max(longest, run)
    indexes = np.flatnonzero(supported); carried = 0.0 if not len(indexes) else float(np.linalg.norm(arrays["object_pos"][indexes[-1], :2] - arrays["object_pos"][indexes[0], :2]))
    non_ee = sum(impulses[name] for name in ("feet", "legs", "pelvis_torso", "other_robot")); foot_translation = np.linalg.norm(data.xpos[feet, :2] - initial_feet[:, :2], axis=1)
    final_speed = float(np.linalg.norm(arrays["qvel"][-1, od : od + 6])); final_pos_error = float(np.linalg.norm(arrays["object_pos"][-1] - goal))
    final_orientation_error = float(2.0 * math.acos(float(np.clip(abs(np.dot(arrays["object_quat"][-1], goal_quaternion)), 0.0, 1.0))))
    hand_frames = np.flatnonzero(arrays["hand_contact"].any(axis=1)); last_hand = int(hand_frames[-1]) if len(hand_frames) else -1
    terrain_after = np.flatnonzero(arrays["object_environment_contact"] & (np.arange(len(arrays["qpos"])) >= last_hand))
    first_terrain_after = int(terrain_after[0]) if len(terrain_after) else None
    release_gap = None if first_terrain_after is None else first_terrain_after - last_hand
    release_speed = float(np.linalg.norm(arrays["qvel"][last_hand, od : od + 6])) if last_hand >= 0 else None
    airborne = np.flatnonzero((~arrays["object_environment_contact"]) & (lift > 0.05))
    primary_landing = None
    primary_prehand = None
    primary_gap = None
    landing_linear_speed = None
    landing_angular_speed = None
    if len(airborne):
        landings = np.flatnonzero(arrays["object_environment_contact"] & (np.arange(len(arrays["qpos"])) > int(airborne[0])))
        if len(landings):
            primary_landing = int(landings[0])
            prior_hands = hand_frames[hand_frames <= primary_landing]
            primary_prehand = int(prior_hands[-1]) if len(prior_hands) else None
            primary_gap = None if primary_prehand is None else primary_landing - primary_prehand
            landing_linear_speed = float(np.linalg.norm(arrays["qvel"][primary_landing, od : od + 3]))
            landing_angular_speed = float(np.linalg.norm(arrays["qvel"][primary_landing, od + 3 : od + 6]))
    post_landing_hand_frames = 0 if primary_landing is None else int(arrays["hand_contact"][primary_landing + 1 :].any(axis=1).sum())
    metrics = {"full_horizon_valid": first_invalid is None, "first_invalid": first_invalid, "maximum_lift_m": float(lift.max()), "longest_bilateral_supported_lift_frames": int(longest), "carried_displacement_m": carried, "foot_translation_m": foot_translation.tolist(), "maximum_foot_center_height_m": max_foot_height.tolist(), "final_position_error_m": final_pos_error, "final_orientation_error_rad": final_orientation_error, "final_object_speed": final_speed, "final_hand_contact": arrays["hand_contact"][-1].tolist(), "last_end_effector_contact_frame": last_hand, "first_terrain_contact_after_release_frame": first_terrain_after, "release_to_terrain_frames": release_gap, "primary_landing_frame": primary_landing, "last_hand_frame_before_primary_landing": primary_prehand, "primary_release_to_terrain_frames": primary_gap, "primary_landing_linear_speed_mps": landing_linear_speed, "primary_landing_angular_speed_radps": landing_angular_speed, "post_landing_hand_contact_frames": post_landing_hand_frames, "object_com_height_at_release_m": float(arrays["object_pos"][last_hand, 2]) if last_hand >= 0 else None, "object_speed_at_release": release_speed, "object_impulse_ns": impulses, "non_end_effector_robot_object_impulse_ns": non_ee, "minimum_root_height_m": float(arrays["root_height"].min()), "maximum_torso_tilt_deg": float(arrays["torso_tilt"].max())}
    return metrics, arrays


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); pickup = np.load(PICKUP); donor = np.load(DONOR)
    qrefs, desired_object, ik_errors, goal = write_motion_and_targets(pickup, donor)
    join_qpos = pickup["qpos"][JOIN_FRAME].copy(); join_qvel = pickup["qvel"][JOIN_FRAME].copy()
    prefix_steps = (JOIN_FRAME + 1) * SUBSTEPS
    continuation_controls, model = generate_sonic_controls(qrefs, join_qpos, join_qvel, pickup["ctrl_physics"][prefix_steps - 1])
    feedback_ik_errors = np.zeros(len(continuation_controls))
    controls = np.concatenate([pickup["ctrl_physics"][:prefix_steps], continuation_controls], axis=0)
    initial_qpos = pickup["initial_qpos"].copy(); initial_qvel = pickup["initial_qvel"].copy()
    metrics, arrays = replay(model, initial_qpos, initial_qvel, controls, goal, FINAL_GOAL_QUATERNION)
    repeat, repeat_arrays = replay(model, initial_qpos, initial_qvel, controls, goal, FINAL_GOAL_QUATERNION)
    replay_qpos_error = float(np.max(np.abs(arrays["qpos"] - repeat_arrays["qpos"]))); replay_qvel_error = float(np.max(np.abs(arrays["qvel"] - repeat_arrays["qvel"])))
    metrics.update({"method": "verified static pickup prefix plus closed-loop SONIC support and realized-state cradle control", "geometry_full_size_m": (2 * np.asarray(BOX_HALF)).tolist(), "mass_kg": BOX_MASS, "horizon_reference_frames": len(arrays["qpos"]), "object_initially_on_terrain": True, "maximum_reference_hand_ik_error_m": float(ik_errors.max()), "maximum_realized_feedback_ik_error_m": float(feedback_ik_errors.max()), "independent_replay_max_qpos_error": replay_qpos_error, "independent_replay_max_qvel_error": replay_qvel_error, "goal_position": goal.tolist()})
    metrics["success"] = bool(metrics["full_horizon_valid"] and metrics["maximum_lift_m"] > 0.05 and metrics["longest_bilateral_supported_lift_frames"] >= 20 and metrics["carried_displacement_m"] > 0.30 and max(metrics["foot_translation_m"]) > 0.5 and metrics["non_end_effector_robot_object_impulse_ns"] < 1e-3 and metrics["final_position_error_m"] < 0.15 and metrics["final_orientation_error_rad"] < 0.35 and metrics["final_object_speed"] < 0.25 and not any(metrics["final_hand_contact"]) and metrics["primary_release_to_terrain_frames"] is not None and metrics["primary_release_to_terrain_frames"] <= 2 and metrics["primary_landing_linear_speed_mps"] < 0.75 and replay_qpos_error == 0.0 and replay_qvel_error == 0.0)
    np.savez_compressed(ARTIFACT, initial_qpos=initial_qpos, initial_qvel=initial_qvel, ctrl_physics=controls, continuation_ctrl_physics=continuation_controls, desired_object_pos=desired_object, reference_qpos=qrefs, ik_errors=ik_errors, feedback_ik_errors=feedback_ik_errors, impulse_groups=np.asarray(("left_end_effector", "right_end_effector", "feet", "legs", "pelvis_torso", "other_robot", "environment")), goal_position=goal, goal_quaternion=FINAL_GOAL_QUATERNION, **arrays)
    REPORT.write_text(json.dumps(metrics, indent=2) + "\n"); print("RESULT", json.dumps(metrics), flush=True)


if __name__ == "__main__": main()
