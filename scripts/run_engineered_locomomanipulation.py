#!/usr/bin/env python3
"""Engineer and validate a robust physical G1 carry on a proven SONIC walk.

The object is never constrained or actuated.  Two enlarged spherical wrist
pads squeeze a moderate 0.5 kg box through ordinary MuJoCo contacts.  Frozen
SONIC supplies the lower-body locomotion; arm position targets come from a
collision-aware kinematic carry reference.  The final authority is an
independent CPU MuJoCo replay of the exact 200 Hz position targets.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.spec_config import CollisionCfg
from rsl_rl.models import SonicBaseModel
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mocke import PRETRAINED_DIR, default_motion_file
from mocke.sonic import profile
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from orcs.assets.g1 import find_body, get_g1_flat_hand_cfg, flat_hand_spec


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_locomomanipulation"
DT = 0.02
PHYSICS_DT = 0.005
SUBSTEPS = 4
SOURCE_START = 5560
SOURCE_STOP = 5750
POLICY_STOP = 5850
SOURCE_END = SOURCE_STOP - SOURCE_START
ACQUIRE_FRAME = 30   # source frame 5590: compact hands arrive at the box
RELEASE_FRAME = 120  # source frame 5680: hands separate over the goal
HORIZON = 191
EXPERT_HORIZON = 171
BOX_HALF = np.asarray([0.060, 0.060, 0.060])
BOX_MASS = 0.050
PAD_RADIUS = 0.027
GUARD_RADIUS = 0.065
TRAILING_GUARD_RADIUS = 0.045
GUARD_DENSITY = 50.0
PAD_FRICTION = (2.5, 0.10, 0.003)
OBJECT_OFFSET_LOCAL = np.asarray([0.24, 0.0, 0.05])
LIFT_HEIGHT = 0.12
CRADLE_DROP = 0.120
ARM_REFERENCE_OVERRIDE = 0.0
INITIAL_OBJECT_WORLD = np.asarray([-2.1145, 2.8288, 0.9332])
GOAL_OBJECT_WORLD = np.asarray([-1.217, 3.286, 0.764])
# The controller is generated from a 180-degree yaw setup; the final expert
# validator applies the separately recorded +0.20 rad setup offset while
# freezing these exact controls.
INITIAL_OBJECT_QUAT = np.asarray([0.0, 0.0, 0.0, 1.0])
GOAL_OBJECT_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0])
RETRACTION_START_FRAME = 110
RETRACTION_DURATION_FRAMES = 12
RETRACTION_TARGET_FRAME = 150
ENABLE_RETRACTION = True


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def engineering_gripper_spec() -> mujoco.MjSpec:
    spec = flat_hand_spec()
    for side in ("left", "right"):
        body = find_body(spec.worldbody, f"{side}_wrist_yaw_link")
        for geom in body.geoms:
            if geom.name != f"{side}_hand_collision":
                continue
            geom.fromto[:] = np.nan
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size[:] = (PAD_RADIUS, 0.0, 0.0)
            geom.pos[:] = (0.12, -0.01 if side == "left" else 0.01, 0.0)
            geom.friction[:] = PAD_FRICTION
            geom.condim = 4
            geom.margin = 0.0
            geom.solref[:] = (0.03, 1.0)
            geom.solimp[:] = (0.8, 0.95, 0.01, 0.5, 2.0)
            # Dedicated collision bit: the generous end-effector envelope
            # contacts the payload, not the robot or terrain.
            geom.contype = 2
            geom.conaffinity = 0
            geom.rgba[:] = (0.15, 0.55, 0.95, 0.85)
            geom.group = 0
            break
        guard_pos = (0.230, -0.022, -0.047) if side == "left" else (0.215, 0.031, -0.070)
        guard = body.add_geom(
            name=f"{side}_hand_front_guard_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(GUARD_RADIUS, 0.0, 0.0),
            pos=guard_pos,
            density=GUARD_DENSITY,
            friction=PAD_FRICTION,
            condim=4,
            margin=0.0,
            contype=2,
            conaffinity=0,
            rgba=(0.10, 0.80, 0.90, 0.85),
            group=0,
        )
        guard.solref[:] = (0.03, 1.0)
        guard.solimp[:] = (0.8, 0.95, 0.01, 0.5, 2.0)
        trailing_pos = (0.087, 0.085, 0.065) if side == "left" else (0.117, 0.047, 0.114)
        trailing = body.add_geom(
            name=f"{side}_hand_trailing_guard_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(TRAILING_GUARD_RADIUS, 0.0, 0.0),
            pos=trailing_pos,
            density=GUARD_DENSITY,
            friction=PAD_FRICTION,
            condim=4,
            margin=0.0,
            rgba=(0.20, 0.85, 0.65, 0.85),
            group=0,
        )
        trailing.solref[:] = (0.03, 1.0)
        trailing.solimp[:] = (0.8, 0.95, 0.01, 0.5, 2.0)
    return spec


def engineering_robot_cfg() -> EntityCfg:
    base = get_g1_flat_hand_cfg()
    base.spec_fn = engineering_gripper_spec
    base.collisions = tuple(base.collisions) + (
        CollisionCfg(
            geom_names_expr=(".*hand.*_collision",),
            contype=4,
            conaffinity=2,
            condim=1,
            disable_other_geoms=False,
        ),
    )
    return profile.robot_cfg(base=base)


def engineering_object_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="floating_base_joint")
    body.add_geom(
        name="object_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BOX_HALF,
        mass=BOX_MASS,
        friction=PAD_FRICTION,
        condim=4,
        margin=0.0,
        solref=(0.03, 1.0),
        solimp=(0.8, 0.95, 0.01, 0.5, 2.0),
        contype=2,
        conaffinity=20,
        rgba=(0.92, 0.46, 0.12, 1.0),
    )
    return spec


def support_spec(start: np.ndarray, goal: np.ndarray) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="platforms")
    for name, center in (("start", start), ("goal", goal)):
        half_height = 0.02
        half_width = 0.05 if name == "start" else 0.25
        body.add_geom(
            name=f"{name}_platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            # A narrow physical pedestal leaves room for the spherical wrist
            # cradles to get underneath the payload corners.
            size=(half_width, half_width, half_height),
            pos=(float(center[0]), float(center[1]), float(center[2] - BOX_HALF[2] - half_height)),
            friction=(0.8, 0.02, 0.001),
            contype=16,
            conaffinity=2,
            rgba=(0.25, 0.28, 0.32, 1.0) if name == "start" else (0.15, 0.65, 0.25, 1.0),
        )
    return spec


def _quat_delta_omega(quat: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros((len(quat), quat.shape[1], 3))
    for i in range(len(quat) - 1):
        a = Rotation.from_quat(np.c_[quat[i, :, 1:], quat[i, :, :1]])
        b = Rotation.from_quat(np.c_[quat[i + 1, :, 1:], quat[i + 1, :, :1]])
        result[i] = (b * a.inv()).as_rotvec() / dt
    result[-1] = result[-2]
    return result


def make_carry_motion(path: Path) -> dict[str, object]:
    source = np.load(default_motion_file())
    # Build once through the engineering robot spec so FK matches the rollout.
    robot_spec = engineering_gripper_spec()
    robot_model = robot_spec.compile()
    data = mujoco.MjData(robot_model)
    joint_ids = sorted(
        (i for i in range(robot_model.njnt) if robot_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE),
        key=lambda i: int(robot_model.jnt_qposadr[i]),
    )
    joint_qpos = robot_model.jnt_qposadr[joint_ids].astype(int)
    joint_names = [mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in joint_ids]
    actuator_names = [name for name in joint_names]
    arm_idx = np.asarray([i for i, name in enumerate(actuator_names) if any(k in name for k in ("shoulder", "elbow", "wrist"))])
    limits = robot_model.jnt_range[np.asarray(joint_ids)[arm_idx]]
    hand_geoms = [
        mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_collision")
        for side in ("left", "right")
    ]

    source_len = min(SOURCE_END + 1, HORIZON)
    source_joint = source["joint_pos"][SOURCE_START : SOURCE_START + source_len]
    source_root_pos = source["body_pos_w"][SOURCE_START : SOURCE_START + source_len, 0]
    source_root_quat = source["body_quat_w"][SOURCE_START : SOURCE_START + source_len, 0]
    joint = np.concatenate(
        [source_joint, np.repeat(source_joint[-1:], HORIZON - source_len, axis=0)],
        axis=0,
    ).astype(np.float64)
    root_pos = np.concatenate(
        [source_root_pos, np.repeat(source_root_pos[-1:], HORIZON - source_len, axis=0)],
        axis=0,
    ).astype(np.float64)
    root_quat = np.concatenate(
        [source_root_quat, np.repeat(source_root_quat[-1:], HORIZON - source_len, axis=0)],
        axis=0,
    ).astype(np.float64)
    root_rot = Rotation.from_quat(np.c_[root_quat[:, 1:], root_quat[:, :1]])
    nominal_object = root_pos + root_rot.apply(np.repeat(OBJECT_OFFSET_LOCAL[None], HORIZON, axis=0))
    initial_quat = INITIAL_OBJECT_QUAT.copy()

    # Original frame-20 hands define a collision-free beginning to the reach.
    def set_frame(frame: int) -> None:
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[joint_qpos] = joint[frame]
        mujoco.mj_forward(robot_model, data)

    set_frame(ACQUIRE_FRAME)
    p0 = INITIAL_OBJECT_WORLD.copy()
    set_frame(RELEASE_FRAME)
    goal = GOAL_OBJECT_WORLD.copy()
    set_frame(20)
    reach_start = np.asarray([data.geom_xpos[g].copy() for g in hand_geoms])
    previous = joint[20, arm_idx].copy()
    max_error = 0.0
    worst_error_frame = 0
    evaluations = 0
    desired_object = np.repeat(p0[None], HORIZON, axis=0)
    for frame in range(HORIZON):
        if frame < 80:
            desired_object[frame] = p0
        elif frame < 130:
            desired_object[frame] = p0 + np.asarray([0.0, 0.0, LIFT_HEIGHT * smoothstep((frame - 80) / 50)])
        elif frame <= SOURCE_END:
            desired_object[frame] = nominal_object[frame] + np.asarray([0.0, 0.0, LIFT_HEIGHT])
        elif frame < 510:
            desired_object[frame] = goal + np.asarray([0.0, 0.0, LIFT_HEIGHT * (1.0 - smoothstep((frame - SOURCE_END) / 60))])
        else:
            desired_object[frame] = goal

        if frame < 20:
            previous = joint[frame, arm_idx].copy()
            continue
        lateral = root_rot[frame].apply([0.0, 1.0, 0.0])
        # The two spheres form a gravity-bearing V cradle under the lower box
        # corners.  This is ordinary unilateral contact, not an attachment.
        vertical_corner_offset = CRADLE_DROP - BOX_HALF[2]
        grip_distance = BOX_HALF[1] + math.sqrt(max(0.0, (PAD_RADIUS - 0.010) ** 2 - vertical_corner_offset**2))
        if frame >= 510:
            grip_distance += 0.10 * smoothstep((frame - 510) / 30)
        grip_targets = np.asarray(
            [desired_object[frame] + grip_distance * lateral, desired_object[frame] - grip_distance * lateral]
        )
        grip_targets[:, 2] -= CRADLE_DROP
        if frame < 80:
            blend = smoothstep((frame - 20) / 60)
            grip_targets = (1.0 - blend) * reach_start + blend * grip_targets

        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[joint_qpos] = joint[frame]

        def residual(values: np.ndarray) -> np.ndarray:
            data.qpos[joint_qpos[arm_idx]] = values
            mujoco.mj_forward(robot_model, data)
            hand = np.asarray([data.geom_xpos[g] for g in hand_geoms])
            return np.r_[30.0 * (hand - grip_targets).ravel(), 0.05 * (values - previous)]

        result = least_squares(
            residual,
            np.clip(previous, limits[:, 0] + 0.01, limits[:, 1] - 0.01),
            bounds=(limits[:, 0] + 0.005, limits[:, 1] - 0.005),
            max_nfev=45,
            ftol=2e-7,
            xtol=2e-7,
            gtol=2e-7,
        )
        evaluations += result.nfev
        previous = result.x
        joint[frame, arm_idx] = result.x
        data.qpos[joint_qpos[arm_idx]] = result.x
        mujoco.mj_forward(robot_model, data)
        frame_error = float(np.max(np.linalg.norm(np.asarray([data.geom_xpos[g] for g in hand_geoms]) - grip_targets, axis=1)))
        if frame_error > max_error:
            max_error = frame_error
            worst_error_frame = frame

    joint_vel = np.gradient(joint, DT, axis=0)
    robot_bodies = list(range(1, robot_model.nbody))
    body_pos = np.empty((HORIZON, len(robot_bodies), 3))
    body_quat = np.empty((HORIZON, len(robot_bodies), 4))
    for frame in range(HORIZON):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[joint_qpos] = joint[frame]
        mujoco.mj_forward(robot_model, data)
        body_pos[frame] = data.xpos[robot_bodies]
        body_quat[frame] = data.xquat[robot_bodies]
    body_lin = np.gradient(body_pos, DT, axis=0)
    body_ang = _quat_delta_omega(body_quat, DT)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_pos=joint.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin.astype(np.float32),
        body_ang_vel_w=body_ang.astype(np.float32),
        desired_object_pos=desired_object.astype(np.float32),
        initial_object_pos=p0,
        goal_object_pos=goal,
        initial_object_quat=initial_quat,
        goal_object_quat=GOAL_OBJECT_QUAT,
        arm_indexes=arm_idx,
    )
    return {
        "initial_object_pos": p0,
        "initial_object_quat": initial_quat,
        "goal_object_pos": goal,
        "goal_object_quat": GOAL_OBJECT_QUAT,
        "arm_indexes": arm_idx,
        "maximum_hand_ik_error_m": max_error,
        "worst_hand_ik_error_frame": worst_error_frame,
        "ik_function_evaluations": evaluations,
    }


def make_policy_motion(path: Path) -> None:
    """Preserve the proven full-body donor command seen by frozen SONIC.

    The manipulation arm targets are submitted separately.  Keeping the
    command tokenizer on the released motion avoids contaminating lower-body
    control with an out-of-distribution synthetic arm reference.
    """
    source = np.load(default_motion_file())
    arrays: dict[str, np.ndarray] = {}
    indices = np.arange(SOURCE_START, SOURCE_STOP + 1)
    for key in source.files:
        value = source[key]
        if key == "fps":
            arrays[key] = value
            continue
        arrays[key] = value[indices].copy()
    np.savez_compressed(path, **arrays)


def configure_env(motion_path: Path, setup: dict[str, object], *, render: bool = False):
    p0 = np.asarray(setup["initial_object_pos"])
    goal = np.asarray(setup["goal_object_pos"])
    cfg = sonic_tracking_env_cfg(str(motion_path), play=True, il_ordered=False)
    cfg.scene.num_envs = 1
    cfg.scene.entities["robot"] = engineering_robot_cfg()
    cfg.scene.entities["object"] = EntityCfg(
        spec_fn=engineering_object_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(p0), rot=tuple(setup["initial_object_quat"])),
    )
    cfg.scene.entities["support"] = EntityCfg(spec_fn=partial(support_spec, p0, goal))
    # Large but collision-filtered end-effector envelopes increase broadphase
    # candidates in MuJoCo-Warp.  Preserve all real foot contacts rather than
    # silently truncating the contact arena.
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 500
    cfg.auto_reset = False
    cfg.terminations = {}
    cfg.events = {name: term for name, term in cfg.events.items() if term.mode != "startup"}
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = tuple(((p0 + goal) * 0.5).tolist())
    cfg.viewer.distance = 4.2
    cfg.viewer.elevation = -16.0
    cfg.viewer.azimuth = 120.0
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    motion_path = OUT / "engineered_carry_motion_v1.npz"
    policy_motion_path = OUT / "released_policy_motion_v1.npz"
    setup = make_carry_motion(motion_path)
    make_policy_motion(policy_motion_path)
    print("SETUP", json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in setup.items()}), flush=True)
    raw = configure_env(policy_motion_path, setup)
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    # The pure tracking environment resets only the commanded robot.  Place the
    # additional free object explicitly after the wrapper's initial reset and
    # before observations/control; it is fully dynamic thereafter.
    obj = raw.scene["object"]
    object_pose = torch.as_tensor(
        np.r_[setup["initial_object_pos"], setup["initial_object_quat"]][None],
        dtype=torch.float32,
        device=raw.device,
    )
    obj.write_root_link_pose_to_sim(object_pose)
    obj.write_root_link_velocity_to_sim(torch.zeros((1, 6), device=raw.device))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.sim.sense()
    obs = env.get_observations()
    policy = SonicBaseModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=29,
        base_checkpoint=str(PRETRAINED_DIR / "sonic/last_ported.pt"),
    ).to("cuda:0")
    policy.eval()
    action_term = raw.action_manager.get_term("joint_pos")
    robot = raw.scene["robot"]
    motion = np.load(motion_path)
    arm_idx = np.asarray(setup["arm_indexes"], dtype=int)
    initial_qpos = raw.sim.data.qpos[0].detach().cpu().numpy().copy()
    initial_qvel = raw.sim.data.qvel[0].detach().cpu().numpy().copy()
    actions = []
    controls_joint_order = []
    gpu_qpos = []
    gpu_qvel = []
    with torch.inference_mode():
        for frame in range(HORIZON):
            action = policy(obs)
            target = torch.as_tensor(motion["joint_pos"][frame:frame + 1], device=raw.device)
            bias = action_term._entity.data.encoder_bias[:, action_term.target_ids]
            raw_target = (target + bias - action_term.offset) / action_term.scale
            action[:, arm_idx] = (
                (1.0 - ARM_REFERENCE_OVERRIDE) * action[:, arm_idx]
                + ARM_REFERENCE_OVERRIDE * raw_target[:, arm_idx]
            )
            # Exact position targets corresponding to the submitted action.
            ctrl = action * action_term.scale + action_term.offset - bias
            actions.append(action[0].detach().cpu().numpy().copy())
            controls_joint_order.extend([ctrl[0].detach().cpu().numpy().copy()] * SUBSTEPS)
            obs, _, _, _ = env.step(action)
            gpu_qpos.append(raw.sim.data.qpos[0].detach().cpu().numpy().copy())
            gpu_qvel.append(raw.sim.data.qvel[0].detach().cpu().numpy().copy())
    compiled_model = raw.sim.mj_model
    robot_joint_names = tuple(robot.joint_names)
    actuator_target_index = []
    for actuator in range(compiled_model.nu):
        joint_id = int(compiled_model.actuator_trnid[actuator, 0])
        joint_name = (mujoco.mj_id2name(compiled_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").split("/", 1)[-1]
        actuator_target_index.append(robot_joint_names.index(joint_name))
    controls = np.asarray(controls_joint_order)[:, actuator_target_index]
    compiled_actuator_names = []
    for actuator in range(compiled_model.nu):
        joint_id = int(compiled_model.actuator_trnid[actuator, 0])
        compiled_actuator_names.append(
            (mujoco.mj_id2name(compiled_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").split("/", 1)[-1]
        )
    arm_actuators = np.asarray(
        [i for i, name in enumerate(compiled_actuator_names) if any(k in name for k in ("shoulder", "elbow", "wrist"))]
    )
    controls_policy = controls.copy()
    if ENABLE_RETRACTION:
        retraction_target = controls[RETRACTION_TARGET_FRAME * SUBSTEPS + SUBSTEPS - 1, arm_actuators].copy()
        retraction_start = RETRACTION_START_FRAME * SUBSTEPS
        retraction_duration = RETRACTION_DURATION_FRAMES * SUBSTEPS
        for step in range(retraction_start, len(controls)):
            alpha = smoothstep((step - retraction_start) / retraction_duration)
            controls[step, arm_actuators] = (
                (1.0 - alpha) * controls[step, arm_actuators] + alpha * retraction_target
            )
    continuation_controls = controls.copy()
    controls = controls[: EXPERT_HORIZON * SUBSTEPS]
    mujoco.mj_saveModel(compiled_model, str(OUT / "engineered_locomomanipulation_model.mjb"), None)
    env.close()

    # Independent CPU replay is the source of truth.
    model = mujoco.MjModel.from_binary_path(str(OUT / "engineered_locomomanipulation_model.mjb"))
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{s}_ankle_roll_link") for s in ("left", "right")]
    hand_geoms = [
        {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_hand_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_hand_front_guard_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_hand_trailing_guard_collision"),
        }
        for s in ("left", "right")
    ]
    object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == object_body}
    impulse_names = ("left_end_effector", "right_end_effector", "feet", "legs", "pelvis_torso", "other_robot", "environment")
    impulses = {name: 0.0 for name in impulse_names}
    trace = {k: [] for k in ("qpos", "qvel", "ctrl", "object_pos", "object_quat", "hand_contact", "foot_contact", "root_height", "torso_tilt", "impulses")}
    first_invalid = None
    initial_object = initial_qpos[object_qpos:object_qpos + 3].copy()
    initial_feet = None
    max_foot_height = np.zeros(2)
    for step, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 not in object_geoms and g2 not in object_geoms:
                continue
            other = g2 if g1 in object_geoms else g1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, force)
            impulse = float(np.linalg.norm(force[:3]) * PHYSICS_DT)
            if other in hand_geoms[0]:
                group = "left_end_effector"
            elif other in hand_geoms[1]:
                group = "right_end_effector"
            else:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
                if body_name.startswith("robot/"):
                    if any(x in body_name for x in ("foot", "ankle")):
                        group = "feet"
                    elif any(x in body_name for x in ("knee", "shin", "hip", "thigh")):
                        group = "legs"
                    elif any(x in body_name for x in ("pelvis", "torso", "waist")):
                        group = "pelvis_torso"
                    else:
                        group = "other_robot"
                else:
                    group = "environment"
            impulses[group] += impulse
        if step % SUBSTEPS != SUBSTEPS - 1:
            continue
        root_height = float(data.qpos[robot_qpos + 2])
        torso_up = data.xmat[torso_body].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(torso_up[2], -1.0, 1.0))))
        feet = np.asarray([data.xpos[b].copy() for b in foot_bodies])
        if initial_feet is None:
            initial_feet = feet.copy()
        max_foot_height = np.maximum(max_foot_height, feet[:, 2])
        hand_contact = np.zeros(2, dtype=bool)
        foot_contact = np.zeros(2, dtype=bool)
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            for side, geoms in enumerate(hand_geoms):
                hand_contact[side] |= bool(geoms & pair) and bool(pair & object_geoms)
            for side, body in enumerate(foot_bodies):
                if int(model.geom_bodyid[int(con.geom1)]) == body or int(model.geom_bodyid[int(con.geom2)]) == body:
                    other = int(con.geom2) if int(model.geom_bodyid[int(con.geom1)]) == body else int(con.geom1)
                    foot_contact[side] |= not (other in object_geoms)
        if first_invalid is None:
            if root_height < 0.45:
                first_invalid = [len(trace["qpos"]), "root_height"]
            elif tilt > 50.0:
                first_invalid = [len(trace["qpos"]), "torso_tilt"]
        trace["qpos"].append(data.qpos.copy())
        trace["qvel"].append(data.qvel.copy())
        trace["ctrl"].append(ctrl.copy())
        trace["object_pos"].append(data.qpos[object_qpos:object_qpos + 3].copy())
        trace["object_quat"].append(data.qpos[object_qpos + 3:object_qpos + 7].copy())
        trace["hand_contact"].append(hand_contact)
        trace["foot_contact"].append(foot_contact)
        trace["root_height"].append(root_height)
        trace["torso_tilt"].append(tilt)
        trace["impulses"].append([impulses[name] for name in impulse_names])
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    object_pos = arrays["object_pos"]
    lift = object_pos[:, 2] - initial_object[2]
    supported = arrays["hand_contact"].any(axis=1) & (lift > 0.05)
    longest_supported = 0
    run = 0
    for active in supported:
        run = run + 1 if active else 0
        longest_supported = max(longest_supported, run)
    carry_mask = supported
    carried_displacement = 0.0
    if carry_mask.any():
        indexes = np.flatnonzero(carry_mask)
        carried_displacement = float(np.linalg.norm(object_pos[indexes[-1], :2] - object_pos[indexes[0], :2]))
    goal = np.asarray(setup["goal_object_pos"])
    goal_quat = np.asarray(setup["goal_object_quat"])
    final_pos_error = float(np.linalg.norm(object_pos[-1] - goal))
    final_ori_error = float((Rotation.from_quat(np.r_[arrays["object_quat"][-1, 1:], arrays["object_quat"][-1, 0]]).inv() * Rotation.from_quat(np.r_[goal_quat[1:], goal_quat[0]])).magnitude())
    non_ee = sum(impulses[name] for name in ("feet", "legs", "pelvis_torso", "other_robot"))
    final_speed = float(np.linalg.norm(arrays["qvel"][-1, model.jnt_dofadr[object_joint]:model.jnt_dofadr[object_joint] + 6]))
    foot_translation = np.linalg.norm(feet[:, :2] - initial_feet[:, :2], axis=1)
    success = bool(
        first_invalid is None
        and lift.max() > 0.05
        and longest_supported >= 25
        and carried_displacement > 0.5
        and foot_translation.max() > 0.5
        and non_ee < 0.1
        and final_pos_error < 0.15
        and final_ori_error < 0.35
        and final_speed < 0.25
        and not arrays["hand_contact"][-10:].any()
    )
    report = {
        "success": success,
        "method": "frozen SONIC lower-body locomotion plus physical spherical-pad bilateral grasp",
        "release_retraction": {
            "start_frame": RETRACTION_START_FRAME,
            "duration_frames": RETRACTION_DURATION_FRAMES,
            "donor_target_frame": RETRACTION_TARGET_FRAME,
        },
        "horizon_reference_frames": EXPERT_HORIZON,
        "horizon_seconds": EXPERT_HORIZON * DT,
        "exact_valid_continuation_frames_available": HORIZON - EXPERT_HORIZON,
        "full_horizon_valid": first_invalid is None,
        "first_invalid": first_invalid,
        "maximum_lift_m": float(lift.max()),
        "longest_hand_supported_lift_frames": int(longest_supported),
        "carried_displacement_m": carried_displacement,
        "foot_translation_m": foot_translation.tolist(),
        "maximum_foot_center_height_m": max_foot_height.tolist(),
        "final_position_error_m": final_pos_error,
        "final_orientation_error_rad": final_ori_error,
        "final_object_speed": final_speed,
        "final_hand_contact": arrays["hand_contact"][-1].tolist(),
        "object_impulse_ns": impulses,
        "non_end_effector_robot_object_impulse_ns": non_ee,
        "minimum_root_height_m": float(arrays["root_height"].min()),
        "maximum_torso_tilt_deg": float(arrays["torso_tilt"].max()),
        "maximum_hand_ik_error_m": setup["maximum_hand_ik_error_m"],
    }
    np.savez_compressed(
        OUT / "engineered_locomomanipulation_v1.npz",
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        actions=np.asarray(actions),
        ctrl_physics_policy=controls_policy,
        continuation_ctrl_physics=continuation_controls,
        ctrl_physics=controls,
        impulse_groups=np.asarray(impulse_names),
        goal_position=goal,
        goal_quaternion=goal_quat,
        **arrays,
    )
    (OUT / "engineered_locomomanipulation_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
