"""Simulation-rate diagnostics for Stage-A support tracking.

This module only replays an existing position-target trajectory.  It does not
change targets, gains, reference state, model parameters, or physical state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    RolloutResult,
    _DynamicRollout,
    _np,
)
from orcs.tasks.uolm.generated_reference import GeneratedReference

__all__ = [
    "SupportDiagnosticResult",
    "instrument_support_rollout",
]


@dataclass(slots=True)
class SupportDiagnosticResult:
    label: str
    data: dict[str, np.ndarray]
    summary: dict[str, Any]
    rollout: RolloutResult


def _body_name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the 2-D monotone-chain hull without an external geometry dep."""
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(
        origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed minimum distance: positive inside a CCW convex polygon."""
    if len(polygon) < 3 or not np.all(np.isfinite(point)):
        return float("nan")
    margin = float("inf")
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        edge = second - first
        norm = float(np.linalg.norm(edge))
        if norm <= 1.0e-12:
            continue
        offset = point - first
        cross = edge[0] * offset[1] - edge[1] * offset[0]
        margin = min(margin, float(cross / norm))
    return margin


def _reference_com_path(
    engine: _DynamicRollout,
    robot_root_body: int,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(engine.model)
    path = np.empty((end_frame + 1, 3), dtype=np.float64)
    velocity = np.empty_like(path)
    angular_momentum = np.empty_like(path)
    for frame in range(end_frame + 1):
        data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = engine.reference_root[
            frame
        ]
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = _np(
            engine.reference.root_quat_wxyz[0, frame]
        )
        data.qpos[engine.joint_qpos] = engine.reference_joint[frame]
        data.qvel[engine.robot_dof : engine.robot_dof + 35] = engine.reference_mj_qvel[
            frame
        ]
        mujoco.mj_forward(engine.model, data)
        path[frame] = data.subtree_com[robot_root_body]
        velocity[frame] = data.subtree_linvel[robot_root_body]
        angular_momentum[frame] = data.subtree_angmom[robot_root_body]
    return path, velocity, angular_momentum


def _hermite_interpolate(
    values: np.ndarray,
    velocities: np.ndarray,
    time: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C1 interpolation with analytic velocity and acceleration."""
    segment = np.minimum((time / dt).astype(np.int32), len(values) - 2)
    phase = ((time - segment * dt) / dt)[:, None]
    phase2 = phase * phase
    phase3 = phase2 * phase
    lower, upper = values[segment], values[segment + 1]
    lower_velocity, upper_velocity = velocities[segment], velocities[segment + 1]
    position = (
        (2.0 * phase3 - 3.0 * phase2 + 1.0) * lower
        + (phase3 - 2.0 * phase2 + phase) * dt * lower_velocity
        + (-2.0 * phase3 + 3.0 * phase2) * upper
        + (phase3 - phase2) * dt * upper_velocity
    )
    velocity = (
        (6.0 * phase2 - 6.0 * phase) * lower
        + (3.0 * phase2 - 4.0 * phase + 1.0) * dt * lower_velocity
        + (-6.0 * phase2 + 6.0 * phase) * upper
        + (3.0 * phase2 - 2.0 * phase) * dt * upper_velocity
    ) / dt
    acceleration = (
        (12.0 * phase - 6.0) * lower
        + (6.0 * phase - 4.0) * dt * lower_velocity
        + (-12.0 * phase + 6.0) * upper
        + (6.0 * phase - 2.0) * dt * upper_velocity
    ) / (dt * dt)
    return position, velocity, acceleration


def _linear_interpolate(values: np.ndarray, time: np.ndarray, dt: float) -> np.ndarray:
    source_time = np.arange(len(values), dtype=np.float64) * dt
    return np.stack(
        [
            np.interp(time, source_time, values[:, axis])
            for axis in range(values.shape[1])
        ],
        axis=1,
    )


def _derivatives(position: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.gradient(position, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    return velocity, acceleration


def _first_sustained(mask: np.ndarray, samples: int = 4) -> int | None:
    if len(mask) < samples:
        return None
    runs = np.convolve(mask.astype(np.int32), np.ones(samples, dtype=np.int32), "valid")
    indexes = np.flatnonzero(runs == samples)
    return int(indexes[0]) if len(indexes) else None


def _contact_wrench(
    engine: _DynamicRollout,
    data: mujoco.MjData,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return per-foot wrench/contact data, expressed in world coordinates."""
    forces = np.zeros((2, 3), dtype=np.float64)
    moments = np.zeros((2, 3), dtype=np.float64)
    normal = np.zeros(2, dtype=np.float64)
    slip = np.zeros(2, dtype=np.float64)
    contact_points: list[list[np.ndarray]] = [[], []]
    all_robot_force = np.zeros(3, dtype=np.float64)
    result = np.empty(6, dtype=np.float64)
    velocity = np.empty(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in engine.terrain_geoms and second not in engine.terrain_geoms:
            continue
        robot_geom = second if first in engine.terrain_geoms else first
        robot_body = int(engine.model.geom_bodyid[robot_geom])
        if robot_body not in engine.robot_bodies:
            continue
        mujoco.mj_contactForce(engine.model, data, contact_index, result)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame.T @ result[:3]
        torque_world = frame.T @ result[3:]
        # mj_contactForce is the force on geom2. Reverse it if the robot is geom1.
        if robot_geom == first:
            force_world *= -1.0
            torque_world *= -1.0
        all_robot_force += force_world
        if robot_body not in engine.allowed_terrain_bodies:
            continue
        side = engine.foot_bodies.index(robot_body)
        point = np.asarray(contact.pos, dtype=np.float64).copy()
        forces[side] += force_world
        moments[side] += np.cross(point, force_world) + torque_world
        normal[side] += max(0.0, float(result[0]))
        contact_points[side].append(point)
        mujoco.mj_objectVelocity(
            engine.model,
            data,
            mujoco.mjtObj.mjOBJ_GEOM,
            robot_geom,
            velocity,
            0,
        )
        normal_world = frame[0]
        tangential = velocity[3:] - normal_world * np.dot(velocity[3:], normal_world)
        slip[side] = max(slip[side], float(np.linalg.norm(tangential)))

    points = [point for group in contact_points for point in group]
    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return forces, moments, normal, slip, point_array, all_robot_force


def instrument_support_rollout(
    model: mujoco.MjModel,
    reference: GeneratedReference,
    joint_target: np.ndarray,
    *,
    label: str,
    config: DynamicsRefinementConfig | None = None,
    end_frame: int = 40,
) -> SupportDiagnosticResult:
    """Replay one controller target and record frames 0..40 at 200 Hz."""
    cfg = config or DynamicsRefinementConfig(run_stage_b=False)
    engine = _DynamicRollout(model, reference, cfg)
    if joint_target.shape != engine.reference_joint.shape:
        raise ValueError("joint target shape does not match the reference")
    if not 1 <= end_frame < engine.horizon:
        raise ValueError("invalid diagnostic end frame")

    rollout = engine.rollout(joint_target, stop_on_invalid=False)
    samples = end_frame * cfg.substeps + 1
    time = np.arange(samples, dtype=np.float64) * cfg.physics_dt
    data = engine._new_initialized_data()
    robot_root_body = int(model.jnt_bodyid[engine.robot_free_joint])
    total_mass = float(model.body_subtreemass[robot_root_body])
    (
        reference_com_50hz,
        reference_com_velocity_50hz,
        reference_angular_momentum_50hz,
    ) = _reference_com_path(engine, robot_root_body, end_frame)
    reference_com, reference_com_velocity, _ = _hermite_interpolate(
        reference_com_50hz,
        reference_com_velocity_50hz,
        time,
        cfg.control_dt,
    )
    reference_com_acceleration = _linear_interpolate(
        np.gradient(
            reference_com_velocity_50hz,
            cfg.control_dt,
            axis=0,
            edge_order=2,
        ),
        time,
        cfg.control_dt,
    )
    root_velocity_50hz = _np(reference.root_lin_vel_w[0, : end_frame + 1])
    reference_root, reference_root_velocity, _ = _hermite_interpolate(
        engine.reference_root[: end_frame + 1],
        root_velocity_50hz,
        time,
        cfg.control_dt,
    )
    reference_root_acceleration = _linear_interpolate(
        np.gradient(root_velocity_50hz, cfg.control_dt, axis=0, edge_order=2),
        time,
        cfg.control_dt,
    )
    gravity = abs(float(model.opt.gravity[2]))
    reference_angular_momentum = _linear_interpolate(
        reference_angular_momentum_50hz, time, cfg.control_dt
    )
    reference_angular_momentum_rate = _linear_interpolate(
        np.gradient(
            reference_angular_momentum_50hz,
            cfg.control_dt,
            axis=0,
            edge_order=2,
        ),
        time,
        cfg.control_dt,
    )
    torso_reference_index = reference.tracked_body_names.index("torso_link")
    reference_torso_angular_velocity = _linear_interpolate(
        _np(reference.body_ang_vel_w[0, : end_frame + 1, torso_reference_index]),
        time,
        cfg.control_dt,
    )
    reference_required_force = total_mass * reference_com_acceleration
    reference_required_force[:, 2] += total_mass * gravity
    reference_required_cop = np.full((samples, 2), np.nan, dtype=np.float64)
    defined_cop = reference_required_force[:, 2] > 0.25 * total_mass * gravity
    reference_required_cop[defined_cop, 0] = (
        reference_com[defined_cop, 0]
        - reference_com[defined_cop, 2]
        * reference_required_force[defined_cop, 0]
        / reference_required_force[defined_cop, 2]
        - reference_angular_momentum_rate[defined_cop, 1]
        / reference_required_force[defined_cop, 2]
    )
    reference_required_cop[defined_cop, 1] = (
        reference_com[defined_cop, 1]
        - reference_com[defined_cop, 2]
        * reference_required_force[defined_cop, 1]
        / reference_required_force[defined_cop, 2]
        + reference_angular_momentum_rate[defined_cop, 0]
        / reference_required_force[defined_cop, 2]
    )

    root_position = np.empty((samples, 3), dtype=np.float64)
    root_quaternion = np.empty((samples, 4), dtype=np.float64)
    root_velocity = np.empty((samples, 3), dtype=np.float64)
    root_angular_velocity = np.empty((samples, 3), dtype=np.float64)
    com_position = np.empty((samples, 3), dtype=np.float64)
    com_velocity = np.empty((samples, 3), dtype=np.float64)
    centroidal_angular_momentum = np.empty((samples, 3), dtype=np.float64)
    torso_angular_velocity = np.empty((samples, 3), dtype=np.float64)
    torso_roll_pitch = np.empty((samples, 2), dtype=np.float64)
    arm_subtree_angular_momentum = np.empty((samples, 2, 3), dtype=np.float64)
    foot_force = np.zeros((samples, 2, 3), dtype=np.float64)
    foot_moment = np.zeros((samples, 2, 3), dtype=np.float64)
    foot_normal_force = np.zeros((samples, 2), dtype=np.float64)
    foot_slip_velocity = np.zeros((samples, 2), dtype=np.float64)
    foot_contact = np.zeros((samples, 2), dtype=bool)
    nonfoot_terrain_contact = np.zeros(samples, dtype=bool)
    wrist_terrain_contact = np.zeros(samples, dtype=bool)
    total_robot_ground_force = np.zeros((samples, 3), dtype=np.float64)
    total_support_force = np.zeros((samples, 3), dtype=np.float64)
    total_support_moment = np.zeros((samples, 3), dtype=np.float64)
    cop = np.full((samples, 2), np.nan, dtype=np.float64)
    support_bounds = np.full((samples, 4), np.nan, dtype=np.float64)
    support_polygon_margin_com = np.full(samples, np.nan, dtype=np.float64)
    support_polygon_margin_cop = np.full(samples, np.nan, dtype=np.float64)
    support_polygon_margin_reference_com = np.full(samples, np.nan, dtype=np.float64)
    support_polygon_margin_reference_cop = np.full(samples, np.nan, dtype=np.float64)
    support_contact_count = np.zeros(samples, dtype=np.int32)
    contact_capacity = max(64, int(model.nconmax))
    contact_points = np.full((samples, contact_capacity, 3), np.nan, dtype=np.float64)
    actuator_torque = np.empty((samples, model.nu), dtype=np.float64)
    generalized_actuator_torque = np.empty((samples, model.nu), dtype=np.float64)
    normalized_torque_usage = np.empty((samples, model.nu), dtype=np.float64)
    actuator_length = np.empty((samples, model.nu), dtype=np.float64)
    actuator_transmission_velocity = np.empty((samples, model.nu), dtype=np.float64)
    actuator_position = np.empty((samples, model.nu), dtype=np.float64)
    actuator_velocity = np.empty((samples, model.nu), dtype=np.float64)
    actuator_target = np.empty((samples, model.nu), dtype=np.float64)
    actuator_position_error = np.empty((samples, model.nu), dtype=np.float64)
    foot_body_position = np.empty((samples, 2, 3), dtype=np.float64)
    control_frame = np.empty(samples, dtype=np.int32)
    lower_body_target = np.empty(
        (samples, len(engine.balance_indexes)), dtype=np.float64
    )

    actuator_dof = np.asarray(
        [model.jnt_dofadr[model.actuator_trnid[index, 0]] for index in range(model.nu)],
        dtype=np.int32,
    )
    actuator_names = tuple(
        (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[index, 0]),
            )
            or ""
        ).split("/", 1)[-1]
        for index in range(model.nu)
    )
    arm_root_bodies = (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot/left_shoulder_pitch_link"
        ),
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_shoulder_pitch_link"
        ),
    )
    if any(body < 0 for body in arm_root_bodies):
        raise ValueError("compiled model is missing a proximal-arm subtree root")

    def record(sample: int, frame: int) -> None:
        forces, moments, normal, slip, points, all_force = _contact_wrench(engine, data)
        root_position[sample] = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        root_quaternion[sample] = data.qpos[
            engine.robot_qpos + 3 : engine.robot_qpos + 7
        ]
        root_velocity[sample] = data.qvel[engine.robot_dof : engine.robot_dof + 3]
        root_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(root_rotation, root_quaternion[sample])
        root_angular_velocity[sample] = (
            root_rotation.reshape(3, 3)
            @ data.qvel[engine.robot_dof + 3 : engine.robot_dof + 6]
        )
        com_position[sample] = data.subtree_com[robot_root_body]
        com_velocity[sample] = data.subtree_linvel[robot_root_body]
        centroidal_angular_momentum[sample] = data.subtree_angmom[robot_root_body]
        torso_angular_velocity[sample] = data.cvel[engine.torso_body, :3]
        torso_rotation = data.xmat[engine.torso_body].reshape(3, 3)
        torso_roll_pitch[sample] = [
            np.arctan2(torso_rotation[2, 1], torso_rotation[2, 2]),
            np.arcsin(np.clip(-torso_rotation[2, 0], -1.0, 1.0)),
        ]
        arm_subtree_angular_momentum[sample] = data.subtree_angmom[
            list(arm_root_bodies)
        ]
        foot_force[sample] = forces
        foot_moment[sample] = moments
        foot_normal_force[sample] = normal
        foot_slip_velocity[sample] = slip
        foot_contact[sample] = normal > 1.0e-6
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in engine.terrain_geoms and second not in engine.terrain_geoms:
                continue
            robot_geom = second if first in engine.terrain_geoms else first
            body = int(model.geom_bodyid[robot_geom])
            if (
                body in engine.robot_bodies
                and body not in engine.allowed_terrain_bodies
            ):
                nonfoot_terrain_contact[sample] = True
                name = _body_name(model, body)
                wrist_terrain_contact[sample] |= "wrist" in name or "hand" in name
        total_robot_ground_force[sample] = all_force
        total_support_force[sample] = np.sum(forces, axis=0)
        total_support_moment[sample] = np.sum(moments, axis=0)
        support_contact_count[sample] = len(points)
        if len(points):
            stored_points = points[:contact_capacity]
            contact_points[sample, : len(stored_points)] = stored_points
            support_bounds[sample] = [
                np.min(points[:, 0]),
                np.max(points[:, 0]),
                np.min(points[:, 1]),
                np.max(points[:, 1]),
            ]
            hull = _convex_hull(points[:, :2])
            support_polygon_margin_com[sample] = _polygon_margin(
                com_position[sample, :2], hull
            )
            support_polygon_margin_reference_com[sample] = _polygon_margin(
                reference_com[sample, :2], hull
            )
            support_polygon_margin_reference_cop[sample] = _polygon_margin(
                reference_required_cop[sample], hull
            )
            vertical_force = total_support_force[sample, 2]
            if vertical_force > 1.0:
                cop[sample] = [
                    -total_support_moment[sample, 1] / vertical_force,
                    total_support_moment[sample, 0] / vertical_force,
                ]
                support_polygon_margin_cop[sample] = _polygon_margin(cop[sample], hull)
        actuator_torque[sample] = data.actuator_force
        generalized_actuator_torque[sample] = data.qfrc_actuator[actuator_dof]
        normalized_torque_usage[sample] = (
            np.abs(data.actuator_force) / engine.force_scale
        )
        actuator_length[sample] = data.actuator_length
        actuator_transmission_velocity[sample] = data.actuator_velocity
        actuator_position[sample] = data.qpos[engine.actuator_qpos]
        actuator_velocity[sample] = data.qvel[actuator_dof]
        actuator_target[sample] = data.ctrl
        actuator_position_error[sample] = data.ctrl - actuator_position[sample]
        foot_body_position[sample] = data.xpos[list(engine.foot_bodies)]
        control_frame[sample] = frame
        lower_body_target[sample] = joint_target[frame, engine.balance_indexes]

    data.ctrl[:] = np.clip(
        joint_target[0, engine.actuator_joint_indexes],
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    mujoco.mj_forward(model, data)
    record(0, 0)
    sample = 1
    for frame in range(1, end_frame + 1):
        data.ctrl[:] = np.clip(
            joint_target[frame, engine.actuator_joint_indexes],
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        for _ in range(cfg.substeps):
            mujoco.mj_step(model, data)
            record(sample, frame)
            sample += 1

    root_velocity_fd, root_acceleration = _derivatives(root_position, cfg.physics_dt)
    com_acceleration = np.gradient(com_velocity, cfg.physics_dt, axis=0, edge_order=2)
    centroidal_angular_momentum_rate = np.gradient(
        centroidal_angular_momentum, cfg.physics_dt, axis=0, edge_order=2
    )
    actuator_target_velocity = (
        np.diff(actuator_target, axis=0, prepend=actuator_target[:1]) / cfg.physics_dt
    )
    actuator_target_acceleration = (
        np.diff(
            actuator_target_velocity,
            axis=0,
            prepend=actuator_target_velocity[:1],
        )
        / cfg.physics_dt
    )
    actuator_acceleration = np.gradient(
        actuator_velocity, cfg.physics_dt, axis=0, edge_order=2
    )
    actuator_velocity_error = actuator_target_velocity - actuator_velocity
    actuator_gain_term = actuator_target * model.actuator_gainprm[:, 0]
    actuator_bias_term = (
        model.actuator_biasprm[:, 0]
        + actuator_length * model.actuator_biasprm[:, 1]
        + actuator_transmission_velocity * model.actuator_biasprm[:, 2]
    )
    reconstructed_unclipped_actuator_torque = actuator_gain_term + actuator_bias_term
    foot_body_velocity = np.gradient(
        foot_body_position, cfg.physics_dt, axis=0, edge_order=2
    )
    required_vertical_force = total_mass * (gravity + reference_com_acceleration[:, 2])
    actual_equivalent_vertical_force = total_mass * (gravity + com_acceleration[:, 2])
    vertical_momentum = total_mass * com_velocity[:, 2]
    vertical_momentum_rate = total_mass * com_acceleration[:, 2]
    foot_impulse_error = np.zeros(samples, dtype=np.float64)
    all_ground_impulse_error = np.zeros(samples, dtype=np.float64)
    foot_force_error = total_support_force[:, 2] - required_vertical_force
    all_force_error = total_robot_ground_force[:, 2] - required_vertical_force
    foot_impulse_error[1:] = np.cumsum(
        0.5 * (foot_force_error[1:] + foot_force_error[:-1]) * cfg.physics_dt
    )
    all_ground_impulse_error[1:] = np.cumsum(
        0.5 * (all_force_error[1:] + all_force_error[:-1]) * cfg.physics_dt
    )

    data_out: dict[str, np.ndarray] = {
        "time_s": time,
        "control_frame": control_frame,
        "root_position_m": root_position,
        "root_quaternion_wxyz": root_quaternion,
        "root_linear_velocity_m_s": root_velocity,
        "root_linear_velocity_fd_m_s": root_velocity_fd,
        "root_linear_acceleration_m_s2": root_acceleration,
        "root_angular_velocity_rad_s": root_angular_velocity,
        "com_position_m": com_position,
        "com_velocity_m_s": com_velocity,
        "com_acceleration_m_s2": com_acceleration,
        "centroidal_angular_momentum_kg_m2_s": centroidal_angular_momentum,
        "centroidal_angular_momentum_rate_nm": centroidal_angular_momentum_rate,
        "torso_angular_velocity_rad_s": torso_angular_velocity,
        "torso_roll_pitch_rad": torso_roll_pitch,
        # Approximation: each value is MuJoCo's angular momentum for the
        # shoulder-rooted body subtree, expressed about that subtree's COM.
        "arm_subtree_angular_momentum_kg_m2_s": arm_subtree_angular_momentum,
        "reference_root_position_m": reference_root,
        "reference_root_velocity_m_s": reference_root_velocity,
        "reference_root_acceleration_m_s2": reference_root_acceleration,
        "reference_com_position_m": reference_com,
        "reference_com_velocity_m_s": reference_com_velocity,
        "reference_com_acceleration_m_s2": reference_com_acceleration,
        "reference_angular_momentum_kg_m2_s": reference_angular_momentum,
        "reference_angular_momentum_rate_nm": reference_angular_momentum_rate,
        "reference_torso_angular_velocity_rad_s": (reference_torso_angular_velocity),
        "reference_required_force_world_n": reference_required_force,
        "reference_required_cop_xy_m": reference_required_cop,
        "robot_mass_kg": np.asarray([total_mass], dtype=np.float64),
        "required_vertical_force_n": required_vertical_force,
        "actual_equivalent_vertical_force_n": actual_equivalent_vertical_force,
        "vertical_momentum_kg_m_s": vertical_momentum,
        "vertical_momentum_rate_n": vertical_momentum_rate,
        "foot_vertical_impulse_error_n_s": foot_impulse_error,
        "all_ground_vertical_impulse_error_n_s": all_ground_impulse_error,
        "foot_force_world_n": foot_force,
        "foot_moment_world_origin_nm": foot_moment,
        "foot_normal_force_n": foot_normal_force,
        "foot_slip_velocity_m_s": foot_slip_velocity,
        "foot_contact": foot_contact,
        "nonfoot_terrain_contact": nonfoot_terrain_contact,
        "wrist_terrain_contact": wrist_terrain_contact,
        "total_support_force_world_n": total_support_force,
        "total_support_moment_world_origin_nm": total_support_moment,
        "total_robot_ground_force_world_n": total_robot_ground_force,
        "cop_xy_m": cop,
        "support_bounds_xy_m": support_bounds,
        "support_polygon_margin_com_m": support_polygon_margin_com,
        "support_polygon_margin_cop_m": support_polygon_margin_cop,
        "support_polygon_margin_reference_com_m": (
            support_polygon_margin_reference_com
        ),
        "support_polygon_margin_reference_cop_m": (
            support_polygon_margin_reference_cop
        ),
        "support_contact_count": support_contact_count,
        "support_contact_points_m": contact_points,
        "actuator_torque_nm": actuator_torque,
        "generalized_actuator_torque_nm": generalized_actuator_torque,
        "reconstructed_unclipped_actuator_torque_nm": (
            reconstructed_unclipped_actuator_torque
        ),
        "actuator_gain_term_nm": actuator_gain_term,
        "actuator_bias_term_nm": actuator_bias_term,
        "actuator_torque_limit_nm": engine.force_scale.copy(),
        "normalized_torque_usage": normalized_torque_usage,
        "actuator_position_rad": actuator_position,
        "actuator_velocity_rad_s": actuator_velocity,
        "actuator_transmission_velocity_rad_s": actuator_transmission_velocity,
        "actuator_acceleration_rad_s2": actuator_acceleration,
        "actuator_target_rad": actuator_target,
        "actuator_target_velocity_rad_s": actuator_target_velocity,
        "actuator_target_acceleration_rad_s2": actuator_target_acceleration,
        "actuator_position_error_rad": actuator_position_error,
        "actuator_velocity_error_rad_s": actuator_velocity_error,
        "foot_body_position_m": foot_body_position,
        "foot_body_velocity_m_s": foot_body_velocity,
        "lower_body_target_rad": lower_body_target,
        "actuator_names": np.asarray(actuator_names),
        "balance_joint_names": np.asarray(
            [engine.joint_names[index] for index in engine.balance_indexes]
        ),
    }

    root_error = reference_root[:, 2] - root_position[:, 2]
    com_height_error = reference_com[:, 2] - com_position[:, 2]
    velocity_error = reference_com_velocity[:, 2] - com_velocity[:, 2]
    force_deficit = required_vertical_force - total_support_force[:, 2]
    any_support = np.any(foot_contact, axis=1)
    downward_velocity_excess = reference_com_velocity[:, 2] - com_velocity[:, 2]
    earliest = {
        "root_height_error_gt_10mm_sample": _first_sustained(root_error > 0.010),
        "com_height_error_gt_10mm_sample": _first_sustained(com_height_error > 0.010),
        "vertical_velocity_error_gt_0p10m_s_sample": _first_sustained(
            np.abs(velocity_error) > 0.10
        ),
        "vertical_force_deficit_gt_10pct_sample": _first_sustained(
            force_deficit > 0.10 * np.maximum(required_vertical_force, 1.0)
        ),
        "supported_force_deficit_gt_10pct_sample": _first_sustained(
            any_support
            & (force_deficit > 0.10 * np.maximum(required_vertical_force, 1.0))
        ),
        "actual_downward_velocity_excess_gt_0p10m_s_sample": _first_sustained(
            downward_velocity_excess > 0.10
        ),
        "any_torque_usage_gt_80pct_sample": _first_sustained(
            np.max(normalized_torque_usage, axis=1) > 0.80
        ),
        "lower_body_error_gt_50mrad_sample": _first_sustained(
            np.max(
                np.abs(
                    actuator_position_error[
                        :,
                        [
                            actuator_names.index(engine.joint_names[index])
                            for index in engine.balance_indexes
                        ],
                    ]
                ),
                axis=1,
            )
            > 0.05
        ),
    }
    for key, value in tuple(earliest.items()):
        earliest[key.replace("_sample", "_time_s")] = (
            None if value is None else float(time[value])
        )
        earliest[key.replace("_sample", "_frame")] = (
            None if value is None else int(control_frame[value])
        )

    relevant_names = [
        name
        for name in actuator_names
        if any(token in name for token in ("hip", "knee", "ankle", "waist"))
    ]
    actuator_summary: dict[str, Any] = {}
    for name in relevant_names:
        index = actuator_names.index(name)
        usage = normalized_torque_usage[:, index]
        torque = actuator_torque[:, index]
        error = actuator_position_error[:, index]
        velocity = actuator_velocity[:, index]
        peak_index = int(np.argmax(usage))
        actuator_summary[name] = {
            "peak_normalized_usage": float(usage[peak_index]),
            "peak_usage_time_s": float(time[peak_index]),
            "peak_torque_nm": float(torque[peak_index]),
            "torque_limit_nm": float(engine.force_scale[index]),
            "duration_above_80pct_s": float(np.sum(usage > 0.80) * cfg.physics_dt),
            "duration_above_95pct_s": float(np.sum(usage > 0.95) * cfg.physics_dt),
            "positive_torque_duration_s": float(np.sum(torque > 0.0) * cfg.physics_dt),
            "negative_torque_duration_s": float(np.sum(torque < 0.0) * cfg.physics_dt),
            "maximum_abs_position_error_rad": float(np.max(np.abs(error))),
            "rms_position_error_rad": float(np.sqrt(np.mean(np.square(error)))),
            "maximum_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
        }

    minimum_velocity_sample = int(np.argmin(com_velocity[:, 2]))
    reversal_indexes = np.flatnonzero(
        (np.arange(samples) > minimum_velocity_sample) & (com_velocity[:, 2] >= 0.0)
    )
    reversal_sample = int(reversal_indexes[0]) if len(reversal_indexes) else None
    first_left = np.flatnonzero(foot_contact[:, 0])
    first_right = np.flatnonzero(foot_contact[:, 1])
    first_nonfoot = np.flatnonzero(nonfoot_terrain_contact)
    first_wrist = np.flatnonzero(wrist_terrain_contact)
    failure_sample = min(
        samples - 1,
        (rollout.first_invalid_frame or end_frame) * cfg.substeps,
    )

    summary: dict[str, Any] = {
        "label": label,
        "sampling_hz": 1.0 / cfg.physics_dt,
        "reference_rate_hz": 1.0 / cfg.control_dt,
        "end_frame": end_frame,
        "robot_mass_kg": total_mass,
        "rollout_first_invalid_frame": rollout.first_invalid_frame,
        "rollout_invalid_reason": rollout.invalid_reason,
        "earliest_divergence": earliest,
        "minimum_root_height_m": float(np.min(root_position[:, 2])),
        "maximum_root_height_deficit_m": float(np.max(root_error)),
        "maximum_com_height_deficit_m": float(np.max(com_height_error)),
        "minimum_actual_com_vertical_velocity_m_s": float(np.min(com_velocity[:, 2])),
        "minimum_reference_com_vertical_velocity_m_s": float(
            np.min(reference_com_velocity[:, 2])
        ),
        "final_vertical_momentum_kg_m_s": float(vertical_momentum[-1]),
        "minimum_vertical_momentum_kg_m_s": float(np.min(vertical_momentum)),
        "minimum_vertical_momentum_time_s": float(time[minimum_velocity_sample]),
        "downward_momentum_reversal_time_s": (
            None if reversal_sample is None else float(time[reversal_sample])
        ),
        "first_left_foot_contact_time_s": (
            None if not len(first_left) else float(time[first_left[0]])
        ),
        "first_right_foot_contact_time_s": (
            None if not len(first_right) else float(time[first_right[0]])
        ),
        "first_nonfoot_terrain_contact_time_s": (
            None if not len(first_nonfoot) else float(time[first_nonfoot[0]])
        ),
        "first_wrist_terrain_contact_time_s": (
            None if not len(first_wrist) else float(time[first_wrist[0]])
        ),
        "foot_vertical_impulse_error_at_invalid_n_s": float(
            foot_impulse_error[failure_sample]
        ),
        "all_ground_vertical_impulse_error_at_invalid_n_s": float(
            all_ground_impulse_error[failure_sample]
        ),
        "minimum_support_polygon_margin_com_m": float(
            np.nanmin(support_polygon_margin_com)
        ),
        "minimum_support_polygon_margin_cop_m": float(
            np.nanmin(support_polygon_margin_cop)
        ),
        "minimum_support_polygon_margin_reference_com_m": float(
            np.nanmin(support_polygon_margin_reference_com)
        ),
        "minimum_support_polygon_margin_reference_cop_m": float(
            np.nanmin(support_polygon_margin_reference_cop)
        ),
        "cop_defined_fraction": float(np.mean(np.all(np.isfinite(cop), axis=1))),
        "reference_cop_note": (
            "required ground-wrench location from COM acceleration and the "
            "finite-difference rate of MuJoCo subtree angular momentum; assumes "
            "terrain is the only external wrench on the robot"
        ),
        "both_feet_contact_fraction": float(np.mean(np.all(foot_contact, axis=1))),
        "actuation": actuator_summary,
        "command_semantics": (
            "ctrl is a position target; actuator_force is the generated actuator "
            "torque and qfrc_actuator is the transmitted generalized torque"
        ),
    }
    return SupportDiagnosticResult(
        label=label,
        data=data_out,
        summary=summary,
        rollout=rollout,
    )
