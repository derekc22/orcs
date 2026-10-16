"""Direct MuJoCo contact-dynamics refinement of lower-body position targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import mujoco
import numpy as np

from orcs.tasks.uolm.balance_mpc import _rotation_error
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
    _np,
)
from orcs.tasks.uolm.generated_reference import GeneratedReference
from orcs.tasks.uolm.support_diagnostics import (
    _contact_wrench,
    _convex_hull,
    _polygon_margin,
)

__all__ = [
    "ContactDynamicsConfig",
    "ContactDynamicsResult",
    "refine_contact_dynamics",
]


@dataclass(frozen=True, slots=True)
class ContactDynamicsWeights:
    root_height: float = 300.0
    root_horizontal: float = 8.0
    root_velocity: float = 30.0
    torso_orientation: float = 100.0
    torso_tilt: float = 120.0
    angular_momentum: float = 2.0
    both_feet_off: float = 500.0
    single_foot_off: float = 8.0
    low_support_force: float = 40.0
    foot_slip: float = 4.0
    cop_edge: float = 120.0
    joint_tracking: float = 20.0
    joint_velocity: float = 0.3
    actuator_saturation: float = 25.0
    residual: float = 4.0
    residual_rate: float = 2.0
    residual_acceleration: float = 0.2
    terminal: float = 12.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ContactDynamicsConfig:
    anchor_frame: int = 85
    optimize_end_frame: int = 140
    evaluation_end_frame: int = 160
    knot_frames: tuple[int, ...] = (85, 95, 110, 125, 140, 160)
    residual_bound_rad: float = 0.15
    population: int = 36
    elite_count: int = 6
    generations: int = 10
    initial_std_rad: float = 0.05
    minimum_std_rad: float = 0.012
    seed: int = 545
    nominal_alignment_end_frame: int = 100
    cop_target_margin_m: float = 0.005
    minimum_support_force_fraction: float = 0.55
    terminal_frames: int = 6
    decision_joint_names: tuple[str, ...] | None = None
    weights: ContactDynamicsWeights = field(default_factory=ContactDynamicsWeights)


@dataclass(slots=True)
class ContactDynamicsEvaluation:
    target: np.ndarray
    correction: np.ndarray
    valid: bool
    recoverable_terminal: bool
    first_invalid_frame: int | None
    first_invalid_step: int | None
    invalid_reason: str | None
    cost: float
    metrics: dict[str, Any]
    control: dict[str, np.ndarray]
    simulation: dict[str, np.ndarray]


@dataclass(slots=True)
class ContactDynamicsResult:
    best: ContactDynamicsEvaluation
    zero: ContactDynamicsEvaluation
    through95_seed: ContactDynamicsEvaluation
    warm_start: ContactDynamicsEvaluation | None
    parameters: np.ndarray
    history: list[dict[str, Any]]
    optimizer: dict[str, Any]
    prefix: dict[str, Any]
    evaluations: int
    optimization_seconds: float


def _smootherstep(value: np.ndarray) -> np.ndarray:
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _residual_path(
    parameters: np.ndarray,
    frames: int,
    joints: int,
    knot_frames: tuple[int, ...],
) -> np.ndarray:
    """C2 residual with fixed-zero entry and exit knots."""
    knots = np.zeros((len(knot_frames), joints), dtype=np.float64)
    knots[1:-1] = parameters.reshape(len(knot_frames) - 2, joints)
    result = np.zeros((frames, joints), dtype=np.float64)
    for segment in range(len(knot_frames) - 1):
        lower, upper = knot_frames[segment : segment + 2]
        sample = np.arange(lower, upper + 1)
        blend = _smootherstep((sample - lower) / (upper - lower))[:, None]
        result[sample] = (1.0 - blend) * knots[segment] + blend * knots[segment + 1]
    return result


def _contact_state(
    engine: _DynamicRollout, data: mujoco.MjData
) -> tuple[set[int], float, bool]:
    terrain_bodies: set[int] = set()
    maximum_penetration = 0.0
    wrist_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        maximum_penetration = max(maximum_penetration, max(0.0, -float(contact.dist)))
        if not pair & engine.terrain_geoms:
            continue
        other = second if first in engine.terrain_geoms else first
        body = int(engine.model.geom_bodyid[other])
        if body not in engine.robot_bodies:
            continue
        terrain_bodies.add(body)
        name = engine._name(mujoco.mjtObj.mjOBJ_BODY, body)
        wrist_contact |= "wrist" in name or "hand" in name
    return terrain_bodies, maximum_penetration, wrist_contact


def _reference_centroidal(
    engine: _DynamicRollout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_body = int(engine.model.jnt_bodyid[engine.robot_free_joint])
    com = np.empty((engine.horizon, 3), dtype=np.float64)
    velocity = np.empty_like(com)
    angular_momentum = np.empty_like(com)
    data = mujoco.MjData(engine.model)
    for frame in range(engine.horizon):
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
        com[frame] = data.subtree_com[root_body]
        velocity[frame] = data.subtree_linvel[root_body]
        angular_momentum[frame] = data.subtree_angmom[root_body]
    return com, velocity, angular_momentum


def _copy_data(model: mujoco.MjModel, source: mujoco.MjData) -> mujoco.MjData:
    target = mujoco.MjData(model)
    mujoco.mj_copyData(target, model, source)
    return target


class _ContactDynamicsSearch:
    def __init__(
        self,
        engine: _DynamicRollout,
        successful_prefix_target: np.ndarray,
        analytical_target: np.ndarray,
        config: ContactDynamicsConfig,
    ) -> None:
        self.engine = engine
        self.config = config
        self.successful_target = np.asarray(successful_prefix_target, dtype=np.float64)
        self.analytical_target = np.asarray(analytical_target, dtype=np.float64)
        if self.successful_target.shape != engine.reference_joint.shape:
            raise ValueError("successful-prefix target has the wrong shape")
        if self.analytical_target.shape != engine.reference_joint.shape:
            raise ValueError("analytical target has the wrong shape")
        if config.knot_frames[0] != config.anchor_frame:
            raise ValueError("first knot must equal the anchor frame")
        if config.knot_frames[-1] != config.evaluation_end_frame:
            raise ValueError("last knot must equal the evaluation end frame")
        if not (
            config.anchor_frame
            < config.optimize_end_frame
            < config.evaluation_end_frame
            < engine.horizon
        ):
            raise ValueError("invalid direct-dynamics interval")
        self.reference_com, self.reference_com_velocity, self.reference_h = (
            _reference_centroidal(engine)
        )
        names = engine.reference.tracked_body_names
        self.torso_reference_index = names.index("torso_link")
        self.reference_torso_quat = _np(
            engine.reference.body_quat_wxyz[0, :, self.torso_reference_index]
        )
        self.reference_joint_velocity = _np(engine.reference.joint_vel[0])
        self.decision_indexes = np.asarray(
            engine.balance_indexes
            if config.decision_joint_names is None
            else [
                engine.joint_names.index(name) for name in config.decision_joint_names
            ],
            dtype=np.int32,
        )
        self.decision_joint_names = tuple(
            engine.joint_names[index] for index in self.decision_indexes
        )
        self.arm_contact_bodies = {
            body
            for body in engine.robot_bodies
            if any(
                token in engine._name(mujoco.mjtObj.mjOBJ_BODY, body)
                for token in ("shoulder", "elbow", "wrist", "hand")
            )
        }
        self.baseline_arm_object_contact: np.ndarray | None = None
        self.robot_root_body = int(engine.model.jnt_bodyid[engine.robot_free_joint])
        self.mass = float(engine.model.body_subtreemass[self.robot_root_body])
        self.nominal_target = self._aligned_nominal_target()
        self.prefix_data, self.prefix = self._execute_prefix()

    def _aligned_nominal_target(self) -> np.ndarray:
        cfg = self.config
        target = self.analytical_target.copy()
        target[: cfg.anchor_frame + 1] = self.successful_target[: cfg.anchor_frame + 1]
        offset = (
            self.successful_target[cfg.anchor_frame]
            - self.analytical_target[cfg.anchor_frame]
        )
        sample = np.arange(cfg.anchor_frame, cfg.nominal_alignment_end_frame + 1)
        phase = (sample - cfg.anchor_frame) / (
            cfg.nominal_alignment_end_frame - cfg.anchor_frame
        )
        target[sample] += (1.0 - _smootherstep(phase))[:, None] * offset
        target[cfg.anchor_frame] = self.successful_target[cfg.anchor_frame]
        return target

    def _execute_prefix(self) -> tuple[mujoco.MjData, dict[str, Any]]:
        engine, anchor = self.engine, self.config.anchor_frame
        data = engine._new_initialized_data()
        initial_torso = data.xmat[engine.torso_body].reshape(3, 3).copy()
        self.torso_local_up = initial_torso.T @ np.asarray([0.0, 0.0, 1.0])
        data.ctrl[:] = np.clip(
            self.successful_target[0, engine.actuator_joint_indexes],
            engine.model.actuator_ctrlrange[:, 0],
            engine.model.actuator_ctrlrange[:, 1],
        )
        for frame in range(1, anchor + 1):
            data.ctrl[:] = np.clip(
                self.successful_target[frame, engine.actuator_joint_indexes],
                engine.model.actuator_ctrlrange[:, 0],
                engine.model.actuator_ctrlrange[:, 1],
            )
            for _ in range(engine.config.substeps):
                mujoco.mj_step(engine.model, data)
        terrain, penetration, _ = _contact_state(engine, data)
        return _copy_data(engine.model, data), {
            "anchor_frame": anchor,
            "root_position_m": data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ].copy(),
            "root_quaternion_wxyz": data.qpos[
                engine.robot_qpos + 3 : engine.robot_qpos + 7
            ].copy(),
            "root_velocity_local_m_s_rad_s": data.qvel[
                engine.robot_dof : engine.robot_dof + 6
            ].copy(),
            "joint_position_rad": data.qpos[engine.joint_qpos].copy(),
            "joint_velocity_rad_s": data.qvel[engine.joint_dof].copy(),
            "object_position_m": data.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ].copy(),
            "object_quaternion_wxyz": data.qpos[
                engine.object_qpos + 3 : engine.object_qpos + 7
            ].copy(),
            "object_velocity_local_m_s_rad_s": data.qvel[
                engine.object_dof : engine.object_dof + 6
            ].copy(),
            "foot_contact_left_right": [body in terrain for body in engine.foot_bodies],
            "maximum_penetration_m": penetration,
        }

    def decode(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        engine, cfg = self.engine, self.config
        residual = _residual_path(
            parameters,
            engine.horizon,
            len(self.decision_indexes),
            cfg.knot_frames,
        )
        target = self.nominal_target.copy()
        target[:, self.decision_indexes] += residual
        return target, residual

    def set_arm_object_contact_baseline(
        self, evaluation: ContactDynamicsEvaluation
    ) -> None:
        self.baseline_arm_object_contact = evaluation.simulation[
            "arm_object_contact"
        ].copy()

    def evaluate(
        self, parameters: np.ndarray, *, detailed: bool = False
    ) -> ContactDynamicsEvaluation:
        engine, cfg, weights = self.engine, self.config, self.config.weights
        target, residual = self.decode(parameters)
        data = _copy_data(engine.model, self.prefix_data)
        start, end = cfg.anchor_frame, cfg.evaluation_end_frame
        control_count = end - start + 1
        sim_count = (end - start) * engine.config.substeps + 1
        control_root = np.empty((control_count, 3), dtype=np.float64)
        control_root_quat = np.empty((control_count, 4), dtype=np.float64)
        control_root_velocity = np.empty((control_count, 6), dtype=np.float64)
        control_joint = np.empty((control_count, len(engine.joint_names)))
        control_joint_velocity = np.empty_like(control_joint)
        control_object_pos = np.empty((control_count, 3), dtype=np.float64)
        control_object_quat = np.empty((control_count, 4), dtype=np.float64)
        sim_contact = np.zeros((sim_count, 2), dtype=bool)
        sim_normal = np.zeros((sim_count, 2), dtype=np.float64)
        sim_force = np.zeros((sim_count, 2, 3), dtype=np.float64)
        sim_slip = np.zeros((sim_count, 2), dtype=np.float64)
        sim_root_height = np.empty(sim_count, dtype=np.float64)
        sim_vertical_momentum = np.empty(sim_count, dtype=np.float64)
        sim_torso_tilt = np.empty(sim_count, dtype=np.float64)
        sim_usage = np.empty((sim_count, engine.model.nu), dtype=np.float64)
        sim_tracking = np.empty((sim_count, engine.model.nu), dtype=np.float64)
        sim_torque = np.empty((sim_count, engine.model.nu), dtype=np.float64)
        sim_cop_margin = np.full(sim_count, np.nan, dtype=np.float64)
        sim_h = np.empty((sim_count, 3), dtype=np.float64)
        sim_frame = np.empty(sim_count, dtype=np.int32)
        sim_wrist = np.zeros(sim_count, dtype=bool)
        sim_arm_object = np.zeros(sim_count, dtype=bool)
        max_penetration = 0.0
        forbidden_run = 0
        severe_run = 0
        new_arm_object_run = 0
        first_invalid_frame: int | None = None
        first_invalid_step: int | None = None
        invalid_reason: str | None = None
        cost = 0.0
        sample = 0

        target_below = np.maximum(engine.joint_limits[:, 0] - target, 0.0)
        target_above = np.maximum(target - engine.joint_limits[:, 1], 0.0)
        max_target_violation = float(np.max(target_below + target_above))
        if max_target_violation > engine.config.joint_limit_margin_rad:
            first_invalid_frame = start
            first_invalid_step = 0
            invalid_reason = "catastrophic_joint_target_limit"

        def record(frame: int, index: int) -> tuple[set[int], float]:
            nonlocal max_penetration
            forces, _, normal, slip, points, _ = _contact_wrench(engine, data)
            terrain, penetration, wrist = _contact_state(engine, data)
            max_penetration = max(max_penetration, penetration)
            support = np.asarray(
                [body in terrain for body in engine.foot_bodies], dtype=bool
            )
            rotation = data.xmat[engine.torso_body].reshape(3, 3)
            torso_up = rotation @ self.torso_local_up
            tilt = float(np.arccos(np.clip(torso_up[2], -1.0, 1.0)))
            usage = np.abs(data.actuator_force) / engine.force_scale
            tracking = data.ctrl - data.qpos[engine.actuator_qpos]
            sim_contact[index] = support
            sim_normal[index] = normal
            sim_force[index] = forces
            sim_slip[index] = slip
            sim_root_height[index] = data.qpos[engine.robot_qpos + 2]
            sim_vertical_momentum[index] = (
                self.mass * data.subtree_linvel[self.robot_root_body, 2]
            )
            sim_torso_tilt[index] = tilt
            sim_usage[index] = usage
            sim_tracking[index] = tracking
            sim_torque[index] = data.actuator_force
            sim_h[index] = data.subtree_angmom[self.robot_root_body]
            sim_frame[index] = frame
            sim_wrist[index] = wrist
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if not {first, second} & engine.object_geoms:
                    continue
                other = second if first in engine.object_geoms else first
                if int(engine.model.geom_bodyid[other]) in self.arm_contact_bodies:
                    sim_arm_object[index] = True
                    break
            if len(points):
                total_force = forces.sum(axis=0)
                if total_force[2] > 1.0:
                    moments = np.zeros(3, dtype=np.float64)
                    # _contact_wrench returns each foot moment about world origin.
                    _, foot_moment, _, _, _, _ = _contact_wrench(engine, data)
                    moments[:] = foot_moment.sum(axis=0)
                    cop = np.asarray(
                        [-moments[1] / total_force[2], moments[0] / total_force[2]]
                    )
                    sim_cop_margin[index] = _polygon_margin(
                        cop, _convex_hull(points[:, :2])
                    )
            return terrain, tilt

        def state_cost(frame: int, index: int, terminal: bool) -> float:
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            torso_rotation = data.xmat[engine.torso_body].reshape(3, 3)
            root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
            velocity_error = (
                data.subtree_linvel[self.robot_root_body]
                - self.reference_com_velocity[frame]
            )
            h_error = sim_h[index, :2] - self.reference_h[frame, :2]
            support_count = int(sim_contact[index].sum())
            support_force = float(sim_force[index, :, 2].sum())
            force_deficit = max(
                0.0,
                cfg.minimum_support_force_fraction * self.mass * 9.81 - support_force,
            ) / (self.mass * 9.81)
            cop_edge = (
                max(0.0, cfg.cop_target_margin_m - sim_cop_margin[index])
                if np.isfinite(sim_cop_margin[index])
                else float(support_count == 0)
            )
            joint_velocity_error = (
                data.qvel[engine.joint_dof[engine.balance_indexes]]
                - self.reference_joint_velocity[frame, engine.balance_indexes]
            )
            value = (
                weights.root_height * (root[2] - engine.reference_root[frame, 2]) ** 2
                + weights.root_horizontal
                * float(np.square(root[:2] - engine.reference_root[frame, :2]).sum())
                + weights.root_velocity * float(np.square(velocity_error).sum())
                + weights.torso_orientation
                * _rotation_error(
                    torso_rotation,
                    _quat_rotation(self.reference_torso_quat[frame]),
                )
                ** 2
                + weights.torso_tilt * sim_torso_tilt[index] ** 2
                + weights.angular_momentum * float(np.square(h_error).sum())
                + weights.both_feet_off * float(support_count == 0)
                + weights.single_foot_off * float(support_count == 1)
                + weights.low_support_force * force_deficit**2
                + weights.foot_slip * float(np.square(sim_slip[index]).sum())
                + weights.cop_edge * cop_edge**2
                + weights.joint_tracking * float(np.square(sim_tracking[index]).mean())
                + weights.joint_velocity * float(np.square(joint_velocity_error).mean())
                + weights.actuator_saturation
                * float(np.square(np.maximum(sim_usage[index] - 0.8, 0.0)).mean())
            )
            if terminal:
                downward = max(0.0, -sim_vertical_momentum[index] - 2.0)
                root_deficit = max(0.0, engine.reference_root[frame, 2] - root[2])
                angular_speed = float(np.linalg.norm(data.cvel[engine.torso_body, :2]))
                value += weights.terminal * (
                    value
                    + 3.0 * downward**2
                    + 300.0 * root_deficit**2
                    + 8.0 * angular_speed**2
                    + 80.0 * float(support_count == 0)
                )
            # Retain direct access to the actual base state in the objective.
            value += 0.1 * float(np.square(root_velocity).sum())
            return float(value)

        record(start, sample)
        for frame in range(start, end + 1):
            local = frame - start
            control_root[local] = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            control_root_quat[local] = data.qpos[
                engine.robot_qpos + 3 : engine.robot_qpos + 7
            ]
            control_root_velocity[local] = data.qvel[
                engine.robot_dof : engine.robot_dof + 6
            ]
            control_joint[local] = data.qpos[engine.joint_qpos]
            control_joint_velocity[local] = data.qvel[engine.joint_dof]
            control_object_pos[local] = data.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ]
            control_object_quat[local] = data.qpos[
                engine.object_qpos + 3 : engine.object_qpos + 7
            ]
            if frame == end or first_invalid_frame is not None:
                break
            next_frame = frame + 1
            data.ctrl[:] = np.clip(
                target[next_frame, engine.actuator_joint_indexes],
                engine.model.actuator_ctrlrange[:, 0],
                engine.model.actuator_ctrlrange[:, 1],
            )
            for _ in range(engine.config.substeps):
                mujoco.mj_step(engine.model, data)
                sample += 1
                terrain, tilt = record(next_frame, sample)
                forbidden = terrain - engine.allowed_terrain_bodies
                forbidden_run = forbidden_run + 1 if forbidden else 0
                severe = (
                    np.mean(sim_usage[sample] >= engine.config.severe_saturation_usage)
                    >= engine.config.severe_saturation_actuator_fraction
                )
                severe_run = severe_run + 1 if severe else 0
                baseline_arm_contact = (
                    False
                    if self.baseline_arm_object_contact is None
                    or sample >= len(self.baseline_arm_object_contact)
                    else bool(self.baseline_arm_object_contact[sample])
                )
                # The zero-residual rollout establishes the contact baseline, so it
                # cannot itself introduce a contact relative to that baseline.
                new_arm_object = (
                    self.baseline_arm_object_contact is not None
                    and sim_arm_object[sample]
                    and not baseline_arm_contact
                )
                new_arm_object_run = new_arm_object_run + 1 if new_arm_object else 0
                reason = None
                if data.qpos[engine.robot_qpos + 2] < engine.minimum_root_height:
                    reason = "root_height"
                elif tilt > engine.maximum_torso_tilt:
                    reason = "torso_tilt"
                elif forbidden & engine.fall_terrain_bodies:
                    reason = "fall_body_terrain_contact"
                elif forbidden_run >= engine.sustained_terrain_steps:
                    reason = "sustained_nonfoot_terrain_contact"
                elif max_penetration > engine.config.hard_penetration_m:
                    reason = "pathological_penetration"
                elif severe_run >= engine.severe_saturation_steps:
                    reason = "sustained_actuator_saturation"
                elif new_arm_object_run >= engine.sustained_terrain_steps:
                    reason = "new_sustained_arm_object_contact"
                if reason is not None:
                    first_invalid_frame = next_frame
                    first_invalid_step = sample
                    invalid_reason = reason
                    break
                terminal = next_frame >= end - cfg.terminal_frames + 1
                cost += state_cost(next_frame, sample, terminal)
            if first_invalid_frame is not None:
                break

        simulated_samples = sample + 1
        residual_window = residual[start : end + 1]
        residual_velocity = np.diff(residual_window, axis=0) / engine.config.control_dt
        residual_acceleration = (
            np.diff(residual_velocity, axis=0) / engine.config.control_dt
        )
        cost += weights.residual * float(np.square(residual_window).mean())
        cost += weights.residual_rate * float(np.square(residual_velocity).mean())
        cost += weights.residual_acceleration * float(
            np.square(residual_acceleration).mean()
        )
        support_last = sim_contact[sample]
        recoverable = bool(
            first_invalid_frame is None
            and support_last.any()
            and sim_root_height[sample] >= engine.minimum_root_height + 0.03
            and sim_vertical_momentum[sample] >= -5.0
            and sim_torso_tilt[sample] <= np.deg2rad(35.0)
            and np.linalg.norm(data.cvel[engine.torso_body, :2]) <= 3.0
        )
        metrics = {
            "first_invalid_frame": first_invalid_frame,
            "first_invalid_step": first_invalid_step,
            "invalid_reason": invalid_reason,
            "simulated_through_frame": int(sim_frame[sample]),
            "recoverable_terminal": recoverable,
            "minimum_root_height_m": float(np.min(sim_root_height[:simulated_samples])),
            "maximum_torso_tilt_deg": float(
                np.rad2deg(np.max(sim_torso_tilt[:simulated_samples]))
            ),
            "maximum_actuator_usage": float(np.max(sim_usage[:simulated_samples])),
            "maximum_joint_tracking_error_rad": float(
                np.max(np.abs(sim_tracking[:simulated_samples]))
            ),
            "both_feet_off_duration_s": float(
                np.sum(~np.any(sim_contact[:simulated_samples], axis=1))
                * engine.config.physics_dt
            ),
            "minimum_actual_cop_margin_m": (
                float(np.nanmin(sim_cop_margin[:simulated_samples]))
                if np.any(np.isfinite(sim_cop_margin[:simulated_samples]))
                else None
            ),
            "maximum_penetration_m": float(max_penetration),
            "arm_object_contact_duration_s": float(
                np.sum(sim_arm_object[:simulated_samples]) * engine.config.physics_dt
            ),
            "terminal_root_height_m": float(sim_root_height[sample]),
            "terminal_vertical_momentum_kg_m_s": float(sim_vertical_momentum[sample]),
            "terminal_torso_tilt_deg": float(np.rad2deg(sim_torso_tilt[sample])),
            "terminal_foot_contact_left_right": support_last.tolist(),
            "residual_rms_rad": float(np.sqrt(np.mean(np.square(residual_window)))),
            "residual_max_rad": float(np.max(np.abs(residual_window))),
            "maximum_target_joint_limit_violation_rad": max_target_violation,
        }
        control = {
            "root_position_m": control_root[: local + 1],
            "root_quaternion_wxyz": control_root_quat[: local + 1],
            "root_velocity_local_m_s_rad_s": control_root_velocity[: local + 1],
            "joint_position_rad": control_joint[: local + 1],
            "joint_velocity_rad_s": control_joint_velocity[: local + 1],
            "object_position_m": control_object_pos[: local + 1],
            "object_quaternion_wxyz": control_object_quat[: local + 1],
        }
        simulation = {
            "control_frame": sim_frame[:simulated_samples],
            "foot_contact": sim_contact[:simulated_samples],
            "foot_normal_force_n": sim_normal[:simulated_samples],
            "foot_force_world_n": sim_force[:simulated_samples],
            "foot_slip_velocity_m_s": sim_slip[:simulated_samples],
            "root_height_m": sim_root_height[:simulated_samples],
            "vertical_momentum_kg_m_s": sim_vertical_momentum[:simulated_samples],
            "torso_tilt_rad": sim_torso_tilt[:simulated_samples],
            "normalized_actuator_usage": sim_usage[:simulated_samples],
            "joint_tracking_error_rad": sim_tracking[:simulated_samples],
            "actuator_torque_nm": sim_torque[:simulated_samples],
            "actual_cop_margin_m": sim_cop_margin[:simulated_samples],
            "centroidal_angular_momentum_kg_m2_s": sim_h[:simulated_samples],
            "wrist_terrain_contact": sim_wrist[:simulated_samples],
            "arm_object_contact": sim_arm_object[:simulated_samples],
        }
        if not detailed:
            control = {}
            simulation = {}
        return ContactDynamicsEvaluation(
            target=target,
            correction=residual,
            valid=first_invalid_frame is None,
            recoverable_terminal=recoverable,
            first_invalid_frame=first_invalid_frame,
            first_invalid_step=first_invalid_step,
            invalid_reason=invalid_reason,
            cost=float(cost),
            metrics=metrics,
            control=control,
            simulation=simulation,
        )


def _quat_rotation(quaternion: np.ndarray) -> np.ndarray:
    value = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(value, quaternion)
    return value.reshape(3, 3)


def _rank(evaluation: ContactDynamicsEvaluation) -> tuple[int, int, float]:
    if evaluation.valid:
        return (0 if evaluation.recoverable_terminal else 1, 0, evaluation.cost)
    step = evaluation.first_invalid_step or 0
    return (2, -step, evaluation.cost)


def refine_contact_dynamics(
    model: mujoco.MjModel,
    reference: GeneratedReference,
    successful_prefix_target: np.ndarray,
    analytical_target: np.ndarray,
    dynamics_config: DynamicsRefinementConfig | None = None,
    config: ContactDynamicsConfig | None = None,
    warm_start_parameters: np.ndarray | None = None,
) -> ContactDynamicsResult:
    """Optimize lower-body targets using actual MuJoCo contact dynamics."""
    cfg = config or ContactDynamicsConfig()
    engine = _DynamicRollout(
        model,
        reference,
        dynamics_config or DynamicsRefinementConfig(run_stage_b=False),
    )
    search = _ContactDynamicsSearch(
        engine, successful_prefix_target, analytical_target, cfg
    )
    joints = len(search.decision_indexes)
    free_knots = len(cfg.knot_frames) - 2
    dimensions = free_knots * joints
    lower = np.full(dimensions, -cfg.residual_bound_rad, dtype=np.float64)
    upper = -lower
    rng = np.random.default_rng(cfg.seed)
    mean = np.zeros(dimensions, dtype=np.float64)
    std = np.full(dimensions, cfg.initial_std_rad, dtype=np.float64)
    zero_parameters = np.zeros(dimensions, dtype=np.float64)
    if warm_start_parameters is not None:
        warm_start_parameters = np.asarray(
            warm_start_parameters, dtype=np.float64
        ).reshape(-1)
        if warm_start_parameters.shape != (dimensions,):
            raise ValueError(
                f"warm start must have shape {(dimensions,)}, got "
                f"{warm_start_parameters.shape}"
            )
        warm_start_parameters = np.clip(warm_start_parameters, lower, upper)
    through95_parameters = np.clip(
        np.stack(
            [
                successful_prefix_target[frame, search.decision_indexes]
                - search.nominal_target[frame, search.decision_indexes]
                for frame in cfg.knot_frames[1:-1]
            ]
        ).reshape(-1),
        lower,
        upper,
    )
    started = perf_counter()
    zero = search.evaluate(zero_parameters, detailed=True)
    search.set_arm_object_contact_baseline(zero)
    through95_seed = search.evaluate(through95_parameters)
    warm_start = (
        None
        if warm_start_parameters is None
        else search.evaluate(warm_start_parameters)
    )
    evaluations = 2 + int(warm_start is not None)
    initial_candidates = [
        (zero, zero_parameters),
        (through95_seed, through95_parameters),
    ]
    if warm_start is not None and warm_start_parameters is not None:
        initial_candidates.append((warm_start, warm_start_parameters))
    best, best_parameters = min(initial_candidates, key=lambda item: _rank(item[0]))
    best_parameters = best_parameters.copy()
    mean = best_parameters.copy()
    history: list[dict[str, Any]] = []
    for generation in range(cfg.generations):
        samples = np.clip(
            rng.normal(mean, std, size=(cfg.population, dimensions)), lower, upper
        )
        samples[0] = best_parameters
        if warm_start_parameters is not None:
            samples[1] = warm_start_parameters
            samples[2] = zero_parameters
            samples[3] = through95_parameters
        else:
            samples[1] = zero_parameters
            samples[2] = through95_parameters
        results = [search.evaluate(sample) for sample in samples]
        evaluations += len(results)
        order = sorted(range(len(results)), key=lambda index: _rank(results[index]))
        elites = samples[order[: cfg.elite_count]]
        mean = np.clip(0.25 * mean + 0.75 * elites.mean(axis=0), lower, upper)
        std = np.maximum(
            cfg.minimum_std_rad,
            0.25 * std + 0.75 * elites.std(axis=0),
        )
        candidate = results[order[0]]
        if _rank(candidate) < _rank(best):
            best = candidate
            best_parameters = samples[order[0]].copy()
        history.append(
            {
                "generation": generation,
                "valid": best.valid,
                "recoverable_terminal": best.recoverable_terminal,
                "first_invalid_frame": best.first_invalid_frame,
                "invalid_reason": best.invalid_reason,
                "cost": best.cost,
                "both_feet_off_duration_s": best.metrics["both_feet_off_duration_s"],
                "minimum_root_height_m": best.metrics["minimum_root_height_m"],
                "maximum_torso_tilt_deg": best.metrics["maximum_torso_tilt_deg"],
                "terminal_vertical_momentum_kg_m_s": best.metrics[
                    "terminal_vertical_momentum_kg_m_s"
                ],
            }
        )
    best = search.evaluate(best_parameters, detailed=True)
    evaluations += 1
    return ContactDynamicsResult(
        best=best,
        zero=zero,
        through95_seed=through95_seed,
        warm_start=warm_start,
        parameters=best_parameters,
        history=history,
        optimizer={
            "method": "cross_entropy",
            "population": cfg.population,
            "elite_count": cfg.elite_count,
            "generations": cfg.generations,
            "dimensions": dimensions,
            "physical_candidate_rollouts": evaluations,
            "knot_frames": list(cfg.knot_frames),
            "free_knot_frames": list(cfg.knot_frames[1:-1]),
            "residual_bound_rad": cfg.residual_bound_rad,
            "decision_joint_names": list(search.decision_joint_names),
            "weights": cfg.weights.as_dict(),
            "nominal_prior": (
                "analytical through-140 target, C2-aligned to the successful "
                "through-95 prefix at frame 85"
            ),
        },
        prefix=search.prefix,
        evaluations=evaluations,
        optimization_seconds=perf_counter() - started,
    )
