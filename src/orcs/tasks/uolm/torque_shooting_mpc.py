"""Short-horizon torque-residual MPC evaluated only by MuJoCo shooting.

The controller contains no analytical contact constraints.  It copies the
current physical state, rolls candidate torque residuals through the compiled
position actuators and MuJoCo contacts, applies the first replan interval, and
then solves again from the newly observed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import mujoco
import numpy as np

from orcs.tasks.uolm.contact_dynamics_refinement import _contact_state, _copy_data
from orcs.tasks.uolm.dynamics_refinement import _DynamicRollout
from orcs.tasks.uolm.support_diagnostics import _contact_wrench
from orcs.tasks.uolm.whole_body_feedback import _execute_prefix

__all__ = [
    "TorqueShootingMPCConfig",
    "TorqueShootingMPCResult",
    "track_with_torque_shooting_mpc",
]


@dataclass(frozen=True, slots=True)
class TorqueShootingWeights:
    root_height: float = 600.0
    root_horizontal: float = 20.0
    com_position: float = 180.0
    com_velocity: float = 90.0
    downward_velocity: float = 140.0
    torso_orientation: float = 260.0
    torso_tilt: float = 220.0
    torso_angular_velocity: float = 35.0
    centroidal_momentum: float = 1.5
    both_feet_off: float = 800.0
    single_foot_off: float = 15.0
    low_support_force: float = 80.0
    foot_slip: float = 8.0
    forbidden_contact: float = 1200.0
    joint_position: float = 12.0
    joint_velocity: float = 0.25
    torque_residual: float = 2.0
    torque_rate: float = 5.0
    torque_saturation: float = 30.0
    terminal_multiplier: float = 10.0


@dataclass(frozen=True, slots=True)
class TorqueShootingMPCConfig:
    anchor_frame: int = 85
    evaluation_end_frame: int = 160
    horizon_frames: int = 12
    replan_interval_frames: int = 2
    future_knots: int = 2
    residual_limit_fraction: float = 0.25
    control_all_actuators: bool = False
    arm_residual_limit_fraction: float = 0.15
    population: int = 24
    elite_count: int = 5
    generations: int = 2
    initial_std_fraction: float = 0.10
    minimum_std_fraction: float = 0.025
    distribution_smoothing: float = 0.70
    seed: int = 884
    weights: TorqueShootingWeights = field(default_factory=TorqueShootingWeights)


@dataclass(slots=True)
class _Candidate:
    parameters: np.ndarray
    residual_path: np.ndarray
    cost: float
    first_invalid_step: int | None
    invalid_reason: str | None
    simulated_steps: int
    terminal_root_height_m: float
    terminal_vertical_velocity_m_s: float
    terminal_torso_tilt_rad: float
    terminal_support_count: int
    clipped_commands: int


@dataclass(slots=True)
class TorqueShootingMPCResult:
    valid: bool
    recoverable_terminal: bool
    first_invalid_frame: int | None
    first_invalid_sample: int | None
    invalid_reason: str | None
    data: dict[str, np.ndarray]
    replans: list[dict[str, Any]]
    summary: dict[str, Any]
    prefix: dict[str, Any]


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _rotation_error(actual: np.ndarray, desired: np.ndarray) -> float:
    relative = desired.T @ actual
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _smootherstep(value: np.ndarray) -> np.ndarray:
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _decode_residual(
    parameters: np.ndarray,
    current_residual: np.ndarray,
    horizon_steps: int,
    future_knots: int,
) -> np.ndarray:
    joints = len(current_residual)
    future = parameters.reshape(future_knots, joints)
    knot_steps = np.linspace(0, horizon_steps, future_knots + 1, dtype=np.int32)
    knots = np.vstack((current_residual, future))
    path = np.empty((horizon_steps + 1, joints), dtype=np.float64)
    for segment in range(future_knots):
        lower, upper = knot_steps[segment : segment + 2]
        indexes = np.arange(lower, upper + 1)
        phase = (indexes - lower) / max(upper - lower, 1)
        blend = _smootherstep(phase)[:, None]
        path[indexes] = (1.0 - blend) * knots[segment] + blend * knots[segment + 1]
    path[0] = current_residual
    return path


class _Reference:
    def __init__(
        self,
        engine: _DynamicRollout,
        artifact: dict[str, np.ndarray],
        config: TorqueShootingMPCConfig,
        warm_start_residual: np.ndarray | None = None,
    ) -> None:
        start = config.anchor_frame * engine.config.substeps
        end = config.evaluation_end_frame * engine.config.substeps + 1
        self.start_sample = start
        self.root_position = np.asarray(
            artifact["actual_root_position_m"][start:end], dtype=np.float64
        )
        self.root_quaternion = np.asarray(
            artifact["actual_root_quaternion_wxyz"][start:end], dtype=np.float64
        )
        self.com_position = np.asarray(
            artifact["actual_com_position_m"][start:end], dtype=np.float64
        )
        self.com_velocity = np.asarray(
            artifact["actual_com_velocity_m_s"][start:end], dtype=np.float64
        )
        self.angular_momentum = np.asarray(
            artifact["actual_centroidal_angular_momentum"][start:end],
            dtype=np.float64,
        )
        self.joint_position = np.asarray(
            artifact["actual_joint_position_rad"][start:end], dtype=np.float64
        )
        self.joint_velocity = np.asarray(
            artifact["actual_joint_velocity_rad_s"][start:end], dtype=np.float64
        )
        self.feedforward_torque = np.asarray(
            artifact["actual_actuator_torque_nm"][start:end], dtype=np.float64
        )
        count = end - start
        self.torso_rotation = np.empty((count, 3, 3), dtype=np.float64)
        self.torso_angular_velocity = np.empty((count, 3), dtype=np.float64)
        scratch = mujoco.MjData(engine.model)
        actuator_dofs = np.asarray(
            [
                engine.model.jnt_dofadr[engine.model.actuator_trnid[index, 0]]
                for index in range(engine.model.nu)
            ],
            dtype=np.int32,
        )
        for sample in range(count):
            scratch.qpos[engine.robot_qpos : engine.robot_qpos + 3] = (
                self.root_position[sample]
            )
            scratch.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = (
                self.root_quaternion[sample]
            )
            scratch.qpos[engine.actuator_qpos] = self.joint_position[sample]
            scratch.qvel[engine.robot_dof : engine.robot_dof + 3] = artifact[
                "actual_root_linear_velocity_m_s"
            ][start + sample]
            rotation = _rotation(self.root_quaternion[sample])
            scratch.qvel[engine.robot_dof + 3 : engine.robot_dof + 6] = (
                rotation.T
                @ artifact["actual_root_angular_velocity_rad_s"][start + sample]
            )
            scratch.qvel[actuator_dofs] = self.joint_velocity[sample]
            mujoco.mj_forward(engine.model, scratch)
            self.torso_rotation[sample] = scratch.xmat[engine.torso_body].reshape(3, 3)
            self.torso_angular_velocity[sample] = scratch.cvel[engine.torso_body, :3]


class _TorqueShootingMPC:
    def __init__(
        self,
        engine: _DynamicRollout,
        nominal_target: np.ndarray,
        artifact: dict[str, np.ndarray],
        config: TorqueShootingMPCConfig,
        warm_start_residual: np.ndarray | None = None,
    ) -> None:
        self.engine = engine
        self.model = engine.model
        self.nominal_target = np.asarray(nominal_target, dtype=np.float64)
        self.config = config
        if config.population < 3 or not 1 <= config.elite_count <= config.population:
            raise ValueError("CEM requires population >= 3 and valid elite count")
        if config.future_knots != 2:
            raise ValueError("the prototype warm start currently requires two knots")
        self.reference = _Reference(engine, artifact, config)
        self.rng = np.random.default_rng(config.seed)
        self.actuator_dofs = np.asarray(
            [
                self.model.jnt_dofadr[self.model.actuator_trnid[index, 0]]
                for index in range(self.model.nu)
            ],
            dtype=np.int32,
        )
        self.balance_actuators = np.flatnonzero(
            np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
        )
        self.balance_names = tuple(
            engine.joint_names[engine.actuator_joint_indexes[index]]
            for index in self.balance_actuators
        )
        self.control_actuators = (
            np.arange(self.model.nu, dtype=np.int32)
            if config.control_all_actuators
            else self.balance_actuators.copy()
        )
        self.control_names = tuple(
            engine.joint_names[engine.actuator_joint_indexes[index]]
            for index in self.control_actuators
        )
        self.kp = self.model.actuator_gainprm[:, 0].copy()
        self.kd = -self.model.actuator_biasprm[:, 2].copy()
        if np.any(self.kp <= 0.0):
            raise ValueError("torque-equivalent targets require positive actuator Kp")
        self.torque_limit = np.max(np.abs(self.model.actuator_forcerange), axis=1)
        controlled_is_balance = np.isin(self.control_actuators, self.balance_actuators)
        residual_fraction = np.where(
            controlled_is_balance,
            config.residual_limit_fraction,
            config.arm_residual_limit_fraction,
        )
        self.residual_bound = (
            residual_fraction * self.torque_limit[self.control_actuators]
        )
        self.warm_start_residual = np.zeros(
            (len(self.reference.root_position), self.model.nu), dtype=np.float64
        )
        if warm_start_residual is not None:
            supplied = np.asarray(warm_start_residual, dtype=np.float64)
            if supplied.shape != self.warm_start_residual.shape:
                raise ValueError(
                    "warm-start residual must have shape "
                    f"{self.warm_start_residual.shape}, got {supplied.shape}"
                )
            self.warm_start_residual[:] = supplied
        if config.control_all_actuators:
            arm_actuators = np.setdiff1d(self.control_actuators, self.balance_actuators)
            self.warm_start_residual[:, arm_actuators] = 0.0
        self.root_body = int(self.model.jnt_bodyid[engine.robot_free_joint])
        self.mass = float(self.model.body_subtreemass[self.root_body])
        self.arm_bodies = {
            body
            for body in engine.robot_bodies
            if any(
                token in engine._name(mujoco.mjtObj.mjOBJ_BODY, body)
                for token in ("shoulder", "elbow", "wrist", "hand")
            )
        }
        self.prefix_data, self.prefix, self.torso_local_up = _execute_prefix(
            engine, self.nominal_target, config.anchor_frame
        )
        anchor_reference = self.reference
        self.prefix["saved_reference_anchor_error"] = {
            "root_position_m": float(
                np.linalg.norm(
                    self.prefix_data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
                    - anchor_reference.root_position[0]
                )
            ),
            "joint_position_rad": float(
                np.max(
                    np.abs(
                        self.prefix_data.qpos[engine.actuator_qpos]
                        - anchor_reference.joint_position[0]
                    )
                )
            ),
            "joint_velocity_rad_s": float(
                np.max(
                    np.abs(
                        self.prefix_data.qvel[self.actuator_dofs]
                        - anchor_reference.joint_velocity[0]
                    )
                )
            ),
        }
        self.baseline_arm_object_contact = self._baseline_arm_object_schedule()

    def _baseline_arm_object_schedule(self) -> np.ndarray:
        """Measure intended arm/object contact from zero-residual torque replay."""
        samples = len(self.reference.root_position)
        result = np.zeros(samples, dtype=bool)
        data = _copy_data(self.model, self.prefix_data)
        result[0] = self._physical_state(data)[7]
        zero = np.zeros(len(self.control_actuators), dtype=np.float64)
        for sample in range(1, samples):
            command, _, _, _, _ = self._command(data, sample, zero)
            data.ctrl[:] = command
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            result[sample] = self._physical_state(data)[7]
        return result

    def _command(
        self,
        data: mujoco.MjData,
        reference_sample: int,
        residual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feedforward = self.reference.feedforward_torque[reference_sample].copy()
        requested = feedforward.copy()
        requested[self.control_actuators] += residual
        limited = np.clip(requested, -self.torque_limit, self.torque_limit)
        joint_position = data.qpos[self.engine.actuator_qpos]
        joint_velocity = data.qvel[self.actuator_dofs]
        target_raw = joint_position + (limited + self.kd * joint_velocity) / self.kp
        target = np.clip(
            target_raw,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        clipped = (np.abs(requested - limited) > 1.0e-9) | (
            np.abs(target_raw - target) > 1.0e-9
        )
        return target, feedforward, requested, limited, clipped

    def _physical_state(
        self, data: mujoco.MjData
    ) -> tuple[set[int], float, np.ndarray, np.ndarray, np.ndarray, float, bool, bool]:
        terrain, penetration, _ = _contact_state(self.engine, data)
        forces, _, normal, slip, _, _ = _contact_wrench(self.engine, data)
        arm_terrain = bool(terrain & self.arm_bodies)
        arm_object = False
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if not ({first, second} & self.engine.object_geoms):
                continue
            other = second if first in self.engine.object_geoms else first
            arm_object |= int(self.model.geom_bodyid[other]) in self.arm_bodies
        torso = data.xmat[self.engine.torso_body].reshape(3, 3)
        torso_up = torso @ self.torso_local_up
        tilt = float(np.arccos(np.clip(torso_up[2], -1.0, 1.0)))
        return (
            terrain,
            penetration,
            forces,
            normal,
            slip,
            tilt,
            arm_terrain,
            arm_object,
        )

    def _state_cost(
        self,
        data: mujoco.MjData,
        reference_sample: int,
        residual: np.ndarray,
        previous_residual: np.ndarray,
        physical: tuple[
            set[int], float, np.ndarray, np.ndarray, np.ndarray, float, bool, bool
        ],
        clipped: np.ndarray,
        new_arm_object: bool,
        *,
        terminal: bool,
    ) -> float:
        weights, ref = self.config.weights, self.reference
        terrain, _, forces, normal, slip, tilt, arm_terrain, _arm_object = physical
        root = data.qpos[self.engine.robot_qpos : self.engine.robot_qpos + 3]
        com = data.subtree_com[self.root_body]
        com_velocity = data.subtree_linvel[self.root_body]
        torso_rotation = data.xmat[self.engine.torso_body].reshape(3, 3)
        torso_omega = data.cvel[self.engine.torso_body, :3]
        support_count = int(np.sum(normal > 1.0e-6))
        support_force = float(np.sum(forces[:, 2]))
        force_deficit = max(0.0, 0.55 * self.mass * 9.81 - support_force) / (
            self.mass * 9.81
        )
        q_error = (
            data.qpos[self.engine.actuator_qpos] - ref.joint_position[reference_sample]
        )
        qdot_error = (
            data.qvel[self.actuator_dofs] - ref.joint_velocity[reference_sample]
        )
        h_error = (
            data.subtree_angmom[self.root_body] - ref.angular_momentum[reference_sample]
        )
        residual_normalized = residual / self.residual_bound
        rate_normalized = (residual - previous_residual) / self.residual_bound
        saturation = np.maximum(
            np.abs(data.actuator_force) / self.engine.force_scale - 0.9, 0.0
        )
        forbidden = bool(terrain - self.engine.allowed_terrain_bodies)
        value = (
            weights.root_height
            * (root[2] - ref.root_position[reference_sample, 2]) ** 2
            + weights.root_horizontal
            * float(np.square(root[:2] - ref.root_position[reference_sample, :2]).sum())
            + weights.com_position
            * float(np.square(com - ref.com_position[reference_sample]).sum())
            + weights.com_velocity
            * float(np.square(com_velocity - ref.com_velocity[reference_sample]).sum())
            + weights.downward_velocity * max(0.0, -com_velocity[2]) ** 2
            + weights.torso_orientation
            * _rotation_error(torso_rotation, ref.torso_rotation[reference_sample]) ** 2
            + weights.torso_tilt * tilt**2
            + weights.torso_angular_velocity
            * float(
                np.square(
                    torso_omega - ref.torso_angular_velocity[reference_sample]
                ).sum()
            )
            + weights.centroidal_momentum * float(np.square(h_error).sum())
            + weights.both_feet_off * float(support_count == 0)
            + weights.single_foot_off * float(support_count == 1)
            + weights.low_support_force * force_deficit**2
            + weights.foot_slip * float(np.square(slip).sum())
            + weights.forbidden_contact
            * float(forbidden or arm_terrain or new_arm_object)
            + weights.joint_position * float(np.square(q_error).mean())
            + weights.joint_velocity * float(np.square(qdot_error).mean())
            + weights.torque_residual * float(np.square(residual_normalized).mean())
            + weights.torque_rate * float(np.square(rate_normalized).mean())
            + weights.torque_saturation
            * (
                float(np.square(saturation).mean())
                + float(np.mean(clipped.astype(np.float64)))
            )
        )
        if terminal:
            root_deficit = max(0.0, ref.root_position[reference_sample, 2] - root[2])
            downward = max(0.0, -com_velocity[2])
            angular_speed = float(np.linalg.norm(torso_omega[:2]))
            value += weights.terminal_multiplier * (
                500.0 * root_deficit**2
                + 180.0 * downward**2
                + 180.0 * tilt**2
                + 30.0 * angular_speed**2
                + 1.5 * float(np.square(data.subtree_angmom[self.root_body, :2]).sum())
                + 1000.0 * float(support_count == 0)
            )
        return float(value)

    def _evaluate(
        self,
        source: mujoco.MjData,
        current_sample: int,
        current_residual: np.ndarray,
        parameters: np.ndarray,
        forbidden_run: int,
        severe_run: int,
        arm_run: int,
        maximum_penetration: float,
    ) -> _Candidate:
        engine, cfg = self.engine, self.config
        remaining = len(self.reference.root_position) - 1 - current_sample
        horizon = min(cfg.horizon_frames * engine.config.substeps, remaining)
        residual_path = _decode_residual(
            parameters, current_residual, horizon, cfg.future_knots
        )
        data = _copy_data(self.model, source)
        cost = 0.0
        first_invalid: int | None = None
        invalid_reason: str | None = None
        simulated = 0
        clipped_commands = 0
        last_physical = self._physical_state(data)
        for step in range(1, horizon + 1):
            reference_sample = current_sample + step
            command, _, _, _, clipped = self._command(
                data, reference_sample, residual_path[step]
            )
            clipped_commands += int(np.sum(clipped))
            data.ctrl[:] = command
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            simulated += 1
            physical = self._physical_state(data)
            terrain, penetration, _, _, _, tilt, arm_terrain, arm_object = physical
            new_arm_object = bool(
                arm_object and not self.baseline_arm_object_contact[reference_sample]
            )
            maximum_penetration = max(maximum_penetration, penetration)
            forbidden = terrain - engine.allowed_terrain_bodies
            forbidden_run = forbidden_run + 1 if forbidden else 0
            arm_run = arm_run + 1 if arm_terrain or new_arm_object else 0
            usage = np.abs(data.actuator_force) / engine.force_scale
            severe = (
                np.mean(usage >= engine.config.severe_saturation_usage)
                >= engine.config.severe_saturation_actuator_fraction
            )
            severe_run = severe_run + 1 if severe else 0
            reason = None
            if data.qpos[engine.robot_qpos + 2] < engine.minimum_root_height:
                reason = "root_height"
            elif tilt > engine.maximum_torso_tilt:
                reason = "torso_tilt"
            elif forbidden & engine.fall_terrain_bodies:
                reason = "fall_body_terrain_contact"
            elif forbidden_run >= engine.sustained_terrain_steps:
                reason = "sustained_nonfoot_terrain_contact"
            elif arm_run >= engine.sustained_terrain_steps:
                reason = "sustained_arm_support"
            elif maximum_penetration > engine.config.hard_penetration_m:
                reason = "pathological_penetration"
            elif severe_run >= engine.severe_saturation_steps:
                reason = "sustained_actuator_saturation"
            cost += self._state_cost(
                data,
                reference_sample,
                residual_path[step],
                residual_path[step - 1],
                physical,
                clipped,
                new_arm_object,
                terminal=step == horizon,
            )
            last_physical = physical
            if reason is not None:
                first_invalid = step
                invalid_reason = reason
                cost += 1.0e9 + (horizon - step) * 1.0e7
                break
        return _Candidate(
            parameters=parameters.copy(),
            residual_path=residual_path,
            cost=cost,
            first_invalid_step=first_invalid,
            invalid_reason=invalid_reason,
            simulated_steps=simulated,
            terminal_root_height_m=float(data.qpos[engine.robot_qpos + 2]),
            terminal_vertical_velocity_m_s=float(
                data.subtree_linvel[self.root_body, 2]
            ),
            terminal_torso_tilt_rad=float(last_physical[5]),
            terminal_support_count=int(np.sum(last_physical[3] > 1.0e-6)),
            clipped_commands=clipped_commands,
        )

    @staticmethod
    def _rank(candidate: _Candidate) -> tuple[int, int, float]:
        if candidate.first_invalid_step is None:
            return 0, 0, candidate.cost
        return 1, -candidate.first_invalid_step, candidate.cost

    def _solve(
        self,
        data: mujoco.MjData,
        current_sample: int,
        current_residual: np.ndarray,
        warm_mean: np.ndarray,
        forbidden_run: int,
        severe_run: int,
        arm_run: int,
        maximum_penetration: float,
    ) -> tuple[_Candidate, np.ndarray, int, int, list[dict[str, Any]]]:
        cfg = self.config
        bounds = np.tile(self.residual_bound, cfg.future_knots)
        mean = np.clip(warm_mean, -bounds, bounds)
        horizon = min(
            cfg.horizon_frames * self.engine.config.substeps,
            len(self.reference.root_position) - 1 - current_sample,
        )
        midpoint = current_sample + horizon // 2
        endpoint = current_sample + horizon
        baseline_parameters = np.concatenate(
            (
                self.warm_start_residual[midpoint, self.control_actuators],
                self.warm_start_residual[endpoint, self.control_actuators],
            )
        )
        baseline_parameters = np.clip(baseline_parameters, -bounds, bounds)
        std = cfg.initial_std_fraction * bounds
        minimum_std = cfg.minimum_std_fraction * bounds
        best = self._evaluate(
            data,
            current_sample,
            current_residual,
            mean,
            forbidden_run,
            severe_run,
            arm_run,
            maximum_penetration,
        )
        evaluations = 1
        simulated_steps = best.simulated_steps
        progress: list[dict[str, Any]] = []
        for generation in range(cfg.generations):
            population = np.clip(
                self.rng.normal(mean, std, size=(cfg.population, len(bounds))),
                -bounds,
                bounds,
            )
            population[0] = 0.0
            population[1] = best.parameters
            population[2] = baseline_parameters
            evaluated = [
                self._evaluate(
                    data,
                    current_sample,
                    current_residual,
                    parameters,
                    forbidden_run,
                    severe_run,
                    arm_run,
                    maximum_penetration,
                )
                for parameters in population
            ]
            evaluations += len(evaluated)
            simulated_steps += sum(item.simulated_steps for item in evaluated)
            order = sorted(
                range(cfg.population), key=lambda index: self._rank(evaluated[index])
            )
            generation_best = evaluated[order[0]]
            if self._rank(generation_best) < self._rank(best):
                best = generation_best
            elite = population[order[: cfg.elite_count]]
            alpha = cfg.distribution_smoothing
            mean = alpha * np.mean(elite, axis=0) + (1.0 - alpha) * mean
            std = np.maximum(
                alpha * np.std(elite, axis=0) + (1.0 - alpha) * std,
                minimum_std,
            )
            progress.append(
                {
                    "generation": generation,
                    "best_first_invalid_step": generation_best.first_invalid_step,
                    "best_invalid_reason": generation_best.invalid_reason,
                    "best_cost": generation_best.cost,
                    "baseline_candidate_rank": int(order.index(2)),
                    "baseline_first_invalid_step": evaluated[2].first_invalid_step,
                    "baseline_invalid_reason": evaluated[2].invalid_reason,
                }
            )
        apply_steps = cfg.replan_interval_frames * self.engine.config.substeps
        horizon = len(best.residual_path) - 1
        midpoint = min(apply_steps + horizon // 2, horizon)
        shifted_mean = np.concatenate(
            (best.residual_path[midpoint], best.residual_path[-1])
        )
        return best, shifted_mean, evaluations, simulated_steps, progress

    def run(self) -> TorqueShootingMPCResult:
        engine, cfg = self.engine, self.config
        data = _copy_data(self.model, self.prefix_data)
        samples = (
            cfg.evaluation_end_frame - cfg.anchor_frame
        ) * engine.config.substeps + 1
        nu = self.model.nu
        vector3_keys = (
            "root_position",
            "actual_com",
            "actual_com_velocity",
            "actual_angular_momentum",
        )
        recorded: dict[str, np.ndarray] = {
            key: np.full((samples, 3), np.nan, dtype=np.float64) for key in vector3_keys
        }
        recorded.update(
            {
                "joint_position": np.full((samples, nu), np.nan),
                "joint_velocity": np.full((samples, nu), np.nan),
                "joint_position_error": np.full((samples, nu), np.nan),
                "joint_velocity_error": np.full((samples, nu), np.nan),
                "feedforward_torque_nm": np.full((samples, nu), np.nan),
                "torque_residual_nm": np.zeros((samples, nu)),
                "requested_torque_nm": np.full((samples, nu), np.nan),
                "limited_torque_nm": np.full((samples, nu), np.nan),
                "actual_torque_nm": np.full((samples, nu), np.nan),
                "torque_usage": np.full((samples, nu), np.nan),
                "torque_clipped": np.zeros((samples, nu), dtype=bool),
                "torso_tilt_rad": np.full(samples, np.nan),
                "foot_contact": np.zeros((samples, 2), dtype=bool),
                "foot_normal_force_n": np.zeros((samples, 2)),
                "foot_slip_m_s": np.zeros((samples, 2)),
                "foot_height_m": np.full((samples, 2), np.nan),
                "nonfoot_terrain": np.zeros(samples, dtype=bool),
                "arm_terrain_contact": np.zeros(samples, dtype=bool),
                "arm_object_contact": np.zeros(samples, dtype=bool),
                "new_arm_object_contact": np.zeros(samples, dtype=bool),
                "maximum_penetration_m": np.zeros(samples),
                "control_target": np.full((samples, nu), np.nan),
            }
        )
        current_residual = self.warm_start_residual[0, self.control_actuators].copy()
        first_horizon = min(cfg.horizon_frames * engine.config.substeps, samples - 1)
        warm_mean = np.concatenate(
            (
                self.warm_start_residual[first_horizon // 2, self.control_actuators],
                self.warm_start_residual[first_horizon, self.control_actuators],
            )
        )
        forbidden_run = 0
        severe_run = 0
        arm_run = 0
        maximum_penetration = 0.0
        first_invalid_sample: int | None = None
        invalid_reason: str | None = None
        replans: list[dict[str, Any]] = []
        candidate_rollouts = 0
        candidate_steps = 0
        applied_steps = 0
        started = perf_counter()

        def record(index: int, residual: np.ndarray) -> None:
            nonlocal maximum_penetration
            physical = self._physical_state(data)
            terrain, penetration, _, normal, slip, tilt, arm_terrain, arm_object = (
                physical
            )
            maximum_penetration = max(maximum_penetration, penetration)
            ref = self.reference
            recorded["root_position"][index] = data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ]
            recorded["actual_com"][index] = data.subtree_com[self.root_body]
            recorded["actual_com_velocity"][index] = data.subtree_linvel[self.root_body]
            recorded["actual_angular_momentum"][index] = data.subtree_angmom[
                self.root_body
            ]
            recorded["joint_position"][index] = data.qpos[engine.actuator_qpos]
            recorded["joint_velocity"][index] = data.qvel[self.actuator_dofs]
            recorded["joint_position_error"][index] = (
                ref.joint_position[index] - data.qpos[engine.actuator_qpos]
            )
            recorded["joint_velocity_error"][index] = (
                ref.joint_velocity[index] - data.qvel[self.actuator_dofs]
            )
            recorded["actual_torque_nm"][index] = data.actuator_force
            recorded["torque_usage"][index] = (
                np.abs(data.actuator_force) / engine.force_scale
            )
            recorded["torso_tilt_rad"][index] = tilt
            recorded["foot_contact"][index] = normal > 1.0e-6
            recorded["foot_normal_force_n"][index] = normal
            recorded["foot_slip_m_s"][index] = slip
            recorded["foot_height_m"][index] = data.xpos[list(engine.foot_bodies), 2]
            recorded["nonfoot_terrain"][index] = bool(
                terrain - engine.allowed_terrain_bodies
            )
            recorded["arm_terrain_contact"][index] = arm_terrain
            recorded["arm_object_contact"][index] = arm_object
            recorded["new_arm_object_contact"][index] = bool(
                arm_object and not self.baseline_arm_object_contact[index]
            )
            recorded["maximum_penetration_m"][index] = maximum_penetration
            recorded["torque_residual_nm"][index, self.control_actuators] = residual

        (
            initial_command,
            initial_feedforward,
            initial_requested,
            initial_limited,
            (initial_clipped),
        ) = self._command(data, 0, current_residual)
        recorded["feedforward_torque_nm"][0] = initial_feedforward
        recorded["requested_torque_nm"][0] = initial_requested
        recorded["limited_torque_nm"][0] = initial_limited
        recorded["torque_clipped"][0] = initial_clipped
        recorded["control_target"][0] = initial_command
        record(0, current_residual)
        current_sample = 0
        while current_sample < samples - 1:
            best, warm_mean, evaluations, simulated, progress = self._solve(
                data,
                current_sample,
                current_residual,
                warm_mean,
                forbidden_run,
                severe_run,
                arm_run,
                maximum_penetration,
            )
            candidate_rollouts += evaluations
            candidate_steps += simulated
            apply_steps = min(
                cfg.replan_interval_frames * engine.config.substeps,
                samples - 1 - current_sample,
            )
            replans.append(
                {
                    "frame": cfg.anchor_frame + current_sample / engine.config.substeps,
                    "predicted_valid": best.first_invalid_step is None,
                    "predicted_first_invalid_step": best.first_invalid_step,
                    "predicted_invalid_reason": best.invalid_reason,
                    "best_cost": best.cost,
                    "terminal_root_height_m": best.terminal_root_height_m,
                    "terminal_vertical_velocity_m_s": (
                        best.terminal_vertical_velocity_m_s
                    ),
                    "terminal_torso_tilt_deg": float(
                        np.rad2deg(best.terminal_torso_tilt_rad)
                    ),
                    "terminal_support_count": best.terminal_support_count,
                    "residual_rms_nm": float(
                        np.sqrt(np.mean(np.square(best.residual_path)))
                    ),
                    "residual_max_nm": float(np.max(np.abs(best.residual_path))),
                    "candidate_rollouts": evaluations,
                    "candidate_simulated_steps": simulated,
                    "generations": progress,
                }
            )
            for step in range(1, apply_steps + 1):
                index = current_sample + step
                residual = best.residual_path[step]
                command, feedforward, requested, limited, clipped = self._command(
                    data, index, residual
                )
                data.ctrl[:] = command
                mujoco.mj_step(self.model, data)
                mujoco.mj_forward(self.model, data)
                applied_steps += 1
                physical = self._physical_state(data)
                terrain, penetration, _, _, _, tilt, arm_terrain, arm_object = physical
                new_arm_object = bool(
                    arm_object and not self.baseline_arm_object_contact[index]
                )
                maximum_penetration = max(maximum_penetration, penetration)
                forbidden = terrain - engine.allowed_terrain_bodies
                forbidden_run = forbidden_run + 1 if forbidden else 0
                arm_run = arm_run + 1 if arm_terrain or new_arm_object else 0
                usage = np.abs(data.actuator_force) / engine.force_scale
                severe = (
                    np.mean(usage >= engine.config.severe_saturation_usage)
                    >= engine.config.severe_saturation_actuator_fraction
                )
                severe_run = severe_run + 1 if severe else 0
                reason = None
                if data.qpos[engine.robot_qpos + 2] < engine.minimum_root_height:
                    reason = "root_height"
                elif tilt > engine.maximum_torso_tilt:
                    reason = "torso_tilt"
                elif forbidden & engine.fall_terrain_bodies:
                    reason = "fall_body_terrain_contact"
                elif forbidden_run >= engine.sustained_terrain_steps:
                    reason = "sustained_nonfoot_terrain_contact"
                elif arm_run >= engine.sustained_terrain_steps:
                    reason = "sustained_arm_support"
                elif maximum_penetration > engine.config.hard_penetration_m:
                    reason = "pathological_penetration"
                elif severe_run >= engine.severe_saturation_steps:
                    reason = "sustained_actuator_saturation"
                recorded["feedforward_torque_nm"][index] = feedforward
                recorded["requested_torque_nm"][index] = requested
                recorded["limited_torque_nm"][index] = limited
                recorded["torque_clipped"][index] = clipped
                recorded["control_target"][index] = command
                record(index, residual)
                if reason is not None and first_invalid_sample is None:
                    first_invalid_sample = index
                    invalid_reason = reason
            current_sample += apply_steps
            current_residual = best.residual_path[apply_steps].copy()
        elapsed = perf_counter() - started
        first_invalid_frame = (
            None
            if first_invalid_sample is None
            else cfg.anchor_frame
            + int(np.ceil(first_invalid_sample / engine.config.substeps))
        )
        stop = samples if first_invalid_sample is None else first_invalid_sample + 1
        final_window = slice(max(0, samples - 8), samples)
        recoverable = bool(
            first_invalid_sample is None
            and np.any(recorded["foot_contact"][final_window], axis=1).all()
            and recorded["root_position"][-1, 2] >= engine.minimum_root_height + 0.02
            and abs(recorded["actual_com_velocity"][-1, 2]) < 0.5
            and recorded["torso_tilt_rad"][-1] < 0.8 * engine.maximum_torso_tilt
        )
        torque_clipped = recorded["torque_clipped"][:stop]
        summary = {
            "candidate_rollouts": candidate_rollouts,
            "candidate_simulated_steps": candidate_steps,
            "applied_simulated_steps": applied_steps,
            "total_mujoco_steps": candidate_steps + applied_steps,
            "replans": len(replans),
            "runtime_s": elapsed,
            "maximum_joint_position_error_rad": float(
                np.nanmax(np.abs(recorded["joint_position_error"][:stop]))
            ),
            "maximum_joint_velocity_error_rad_s": float(
                np.nanmax(np.abs(recorded["joint_velocity_error"][:stop]))
            ),
            "maximum_balance_torque_usage": float(
                np.nanmax(recorded["torque_usage"][:stop, self.balance_actuators])
            ),
            "maximum_controlled_torque_usage": float(
                np.nanmax(recorded["torque_usage"][:stop, self.control_actuators])
            ),
            "torque_clipping_duration_s": float(
                np.sum(np.any(torque_clipped, axis=1)) * engine.config.physics_dt
            ),
            "torque_clipping_fraction": float(np.mean(torque_clipped)),
            "arm_terrain_contact_duration_s": float(
                np.sum(recorded["arm_terrain_contact"]) * engine.config.physics_dt
            ),
            "arm_terrain_contact_before_invalidation_s": float(
                np.sum(recorded["arm_terrain_contact"][:stop])
                * engine.config.physics_dt
            ),
            "arm_object_contact_duration_s": float(
                np.sum(recorded["arm_object_contact"]) * engine.config.physics_dt
            ),
            "new_arm_object_contact_duration_s": float(
                np.sum(recorded["new_arm_object_contact"]) * engine.config.physics_dt
            ),
            "root_position_rms_error_m": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            recorded["root_position"][:stop]
                            - self.reference.root_position[:stop]
                        )
                    )
                )
            ),
            "com_position_rms_error_m": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            recorded["actual_com"][:stop]
                            - self.reference.com_position[:stop]
                        )
                    )
                )
            ),
            "com_velocity_rms_error_m_s": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            recorded["actual_com_velocity"][:stop]
                            - self.reference.com_velocity[:stop]
                        )
                    )
                )
            ),
            "maximum_penetration_m": maximum_penetration,
        }
        return TorqueShootingMPCResult(
            valid=first_invalid_sample is None,
            recoverable_terminal=recoverable,
            first_invalid_frame=first_invalid_frame,
            first_invalid_sample=first_invalid_sample,
            invalid_reason=invalid_reason,
            data=recorded,
            replans=replans,
            summary=summary,
            prefix=self.prefix,
        )


def track_with_torque_shooting_mpc(
    engine: _DynamicRollout,
    nominal_target: np.ndarray,
    artifact: dict[str, np.ndarray],
    config: TorqueShootingMPCConfig | None = None,
    warm_start_residual: np.ndarray | None = None,
) -> TorqueShootingMPCResult:
    """Execute one deterministic frame-85 anchored torque-shooting MPC run."""
    controller = _TorqueShootingMPC(
        engine,
        nominal_target,
        artifact,
        config or TorqueShootingMPCConfig(),
        warm_start_residual,
    )
    return controller.run()
