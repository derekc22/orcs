"""Unified MuJoCo-in-the-loop generation of physically realized references.

The diffusion sample supplies the object initial/goal state and a soft motion
prior.  Candidate actuator targets are evaluated only by MuJoCo.  Successful
control prefixes are committed from the actual state and the uninterrupted
realized trajectory is the output reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from orcs.tasks.uolm.contact_dynamics_refinement import _copy_data
from orcs.tasks.uolm.dynamics_refinement import _DynamicRollout
from orcs.tasks.uolm.torque_shooting_mpc import _decode_residual, _rotation_error


@dataclass(frozen=True, slots=True)
class PhysicalReferenceConfig:
    horizon_frames: int = 25
    replan_frames: int = 5
    future_knots: int = 2
    population: int = 40
    elites: int = 6
    generations: int = 3
    residual_bound_rad: float = 0.45
    initial_std_rad: float = 0.12
    minimum_std_rad: float = 0.025
    smoothing: float = 0.7
    robot_clearance_m: float = 0.60
    desired_stance_height_m: float = 0.75
    optimize_initial_stance: bool = True
    stance_horizon_frames: int = 60
    stance_population: int = 64
    stance_elites: int = 8
    stance_generations: int = 20
    stance_parameters: tuple[float, ...] | None = None
    use_lqr_executor: bool = True
    lqr_settle_frames: int = 60
    seed: int = 1291


@dataclass(slots=True)
class PhysicalReferenceResult:
    valid: bool
    first_invalid_frame: int | None
    invalid_reason: str | None
    data: dict[str, np.ndarray]
    controls: np.ndarray
    summary: dict[str, Any]
    replans: list[dict[str, Any]]


@dataclass(slots=True)
class _Shot:
    parameters: np.ndarray
    path: np.ndarray
    cost: float
    first_invalid_step: int | None
    invalid_reason: str | None
    steps: int


class PhysicalReferenceGenerator:
    """Rolling physical control optimizer with no commanded floating base."""

    def __init__(
        self,
        engine: _DynamicRollout,
        config: PhysicalReferenceConfig | None = None,
    ) -> None:
        self.engine = engine
        self.model = engine.model
        self.config = config or PhysicalReferenceConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.balance_actuators = np.flatnonzero(
            np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
        )
        self.nominal_control = self.model.key_qpos[0, engine.actuator_qpos].copy()
        self.root_body = int(self.model.jnt_bodyid[engine.robot_free_joint])
        self.stance_search: dict[str, Any] = {}
        self.initial_data, self.desired_rotation, self.initial_root = (
            self._initial_state()
        )
        self.torso_local_up = self.initial_data.xmat[engine.torso_body].reshape(
            3, 3
        ).T @ np.asarray([0.0, 0.0, 1.0])
        if self.config.stance_parameters is not None:
            parameters = np.asarray(self.config.stance_parameters, dtype=np.float64)
            if parameters.shape != (10,):
                raise ValueError("stance_parameters must contain 10 values")
            self.initial_data, self.nominal_control = self._stance_state(parameters)
            self.desired_rotation = (
                self.initial_data.xmat[self.root_body].reshape(3, 3).copy()
            )
            self.initial_root = self.initial_data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ].copy()
            self.stance_search = {"parameters": parameters.tolist(), "reused": True}
        elif self.config.optimize_initial_stance:
            self._optimize_initial_stance()
        self.torso_local_up = self.initial_data.xmat[engine.torso_body].reshape(
            3, 3
        ).T @ np.asarray([0.0, 0.0, 1.0])

    def _initial_state(self) -> tuple[mujoco.MjData, np.ndarray, np.ndarray]:
        """Construct a supported task-relative stance; object state stays exact."""
        engine = self.engine
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.key_qpos[0]
        data.qvel[:] = 0.0
        object_initial = engine.reference_object_pos[0]
        object_goal = engine.reference_object_pos[-1]
        direction = object_goal[:2] - object_initial[:2]
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        root_xy = object_initial[:2] - self.config.robot_clearance_m * direction
        yaw = float(np.arctan2(direction[1], direction[0]))
        data.qpos[engine.robot_qpos : engine.robot_qpos + 2] = root_xy
        # The stock keyframe is 5 mm above contact in this compiled scene.
        # Start with both soles already carrying support instead of injecting a
        # landing impulse into the first shooting horizon.
        data.qpos[engine.robot_qpos + 2] -= 0.005
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = (
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        )
        data.qpos[engine.object_qpos : engine.object_qpos + 3] = object_initial
        data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = (
            engine.reference_object_quat[0]
        )
        data.qvel[engine.object_dof : engine.object_dof + 6] = 0.0
        mujoco.mj_forward(self.model, data)
        return (
            data,
            data.xmat[self.root_body].reshape(3, 3).copy(),
            data.qpos[engine.robot_qpos : engine.robot_qpos + 3].copy(),
        )

    def _stance_state(self, parameters: np.ndarray) -> tuple[mujoco.MjData, np.ndarray]:
        """Decode a compact symmetric sagittal stance and its hold target."""
        engine = self.engine
        pose = parameters[:4]
        root_pitch, root_z_offset = parameters[4:6]
        target_bias = parameters[6:10]
        data, _, _ = self._initial_state()
        names = [engine.joint_names[index] for index in engine.actuator_joint_indexes]
        groups = (
            ("hip_pitch", pose[0], target_bias[0]),
            ("knee", pose[1], target_bias[1]),
            ("ankle_pitch", pose[2], target_bias[2]),
            ("waist_pitch", pose[3], target_bias[3]),
        )
        control = self.model.key_qpos[0, engine.actuator_qpos].copy()
        for token, pose_value, bias_value in groups:
            selected = [index for index, name in enumerate(names) if token in name]
            data.qpos[engine.actuator_qpos[selected]] += pose_value
            control[selected] += pose_value + bias_value
        data.qpos[engine.robot_qpos + 2] += root_z_offset
        direction = (
            engine.reference_object_pos[-1, :2] - engine.reference_object_pos[0, :2]
        )
        yaw = float(np.arctan2(direction[1], direction[0]))
        cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
        cp, sp = np.cos(0.5 * root_pitch), np.sin(0.5 * root_pitch)
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = (
            cy * cp,
            -sy * sp,
            cy * sp,
            sy * cp,
        )
        data.ctrl[:] = np.clip(
            control,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_forward(self.model, data)
        return data, data.ctrl.copy()

    def _evaluate_stance(
        self, parameters: np.ndarray, horizon_steps: int
    ) -> tuple[tuple[int, int, float], float, int, str | None]:
        data, control = self._stance_state(parameters)
        engine = self.engine
        desired_rotation = self.desired_rotation
        desired_xy = data.qpos[engine.robot_qpos : engine.robot_qpos + 2].copy()
        desired_z = self.config.desired_stance_height_m
        cost = 0.0
        invalid_step = horizon_steps + 1
        reason = None
        for step in range(1, horizon_steps + 1):
            data.ctrl[:] = control
            mujoco.mj_step(self.model, data)
            physical = self._physical(data)
            reason = self._invalid_reason(data, physical)
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            rotation = data.xmat[self.root_body].reshape(3, 3)
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            support = int(np.sum(physical["foot_contact"]))
            cost += (
                5000.0 * (root[2] - desired_z) ** 2
                + 150.0 * float(np.square(root[:2] - desired_xy).sum())
                + 500.0 * _rotation_error(rotation, desired_rotation) ** 2
                + 45.0 * float(np.square(velocity).sum())
                + 500.0 * float(support == 0)
                + 10.0 * float(support == 1)
                + 5.0 * float(np.square(physical["foot_slip"]).sum())
                + 2000.0 * float(bool(physical["forbidden"]))
                + 20.0 * float(np.square(parameters).mean())
            )
            if reason is not None:
                invalid_step = step
                cost += 1.0e9 + (horizon_steps - step) * 1.0e7
                break
        rank = (0, 0, cost) if reason is None else (1, -invalid_step, cost)
        return rank, cost, invalid_step, reason

    def _optimize_initial_stance(self) -> None:
        """Find a contact-settled stance before rolling trajectory generation."""
        cfg = self.config
        lower = np.asarray([-0.8, -0.8, -0.8, -0.5, -0.3, -0.3] + [-0.6] * 4)
        upper = np.asarray([0.8, 0.8, 0.8, 0.5, 0.3, 0.04] + [0.6] * 4)
        mean = np.zeros(10, dtype=np.float64)
        mean[5] = -0.003
        std = 0.22 * (upper - lower)
        horizon = cfg.stance_horizon_frames * self.engine.config.substeps
        best_parameters = mean.copy()
        best = self._evaluate_stance(best_parameters, horizon)
        evaluations = 1
        progress = []
        for generation in range(cfg.stance_generations):
            population = np.clip(
                self.rng.normal(mean, std, size=(cfg.stance_population, 10)),
                lower,
                upper,
            )
            population[0] = best_parameters
            evaluated = [self._evaluate_stance(item, horizon) for item in population]
            evaluations += len(evaluated)
            order = sorted(range(len(evaluated)), key=lambda index: evaluated[index][0])
            if evaluated[order[0]][0] < best[0]:
                best_parameters = population[order[0]].copy()
                best = evaluated[order[0]]
            elite = population[order[: cfg.stance_elites]]
            mean = 0.7 * elite.mean(axis=0) + 0.3 * mean
            std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, 0.01)
            progress.append(
                {
                    "generation": generation,
                    "best_invalid_step": best[2],
                    "best_reason": best[3],
                    "best_cost": best[1],
                }
            )
        self.initial_data, self.nominal_control = self._stance_state(best_parameters)
        self.desired_rotation = (
            self.initial_data.xmat[self.root_body].reshape(3, 3).copy()
        )
        self.initial_root = self.initial_data.qpos[
            self.engine.robot_qpos : self.engine.robot_qpos + 3
        ].copy()
        self.stance_search = {
            "parameters": best_parameters.tolist(),
            "rank": list(best[0]),
            "invalid_step": best[2],
            "invalid_reason": best[3],
            "cost": best[1],
            "evaluations": evaluations,
            "progress": progress,
        }

    def _physical(self, data: mujoco.MjData) -> dict[str, Any]:
        engine = self.engine
        terrain_bodies: set[int] = set()
        foot_contact = np.zeros(2, dtype=bool)
        foot_force = np.zeros(2, dtype=np.float64)
        foot_slip = np.zeros(2, dtype=np.float64)
        maximum_penetration = 0.0
        object_contact = False
        hand_contact = np.zeros(2, dtype=bool)
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            pair = {first, second}
            maximum_penetration = max(maximum_penetration, -float(contact.dist))
            if pair & engine.terrain_geoms:
                other = second if first in engine.terrain_geoms else first
                body = int(self.model.geom_bodyid[other])
                if body in engine.robot_bodies:
                    terrain_bodies.add(body)
                    if body in engine.foot_bodies:
                        side = engine.foot_bodies.index(body)
                        foot_contact[side] = True
                        force = np.zeros(6, dtype=np.float64)
                        mujoco.mj_contactForce(self.model, data, index, force)
                        foot_force[side] += max(0.0, force[0])
                        velocity = np.zeros(6, dtype=np.float64)
                        mujoco.mj_objectVelocity(
                            self.model,
                            data,
                            mujoco.mjtObj.mjOBJ_BODY,
                            body,
                            velocity,
                            0,
                        )
                        foot_slip[side] = max(
                            foot_slip[side], float(np.linalg.norm(velocity[3:5]))
                        )
            if pair & engine.object_geoms:
                other = second if first in engine.object_geoms else first
                body = int(self.model.geom_bodyid[other])
                if body in engine.robot_bodies:
                    object_contact = True
                if other in engine.hand_geoms:
                    hand_contact[int(np.flatnonzero(engine.hand_geoms == other)[0])] = (
                        True
                    )
        torso = data.xmat[engine.torso_body].reshape(3, 3)
        torso_up = torso @ self.torso_local_up
        tilt = float(np.arccos(np.clip(torso_up[2], -1.0, 1.0)))
        return {
            "terrain_bodies": terrain_bodies,
            "forbidden": terrain_bodies - engine.allowed_terrain_bodies,
            "foot_contact": foot_contact,
            "foot_force": foot_force,
            "foot_slip": foot_slip,
            "penetration": maximum_penetration,
            "object_contact": object_contact,
            "hand_contact": hand_contact,
            "tilt": tilt,
        }

    def _invalid_reason(
        self, data: mujoco.MjData, physical: dict[str, Any]
    ) -> str | None:
        engine = self.engine
        if data.qpos[engine.robot_qpos + 2] < engine.minimum_root_height:
            return "root_height"
        if physical["tilt"] > engine.maximum_torso_tilt:
            return "torso_tilt"
        if physical["forbidden"] & engine.fall_terrain_bodies:
            return "fall_body_terrain_contact"
        if physical["penetration"] > engine.config.hard_penetration_m:
            return "pathological_penetration"
        return None

    def _state_cost(
        self,
        data: mujoco.MjData,
        physical: dict[str, Any],
        residual: np.ndarray,
        previous: np.ndarray,
        *,
        terminal: bool,
    ) -> float:
        engine = self.engine
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        rotation = data.xmat[self.root_body].reshape(3, 3)
        root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
        angular_velocity = data.cvel[self.root_body, :3]
        orientation = _rotation_error(rotation, self.desired_rotation)
        support_count = int(np.sum(physical["foot_contact"]))
        value = (
            800.0 * (root[2] - self.initial_root[2]) ** 2
            + 120.0 * float(np.square(root[:2] - self.initial_root[:2]).sum())
            + 450.0 * orientation**2
            + 100.0 * physical["tilt"] ** 2
            + 45.0 * float(np.square(root_velocity).sum())
            + 18.0 * float(np.square(angular_velocity).sum())
            + 600.0 * float(support_count == 0)
            + 12.0 * float(support_count == 1)
            + 4.0 * float(np.square(physical["foot_slip"]).sum())
            + 2.0 * float(np.square(residual / self.config.residual_bound_rad).mean())
            + 4.0
            * float(
                np.square((residual - previous) / self.config.residual_bound_rad).mean()
            )
        )
        if terminal:
            value += 10.0 * (
                1000.0 * max(0.0, self.initial_root[2] - root[2]) ** 2
                + 500.0 * orientation**2
                + 100.0 * float(np.square(root_velocity).sum())
                + 40.0 * float(np.square(angular_velocity).sum())
                + 1000.0 * float(support_count == 0)
            )
        return value

    def _evaluate(
        self,
        source: mujoco.MjData,
        current_residual: np.ndarray,
        parameters: np.ndarray,
        horizon_steps: int,
    ) -> _Shot:
        cfg = self.config
        path = _decode_residual(
            parameters, current_residual, horizon_steps, cfg.future_knots
        )
        data = _copy_data(self.model, source)
        cost = 0.0
        reason = None
        invalid_step = None
        for step in range(1, horizon_steps + 1):
            target = self.nominal_control.copy()
            target[self.balance_actuators] += path[step]
            data.ctrl[:] = np.clip(
                target,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(self.model, data)
            physical = self._physical(data)
            reason = self._invalid_reason(data, physical)
            cost += self._state_cost(
                data,
                physical,
                path[step],
                path[step - 1],
                terminal=step == horizon_steps,
            )
            if reason is not None:
                invalid_step = step
                cost += 1.0e9 + (horizon_steps - step) * 1.0e7
                break
        return _Shot(parameters, path, cost, invalid_step, reason, step)

    @staticmethod
    def _rank(shot: _Shot) -> tuple[int, int, float]:
        return (
            (0, 0, shot.cost)
            if shot.first_invalid_step is None
            else (1, -shot.first_invalid_step, shot.cost)
        )

    def _solve(
        self,
        data: mujoco.MjData,
        current_residual: np.ndarray,
        warm: np.ndarray,
        horizon_steps: int,
    ) -> tuple[_Shot, np.ndarray, int, int]:
        cfg, engine = self.config, self.engine
        dimensions = cfg.future_knots * len(self.balance_actuators)
        bounds = np.full(dimensions, cfg.residual_bound_rad)
        mean = np.clip(warm, -bounds, bounds)
        std = np.full(dimensions, cfg.initial_std_rad)
        best = self._evaluate(data, current_residual, mean, horizon_steps)
        evaluations, simulated = 1, best.steps
        for _ in range(cfg.generations):
            population = np.clip(
                self.rng.normal(mean, std, size=(cfg.population, dimensions)),
                -bounds,
                bounds,
            )
            population[0] = 0.0
            population[1] = best.parameters
            shots = [
                self._evaluate(data, current_residual, candidate, horizon_steps)
                for candidate in population
            ]
            evaluations += len(shots)
            simulated += sum(shot.steps for shot in shots)
            order = sorted(
                range(len(shots)), key=lambda index: self._rank(shots[index])
            )
            generation_best = shots[order[0]]
            if self._rank(generation_best) < self._rank(best):
                best = generation_best
            elite = population[order[: cfg.elites]]
            mean = cfg.smoothing * elite.mean(axis=0) + (1.0 - cfg.smoothing) * mean
            std = np.maximum(
                cfg.smoothing * elite.std(axis=0) + (1.0 - cfg.smoothing) * std,
                cfg.minimum_std_rad,
            )
        apply_steps = cfg.replan_frames * engine.config.substeps
        midpoint = min(apply_steps + horizon_steps // 2, horizon_steps)
        shifted = np.concatenate((best.path[midpoint], best.path[-1]))
        return best, shifted, evaluations, simulated

    def _lqr_state(self) -> tuple[mujoco.MjData, np.ndarray, np.ndarray]:
        """Settle physically, then linearize MuJoCo's contacted dynamics."""
        engine = self.engine
        reference = _copy_data(self.model, self.initial_data)
        for _ in range(self.config.lqr_settle_frames * engine.config.substeps):
            reference.ctrl[:] = self.nominal_control
            mujoco.mj_step(self.model, reference)
        reference.ctrl[:] = self.nominal_control
        mujoco.mj_forward(self.model, reference)
        state_size = 2 * self.model.nv + self.model.na
        transition = np.empty((state_size, state_size), dtype=np.float64)
        control = np.empty((state_size, self.model.nu), dtype=np.float64)
        mujoco.mjd_transitionFD(
            self.model,
            reference,
            1.0e-6,
            1,
            transition,
            control,
            None,
            None,
        )
        robot_nv = 6 + len(engine.joint_names)
        state_indexes = np.r_[np.arange(robot_nv), self.model.nv + np.arange(robot_nv)]
        system = transition[np.ix_(state_indexes, state_indexes)]
        input_matrix = control[np.ix_(state_indexes, self.balance_actuators)]
        diagonal = np.ones(2 * robot_nv, dtype=np.float64)
        diagonal[:3] = 200.0
        diagonal[3:6] = 500.0
        diagonal[6:robot_nv] = 3.0
        diagonal[robot_nv : robot_nv + 3] = 50.0
        diagonal[robot_nv + 3 : robot_nv + 6] = 100.0
        diagonal[robot_nv + 6 :] = 0.3
        state_cost = np.diag(diagonal)
        control_cost = np.eye(len(self.balance_actuators)) * 0.2
        riccati = solve_discrete_are(system, input_matrix, state_cost, control_cost)
        gain = np.linalg.solve(
            control_cost + input_matrix.T @ riccati @ input_matrix,
            input_matrix.T @ riccati @ system,
        )
        return reference, gain, state_indexes

    def _lqr_control(
        self,
        data: mujoco.MjData,
        reference: mujoco.MjData,
        gain: np.ndarray,
        *,
        arm_target: np.ndarray | None = None,
    ) -> np.ndarray:
        engine = self.engine
        robot_nv = 6 + len(engine.joint_names)
        position_error = np.empty(self.model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self.model, position_error, 1.0, reference.qpos, data.qpos
        )
        state_error = np.r_[
            position_error[:robot_nv],
            data.qvel[:robot_nv] - reference.qvel[:robot_nv],
        ]
        command = self.nominal_control.copy()
        command[self.balance_actuators] -= gain @ state_error
        if arm_target is not None:
            arm_actuators = np.setdiff1d(
                np.arange(self.model.nu), self.balance_actuators
            )
            command[arm_actuators] = arm_target
        return np.clip(
            command,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def generate_lqr_balance(self) -> PhysicalReferenceResult:
        """Generate the uninterrupted realized reference with contact LQR."""
        engine = self.engine
        reference, gain, _ = self._lqr_state()
        data = _copy_data(self.model, self.initial_data)
        total_steps = (engine.horizon - 1) * engine.config.substeps
        samples = total_steps + 1
        nu = self.model.nu
        recorded = {
            "root_position_m": np.empty((samples, 3)),
            "root_quaternion_wxyz": np.empty((samples, 4)),
            "root_velocity_m_s": np.empty((samples, 6)),
            "joint_position_rad": np.empty((samples, nu)),
            "joint_velocity_rad_s": np.empty((samples, nu)),
            "object_position_m": np.empty((samples, 3)),
            "object_quaternion_wxyz": np.empty((samples, 4)),
            "object_velocity_m_s": np.empty((samples, 6)),
            "foot_contact": np.zeros((samples, 2), dtype=bool),
            "foot_force_n": np.zeros((samples, 2)),
            "foot_slip_m_s": np.zeros((samples, 2)),
            "torso_tilt_rad": np.empty(samples),
            "actuator_force_nm": np.empty((samples, nu)),
            "control_target_rad": np.empty((samples, nu)),
            "object_contact": np.zeros(samples, dtype=bool),
            "hand_contact": np.zeros((samples, 2), dtype=bool),
            "forbidden_terrain_contact": np.zeros(samples, dtype=bool),
        }
        controls = np.empty((samples, nu), dtype=np.float64)
        dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]].astype(np.int32)

        def record(index: int) -> None:
            physical = self._physical(data)
            recorded["root_position_m"][index] = data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ]
            recorded["root_quaternion_wxyz"][index] = data.qpos[
                engine.robot_qpos + 3 : engine.robot_qpos + 7
            ]
            recorded["root_velocity_m_s"][index] = data.qvel[
                engine.robot_dof : engine.robot_dof + 6
            ]
            recorded["joint_position_rad"][index] = data.qpos[engine.actuator_qpos]
            recorded["joint_velocity_rad_s"][index] = data.qvel[dofs]
            recorded["object_position_m"][index] = data.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ]
            recorded["object_quaternion_wxyz"][index] = data.qpos[
                engine.object_qpos + 3 : engine.object_qpos + 7
            ]
            recorded["object_velocity_m_s"][index] = data.qvel[
                engine.object_dof : engine.object_dof + 6
            ]
            recorded["foot_contact"][index] = physical["foot_contact"]
            recorded["foot_force_n"][index] = physical["foot_force"]
            recorded["foot_slip_m_s"][index] = physical["foot_slip"]
            recorded["torso_tilt_rad"][index] = physical["tilt"]
            recorded["actuator_force_nm"][index] = data.actuator_force
            recorded["control_target_rad"][index] = data.ctrl
            recorded["object_contact"][index] = physical["object_contact"]
            recorded["hand_contact"][index] = physical["hand_contact"]
            recorded["forbidden_terrain_contact"][index] = bool(physical["forbidden"])

        data.ctrl[:] = self.nominal_control
        controls[0] = data.ctrl
        record(0)
        first_invalid = None
        invalid_reason = None
        settle_steps = self.config.lqr_settle_frames * engine.config.substeps
        for sample in range(1, samples):
            command = (
                self.nominal_control
                if sample <= settle_steps
                else self._lqr_control(data, reference, gain)
            )
            data.ctrl[:] = command
            mujoco.mj_step(self.model, data)
            controls[sample] = command
            record(sample)
            reason = self._invalid_reason(data, self._physical(data))
            if reason is not None and first_invalid is None:
                first_invalid = sample
                invalid_reason = reason
        first_invalid_frame = (
            None
            if first_invalid is None
            else int(np.ceil(first_invalid / engine.config.substeps))
        )
        goal_position_error = float(
            np.linalg.norm(
                recorded["object_position_m"][-1] - engine.reference_object_pos[-1]
            )
        )
        dot = abs(
            float(
                np.dot(
                    recorded["object_quaternion_wxyz"][-1],
                    engine.reference_object_quat[-1],
                )
            )
        )
        goal_orientation_error = float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
        summary = {
            "stance_search": self.stance_search,
            "lqr_gain_shape": list(gain.shape),
            "lqr_spectral_radius": None,
            "minimum_root_height_m": float(recorded["root_position_m"][:, 2].min()),
            "maximum_torso_tilt_deg": float(
                np.rad2deg(recorded["torso_tilt_rad"].max())
            ),
            "both_feet_off_duration_s": float(
                np.sum(~np.any(recorded["foot_contact"], axis=1))
                * engine.config.physics_dt
            ),
            "forbidden_support_duration_s": float(
                np.sum(recorded["forbidden_terrain_contact"]) * engine.config.physics_dt
            ),
            "maximum_actuator_usage": float(
                np.max(np.abs(recorded["actuator_force_nm"]) / engine.force_scale)
            ),
            "object_displacement_m": float(
                np.linalg.norm(
                    recorded["object_position_m"][-1] - recorded["object_position_m"][0]
                )
            ),
            "final_goal_position_error_m": goal_position_error,
            "final_goal_orientation_error_rad": goal_orientation_error,
            "at_goal": goal_position_error < 0.15 and goal_orientation_error < 0.35,
        }
        return PhysicalReferenceResult(
            valid=first_invalid is None,
            first_invalid_frame=first_invalid_frame,
            invalid_reason=invalid_reason,
            data=recorded,
            controls=controls,
            summary=summary,
            replans=[],
        )

    def generate_balance(self) -> PhysicalReferenceResult:
        """Generate and validate one uninterrupted 300-frame upright rollout."""
        if self.config.use_lqr_executor:
            return self.generate_lqr_balance()
        cfg, engine = self.config, self.engine
        data = _copy_data(self.model, self.initial_data)
        total_steps = (engine.horizon - 1) * engine.config.substeps
        horizon_nominal = cfg.horizon_frames * engine.config.substeps
        apply_nominal = cfg.replan_frames * engine.config.substeps
        samples = total_steps + 1
        nu = self.model.nu
        recorded = {
            "root_position_m": np.empty((samples, 3)),
            "root_quaternion_wxyz": np.empty((samples, 4)),
            "root_velocity_m_s": np.empty((samples, 6)),
            "joint_position_rad": np.empty((samples, nu)),
            "joint_velocity_rad_s": np.empty((samples, nu)),
            "object_position_m": np.empty((samples, 3)),
            "object_quaternion_wxyz": np.empty((samples, 4)),
            "object_velocity_m_s": np.empty((samples, 6)),
            "foot_contact": np.zeros((samples, 2), dtype=bool),
            "foot_force_n": np.zeros((samples, 2)),
            "foot_slip_m_s": np.zeros((samples, 2)),
            "torso_tilt_rad": np.empty(samples),
            "actuator_force_nm": np.empty((samples, nu)),
            "control_target_rad": np.empty((samples, nu)),
            "object_contact": np.zeros(samples, dtype=bool),
            "hand_contact": np.zeros((samples, 2), dtype=bool),
            "forbidden_terrain_contact": np.zeros(samples, dtype=bool),
        }
        controls = np.repeat(self.nominal_control[None], samples, axis=0)

        def record(index: int) -> None:
            physical = self._physical(data)
            recorded["root_position_m"][index] = data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ]
            recorded["root_quaternion_wxyz"][index] = data.qpos[
                engine.robot_qpos + 3 : engine.robot_qpos + 7
            ]
            recorded["root_velocity_m_s"][index] = data.qvel[
                engine.robot_dof : engine.robot_dof + 6
            ]
            recorded["joint_position_rad"][index] = data.qpos[engine.actuator_qpos]
            dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]].astype(
                np.int32
            )
            recorded["joint_velocity_rad_s"][index] = data.qvel[dofs]
            recorded["object_position_m"][index] = data.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ]
            recorded["object_quaternion_wxyz"][index] = data.qpos[
                engine.object_qpos + 3 : engine.object_qpos + 7
            ]
            recorded["object_velocity_m_s"][index] = data.qvel[
                engine.object_dof : engine.object_dof + 6
            ]
            recorded["foot_contact"][index] = physical["foot_contact"]
            recorded["foot_force_n"][index] = physical["foot_force"]
            recorded["foot_slip_m_s"][index] = physical["foot_slip"]
            recorded["torso_tilt_rad"][index] = physical["tilt"]
            recorded["actuator_force_nm"][index] = data.actuator_force
            recorded["control_target_rad"][index] = data.ctrl
            recorded["object_contact"][index] = physical["object_contact"]
            recorded["hand_contact"][index] = physical["hand_contact"]
            recorded["forbidden_terrain_contact"][index] = bool(physical["forbidden"])

        record(0)
        current = np.zeros(len(self.balance_actuators))
        warm = np.zeros(cfg.future_knots * len(self.balance_actuators))
        replans: list[dict[str, Any]] = []
        evaluations = simulated = 0
        first_invalid = None
        invalid_reason = None
        index = 0
        started = perf_counter()
        while index < total_steps:
            horizon = min(horizon_nominal, total_steps - index)
            best, warm, count, steps = self._solve(data, current, warm, horizon)
            evaluations += count
            simulated += steps
            apply_steps = min(apply_nominal, total_steps - index)
            replans.append(
                {
                    "frame": index / engine.config.substeps,
                    "predicted_invalid_step": best.first_invalid_step,
                    "predicted_reason": best.invalid_reason,
                    "cost": best.cost,
                }
            )
            for local_step in range(1, apply_steps + 1):
                sample = index + local_step
                residual = best.path[local_step]
                target = self.nominal_control.copy()
                target[self.balance_actuators] += residual
                data.ctrl[:] = np.clip(
                    target,
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
                mujoco.mj_step(self.model, data)
                controls[sample] = data.ctrl
                record(sample)
                reason = self._invalid_reason(data, self._physical(data))
                if reason is not None and first_invalid is None:
                    first_invalid = sample
                    invalid_reason = reason
            index += apply_steps
            current = best.path[apply_steps].copy()
        elapsed = perf_counter() - started
        first_invalid_frame = (
            None
            if first_invalid is None
            else int(np.ceil(first_invalid / engine.config.substeps))
        )
        summary = {
            "stance_search": self.stance_search,
            "candidate_rollouts": evaluations,
            "candidate_steps": simulated,
            "runtime_s": elapsed,
            "minimum_root_height_m": float(recorded["root_position_m"][:, 2].min()),
            "maximum_torso_tilt_deg": float(
                np.rad2deg(recorded["torso_tilt_rad"].max())
            ),
            "both_feet_off_duration_s": float(
                np.sum(~np.any(recorded["foot_contact"], axis=1))
                * engine.config.physics_dt
            ),
            "forbidden_support_duration_s": float(
                np.sum(recorded["forbidden_terrain_contact"]) * engine.config.physics_dt
            ),
            "maximum_actuator_usage": float(
                np.max(np.abs(recorded["actuator_force_nm"]) / engine.force_scale)
            ),
            "object_displacement_m": float(
                np.linalg.norm(
                    recorded["object_position_m"][-1] - recorded["object_position_m"][0]
                )
            ),
        }
        return PhysicalReferenceResult(
            valid=first_invalid is None,
            first_invalid_frame=first_invalid_frame,
            invalid_reason=invalid_reason,
            data=recorded,
            controls=controls,
            summary=summary,
            replans=replans,
        )


def generate_dynamically_feasible_reference(
    engine: _DynamicRollout,
    config: PhysicalReferenceConfig | None = None,
) -> PhysicalReferenceResult:
    """Unified experimental API for physical reference generation."""
    return PhysicalReferenceGenerator(engine, config).generate_balance()
