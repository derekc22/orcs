"""Short-horizon feedback balance refinement around an open-loop feedforward."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import mujoco
import numpy as np

from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    DynamicsRefinementResult,
    RolloutResult,
    _DynamicRollout,
    _np,
)
from orcs.tasks.uolm.generated_reference import GeneratedReference

__all__ = [
    "BalanceMPCConfig",
    "BalanceMPCResult",
    "refine_balance_with_mpc",
]


@dataclass(frozen=True, slots=True)
class BalanceMPCWeights:
    root_height: float = 180.0
    root_orientation: float = 50.0
    torso_orientation: float = 80.0
    root_angular_velocity: float = 18.0
    root_horizontal_position: float = 8.0
    root_linear_velocity: float = 8.0
    joint_position: float = 5.0
    joint_velocity: float = 0.3
    missing_support: float = 25.0
    foot_slip: float = 2.0
    nonfoot_contact: float = 80.0
    residual: float = 3.0
    residual_rate: float = 1.0
    actuator_saturation: float = 15.0
    terminal_multiplier: float = 6.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BalanceMPCConfig:
    horizon_frames: int = 12
    replan_interval_frames: int = 2
    num_future_knots: int = 2
    residual_bound_rad: float = 0.15
    residual_rate_limit_rad_per_frame: float = 0.015
    population: int = 12
    elite_count: int = 3
    generations: int = 2
    initial_std_rad: float = 0.04
    minimum_std_rad: float = 0.01
    reference_support_height_m: float = 0.075
    invalid_cost: float = 1.0e9
    seed: int = 142
    weights: BalanceMPCWeights = field(default_factory=BalanceMPCWeights)


@dataclass(slots=True)
class BalanceMPCResult:
    rollout: RolloutResult
    joint_target: np.ndarray
    feedback_residual: np.ndarray
    feedforward_target: np.ndarray
    replans: list[dict[str, Any]]
    evaluations: int
    optimization_seconds: float
    actual_first_invalid_frame: int | None
    actual_invalid_reason: str | None


@dataclass(slots=True)
class _LocalEvaluation:
    cost: float
    valid: bool
    first_invalid_offset: int | None
    invalid_reason: str | None
    residual_path: np.ndarray


def _rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    value = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(value, quat_wxyz)
    return value.reshape(3, 3)


def _rotation_error(actual: np.ndarray, reference: np.ndarray) -> float:
    relative = reference.T @ actual
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _contact_state(
    engine: _DynamicRollout, data: mujoco.MjData
) -> tuple[set[int], float]:
    terrain_bodies: set[int] = set()
    max_penetration = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        if pair & engine.terrain_geoms:
            other = second if first in engine.terrain_geoms else first
            body = int(engine.model.geom_bodyid[other])
            if body in engine.robot_bodies:
                terrain_bodies.add(body)
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
    return terrain_bodies, max_penetration


def _decode_residual(
    parameters: np.ndarray,
    current: np.ndarray,
    horizon: int,
    config: BalanceMPCConfig,
) -> np.ndarray:
    joints = len(current)
    desired = parameters.reshape(config.num_future_knots, joints)
    knot_frames = np.linspace(0, horizon, config.num_future_knots + 1)
    knots = np.empty((config.num_future_knots + 1, joints), dtype=np.float64)
    knots[0] = current
    for index in range(1, len(knots)):
        span = knot_frames[index] - knot_frames[index - 1]
        rate_bound = config.residual_rate_limit_rad_per_frame * span
        knots[index] = np.clip(
            desired[index - 1], knots[index - 1] - rate_bound,
            knots[index - 1] + rate_bound,
        )
        knots[index] = np.clip(
            knots[index], -config.residual_bound_rad, config.residual_bound_rad
        )
    path = np.empty((horizon + 1, joints), dtype=np.float64)
    for frame in range(horizon + 1):
        segment = min(
            int(np.searchsorted(knot_frames, frame, side="right") - 1),
            config.num_future_knots - 1,
        )
        lower, upper = knot_frames[segment : segment + 2]
        fraction = (frame - lower) / max(upper - lower, 1.0)
        path[frame] = (1.0 - fraction) * knots[segment] + fraction * knots[segment + 1]
    path[0] = current
    return path


class _BalanceMPC:
    def __init__(
        self,
        engine: _DynamicRollout,
        feedforward_target: np.ndarray,
        config: BalanceMPCConfig,
    ) -> None:
        self.engine = engine
        self.feedforward = np.asarray(feedforward_target, dtype=np.float64)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        reference = engine.reference
        names = reference.tracked_body_names
        self.torso_reference_index = names.index("torso_link")
        self.foot_reference_indexes = (
            names.index("left_ankle_roll_link"),
            names.index("right_ankle_roll_link"),
        )
        self.reference_torso_quat = _np(
            reference.body_quat_wxyz[0, :, self.torso_reference_index]
        )
        self.reference_foot_height = _np(
            reference.body_pos_local[0, :, self.foot_reference_indexes, 2]
        )
        self.reference_joint_velocity = _np(reference.joint_vel[0])
        self.reference_root_quat = _np(reference.root_quat_wxyz[0])
        self.reference_root_linear_velocity = _np(reference.root_lin_vel_w[0])
        self.reference_root_angular_velocity = _np(reference.root_ang_vel_w[0])
        initial = engine._new_initialized_data()
        initial_torso = initial.xmat[engine.torso_body].reshape(3, 3)
        self.torso_local_up = initial_torso.T @ np.asarray([0.0, 0.0, 1.0])

    def _state_cost(
        self,
        data: mujoco.MjData,
        reference_frame: int,
        residual: np.ndarray,
        previous_residual: np.ndarray,
        terrain_bodies: set[int],
        saturation_sq: float,
        *,
        terminal: bool,
    ) -> float:
        engine, weights = self.engine, self.config.weights
        root_position = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        root_rotation = _rotation(
            data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        )
        torso_rotation = data.xmat[engine.torso_body].reshape(3, 3)
        reference_root_rotation = _rotation(
            self.reference_root_quat[reference_frame]
        )
        reference_torso_rotation = _rotation(
            self.reference_torso_quat[reference_frame]
        )
        root_linear_velocity = data.qvel[
            engine.robot_dof : engine.robot_dof + 3
        ]
        root_angular_velocity = root_rotation @ data.qvel[
            engine.robot_dof + 3 : engine.robot_dof + 6
        ]
        joint_position_error = (
            data.qpos[engine.joint_qpos[engine.balance_indexes]]
            - engine.reference_joint[reference_frame, engine.balance_indexes]
        )
        joint_velocity_error = (
            data.qvel[engine.joint_dof[engine.balance_indexes]]
            - self.reference_joint_velocity[
                reference_frame, engine.balance_indexes
            ]
        )
        expected_support = (
            self.reference_foot_height[reference_frame]
            <= self.config.reference_support_height_m
        )
        actual_support = np.asarray([
            body in terrain_bodies for body in engine.foot_bodies
        ])
        missing_support = float(expected_support.any() and not actual_support.any())
        foot_slip_sq = 0.0
        for supported, body in zip(actual_support, engine.foot_bodies, strict=True):
            if supported:
                foot_slip_sq += float(np.square(data.cvel[body, 3:5]).sum())
        nonfoot = float(bool(terrain_bodies - engine.allowed_terrain_bodies))
        cost = (
            weights.root_height
            * float((root_position[2] - engine.reference_root[reference_frame, 2]) ** 2)
            + weights.root_orientation
            * _rotation_error(root_rotation, reference_root_rotation) ** 2
            + weights.torso_orientation
            * _rotation_error(torso_rotation, reference_torso_rotation) ** 2
            + weights.root_angular_velocity
            * float(np.square(
                root_angular_velocity
                - self.reference_root_angular_velocity[reference_frame]
            ).sum())
            + weights.root_horizontal_position
            * float(np.square(
                root_position[:2] - engine.reference_root[reference_frame, :2]
            ).sum())
            + weights.root_linear_velocity
            * float(np.square(
                root_linear_velocity
                - self.reference_root_linear_velocity[reference_frame]
            ).sum())
            + weights.joint_position * float(np.square(joint_position_error).mean())
            + weights.joint_velocity * float(np.square(joint_velocity_error).mean())
            + weights.missing_support * missing_support
            + weights.foot_slip * foot_slip_sq
            + weights.nonfoot_contact * nonfoot
            + weights.residual * float(np.square(residual).mean())
            + weights.residual_rate
            * float(np.square(residual - previous_residual).mean())
            + weights.actuator_saturation * saturation_sq
        )
        if terminal:
            cost *= weights.terminal_multiplier
        return cost

    def _evaluate(
        self,
        source: mujoco.MjData,
        current_frame: int,
        current_residual: np.ndarray,
        parameters: np.ndarray,
        forbidden_run: int,
        severe_run: int,
        prior_max_penetration: float,
    ) -> _LocalEvaluation:
        engine, cfg = self.engine, self.config
        horizon = min(cfg.horizon_frames, engine.horizon - 1 - current_frame)
        residual_path = _decode_residual(
            parameters, current_residual, horizon, cfg
        )
        data = mujoco.MjData(engine.model)
        mujoco.mj_copyData(data, engine.model, source)
        cost = 0.0
        maximum_penetration = prior_max_penetration
        first_invalid: int | None = None
        invalid_reason: str | None = None
        for offset in range(1, horizon + 1):
            frame = current_frame + offset
            target = self.feedforward[frame].copy()
            target[engine.balance_indexes] += residual_path[offset]
            below = np.maximum(engine.joint_limits[:, 0] - target, 0.0)
            above = np.maximum(target - engine.joint_limits[:, 1], 0.0)
            if float(np.max(below + above)) > engine.config.joint_limit_margin_rad:
                first_invalid = offset
                invalid_reason = "catastrophic_joint_target_limit"
                break
            data.ctrl[:] = np.clip(
                target[engine.actuator_joint_indexes],
                engine.model.actuator_ctrlrange[:, 0],
                engine.model.actuator_ctrlrange[:, 1],
            )
            terrain_bodies: set[int] = set()
            saturation_sq = 0.0
            for _ in range(engine.config.substeps):
                mujoco.mj_step(engine.model, data)
                terrain_bodies, penetration = _contact_state(engine, data)
                maximum_penetration = max(maximum_penetration, penetration)
                forbidden = terrain_bodies - engine.allowed_terrain_bodies
                forbidden_run = forbidden_run + 1 if forbidden else 0
                usage = np.abs(data.actuator_force) / engine.force_scale
                saturation_sq += float(
                    np.square(np.maximum(usage - 0.9, 0.0)).mean()
                )
                severe = (
                    np.mean(usage >= engine.config.severe_saturation_usage)
                    >= engine.config.severe_saturation_actuator_fraction
                )
                severe_run = severe_run + 1 if severe else 0
                torso_up = (
                    data.xmat[engine.torso_body].reshape(3, 3)
                    @ self.torso_local_up
                )
                torso_tilt = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
                if data.qpos[engine.robot_qpos + 2] < engine.minimum_root_height:
                    invalid_reason = "root_height"
                elif torso_tilt > engine.maximum_torso_tilt:
                    invalid_reason = "torso_tilt"
                elif forbidden & engine.fall_terrain_bodies:
                    invalid_reason = "fall_body_terrain_contact"
                elif forbidden_run >= engine.sustained_terrain_steps:
                    invalid_reason = "sustained_nonfoot_terrain_contact"
                elif maximum_penetration > engine.config.hard_penetration_m:
                    invalid_reason = "pathological_penetration"
                elif severe_run >= engine.severe_saturation_steps:
                    invalid_reason = "sustained_actuator_saturation"
                if invalid_reason is not None:
                    first_invalid = offset
                    break
            cost += self._state_cost(
                data,
                frame,
                residual_path[offset],
                residual_path[offset - 1],
                terrain_bodies,
                saturation_sq / engine.config.substeps,
                terminal=offset == horizon,
            )
            if first_invalid is not None:
                break
        if first_invalid is not None:
            cost += cfg.invalid_cost + (horizon - first_invalid) * 1.0e6
        return _LocalEvaluation(
            cost=cost,
            valid=first_invalid is None,
            first_invalid_offset=first_invalid,
            invalid_reason=invalid_reason,
            residual_path=residual_path,
        )

    @staticmethod
    def _rank(result: _LocalEvaluation, horizon: int) -> tuple[int, int, float]:
        if result.valid:
            return 0, 0, result.cost
        return 1, horizon - (result.first_invalid_offset or 0), result.cost

    def _solve(
        self,
        data: mujoco.MjData,
        frame: int,
        current_residual: np.ndarray,
        warm_mean: np.ndarray,
        forbidden_run: int,
        severe_run: int,
        max_penetration: float,
    ) -> tuple[np.ndarray, _LocalEvaluation, np.ndarray, int]:
        cfg = self.config
        dimensions = cfg.num_future_knots * len(current_residual)
        horizon = min(cfg.horizon_frames, self.engine.horizon - 1 - frame)
        mean = warm_mean.copy()
        std = np.full(dimensions, cfg.initial_std_rad, dtype=np.float64)
        best = self._evaluate(
            data, frame, current_residual, mean, forbidden_run, severe_run,
            max_penetration,
        )
        best_parameters = mean.copy()
        evaluations = 1
        for _ in range(cfg.generations):
            samples = np.clip(
                self.rng.normal(mean, std, size=(cfg.population, dimensions)),
                -cfg.residual_bound_rad,
                cfg.residual_bound_rad,
            )
            samples[0] = best_parameters
            results = [
                self._evaluate(
                    data, frame, current_residual, sample, forbidden_run,
                    severe_run, max_penetration,
                )
                for sample in samples
            ]
            evaluations += len(results)
            order = sorted(
                range(cfg.population),
                key=lambda index: self._rank(results[index], horizon),
            )
            elite = samples[order[:cfg.elite_count]]
            mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
            std = np.maximum(
                cfg.minimum_std_rad,
                0.25 * std + 0.75 * elite.std(axis=0),
            )
            generation_best = results[order[0]]
            if self._rank(generation_best, horizon) < self._rank(best, horizon):
                best = generation_best
                best_parameters = samples[order[0]].copy()
        shifted_frame = min(
            cfg.replan_interval_frames + horizon // 2, horizon
        )
        warm_path = best.residual_path
        next_mean = np.concatenate((warm_path[shifted_frame], warm_path[-1]))
        return best_parameters, best, next_mean, evaluations

    def run(self) -> BalanceMPCResult:
        engine, cfg = self.engine, self.config
        data = engine._new_initialized_data()
        target = self.feedforward.copy()
        feedback = np.zeros_like(target)
        current_residual = np.zeros(len(engine.balance_indexes), dtype=np.float64)
        warm_mean = np.zeros(
            cfg.num_future_knots * len(engine.balance_indexes), dtype=np.float64
        )
        forbidden_run = 0
        severe_run = 0
        max_penetration = 0.0
        actual_invalid_frame: int | None = None
        actual_invalid_reason: str | None = None
        replans: list[dict[str, Any]] = []
        evaluations = 0
        start_time = perf_counter()
        frame = 0
        while frame < engine.horizon - 1 and actual_invalid_frame is None:
            parameters, best, warm_mean, count = self._solve(
                data, frame, current_residual, warm_mean, forbidden_run,
                severe_run, max_penetration,
            )
            evaluations += count
            apply_count = min(
                cfg.replan_interval_frames, engine.horizon - 1 - frame
            )
            replans.append({
                "frame": frame,
                "best_cost": float(best.cost),
                "predicted_valid": best.valid,
                "predicted_first_invalid_offset": best.first_invalid_offset,
                "predicted_invalid_reason": best.invalid_reason,
                "root_height_m": float(data.qpos[engine.robot_qpos + 2]),
                "current_residual_l2_rad": float(np.linalg.norm(current_residual)),
                "evaluations": count,
            })
            for offset in range(1, apply_count + 1):
                next_frame = frame + offset
                residual = best.residual_path[offset]
                feedback[next_frame, engine.balance_indexes] = residual
                target[next_frame, engine.balance_indexes] += residual
                data.ctrl[:] = np.clip(
                    target[next_frame, engine.actuator_joint_indexes],
                    engine.model.actuator_ctrlrange[:, 0],
                    engine.model.actuator_ctrlrange[:, 1],
                )
                for _ in range(engine.config.substeps):
                    mujoco.mj_step(engine.model, data)
                    terrain_bodies, penetration = _contact_state(engine, data)
                    max_penetration = max(max_penetration, penetration)
                    forbidden = terrain_bodies - engine.allowed_terrain_bodies
                    forbidden_run = forbidden_run + 1 if forbidden else 0
                    usage = np.abs(data.actuator_force) / engine.force_scale
                    severe = (
                        np.mean(usage >= engine.config.severe_saturation_usage)
                        >= engine.config.severe_saturation_actuator_fraction
                    )
                    severe_run = severe_run + 1 if severe else 0
                    torso_up = (
                        data.xmat[engine.torso_body].reshape(3, 3)
                        @ self.torso_local_up
                    )
                    tilt = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
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
                    if reason is not None:
                        actual_invalid_frame = next_frame
                        actual_invalid_reason = reason
                        break
                current_residual = residual.copy()
                if actual_invalid_frame is not None:
                    break
            frame += apply_count
            if actual_invalid_frame is None:
                current_residual = best.residual_path[apply_count].copy()
        elapsed = perf_counter() - start_time

        rollout = engine.rollout(target, stop_on_invalid=False)
        total_correction = target - engine.reference_joint
        selected_total = total_correction[:, engine.balance_indexes]
        selected_feedback = feedback[:, engine.balance_indexes]
        rollout.metrics.update({
            "balance_rms_correction_rad": float(
                np.sqrt(np.square(selected_total).mean())
            ),
            "balance_max_correction_rad": float(np.abs(selected_total).max()),
            "mpc_feedback_rms_rad": float(
                np.sqrt(np.square(selected_feedback).mean())
            ),
            "mpc_feedback_max_rad": float(np.abs(selected_feedback).max()),
            "mpc_feedback_by_joint_rms_rad": {
                engine.joint_names[value]: float(
                    np.sqrt(np.square(selected_feedback[:, index]).mean())
                )
                for index, value in enumerate(engine.balance_indexes)
            },
            "mpc_feedback_by_joint_max_rad": {
                engine.joint_names[value]: float(
                    np.abs(selected_feedback[:, index]).max()
                )
                for index, value in enumerate(engine.balance_indexes)
            },
            "mpc_replans": len(replans),
            "mpc_evaluations": evaluations,
            "mpc_optimization_seconds": elapsed,
            "mpc_actual_first_invalid_frame": actual_invalid_frame,
            "mpc_actual_invalid_reason": actual_invalid_reason,
        })
        return BalanceMPCResult(
            rollout=rollout,
            joint_target=target,
            feedback_residual=feedback,
            feedforward_target=self.feedforward,
            replans=replans,
            evaluations=evaluations,
            optimization_seconds=elapsed,
            actual_first_invalid_frame=actual_invalid_frame,
            actual_invalid_reason=actual_invalid_reason,
        )


def refine_balance_with_mpc(
    model: mujoco.MjModel,
    generated_reference: GeneratedReference,
    openloop: DynamicsRefinementResult,
    dynamics_config: DynamicsRefinementConfig,
    mpc_config: BalanceMPCConfig | None = None,
) -> BalanceMPCResult:
    """Replan conservative balance residuals around a fixed open-loop target."""
    if openloop.manipulation is not None:
        raise ValueError("MPC Stage A requires a balance-only open-loop result")
    engine = _DynamicRollout(model, generated_reference, dynamics_config)
    controller = _BalanceMPC(
        engine, openloop.balance_joint_target, mpc_config or BalanceMPCConfig()
    )
    return controller.run()
