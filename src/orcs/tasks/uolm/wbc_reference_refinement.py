"""CEM reference refinement evaluated by the frozen compliant-contact WBC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import mujoco
import numpy as np

from orcs.tasks.uolm.dynamics_refinement import _DynamicRollout
from orcs.tasks.uolm.whole_body_feedback import (
    WholeBodyFeedbackConfig,
    WholeBodyFeedbackResult,
    track_with_whole_body_feedback,
)

__all__ = [
    "WBCReferenceRefinementConfig",
    "WBCReferenceRefinementResult",
    "refine_reference_for_wbc",
]


@dataclass(frozen=True, slots=True)
class WBCReferenceRefinementConfig:
    knot_frames: tuple[int, ...] = (85, 95, 105, 120, 140, 160)
    population: int = 36
    elite_count: int = 6
    generations: int = 10
    seed: int = 782
    root_height_bound_m: float = 0.10
    root_roll_pitch_bound_rad: float = 0.12
    balance_joint_bound_rad: float = 0.12
    initial_std_fraction: float = 0.35
    minimum_std_fraction: float = 0.08
    distribution_smoothing: float = 0.70
    wbc: WholeBodyFeedbackConfig = field(default_factory=WholeBodyFeedbackConfig)


@dataclass(slots=True)
class WBCReferenceEvaluation:
    parameters: np.ndarray
    artifact: dict[str, np.ndarray]
    rollout: WholeBodyFeedbackResult
    metrics: dict[str, Any]
    prior_cost: float
    rank: tuple[float, ...]


@dataclass(slots=True)
class WBCReferenceRefinementResult:
    baseline: WBCReferenceEvaluation
    best: WBCReferenceEvaluation
    history: list[dict[str, Any]]
    evaluations: int
    optimization_seconds: float
    config: WBCReferenceRefinementConfig


def _smootherstep(value: np.ndarray) -> np.ndarray:
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _residual_path(
    parameters: np.ndarray,
    samples: int,
    dimensions: int,
    knot_samples: tuple[int, ...],
) -> np.ndarray:
    knots = np.zeros((len(knot_samples), dimensions), dtype=np.float64)
    knots[1:-1] = parameters.reshape(len(knot_samples) - 2, dimensions)
    result = np.zeros((samples, dimensions), dtype=np.float64)
    for segment in range(len(knot_samples) - 1):
        lower, upper = knot_samples[segment : segment + 2]
        indexes = np.arange(lower, upper + 1)
        phase = (indexes - lower) / (upper - lower)
        blend = _smootherstep(phase)[:, None]
        result[indexes] = (1.0 - blend) * knots[segment] + blend * knots[segment + 1]
    return result


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _quat_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).reshape(-1))
    return quaternion


def _candidate_artifact(
    engine: _DynamicRollout,
    source: dict[str, np.ndarray],
    parameters: np.ndarray,
    config: WBCReferenceRefinementConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build a consistent 200 Hz desired state path from spline corrections."""
    start = config.wbc.anchor_frame * engine.config.substeps
    end = config.wbc.evaluation_end_frame * engine.config.substeps
    sample_count = end - start + 1
    dimensions = 3 + len(engine.balance_indexes)
    knot_samples = tuple(
        (frame - config.wbc.anchor_frame) * engine.config.substeps
        for frame in config.knot_frames
    )
    residual = _residual_path(parameters, sample_count, dimensions, knot_samples)
    result = {key: value.copy() for key, value in source.items()}
    # Preserve the baseline artifact bit-for-bit.  Besides being the CEM origin,
    # this is an explicit equivalence test against the previously reported WBC
    # rollout; recomputing even nominal COM fields can introduce avoidable
    # numerical differences into the feedback objective.
    if not np.any(parameters):
        return result, residual
    root_position = result["actual_root_position_m"][start : end + 1].astype(
        np.float64
    )
    root_quaternion = result["actual_root_quaternion_wxyz"][start : end + 1].astype(
        np.float64
    )
    joint_position = result["actual_joint_position_rad"][start : end + 1].astype(
        np.float64
    )
    joint_velocity = result["actual_joint_velocity_rad_s"][start : end + 1].astype(
        np.float64
    )
    root_position[:, 2] += residual[:, 0]
    balance_actuators = np.flatnonzero(
        np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
    )
    joint_position[:, balance_actuators] += residual[:, 3:]
    residual_velocity = np.gradient(
        residual, engine.config.physics_dt, axis=0, edge_order=2
    )
    joint_velocity[:, balance_actuators] += residual_velocity[:, 3:]

    rotations = np.empty((sample_count, 3, 3), dtype=np.float64)
    for sample in range(sample_count):
        nominal = _quat_to_rotation(root_quaternion[sample])
        correction = _rotation_x(residual[sample, 1]) @ _rotation_y(
            residual[sample, 2]
        )
        rotations[sample] = nominal @ correction
        root_quaternion[sample] = _rotation_to_quat(rotations[sample])

    root_linear_velocity = result["actual_root_linear_velocity_m_s"][
        start : end + 1
    ].astype(np.float64)
    root_linear_velocity[:, 2] += residual_velocity[:, 0]
    root_angular_world = result["actual_root_angular_velocity_rad_s"][
        start : end + 1
    ].astype(np.float64)
    local_correction_rate = np.column_stack(
        (residual_velocity[:, 1], residual_velocity[:, 2], np.zeros(sample_count))
    )
    for sample in range(sample_count):
        root_angular_world[sample] += rotations[sample] @ local_correction_rate[sample]

    root_body = int(engine.model.jnt_bodyid[engine.robot_free_joint])
    com_position = np.empty((sample_count, 3), dtype=np.float64)
    com_velocity = np.empty_like(com_position)
    angular_momentum = np.empty_like(com_position)
    scratch = mujoco.MjData(engine.model)
    actuator_dofs = np.asarray(
        [
            engine.model.jnt_dofadr[engine.model.actuator_trnid[index, 0]]
            for index in range(engine.model.nu)
        ],
        dtype=np.int32,
    )
    for sample in range(sample_count):
        scratch.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root_position[sample]
        scratch.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = root_quaternion[
            sample
        ]
        scratch.qpos[engine.actuator_qpos] = joint_position[sample]
        scratch.qvel[engine.robot_dof : engine.robot_dof + 3] = root_linear_velocity[
            sample
        ]
        scratch.qvel[engine.robot_dof + 3 : engine.robot_dof + 6] = (
            rotations[sample].T @ root_angular_world[sample]
        )
        scratch.qvel[actuator_dofs] = joint_velocity[sample]
        mujoco.mj_forward(engine.model, scratch)
        com_position[sample] = scratch.subtree_com[root_body]
        com_velocity[sample] = scratch.subtree_linvel[root_body]
        angular_momentum[sample] = scratch.subtree_angmom[root_body]

    result["actual_root_position_m"][start : end + 1] = root_position
    result["actual_root_quaternion_wxyz"][start : end + 1] = root_quaternion
    result["actual_root_linear_velocity_m_s"][start : end + 1] = root_linear_velocity
    result["actual_root_angular_velocity_rad_s"][start : end + 1] = root_angular_world
    result["actual_joint_position_rad"][start : end + 1] = joint_position
    result["actual_joint_velocity_rad_s"][start : end + 1] = joint_velocity
    result["actual_com_position_m"][start : end + 1] = com_position
    result["actual_com_velocity_m_s"][start : end + 1] = com_velocity
    result["actual_centroidal_angular_momentum"][start : end + 1] = angular_momentum
    return result, residual


def _first_sustained_loss(contact: np.ndarray, side: int, count: int = 4) -> int | None:
    absent = ~contact[:, side]
    runs = np.convolve(absent.astype(np.int32), np.ones(count, dtype=np.int32), "valid")
    found = np.flatnonzero(runs == count)
    return None if not len(found) else int(found[0])


def _metrics(
    engine: _DynamicRollout,
    rollout: WholeBodyFeedbackResult,
    residual: np.ndarray,
    artifact: dict[str, np.ndarray],
    config: WBCReferenceRefinementConfig,
) -> tuple[dict[str, Any], float, tuple[float, ...]]:
    data = rollout.data
    stop = (
        len(data["foot_contact"])
        if rollout.first_invalid_sample is None
        else rollout.first_invalid_sample + 1
    )
    left_loss = _first_sustained_loss(data["foot_contact"], 0)
    right_loss = _first_sustained_loss(data["foot_contact"], 1)
    both_off = ~np.any(data["foot_contact"][:stop], axis=1)
    balance_actuators = np.flatnonzero(
        np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
    )
    frame100 = (100 - config.wbc.anchor_frame) * engine.config.substeps
    frame110 = (110 - config.wbc.anchor_frame) * engine.config.substeps
    root_body = int(engine.model.jnt_bodyid[engine.robot_free_joint])
    mass = float(engine.model.body_subtreemass[root_body])
    arm_terrain_duration_before_invalid = float(
        np.sum(data["arm_terrain_contact"][:stop]) * engine.config.physics_dt
    )
    arm_object_duration_before_invalid = float(
        np.sum(data["arm_object_contact"][:stop]) * engine.config.physics_dt
    )
    arm_terrain_duration = float(
        np.sum(data["arm_terrain_contact"]) * engine.config.physics_dt
    )
    arm_object_duration = float(
        np.sum(data["arm_object_contact"]) * engine.config.physics_dt
    )
    joint_violation = 0.0
    start = config.wbc.anchor_frame * engine.config.substeps
    end = config.wbc.evaluation_end_frame * engine.config.substeps
    candidate_joint = artifact["actual_joint_position_rad"][start : end + 1]
    limits = engine.joint_limits[engine.actuator_joint_indexes]
    joint_violation = float(
        np.max(
            np.maximum(limits[:, 0] - candidate_joint, 0.0)
            + np.maximum(candidate_joint - limits[:, 1], 0.0)
        )
    )
    correction_velocity = np.gradient(
        residual, engine.config.physics_dt, axis=0, edge_order=2
    )
    correction_acceleration = np.gradient(
        correction_velocity, engine.config.physics_dt, axis=0, edge_order=2
    )
    prior_cost = float(
        30.0 * np.mean(np.square(residual[:, 0]))
        + 12.0 * np.mean(np.square(residual[:, 1:3]))
        + 4.0 * np.mean(np.square(residual[:, 3:]))
        + 0.02 * np.mean(np.square(correction_velocity))
        + 2.0e-4 * np.mean(np.square(correction_acceleration))
    )
    hard_bad = int(
        rollout.summary["maximum_hard_constraint_residual"] > 1.0e-3
        or joint_violation > 0.03
    )
    arm_cheat = int(
        arm_terrain_duration >= engine.config.sustained_nonfoot_terrain_contact_s
    )
    survival = (
        config.wbc.evaluation_end_frame + 1
        if rollout.first_invalid_frame is None
        else rollout.first_invalid_frame
    )
    full_valid = int(
        rollout.valid and rollout.recoverable_terminal and not hard_bad and not arm_cheat
    )
    torque_pathology = float(
        np.mean(data["torque_usage"][:stop, balance_actuators] >= 0.995)
    )
    tracking_cost = float(
        np.mean(np.square(data["joint_position_error"][:stop, balance_actuators]))
        + 0.02
        * np.mean(np.square(data["joint_velocity_error"][:stop, balance_actuators]))
        + 5.0 * np.mean(np.square(data["com_error"][:stop]))
    )
    support_cost = float(
        8.0 * np.mean(both_off)
        + 0.5 * np.mean(np.square(data["foot_slip_m_s"][:stop]))
        + 4.0 * torque_pathology
        + 2.0 * arm_object_duration
        + 10.0 * arm_terrain_duration
    )
    frame140_sample = (140 - config.wbc.anchor_frame) * engine.config.substeps + 1
    reference_root = artifact["actual_root_position_m"][
        start : start + frame140_sample
    ]
    root_error_85_140 = (
        reference_root - data["root_position"][:frame140_sample]
    )
    com_error_85_140 = data["com_error"][:frame140_sample]
    rank = (
        float(full_valid),
        float(survival),
        float(1 - hard_bad),
        float(1 - arm_cheat),
        -float(np.sum(both_off)),
        -float(rollout.summary["solver_failures"]),
        -support_cost,
        -tracking_cost,
        -prior_cost,
    )
    metrics = {
        "first_invalid_frame": rollout.first_invalid_frame,
        "invalid_reason": rollout.invalid_reason,
        "first_left_foot_sustained_loss_frame": (
            None
            if left_loss is None
            else config.wbc.anchor_frame
            + int(np.ceil(left_loss / engine.config.substeps))
        ),
        "first_right_foot_sustained_loss_frame": (
            None
            if right_loss is None
            else config.wbc.anchor_frame
            + int(np.ceil(right_loss / engine.config.substeps))
        ),
        "both_feet_off_duration_before_invalidation_s": float(
            np.sum(both_off) * engine.config.physics_dt
        ),
        "root_height_frame100_m": float(data["root_position"][frame100, 2]),
        "root_height_frame110_m": float(data["root_position"][frame110, 2]),
        "vertical_momentum_frame110_kg_m_s": float(
            mass * data["actual_com_velocity"][frame110, 2]
        ),
        "torso_tilt_frame110_deg": float(
            np.rad2deg(data["torso_tilt_rad"][frame110])
        ),
        "maximum_joint_position_error_before_invalidation_rad": float(
            np.max(np.abs(data["joint_position_error"][:stop]))
        ),
        "maximum_joint_velocity_error_before_invalidation_rad_s": float(
            np.max(np.abs(data["joint_velocity_error"][:stop]))
        ),
        "maximum_balance_torque_usage_before_invalidation": float(
            np.max(data["torque_usage"][:stop, balance_actuators])
        ),
        "solver_failure_count": int(rollout.summary["solver_failures"]),
        "maximum_hard_constraint_residual": float(
            rollout.summary["maximum_hard_constraint_residual"]
        ),
        "arm_terrain_contact_duration_before_invalidation_s": (
            arm_terrain_duration_before_invalid
        ),
        "arm_object_contact_duration_before_invalidation_s": (
            arm_object_duration_before_invalid
        ),
        "arm_terrain_contact_duration_full_horizon_s": arm_terrain_duration,
        "arm_object_contact_duration_full_horizon_s": arm_object_duration,
        "root_position_rms_error_frames85_140_m": float(
            np.sqrt(np.mean(np.square(root_error_85_140)))
        ),
        "root_height_rms_error_frames85_140_m": float(
            np.sqrt(np.mean(np.square(root_error_85_140[:, 2])))
        ),
        "com_position_rms_error_frames85_140_m": float(
            np.sqrt(np.mean(np.square(com_error_85_140)))
        ),
        "com_velocity_rms_error_frames85_140_m_s": float(
            np.sqrt(
                np.mean(
                    np.square(data["com_velocity_error"][:frame140_sample])
                )
            )
        ),
        "torque_limit_activity_fraction": torque_pathology,
        "maximum_candidate_joint_limit_violation_rad": joint_violation,
        "support_cost": support_cost,
        "tracking_cost": tracking_cost,
        "prior_cost": prior_cost,
    }
    return metrics, prior_cost, rank


def _evaluate(
    engine: _DynamicRollout,
    nominal_target: np.ndarray,
    source_artifact: dict[str, np.ndarray],
    parameters: np.ndarray,
    config: WBCReferenceRefinementConfig,
) -> WBCReferenceEvaluation:
    artifact, residual = _candidate_artifact(
        engine, source_artifact, parameters, config
    )
    rollout = track_with_whole_body_feedback(
        engine, nominal_target, artifact, config.wbc
    )
    metrics, prior_cost, rank = _metrics(
        engine, rollout, residual, artifact, config
    )
    metrics["reference_residual"] = residual
    return WBCReferenceEvaluation(
        parameters=parameters.copy(),
        artifact=artifact,
        rollout=rollout,
        metrics=metrics,
        prior_cost=prior_cost,
        rank=rank,
    )


def refine_reference_for_wbc(
    engine: _DynamicRollout,
    nominal_target: np.ndarray,
    source_artifact: dict[str, np.ndarray],
    config: WBCReferenceRefinementConfig | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> WBCReferenceRefinementResult:
    """Run one bounded diagonal-CEM search with the WBC held fixed."""
    cfg = config or WBCReferenceRefinementConfig()
    if cfg.knot_frames[0] != cfg.wbc.anchor_frame:
        raise ValueError("first knot must equal the fixed physical anchor")
    if cfg.knot_frames[-1] != cfg.wbc.evaluation_end_frame:
        raise ValueError("last knot must equal the evaluation horizon")
    free_knots = len(cfg.knot_frames) - 2
    bounds_per_knot = np.r_[
        cfg.root_height_bound_m,
        np.full(2, cfg.root_roll_pitch_bound_rad),
        np.full(len(engine.balance_indexes), cfg.balance_joint_bound_rad),
    ]
    bounds = np.tile(bounds_per_knot, free_knots)
    dimensions = len(bounds)
    rng = np.random.default_rng(cfg.seed)
    mean = np.zeros(dimensions, dtype=np.float64)
    std = cfg.initial_std_fraction * bounds
    minimum_std = cfg.minimum_std_fraction * bounds
    baseline = _evaluate(
        engine, nominal_target, source_artifact, np.zeros(dimensions), cfg
    )
    best = baseline
    history: list[dict[str, Any]] = []
    evaluations = 1
    started = perf_counter()
    for generation in range(cfg.generations):
        population = rng.normal(mean, std, size=(cfg.population, dimensions))
        population = np.clip(population, -bounds, bounds)
        population[0] = best.parameters
        if generation == 0:
            population[0] = 0.0
        evaluated: list[WBCReferenceEvaluation] = []
        for parameters in population:
            candidate = _evaluate(
                engine, nominal_target, source_artifact, parameters, cfg
            )
            evaluated.append(candidate)
            evaluations += 1
            if candidate.rank > best.rank:
                best = candidate
        evaluated.sort(key=lambda item: item.rank, reverse=True)
        elite = np.stack(
            [item.parameters for item in evaluated[: cfg.elite_count]], axis=0
        )
        elite_mean = np.mean(elite, axis=0)
        elite_std = np.std(elite, axis=0)
        alpha = cfg.distribution_smoothing
        mean = alpha * elite_mean + (1.0 - alpha) * mean
        std = np.maximum(
            alpha * elite_std + (1.0 - alpha) * std, minimum_std
        )
        history.append(
            generation_result := {
                "generation": generation,
                "best_first_invalid_frame": best.metrics["first_invalid_frame"],
                "best_invalid_reason": best.metrics["invalid_reason"],
                "best_left_loss_frame": best.metrics[
                    "first_left_foot_sustained_loss_frame"
                ],
                "best_right_loss_frame": best.metrics[
                    "first_right_foot_sustained_loss_frame"
                ],
                "best_both_off_duration_s": best.metrics[
                    "both_feet_off_duration_before_invalidation_s"
                ],
                "best_arm_terrain_duration_s": best.metrics[
                    "arm_terrain_contact_duration_before_invalidation_s"
                ],
                "best_maximum_balance_torque_usage": best.metrics[
                    "maximum_balance_torque_usage_before_invalidation"
                ],
                "best_solver_failures": best.metrics["solver_failure_count"],
                "best_prior_cost": best.prior_cost,
                "elite_failure_frames": [
                    item.metrics["first_invalid_frame"]
                    for item in evaluated[: cfg.elite_count]
                ],
            }
        )
        if progress is not None:
            progress(generation_result)
    return WBCReferenceRefinementResult(
        baseline=baseline,
        best=best,
        history=history,
        evaluations=evaluations,
        optimization_seconds=perf_counter() - started,
        config=cfg,
    )
