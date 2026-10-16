"""Diagnose position-servo tracking and foot-support loss for the 5-knot baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from orcs.cli.refine_contact_dynamics import _serializable
from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.support_diagnostics import (
    SupportDiagnosticResult,
    instrument_support_rollout,
)

_POSITION_THRESHOLDS_RAD = (0.05, 0.10, 0.20)
_VELOCITY_THRESHOLDS_RAD_S = (0.5, 1.0, 2.0)
_ANALYSIS_START = 85
_ANALYSIS_END = 140
_REPLAY_END = 160


def _actuator_indexes(result: SupportDiagnosticResult, names: list[str]) -> np.ndarray:
    actuator_names = list(result.data["actuator_names"])
    return np.asarray([actuator_names.index(name) for name in names], dtype=np.int32)


def _first_true(mask: np.ndarray, start: int, end: int) -> int | None:
    indexes = np.flatnonzero(mask[start:end])
    return None if not len(indexes) else int(start + indexes[0])


def _first_loss_sample(
    result: SupportDiagnosticResult, side: int, start_frame: int = 85
) -> int | None:
    mask = ~result.data["foot_contact"][:, side]
    run = np.convolve(mask.astype(np.int32), np.ones(4, dtype=np.int32), "valid")
    indexes = np.flatnonzero((run == 4) & (np.arange(len(run)) >= start_frame * 4))
    return None if not len(indexes) else int(indexes[0])


def _event(data: dict[str, np.ndarray], sample: int | None) -> dict[str, Any] | None:
    if sample is None:
        return None
    return {
        "sample": sample,
        "time_s": float(data["time_s"][sample]),
        "frame": int(data["control_frame"][sample]),
    }


def _target_derivatives(
    target: np.ndarray, indexes: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    selected = target[:, indexes]
    velocity = np.gradient(selected, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    return velocity, acceleration


def _tracking_summary(
    result: SupportDiagnosticResult,
    target: np.ndarray,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = result.data
    names = list(data["balance_joint_names"])
    actuator_indexes = _actuator_indexes(result, names)
    joint_indexes = np.asarray(
        [engine.joint_names.index(name) for name in names], dtype=np.int32
    )
    target_velocity_50hz, target_acceleration_50hz = _target_derivatives(
        target, joint_indexes, engine.config.control_dt
    )
    start = _ANALYSIS_START * engine.config.substeps
    end = min(
        len(data["time_s"]),
        (_ANALYSIS_END + 1) * engine.config.substeps,
    )
    position_error = data["actuator_position_error_rad"][start:end][:, actuator_indexes]
    velocity_error = data["actuator_velocity_error_rad_s"][start:end][
        :, actuator_indexes
    ]
    actual_velocity = data["actuator_velocity_rad_s"][start:end][:, actuator_indexes]
    actual_acceleration = data["actuator_acceleration_rad_s2"][start:end][
        :, actuator_indexes
    ]
    held_target_velocity = data["actuator_target_velocity_rad_s"][start:end][
        :, actuator_indexes
    ]
    held_target_acceleration = data["actuator_target_acceleration_rad_s2"][start:end][
        :, actuator_indexes
    ]
    summaries: dict[str, Any] = {}
    for local, name in enumerate(names):
        position = np.abs(position_error[:, local])
        velocity = np.abs(velocity_error[:, local])
        entry: dict[str, Any] = {
            "peak_target_velocity_50hz_rad_s": float(
                np.max(
                    np.abs(
                        target_velocity_50hz[_ANALYSIS_START : _ANALYSIS_END + 1, local]
                    )
                )
            ),
            "peak_target_acceleration_50hz_rad_s2": float(
                np.max(
                    np.abs(
                        target_acceleration_50hz[
                            _ANALYSIS_START : _ANALYSIS_END + 1, local
                        ]
                    )
                )
            ),
            "peak_held_command_velocity_200hz_rad_s": float(
                np.max(np.abs(held_target_velocity[:, local]))
            ),
            "peak_held_command_acceleration_200hz_rad_s2": float(
                np.max(np.abs(held_target_acceleration[:, local]))
            ),
            "peak_actual_velocity_rad_s": float(
                np.max(np.abs(actual_velocity[:, local]))
            ),
            "peak_actual_acceleration_rad_s2": float(
                np.max(np.abs(actual_acceleration[:, local]))
            ),
            "maximum_abs_position_error_rad": float(np.max(position)),
            "maximum_abs_velocity_error_rad_s": float(np.max(velocity)),
        }
        for threshold in _POSITION_THRESHOLDS_RAD:
            key = f"position_error_gt_{threshold:.2f}rad"
            index = _first_true(position > threshold, 0, len(position))
            entry[f"first_{key}"] = _event(
                data, None if index is None else start + index
            )
            entry[f"duration_{key}_s"] = float(
                np.sum(position > threshold) * engine.config.physics_dt
            )
        for threshold in _VELOCITY_THRESHOLDS_RAD_S:
            key = f"velocity_error_gt_{threshold:.1f}rad_s"
            index = _first_true(velocity > threshold, 0, len(velocity))
            entry[f"first_{key}"] = _event(
                data, None if index is None else start + index
            )
            entry[f"duration_{key}_s"] = float(
                np.sum(velocity > threshold) * engine.config.physics_dt
            )
        summaries[name] = entry
    return summaries


def _rate_error_relationship(
    result: SupportDiagnosticResult,
    target: np.ndarray,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = result.data
    names = list(data["balance_joint_names"])
    actuator_indexes = _actuator_indexes(result, names)
    joint_indexes = np.asarray(
        [engine.joint_names.index(name) for name in names], dtype=np.int32
    )
    velocity, acceleration = _target_derivatives(
        target, joint_indexes, engine.config.control_dt
    )
    frames = np.arange(_ANALYSIS_START, _ANALYSIS_END + 1)
    error = np.abs(
        data["actuator_position_error_rad"][frames * engine.config.substeps][
            :, actuator_indexes
        ]
    )
    speed = np.abs(velocity[frames])
    per_joint = {
        name: float(np.corrcoef(speed[:, index], error[:, index])[0, 1])
        for index, name in enumerate(names)
    }
    flat_order = np.argsort(speed.ravel())[::-1][:8]
    peaks = []
    for flat in flat_order:
        frame_index, joint_index = np.unravel_index(flat, speed.shape)
        peaks.append(
            {
                "frame": int(frames[frame_index]),
                "joint": names[joint_index],
                "target_velocity_rad_s": float(
                    velocity[frames[frame_index], joint_index]
                ),
                "target_acceleration_rad_s2": float(
                    acceleration[frames[frame_index], joint_index]
                ),
                "position_error_rad": float(error[frame_index, joint_index]),
            }
        )
    anchor_error = data["actuator_position_error_rad"][
        _ANALYSIS_START * engine.config.substeps, actuator_indexes
    ]
    return {
        "pooled_same_frame_abs_target_speed_position_error_correlation": float(
            np.corrcoef(speed.ravel(), error.ravel())[0, 1]
        ),
        "pooled_one_frame_lag_correlation": float(
            np.corrcoef(speed[:-1].ravel(), error[1:].ravel())[0, 1]
        ),
        "per_joint_same_frame_correlation": per_joint,
        "largest_target_speed_events": peaks,
        "anchor_frame85_position_errors_over_0p10rad": {
            name: float(anchor_error[index])
            for index, name in enumerate(names)
            if abs(anchor_error[index]) > 0.10
        },
    }


def _actuator_model(
    model: mujoco.MjModel,
    result: SupportDiagnosticResult,
    target: np.ndarray,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = result.data
    names = list(data["balance_joint_names"])
    actuator_indexes = _actuator_indexes(result, names)
    joint_indexes = np.asarray(
        [engine.joint_names.index(name) for name in names], dtype=np.int32
    )
    target_velocity_50hz, _ = _target_derivatives(
        target, joint_indexes, engine.config.control_dt
    )
    start = _ANALYSIS_START * engine.config.substeps
    invalid = result.rollout.first_invalid_frame or _ANALYSIS_END
    end = min(len(data["time_s"]), invalid * engine.config.substeps + 1)
    output: dict[str, Any] = {}
    for local, (name, actuator) in enumerate(zip(names, actuator_indexes, strict=True)):
        joint_id = int(model.actuator_trnid[actuator, 0])
        dof = int(model.jnt_dofadr[joint_id])
        kp = float(model.actuator_gainprm[actuator, 0])
        kd = float(-model.actuator_biasprm[actuator, 2])
        request = data["reconstructed_unclipped_actuator_torque_nm"][
            start:end, actuator
        ]
        actual = data["actuator_torque_nm"][start:end, actuator]
        position_term = data["actuator_gain_term_nm"][start:end, actuator] + (
            model.actuator_biasprm[actuator, 0]
            + data["actuator_position_rad"][start:end, actuator]
            * model.actuator_biasprm[actuator, 1]
        )
        damping_term = (
            data["actuator_transmission_velocity_rad_s"][start:end, actuator]
            * model.actuator_biasprm[actuator, 2]
        )
        clipped = np.clip(
            request,
            model.actuator_forcerange[actuator, 0],
            model.actuator_forcerange[actuator, 1],
        )
        meaningful = np.abs(position_term) > 1.0
        cancels = meaningful & (position_term * damping_term < 0.0)
        output[name] = {
            "actuator_index": int(actuator),
            "kp_nm_rad": kp,
            "kd_nm_s_rad": kd,
            "gain_type": int(model.actuator_gaintype[actuator]),
            "bias_type": int(model.actuator_biastype[actuator]),
            "dynamics_type": int(model.actuator_dyntype[actuator]),
            "armature_kg_m2": float(model.dof_armature[dof]),
            "joint_damping_nm_s_rad": float(model.dof_damping[dof]),
            "gear": model.actuator_gear[actuator].tolist(),
            "control_range_rad": model.actuator_ctrlrange[actuator].tolist(),
            "force_range_nm": model.actuator_forcerange[actuator].tolist(),
            "peak_unclipped_request_nm": float(np.max(np.abs(request))),
            "peak_actual_torque_nm": float(np.max(np.abs(actual))),
            "peak_normalized_usage": float(
                np.max(data["normalized_torque_usage"][start:end, actuator])
            ),
            "duration_above_80pct_s": float(
                np.sum(data["normalized_torque_usage"][start:end, actuator] > 0.80)
                * engine.config.physics_dt
            ),
            "duration_above_95pct_s": float(
                np.sum(data["normalized_torque_usage"][start:end, actuator] > 0.95)
                * engine.config.physics_dt
            ),
            "request_clipped_duration_s": float(
                np.sum(np.abs(request - clipped) > 1.0e-8) * engine.config.physics_dt
            ),
            "reconstruction_rms_error_nm": float(
                np.sqrt(np.mean(np.square(clipped - actual)))
            ),
            "peak_position_term_nm": float(np.max(np.abs(position_term))),
            "peak_damping_term_nm": float(np.max(np.abs(damping_term))),
            "damping_opposes_position_fraction": float(
                np.sum(cancels) / max(1, np.sum(meaningful))
            ),
            "peak_missing_target_velocity_term_nm": float(
                kd
                * np.max(
                    np.abs(
                        target_velocity_50hz[_ANALYSIS_START : _ANALYSIS_END + 1, local]
                    )
                )
            ),
        }
    return {
        "law": (
            "MuJoCo fixed-gain affine position servo: actuator_force = "
            "clip(Kp*ctrl - Kp*actuator_length - Kd*actuator_velocity, "
            "force_range); there is no target-velocity term"
        ),
        "gain_type_0": "fixed",
        "bias_type_1": "affine",
        "dynamics_type_0": "none (no activation state)",
        "target_updates_hz": 50.0,
        "physics_hz": 200.0,
        "target_interpolation": "zero-order hold for four 5-ms simulation steps",
        "target_rate_limit": None,
        "target_clipping": "direct clip to actuator_ctrlrange at each 50-Hz update",
        "joints": output,
    }


def _target_conditioned_foot_positions(
    model: mujoco.MjModel,
    result: SupportDiagnosticResult,
    engine: _DynamicRollout,
) -> np.ndarray:
    data = result.data
    names = list(data["balance_joint_names"])
    balance_actuators = _actuator_indexes(result, names)
    scratch = mujoco.MjData(model)
    positions = np.empty_like(data["foot_body_position_m"])
    joint = np.empty(len(engine.joint_names), dtype=np.float64)
    for sample in range(len(data["time_s"])):
        scratch.qpos[engine.robot_qpos : engine.robot_qpos + 3] = data[
            "root_position_m"
        ][sample]
        scratch.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = data[
            "root_quaternion_wxyz"
        ][sample]
        joint[engine.actuator_joint_indexes] = data["actuator_position_rad"][sample]
        joint[engine.balance_indexes] = data["actuator_target_rad"][
            sample, balance_actuators
        ]
        scratch.qpos[engine.joint_qpos] = joint
        mujoco.mj_forward(model, scratch)
        positions[sample] = scratch.xpos[list(engine.foot_bodies)]
    return positions


def _foot_loss_timeline(
    result: SupportDiagnosticResult,
    target_conditioned_feet: np.ndarray,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = result.data
    names = list(data["balance_joint_names"])
    actuator_indexes = _actuator_indexes(result, names)
    side_names = {
        "left": [name for name in names if name.startswith("left_")],
        "right": [name for name in names if name.startswith("right_")],
    }
    events: dict[str, Any] = {}
    for side, side_index in (("right", 1), ("left", 0)):
        loss = _first_loss_sample(result, side_index)
        if loss is None:
            events[side] = {"loss": None}
            continue
        frames = sorted(
            {
                max(_ANALYSIS_START, int(data["control_frame"][loss]) + offset)
                for offset in (-3, -1, 0, 1, 3)
            }
        )
        samples: dict[str, Any] = {}
        for frame in frames:
            sample = frame * engine.config.substeps
            joint_values: dict[str, Any] = {}
            for name in side_names[side]:
                actuator = int(actuator_indexes[names.index(name)])
                joint_values[name] = {
                    "target_rad": float(data["actuator_target_rad"][sample, actuator]),
                    "actual_rad": float(
                        data["actuator_position_rad"][sample, actuator]
                    ),
                    "error_rad": float(
                        data["actuator_position_error_rad"][sample, actuator]
                    ),
                    "target_velocity_held_rad_s": float(
                        data["actuator_target_velocity_rad_s"][sample, actuator]
                    ),
                    "actual_velocity_rad_s": float(
                        data["actuator_velocity_rad_s"][sample, actuator]
                    ),
                    "torque_nm": float(data["actuator_torque_nm"][sample, actuator]),
                }
            samples[str(frame)] = {
                "contact": bool(data["foot_contact"][sample, side_index]),
                "normal_force_n": float(
                    data["foot_normal_force_n"][sample, side_index]
                ),
                "slip_m_s": float(data["foot_slip_velocity_m_s"][sample, side_index]),
                "actual_foot_body_height_m": float(
                    data["foot_body_position_m"][sample, side_index, 2]
                ),
                "actual_foot_vertical_velocity_m_s": float(
                    data["foot_body_velocity_m_s"][sample, side_index, 2]
                ),
                "target_conditioned_foot_body_height_m": float(
                    target_conditioned_feet[sample, side_index, 2]
                ),
                "vertical_momentum_kg_m_s": float(
                    data["vertical_momentum_kg_m_s"][sample]
                ),
                "joints": joint_values,
            }
        events[side] = {"loss": _event(data, loss), "frames": samples}
    return events


def _smoothed_target(
    target: np.ndarray, balance_indexes: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    filtered = target.copy()
    for joint in balance_indexes:
        padded = np.pad(target[:, joint], (2, 2), mode="edge")
        filtered[:, joint] = np.convolve(padded, kernel, mode="valid")
    weight = np.zeros(len(target), dtype=np.float64)
    weight[90:156] = 1.0
    phase_in = np.linspace(0.0, 1.0, 6)
    weight[85:91] = phase_in * phase_in * (3.0 - 2.0 * phase_in)
    phase_out = np.linspace(1.0, 0.0, 6)
    weight[155:161] = phase_out * phase_out * (3.0 - 2.0 * phase_out)
    output = target.copy()
    output[:, balance_indexes] += weight[:, None] * (
        filtered[:, balance_indexes] - target[:, balance_indexes]
    )
    return output, {
        "filter": "single zero-phase five-frame binomial low-pass",
        "kernel": kernel.tolist(),
        "full_weight_frames": [90, 155],
        "smooth_taper_frames": [[85, 90], [155, 160]],
        "anchor_and_endpoint_exact": True,
    }


def _comparison_row(
    result: SupportDiagnosticResult,
    target: np.ndarray,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = result.data
    names = list(data["balance_joint_names"])
    actuator_indexes = _actuator_indexes(result, names)
    joint_indexes = np.asarray(
        [engine.joint_names.index(name) for name in names], dtype=np.int32
    )
    velocity, acceleration = _target_derivatives(
        target, joint_indexes, engine.config.control_dt
    )
    invalid = result.rollout.first_invalid_frame or _ANALYSIS_END
    end = min(len(data["time_s"]), invalid * engine.config.substeps + 1)
    start = _ANALYSIS_START * engine.config.substeps
    frame125 = 125 * engine.config.substeps
    right = _first_loss_sample(result, 1)
    left = _first_loss_sample(result, 0)
    return {
        "first_invalid_frame": result.rollout.first_invalid_frame,
        "invalid_reason": result.rollout.invalid_reason,
        "first_right_foot_loss_frame": (
            None if right is None else int(data["control_frame"][right])
        ),
        "first_left_foot_loss_frame": (
            None if left is None else int(data["control_frame"][left])
        ),
        "maximum_balance_position_error_rad": float(
            np.max(
                np.abs(
                    data["actuator_position_error_rad"][start:end][:, actuator_indexes]
                )
            )
        ),
        "maximum_balance_velocity_error_rad_s": float(
            np.max(
                np.abs(
                    data["actuator_velocity_error_rad_s"][start:end][
                        :, actuator_indexes
                    ]
                )
            )
        ),
        "maximum_target_velocity_50hz_rad_s": float(
            np.max(np.abs(velocity[_ANALYSIS_START : _ANALYSIS_END + 1]))
        ),
        "maximum_target_acceleration_50hz_rad_s2": float(
            np.max(np.abs(acceleration[_ANALYSIS_START : _ANALYSIS_END + 1]))
        ),
        "maximum_balance_torque_usage": float(
            np.max(data["normalized_torque_usage"][start:end][:, actuator_indexes])
        ),
        "left_foot_slip_frame125_m_s": float(
            data["foot_slip_velocity_m_s"][frame125, 0]
        ),
        "left_foot_force_frame125_n": float(data["foot_normal_force_n"][frame125, 0]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.artifact.resolve()) as artifact:
        trajectory_np = artifact["analytical_through140_trajectory"].copy()
        original_target = artifact["optimized_joint_target"].astype(np.float64)
        realized_position = artifact["actual_joint_position_rad"].astype(np.float64)

    device = torch.device(args.device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        torch.as_tensor(trajectory_np, device=device)
    )
    model = _build_model()
    config = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    engine = _DynamicRollout(model, reference, config)
    balance_names = [engine.joint_names[index] for index in engine.balance_indexes]

    original = instrument_support_rollout(
        model,
        reference,
        original_target,
        label="fixed-arm five-knot direct-dynamics baseline",
        config=config,
        end_frame=_REPLAY_END,
    )
    balance_actuators = _actuator_indexes(original, balance_names)

    raw_replay_target = original_target.copy()
    for frame in range(_ANALYSIS_START + 1, _REPLAY_END + 1):
        raw_replay_target[frame, engine.balance_indexes] = realized_position[
            frame * config.substeps, balance_actuators
        ]
    replay_target = raw_replay_target.copy()
    raw_balance = raw_replay_target[:, engine.balance_indexes]
    below = np.maximum(
        engine.joint_limits[engine.balance_indexes, 0] - raw_balance, 0.0
    )
    above = np.maximum(
        raw_balance - engine.joint_limits[engine.balance_indexes, 1], 0.0
    )
    violation = below + above
    violation_index = np.unravel_index(np.argmax(violation), violation.shape)
    replay_target[:, engine.balance_indexes] = np.clip(
        raw_balance,
        engine.joint_limits[engine.balance_indexes, 0],
        engine.joint_limits[engine.balance_indexes, 1],
    )
    replay = instrument_support_rollout(
        model,
        reference,
        replay_target,
        label="realized balance-joint trajectory replay",
        config=config,
        end_frame=_REPLAY_END,
    )

    original_velocity, original_acceleration = _target_derivatives(
        original_target, engine.balance_indexes, config.control_dt
    )
    peak_velocity = float(
        np.max(np.abs(original_velocity[_ANALYSIS_START : _ANALYSIS_END + 1]))
    )
    peak_acceleration = float(
        np.max(np.abs(original_acceleration[_ANALYSIS_START : _ANALYSIS_END + 1]))
    )
    run_smoothing = peak_velocity > 5.0 or peak_acceleration > 100.0
    smoothed_target: np.ndarray | None = None
    smoothed: SupportDiagnosticResult | None = None
    smoothing: dict[str, Any] | None = None
    if run_smoothing:
        smoothed_target, smoothing = _smoothed_target(
            original_target, engine.balance_indexes
        )
        smoothed = instrument_support_rollout(
            model,
            reference,
            smoothed_target,
            label="single conservatively smoothed balance target diagnostic",
            config=config,
            end_frame=_REPLAY_END,
        )

    target_feet = _target_conditioned_foot_positions(model, original, engine)
    tracking = _tracking_summary(original, original_target, engine)
    actuator_model = _actuator_model(model, original, original_target, engine)
    comparison = {
        "original_targets": _comparison_row(original, original_target, engine),
        "realized_trajectory_replay": _comparison_row(replay, replay_target, engine),
        "smoothed_target": (
            None
            if smoothed is None or smoothed_target is None
            else _comparison_row(smoothed, smoothed_target, engine)
        ),
    }
    anchor_sample = _ANALYSIS_START * config.substeps
    anchor_consistency = {
        "maximum_joint_position_difference_rad": float(
            np.max(
                np.abs(
                    original.data["actuator_position_rad"][anchor_sample]
                    - replay.data["actuator_position_rad"][anchor_sample]
                )
            )
        ),
        "root_position_difference_m": float(
            np.linalg.norm(
                original.data["root_position_m"][anchor_sample]
                - replay.data["root_position_m"][anchor_sample]
            )
        ),
        "root_velocity_difference_m_s": float(
            np.linalg.norm(
                original.data["root_linear_velocity_m_s"][anchor_sample]
                - replay.data["root_linear_velocity_m_s"][anchor_sample]
            )
        ),
    }
    compare_slice = slice(anchor_sample, (_ANALYSIS_END + 1) * config.substeps)
    replay_divergence = {
        "anchor_consistency": anchor_consistency,
        "root_position_rms_difference_frames85_140_m": float(
            np.sqrt(
                np.mean(
                    np.square(
                        original.data["root_position_m"][compare_slice]
                        - replay.data["root_position_m"][compare_slice]
                    )
                )
            )
        ),
        "vertical_momentum_rms_difference_frames85_140_kg_m_s": float(
            np.sqrt(
                np.mean(
                    np.square(
                        original.data["vertical_momentum_kg_m_s"][compare_slice]
                        - replay.data["vertical_momentum_kg_m_s"][compare_slice]
                    )
                )
            )
        ),
    }

    report: dict[str, Any] = {
        "experiment": "position_target_tracking_and_realized_trajectory_replay",
        "source_artifact": str(args.artifact.resolve()),
        "window": {"anchor": 85, "analysis": [85, 140], "evaluation_end": 160},
        "controller": actuator_model,
        "tracking_by_joint": tracking,
        "target_rate_error_relationship": _rate_error_relationship(
            original, original_target, engine
        ),
        "foot_loss_timeline": _foot_loss_timeline(original, target_feet, engine),
        "comparison": comparison,
        "counterfactual_replay": replay_divergence,
        "smoothing_diagnostic": {
            "triggered": run_smoothing,
            "trigger": {
                "peak_target_velocity_50hz_rad_s": peak_velocity,
                "peak_target_acceleration_50hz_rad_s2": peak_acceleration,
                "criterion": "velocity >5 rad/s or acceleration >100 rad/s^2",
            },
            "formulation": smoothing,
        },
    }
    report["counterfactual_replay"]["joint_limit_projection"] = {
        "reason": (
            "the physically realized path exceeded the nominal left-ankle-pitch "
            "limit enough to trigger the unchanged target-validity gate"
        ),
        "maximum_projection_rad": float(np.max(violation)),
        "maximum_projection_frame": int(violation_index[0]),
        "maximum_projection_joint": balance_names[int(violation_index[1])],
        "projected_sample_count": int(np.sum(violation > 0.0)),
        "method": "componentwise projection to the unchanged model joint limits",
    }

    artifact_path = output / "position_target_tracking_diagnostics_start_000.npz"
    arrays: dict[str, np.ndarray] = {
        "original_target": original_target.astype(np.float32),
        "raw_realized_replay_target": raw_replay_target.astype(np.float32),
        "replay_target": replay_target.astype(np.float32),
        "original_actual_position": original.data["actuator_position_rad"].astype(
            np.float32
        ),
        "original_actual_velocity": original.data["actuator_velocity_rad_s"].astype(
            np.float32
        ),
        "original_actual_acceleration": original.data[
            "actuator_acceleration_rad_s2"
        ].astype(np.float32),
        "original_position_error": original.data["actuator_position_error_rad"].astype(
            np.float32
        ),
        "original_velocity_error": original.data[
            "actuator_velocity_error_rad_s"
        ].astype(np.float32),
        "original_target_velocity_200hz": original.data[
            "actuator_target_velocity_rad_s"
        ].astype(np.float32),
        "original_target_acceleration_200hz": original.data[
            "actuator_target_acceleration_rad_s2"
        ].astype(np.float32),
        "original_actuator_torque": original.data["actuator_torque_nm"].astype(
            np.float32
        ),
        "original_reconstructed_torque": original.data[
            "reconstructed_unclipped_actuator_torque_nm"
        ].astype(np.float32),
        "original_foot_contact": original.data["foot_contact"],
        "original_foot_force": original.data["foot_normal_force_n"].astype(np.float32),
        "original_foot_slip": original.data["foot_slip_velocity_m_s"].astype(
            np.float32
        ),
        "original_foot_position": original.data["foot_body_position_m"].astype(
            np.float32
        ),
        "target_conditioned_foot_position": target_feet.astype(np.float32),
        "replay_actual_position": replay.data["actuator_position_rad"].astype(
            np.float32
        ),
        "replay_root_position": replay.data["root_position_m"].astype(np.float32),
        "replay_vertical_momentum": replay.data["vertical_momentum_kg_m_s"].astype(
            np.float32
        ),
        "actuator_names": original.data["actuator_names"],
        "balance_joint_names": original.data["balance_joint_names"],
        "physics_dt": np.asarray([config.physics_dt], dtype=np.float32),
        "control_dt": np.asarray([config.control_dt], dtype=np.float32),
    }
    if smoothed is not None and smoothed_target is not None:
        arrays["smoothed_target"] = smoothed_target.astype(np.float32)
        arrays["smoothed_actual_position"] = smoothed.data[
            "actuator_position_rad"
        ].astype(np.float32)
        arrays["smoothed_root_position"] = smoothed.data["root_position_m"].astype(
            np.float32
        )
    np.savez_compressed(artifact_path, **arrays)
    report["artifact"] = str(artifact_path)
    report_path = output / "report_start_000.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/position_target_tracking_diagnostics"),
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
