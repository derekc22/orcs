"""Track the frozen five-knot physical reference with contact-aware WBC."""

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
from orcs.tasks.uolm.whole_body_feedback import (
    WholeBodyFeedbackConfig,
    WholeBodyFeedbackResult,
    _execute_prefix,
    track_with_whole_body_feedback,
)

_OPEN_LOOP = {
    "first_invalid_frame": 134,
    "invalid_reason": "torso_tilt",
    "first_right_foot_loss_frame": 103,
    "first_left_foot_loss_frame": 128,
    "both_feet_off_duration_s": 0.170,
    "root_height_frame120_m": 0.616,
    "vertical_momentum_frame120_kg_m_s": -5.81,
    "torso_tilt_frame120_deg": 29.4,
    "maximum_joint_position_error_rad": 0.652,
    "maximum_joint_velocity_error_rad_s": 31.22,
    "maximum_balance_torque_usage": 0.412,
    "torque_limit_violations": 0,
    "first_wrist_contact_frame": 59,
}

_RIGID_FOOT_WBC = {
    "first_invalid_frame": 118,
    "invalid_reason": "torso_tilt",
    "first_solver_infeasibility_frame": 87.75,
    "solver_failure_count": 9,
    "first_right_foot_loss_frame": 109,
    "first_left_foot_loss_frame": 107,
    "both_feet_off_duration_s": 0.085,
    "root_height_frame120_m": 0.505,
    "vertical_momentum_frame120_kg_m_s": -2.98,
    "torso_tilt_frame120_deg": 58.1,
    "maximum_joint_position_error_rad": 1.765,
    "maximum_joint_velocity_error_rad_s": 17.69,
    "maximum_balance_torque_usage": 1.0,
    "first_wrist_contact_frame": 59,
}


def _first_sustained_loss(contact: np.ndarray, side: int, count: int = 4) -> int | None:
    absent = ~contact[:, side]
    if len(absent) < count:
        return None
    run = np.convolve(absent.astype(np.int32), np.ones(count, dtype=np.int32), "valid")
    found = np.flatnonzero(run == count)
    return None if not len(found) else int(found[0])


def _first_true(value: np.ndarray) -> int | None:
    found = np.flatnonzero(value)
    return None if not len(found) else int(found[0])


def _frame(config: WholeBodyFeedbackConfig, sample: int | None) -> int | None:
    return None if sample is None else config.anchor_frame + int(np.ceil(sample / 4))


def _comparison(
    result: WholeBodyFeedbackResult, config: WholeBodyFeedbackConfig
) -> dict[str, Any]:
    data = result.data
    right_loss = _first_sustained_loss(data["foot_contact"], 1)
    left_loss = _first_sustained_loss(data["foot_contact"], 0)
    both_off = ~np.any(data["foot_contact"], axis=1)
    first_wrist = _first_true(data["wrist_terrain"])
    balance_names = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    actuator_names = list(data["actuator_names"])
    balance = np.asarray([actuator_names.index(name) for name in balance_names])
    stop = (
        len(both_off)
        if result.first_invalid_sample is None
        else result.first_invalid_sample + 1
    )
    both_off_duration = float(np.sum(both_off[:stop]) * 0.005)
    return {
        "first_invalid_frame": result.first_invalid_frame,
        "invalid_reason": result.invalid_reason,
        "first_right_foot_loss_frame": _frame(config, right_loss),
        "first_left_foot_loss_frame": _frame(config, left_loss),
        "both_feet_off_duration_s": both_off_duration,
        "root_height_frame120_m": result.summary["root_height_frame120_m"],
        "vertical_momentum_frame120_kg_m_s": result.summary[
            "vertical_momentum_frame120_kg_m_s"
        ],
        "torso_tilt_frame120_deg": result.summary["torso_tilt_frame120_deg"],
        "maximum_joint_position_error_rad": float(
            np.max(np.abs(data["joint_position_error"][:stop]))
        ),
        "maximum_joint_velocity_error_rad_s": float(
            np.max(np.abs(data["joint_velocity_error"][:stop]))
        ),
        "maximum_balance_torque_usage": float(
            np.max(data["torque_usage"][:stop, balance])
        ),
        "torque_limit_violations": result.summary["requested_torque_limit_violations"],
        "torque_limit_active_samples": result.summary["torque_limit_active_samples"],
        # The prefix is bit-identical and already contains the known frame-59
        # wrist event.  Keep the required global metric and expose the new
        # segment event separately.
        "first_wrist_contact_frame": 59,
        "first_wrist_contact_frame_in_controlled_segment": _frame(config, first_wrist),
    }


def _support_timeline(
    result: WholeBodyFeedbackResult,
    actuator_names: list[str],
    config: WholeBodyFeedbackConfig,
) -> dict[str, Any]:
    data = result.data
    timeline: dict[str, Any] = {}
    for frame in (85, 95, 100, 103, 110, 120, 128, 135, 140, 150, 160):
        sample = (frame - config.anchor_frame) * 4
        if not 0 <= sample < len(data["root_position"]):
            continue
        torque = data["desired_torque_nm"][sample]
        order = np.argsort(np.abs(torque))[::-1][:5]
        timeline[str(frame)] = {
            "root_height_m": float(data["root_position"][sample, 2]),
            "desired_com_height_m": float(
                data["actual_com"][sample, 2] + data["com_error"][sample, 2]
            ),
            "actual_com_height_m": float(data["actual_com"][sample, 2]),
            "actual_com_vertical_velocity_m_s": float(
                data["actual_com_velocity"][sample, 2]
            ),
            "desired_com_vertical_velocity_m_s": float(
                data["actual_com_velocity"][sample, 2]
                + data["com_velocity_error"][sample, 2]
            ),
            "torso_roll_pitch_error_deg": np.rad2deg(
                data["torso_error"][sample, :2]
            ).tolist(),
            "foot_contact_left_right": data["foot_contact"][sample].tolist(),
            "controller_active_contact_left_right": data["active_contact"][
                sample
            ].tolist(),
            "foot_normal_force_left_right_n": data["foot_normal_force_n"][
                sample
            ].tolist(),
            "foot_slip_left_right_m_s": data["foot_slip_m_s"][sample].tolist(),
            "active_contact_points_left_right": data["active_contact_point_count"][
                sample
            ].tolist(),
            "contact_normal_velocity_left_right_m_s": data[
                "contact_normal_velocity_m_s"
            ][sample].tolist(),
            "contact_normal_acceleration_residual_left_right_m_s2": data[
                "contact_normal_acceleration_residual_m_s2"
            ][sample].tolist(),
            "contact_tangential_speed_left_right_m_s": data[
                "contact_tangential_speed_m_s"
            ][sample].tolist(),
            "dominant_requested_torques_nm": {
                actuator_names[index]: float(torque[index]) for index in order
            },
            "minimum_torque_margin_fraction": float(
                1.0 - np.max(data["torque_usage"][sample])
            ),
            "solver_success": bool(data["solver_success"][sample]),
        }
    return timeline


def _channel_usage(
    result: WholeBodyFeedbackResult, actuator_names: list[str]
) -> dict[str, Any]:
    data = result.data
    window = slice(10 * 4, 50 * 4 + 1)  # global frames 95..135
    usage = np.max(data["torque_usage"][window], axis=0)
    requested = np.sqrt(np.mean(np.square(data["desired_torque_nm"][window]), axis=0))
    order = np.argsort(requested)[::-1]
    return {
        "dominant_requested_torque_rms_nm": {
            actuator_names[index]: float(requested[index]) for index in order[:10]
        },
        "dominant_torque_usage": {
            actuator_names[index]: float(usage[index])
            for index in np.argsort(usage)[::-1][:10]
        },
    }


def _open_loop_contact_durations(
    engine: _DynamicRollout,
    target: np.ndarray,
    config: WholeBodyFeedbackConfig,
) -> dict[str, float]:
    """Measure baseline arm contacts over the identical anchored interval."""
    data, _, _ = _execute_prefix(engine, target, config.anchor_frame)
    arm_bodies = {
        body
        for body in engine.robot_bodies
        if any(
            token in engine._name(mujoco.mjtObj.mjOBJ_BODY, body)
            for token in ("shoulder", "elbow", "wrist", "hand")
        )
    }
    arm_terrain = 0
    arm_object = 0

    def record() -> None:
        nonlocal arm_terrain, arm_object
        terrain = False
        object_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if {first, second} & engine.terrain_geoms:
                other = second if first in engine.terrain_geoms else first
                terrain |= int(engine.model.geom_bodyid[other]) in arm_bodies
            if {first, second} & engine.object_geoms:
                other = second if first in engine.object_geoms else first
                object_contact |= int(engine.model.geom_bodyid[other]) in arm_bodies
        arm_terrain += int(terrain)
        arm_object += int(object_contact)

    record()
    for frame in range(config.anchor_frame + 1, config.evaluation_end_frame + 1):
        data.ctrl[:] = np.clip(
            target[frame, engine.actuator_joint_indexes],
            engine.model.actuator_ctrlrange[:, 0],
            engine.model.actuator_ctrlrange[:, 1],
        )
        for _ in range(engine.config.substeps):
            mujoco.mj_step(engine.model, data)
            record()
    return {
        "arm_terrain_contact_duration_s": arm_terrain * engine.config.physics_dt,
        "arm_object_contact_duration_s": arm_object * engine.config.physics_dt,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact.resolve()
    with np.load(artifact_path) as source:
        artifact = {key: source[key].copy() for key in source.files}
    trajectory = artifact["analytical_through140_trajectory"].copy()
    nominal_target = artifact["optimized_joint_target"].astype(np.float64)
    device = torch.device(args.device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        torch.as_tensor(trajectory, device=device)
    )
    model = _build_model()
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    engine = _DynamicRollout(model, reference, dynamics)
    config = WholeBodyFeedbackConfig()
    baseline_contacts = _open_loop_contact_durations(engine, nominal_target, config)
    result = track_with_whole_body_feedback(engine, nominal_target, artifact, config)
    actuator_names = [
        engine.joint_names[index] for index in engine.actuator_joint_indexes
    ]
    result.data["actuator_names"] = np.asarray(actuator_names)
    comparison = {
        "open_loop_target_replay": _OPEN_LOOP,
        "rigid_foot_wbc": _RIGID_FOOT_WBC,
        "compliant_contact_wbc": _comparison(result, config),
    }
    wbc_arm_terrain_duration = float(
        np.sum(result.data["arm_terrain_contact"]) * dynamics.physics_dt
    )
    wbc_arm_object_duration = float(
        np.sum(result.data["arm_object_contact"]) * dynamics.physics_dt
    )
    new_arm_support = bool(
        wbc_arm_terrain_duration
        > baseline_contacts["arm_terrain_contact_duration_s"]
        + dynamics.sustained_nonfoot_terrain_contact_s
        or wbc_arm_object_duration
        > baseline_contacts["arm_object_contact_duration_s"]
        + dynamics.sustained_nonfoot_terrain_contact_s
    )
    large_hard_residual = result.summary["first_large_hard_residual_frame"]
    if (
        result.valid
        and result.recoverable_terminal
        and not new_arm_support
        and large_hard_residual is None
    ):
        classification = "A"
        basis = "valid through frame 160 with terminal foot support and recoverable momentum"
        # Deterministic reproduction from the identical anchor is a hard success test.
        repeat = track_with_whole_body_feedback(
            engine, nominal_target, artifact, config
        )
        reproducibility = {
            "run": True,
            "first_invalid_frame": repeat.first_invalid_frame,
            "recoverable_terminal": repeat.recoverable_terminal,
            "maximum_root_position_difference_m": float(
                np.max(
                    np.abs(result.data["root_position"] - repeat.data["root_position"])
                )
            ),
            "identical_contact_state": bool(
                np.array_equal(result.data["foot_contact"], repeat.data["foot_contact"])
            ),
        }
        if not repeat.valid or not repeat.recoverable_terminal:
            classification = "E"
            basis = "the nominally successful feedback rollout was not reproducible"
    else:
        reproducibility = {
            "run": False,
            "reason": "controller did not pass success gate",
        }
        if large_hard_residual is not None:
            classification = "D"
            basis = (
                "actual-point normal/dynamics constraints remain numerically "
                f"infeasible from frame {large_hard_residual:.2f}"
            )
        elif (
            result.first_invalid_frame is not None and result.first_invalid_frame >= 145
        ):
            classification = "B"
            basis = "feedback materially delayed physical invalidation beyond frame 134"
        elif (
            comparison["compliant_contact_wbc"]["first_left_foot_loss_frame"] is None
            or comparison["compliant_contact_wbc"]["first_left_foot_loss_frame"] >= 136
        ):
            classification = "B"
            basis = "feedback clearly improved stance-contact retention"
        elif result.summary["torque_limit_active_samples"] > 0:
            classification = "C"
            basis = (
                "contact optimization remains feasible, but physical support still "
                "fails without improving the segment"
            )
        else:
            classification = "C"
            basis = "failure remains near the open-loop result with the same support-loss mechanism"

    report: dict[str, Any] = {
        "experiment": "compliant_contact_point_whole_body_feedback_tracking",
        "source_artifact": str(artifact_path),
        "frozen_reference": True,
        "controller": {
            "frequency_hz": 1.0 / dynamics.physics_dt,
            "formulation": (
                "constrained inverse dynamics over robot qdd, all 29 actuator torques, "
                "and nonnegative friction-ray forces at actual MuJoCo contact points"
            ),
            "state_feedback": [
                "COM position/velocity",
                "torso orientation/angular velocity",
                "centroidal angular momentum",
                "joint q/qdot",
                "actual foot contact, clearance, force, and slip",
            ],
            "contact_handling": (
                "actual MuJoCo points only; hard normal non-separation with compliant "
                "acceleration tolerance; tangential acceleration/slip and foot "
                "roll/pitch are soft; mu=0.60 friction pyramid"
            ),
            "actuator_interface": (
                "tau_des converted to q_target=q+(tau_des+Kd*qdot)/Kp; original "
                "MuJoCo gains and force limits remain compiled and unchanged"
            ),
            "solver": "SciPy SLSQP convex quadratic objective with linear rigid-body dynamics equality and box constraints",
            "optimization_dimension_min_max": result.summary[
                "optimization_dimension_min_max"
            ],
            "average_solve_time_s": result.summary["solver_average_time_s"],
            "worst_solve_time_s": result.summary["solver_worst_time_s"],
            "solver_failures": result.summary["solver_failures"],
            "maximum_dynamics_equality_residual": result.summary[
                "maximum_dynamics_equality_residual"
            ],
            "maximum_normal_constraint_violation": result.summary[
                "maximum_normal_constraint_violation"
            ],
            "maximum_hard_constraint_residual": result.summary[
                "maximum_hard_constraint_residual"
            ],
            "first_large_hard_residual_frame": result.summary[
                "first_large_hard_residual_frame"
            ],
            "contact_tolerances_and_weights": {
                "normal_acceleration_tolerance_m_s2": config.normal_acceleration_tolerance_m_s2,
                "normal_tracking_weight": config.normal_tracking_weight,
                "tangential_velocity_gain": config.tangential_velocity_gain,
                "tangential_acceleration_weight": config.tangential_acceleration_weight,
                "stance_rotation_weight": config.stance_rotation_weight,
            },
            "gains": {
                name: getattr(config, name)
                for name in (
                    "com_kp",
                    "com_kd",
                    "torso_kp",
                    "torso_kd",
                    "momentum_gain",
                    "joint_kp",
                    "joint_kd",
                    "stance_position_kp",
                    "stance_position_kd",
                    "stance_orientation_kp",
                    "stance_orientation_kd",
                )
            },
        },
        "anchor": result.prefix,
        "comparison": comparison,
        "tracking": {
            key: result.summary[key]
            for key in (
                "maximum_joint_position_error_rad",
                "maximum_joint_velocity_error_rad_s",
                "root_position_rms_error_m",
                "root_height_rms_error_m",
                "com_position_rms_error_m",
                "com_velocity_rms_error_m_s",
                "torso_orientation_rms_error_rad",
            )
        },
        "support_timeline_frames85_160": _support_timeline(
            result, actuator_names, config
        ),
        "feedback_channels_frames95_135": _channel_usage(result, actuator_names),
        "actuation": {
            key: result.summary[key]
            for key in (
                "maximum_torque_usage",
                "maximum_actual_torque_usage",
                "torque_limit_active_samples",
                "requested_torque_limit_violations",
            )
        },
        "contact_validity": {
            "new_sustained_hand_or_arm_support": new_arm_support,
            "open_loop_segment": baseline_contacts,
            "wbc_arm_terrain_contact_duration_s": wbc_arm_terrain_duration,
            "wrist_terrain_duration_segment_s": float(
                np.sum(result.data["wrist_terrain"]) * dynamics.physics_dt
            ),
            "wbc_arm_object_contact_duration_s": wbc_arm_object_duration,
        },
        "reproducibility": reproducibility,
        "classification": classification,
        "classification_basis": basis,
    }
    numeric_path = output / "compliant_contact_wbc_start_000.npz"
    np.savez_compressed(
        numeric_path,
        **{
            key: value
            for key, value in result.data.items()
            if isinstance(value, np.ndarray)
        },
        reference_joint_position=artifact["actual_joint_position_rad"][340:641],
        reference_joint_velocity=artifact["actual_joint_velocity_rad_s"][340:641],
        reference_root_position=artifact["actual_root_position_m"][340:641],
        reference_com_position=artifact["actual_com_position_m"][340:641],
        reference_com_velocity=artifact["actual_com_velocity_m_s"][340:641],
        physics_dt=np.asarray([dynamics.physics_dt]),
        anchor_frame=np.asarray([config.anchor_frame]),
        evaluation_end_frame=np.asarray([config.evaluation_end_frame]),
    )
    report["artifact"] = str(numeric_path)
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
        "--output", type=Path, default=Path("logs/compliant_contact_wbc_tracking")
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
