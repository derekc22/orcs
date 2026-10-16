"""Direct simulator-in-the-loop balance-target refinement for frames 85-140."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.contact_dynamics_refinement import (
    ContactDynamicsConfig,
    ContactDynamicsResult,
    refine_contact_dynamics,
)
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    RolloutResult,
    _DynamicRollout,
    _np,
)
from orcs.tasks.uolm.generated_reference import (
    GeneratedReference,
    GeneratedReferenceAdapter,
)
from orcs.tasks.uolm.reference_repair import _quat_to_columns6d
from orcs.tasks.uolm.support_diagnostics import (
    SupportDiagnosticResult,
    instrument_support_rollout,
)


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_sustained(mask: np.ndarray, start: int, samples: int = 4) -> int | None:
    runs = np.convolve(mask[start:].astype(np.int32), np.ones(samples), "valid")
    indexes = np.flatnonzero(runs == samples)
    return None if not len(indexes) else int(start + indexes[0])


def _first_loss(
    result: SupportDiagnosticResult, side: int, start: int = 85
) -> int | None:
    sample = _first_sustained(~result.data["foot_contact"][:, side], start * 4)
    return None if sample is None else int(result.data["control_frame"][sample])


def _first_wrist(result: SupportDiagnosticResult) -> int | None:
    indexes = np.flatnonzero(result.data["wrist_terrain_contact"])
    return None if not len(indexes) else int(result.data["control_frame"][indexes[0]])


def _comparison_row(
    rollout: RolloutResult,
    diagnostic: SupportDiagnosticResult,
    *,
    start: int = 85,
    end: int = 140,
) -> dict[str, Any]:
    data = diagnostic.data
    invalid = rollout.first_invalid_frame
    last_frame = len(rollout.root_pos) - 1 if invalid is None else invalid
    last_sample = min(last_frame * 4, len(data["time_s"]) - 1)
    compare = slice(start * 4, end * 4 + 1)
    desired_momentum = (
        data["robot_mass_kg"][0] * data["reference_com_velocity_m_s"][compare, 2]
    )
    actual_momentum = data["vertical_momentum_kg_m_s"][compare]
    both_off = ~np.any(data["foot_contact"][: last_sample + 1], axis=1)
    frame100 = 100 * 4
    frame120 = 120 * 4
    actuator_names = list(data["actuator_names"])
    balance_indexes = [
        actuator_names.index(name) for name in data["balance_joint_names"]
    ]
    return {
        "first_invalid_frame": invalid,
        "invalid_reason": rollout.invalid_reason,
        "first_left_foot_loss_frame_after85": _first_loss(diagnostic, 0),
        "first_right_foot_loss_frame_after85": _first_loss(diagnostic, 1),
        "both_feet_off_duration_before_invalidation_s": float(np.sum(both_off) * 0.005),
        "first_wrist_contact_frame": _first_wrist(diagnostic),
        "root_height_frame100_m": float(data["root_position_m"][frame100, 2]),
        "vertical_momentum_frame100_kg_m_s": float(
            data["vertical_momentum_kg_m_s"][frame100]
        ),
        "root_height_frame120_m": float(data["root_position_m"][frame120, 2]),
        "vertical_momentum_frame120_kg_m_s": float(
            data["vertical_momentum_kg_m_s"][frame120]
        ),
        "torso_tilt_frame120_deg": float(np.rad2deg(rollout.torso_tilt_rad[120])),
        "maximum_torso_tilt_before_invalidation_deg": float(
            np.rad2deg(np.max(rollout.torso_tilt_rad[: last_frame + 1]))
        ),
        "com_vertical_momentum_rms_error_frames85_140_kg_m_s": float(
            np.sqrt(np.mean(np.square(actual_momentum - desired_momentum)))
        ),
        "maximum_actuator_usage_before_invalidation": float(
            np.max(data["normalized_torque_usage"][: last_sample + 1])
        ),
        "maximum_balance_joint_usage_before_invalidation": float(
            np.max(data["normalized_torque_usage"][: last_sample + 1, balance_indexes])
        ),
        "maximum_joint_tracking_error_before_invalidation_rad": float(
            np.max(np.abs(data["actuator_position_error_rad"][: last_sample + 1]))
        ),
    }


def _physical_timeline(
    result: SupportDiagnosticResult,
    direct: ContactDynamicsResult,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = result.data
    mass = float(data["robot_mass_kg"][0])
    residual = direct.best.correction
    balance_names = [engine.joint_names[index] for index in engine.balance_indexes]
    actuator_names = list(data["actuator_names"])
    timeline: dict[str, Any] = {}
    for frame in (85, 90, 95, 100, 105, 110, 115, 120):
        sample = frame * 4
        residual_frame = residual[frame]
        residual_order = np.argsort(np.abs(residual_frame))[::-1][:3]
        torque = data["actuator_torque_nm"][sample]
        torque_order = np.argsort(np.abs(torque))[::-1][:3]
        error = data["actuator_position_error_rad"][sample]
        error_order = np.argsort(np.abs(error))[::-1][:3]
        timeline[str(frame)] = {
            "foot_contact_left_right": data["foot_contact"][sample].tolist(),
            "foot_normal_force_left_right_n": data["foot_normal_force_n"][
                sample
            ].tolist(),
            "root_height_m": float(data["root_position_m"][sample, 2]),
            "vertical_momentum_kg_m_s": float(data["vertical_momentum_kg_m_s"][sample]),
            "vertical_velocity_m_s": float(data["com_velocity_m_s"][sample, 2]),
            "desired_vertical_momentum_kg_m_s": float(
                mass * data["reference_com_velocity_m_s"][sample, 2]
            ),
            "torso_tilt_deg": float(np.rad2deg(result.rollout.torso_tilt_rad[frame])),
            "foot_slip_left_right_m_s": data["foot_slip_velocity_m_s"][sample].tolist(),
            "actual_cop_margin_m": (
                None
                if not np.isfinite(data["support_polygon_margin_cop_m"][sample])
                else float(data["support_polygon_margin_cop_m"][sample])
            ),
            "largest_target_residuals_rad": {
                balance_names[index]: float(residual_frame[index])
                for index in residual_order
            },
            "largest_actuator_torques_nm": {
                actuator_names[index]: float(torque[index]) for index in torque_order
            },
            "largest_position_errors_rad": {
                actuator_names[index]: float(error[index]) for index in error_order
            },
        }
    window = slice(85 * 4, 121 * 4)
    usage = np.max(data["normalized_torque_usage"][window], axis=0)
    error = np.max(np.abs(data["actuator_position_error_rad"][window]), axis=0)
    residual_rms = np.sqrt(np.mean(np.square(residual[85:141]), axis=0))
    start = 85 * 4
    right_loss = _first_sustained(~data["foot_contact"][:, 1], start)
    right_recovery = None
    if right_loss is not None:
        right_recovery = _first_sustained(data["foot_contact"][:, 1], right_loss + 4)
    left_loss = _first_sustained(~data["foot_contact"][:, 0], start)
    limit = (
        len(data["time_s"])
        if result.rollout.first_invalid_frame is None
        else min(len(data["time_s"]), result.rollout.first_invalid_frame * 4 + 1)
    )
    single_left = data["foot_contact"][:limit, 0] & ~data["foot_contact"][:limit, 1]
    return {
        "frames": timeline,
        "support_summary": {
            "first_right_foot_loss_frame": (
                None if right_loss is None else int(data["control_frame"][right_loss])
            ),
            "first_right_foot_recovery_frame": (
                None
                if right_recovery is None
                else int(data["control_frame"][right_recovery])
            ),
            "first_left_foot_loss_frame": (
                None if left_loss is None else int(data["control_frame"][left_loss])
            ),
            "single_left_support_duration_before_invalidation_s": float(
                np.sum(single_left) * 0.005
            ),
            "both_feet_off_duration_before_invalidation_s": float(
                np.sum(~np.any(data["foot_contact"][:limit], axis=1)) * 0.005
            ),
        },
        "dominant_residual_joints": {
            balance_names[index]: float(residual_rms[index])
            for index in np.argsort(residual_rms)[::-1][:5]
        },
        "dominant_actuator_usage": {
            actuator_names[index]: float(usage[index])
            for index in np.argsort(usage)[::-1][:5]
        },
        "dominant_tracking_error_rad": {
            actuator_names[index]: float(error[index])
            for index in np.argsort(error)[::-1][:5]
        },
    }


def _continuity(
    result: SupportDiagnosticResult,
    reference: GeneratedReference,
    engine: _DynamicRollout,
    entry: int,
    exit_frame: int,
) -> dict[str, Any]:
    data = result.data
    reference_root = _np(reference.root_pos_local[0])
    reference_root_velocity = _np(reference.root_lin_vel_w[0])
    reference_joint = _np(reference.joint_pos[0])
    reference_joint_velocity = _np(reference.joint_vel[0])
    values: dict[str, Any] = {}
    for label, frame in (("entry", entry), ("exit", exit_frame)):
        sample = frame * 4
        joint = reference_joint[frame].copy()
        joint_velocity = reference_joint_velocity[frame].copy()
        joint[engine.actuator_joint_indexes] = data["actuator_position_rad"][sample]
        joint_velocity[engine.actuator_joint_indexes] = data["actuator_velocity_rad_s"][
            sample
        ]
        values[label] = {
            "frame": frame,
            "root_position_difference_m": float(
                np.linalg.norm(data["root_position_m"][sample] - reference_root[frame])
            ),
            "root_velocity_difference_m_s": float(
                np.linalg.norm(
                    data["root_linear_velocity_m_s"][sample]
                    - reference_root_velocity[frame]
                )
            ),
            "joint_position_rms_difference_rad": float(
                np.sqrt(np.mean(np.square(joint - reference_joint[frame])))
            ),
            "joint_position_max_difference_rad": float(
                np.max(np.abs(joint - reference_joint[frame]))
            ),
            "joint_velocity_rms_difference_rad_s": float(
                np.sqrt(
                    np.mean(np.square(joint_velocity - reference_joint_velocity[frame]))
                )
            ),
            "joint_velocity_max_difference_rad_s": float(
                np.max(np.abs(joint_velocity - reference_joint_velocity[frame]))
            ),
        }
    return values


def _realized_trajectory(
    source: np.ndarray,
    diagnostic: SupportDiagnosticResult,
    rollout: RolloutResult,
    engine: _DynamicRollout,
    start: int,
    end: int,
) -> np.ndarray:
    realized = source.copy()
    data = diagnostic.data
    for frame in range(start, end + 1):
        sample = frame * 4
        realized[0, frame, :3] = data["root_position_m"][sample]
        realized[0, frame, 3:9] = _quat_to_columns6d(
            data["root_quaternion_wxyz"][sample]
        )
        joint = _np(engine.reference.joint_pos[0, frame]).copy()
        joint[engine.actuator_joint_indexes] = data["actuator_position_rad"][sample]
        realized[0, frame, 9:38] = joint
        realized[0, frame, 38:41] = rollout.object_pos[frame]
        realized[0, frame, 41:47] = _quat_to_columns6d(rollout.object_quat_wxyz[frame])
    return realized


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.through95_artifact.resolve()) as artifact:
        through95_np = artifact["extended_centroidal_repaired_trajectory"].copy()
        through95_target = artifact["balance_joint_target"].astype(np.float64)
    with np.load(args.through140_artifact.resolve()) as artifact:
        through140_np = artifact["through140_centroidal_repaired_trajectory"].copy()
        through140_target = artifact["balance_joint_target"].astype(np.float64)
    previous_target: np.ndarray | None = None
    warm_start_parameters: np.ndarray | None = None
    if args.warm_start_artifact is not None:
        with np.load(args.warm_start_artifact.resolve()) as artifact:
            previous_target = artifact["optimized_joint_target"].astype(np.float64)
            previous_parameters = artifact["optimizer_parameters"].astype(np.float64)
        previous_knots = previous_parameters.reshape(4, 15)
        interpolation = (117 - 110) / (125 - 110)
        warm_start_parameters = np.vstack(
            [
                previous_knots[0],
                previous_knots[1],
                (1.0 - interpolation) * previous_knots[1]
                + interpolation * previous_knots[2],
                previous_knots[2],
                previous_knots[3],
            ]
        ).reshape(-1)

    device = torch.device(args.device)
    adapter = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)
    through95_tensor = torch.as_tensor(through95_np, device=device)
    through140_tensor = torch.as_tensor(through140_np, device=device)
    through95_reference = adapter(through95_tensor)
    through140_reference = adapter(through140_tensor)
    model = _build_model()
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    knot_frames = (
        tuple(args.knot_frames)
        if args.knot_frames is not None
        else ContactDynamicsConfig().knot_frames
    )
    config = ContactDynamicsConfig(
        knot_frames=knot_frames,
        population=args.population,
        elite_count=min(args.elite_count, args.population),
        generations=args.generations,
        seed=args.seed,
    )
    direct = refine_contact_dynamics(
        model,
        through140_reference,
        through95_target,
        through140_target,
        dynamics,
        config,
        warm_start_parameters=warm_start_parameters,
    )

    through95_engine = _DynamicRollout(model, through95_reference, dynamics)
    through140_engine = _DynamicRollout(model, through140_reference, dynamics)
    through95_rollout = through95_engine.rollout(through95_target)
    analytical_rollout = through140_engine.rollout(through140_target)
    direct_rollout = through140_engine.rollout(direct.best.target)
    cases = {
        "through95": (
            through95_reference,
            through95_target,
            through95_rollout,
        ),
        "analytical_through140": (
            through140_reference,
            through140_target,
            analytical_rollout,
        ),
        "direct_dynamics": (
            through140_reference,
            direct.best.target,
            direct_rollout,
        ),
    }
    if previous_target is not None:
        cases["previous_4knot_dynamics"] = (
            through140_reference,
            previous_target,
            through140_engine.rollout(previous_target),
        )
    diagnostics: dict[str, SupportDiagnosticResult] = {}
    comparison: dict[str, Any] = {}
    for name, (reference, target, rollout) in cases.items():
        diagnostic = instrument_support_rollout(
            model,
            reference,
            target,
            label=name,
            config=dynamics,
            end_frame=299,
        )
        diagnostics[name] = diagnostic
        comparison[name] = _comparison_row(rollout, diagnostic)

    direct_diagnostic = diagnostics["direct_dynamics"]
    physical_timeline = _physical_timeline(direct_diagnostic, direct, through140_engine)
    new_knot_behavior = None
    if direct.warm_start is not None:
        balance_names = [
            through140_engine.joint_names[index]
            for index in through140_engine.balance_indexes
        ]
        new_knot_behavior = {
            "residuals_at_knots_rad": {
                str(frame): {
                    name: float(direct.best.correction[frame, index])
                    for index, name in enumerate(balance_names)
                }
                for frame in (110, 117, 125)
            },
            "frame117_interpolated_warm_start_rad": {
                name: float(direct.warm_start.correction[117, index])
                for index, name in enumerate(balance_names)
            },
            "frame117_change_from_interpolated_warm_start_rad": {
                name: float(
                    direct.best.correction[117, index]
                    - direct.warm_start.correction[117, index]
                )
                for index, name in enumerate(balance_names)
            },
        }
    continuity = _continuity(
        direct_diagnostic,
        through140_reference,
        through140_engine,
        config.anchor_frame,
        config.evaluation_end_frame,
    )
    realized = _realized_trajectory(
        through140_np,
        direct_diagnostic,
        direct_rollout,
        through140_engine,
        config.anchor_frame,
        config.evaluation_end_frame,
    )

    failure = direct_rollout.first_invalid_frame
    if (
        direct.best.valid
        and direct.best.recoverable_terminal
        and (failure is None or failure > config.evaluation_end_frame)
    ):
        classification = "A"
        basis = (
            "physical candidate survives through frame 160 and exits with support, "
            "bounded downward momentum, and bounded torso state"
        )
    elif previous_target is not None:
        previous_failure = cases["previous_4knot_dynamics"][2].first_invalid_frame
        previous_left_loss = comparison["previous_4knot_dynamics"][
            "first_left_foot_loss_frame_after85"
        ]
        new_left_loss = comparison["direct_dynamics"][
            "first_left_foot_loss_frame_after85"
        ]
        if (
            failure is None
            or (previous_failure is not None and failure >= previous_failure + 5)
            or (
                previous_left_loss is not None
                and new_left_loss is not None
                and new_left_loss >= previous_left_loss + 5
            )
        ):
            classification = "B"
            basis = (
                f"five-knot failure/support retention materially improves on "
                f"the four-knot result ({previous_failure} to {failure})"
            )
        elif (
            failure is not None
            and previous_failure is not None
            and failure < previous_failure - 2
        ):
            classification = "D"
            basis = f"five-knot failure regressed from {previous_failure} to {failure}"
        else:
            classification = "C"
            basis = (
                f"five-knot failure remains near the four-knot result "
                f"({previous_failure} versus {failure}) with the same support loss"
            )
    elif failure is None or failure > 122:
        classification = "B"
        basis = f"physical failure delayed from frame 103 to {failure}"
    else:
        search_frames = [
            item["first_invalid_frame"]
            for item in direct.history
            if item["first_invalid_frame"] is not None
        ]
        classification = "C"
        basis = (
            "bounded direct search could not retain support beyond the prior "
            f"frame-122 baseline; best failure is {failure}; generation range "
            f"was {min(search_frames, default=None)}-{max(search_frames, default=None)}"
        )

    report: dict[str, Any] = {
        "experiment": "direct_contact_dynamics_reference_refinement",
        "dynamic_formulation": {
            "anchor": direct.prefix,
            "optimization_interval_frames_inclusive": [
                config.anchor_frame,
                config.optimize_end_frame,
            ],
            "evaluation_end_frame": config.evaluation_end_frame,
            "decision_variables": (
                "position-target residuals for 12 leg + 3 waist joints; floating "
                "root, COM, feet, and arms are not decision variables"
            ),
            "optimizer": direct.optimizer,
            "optimization_seconds": direct.optimization_seconds,
            "history": direct.history,
            "zero_candidate": direct.zero.metrics,
            "through95_seed_candidate": direct.through95_seed.metrics,
            "warm_start_candidate": (
                None if direct.warm_start is None else direct.warm_start.metrics
            ),
            "best_candidate": direct.best.metrics,
            "controller_or_physics_changes": False,
        },
        "physical_support": physical_timeline,
        "new_knot_behavior": new_knot_behavior,
        "comparison": comparison,
        "realized_reference": {
            "segment_frames_inclusive": [
                config.anchor_frame,
                config.evaluation_end_frame,
            ],
            "definition": (
                "root, all joint states, object state, and velocities/contact "
                "diagnostics are taken from the actual MuJoCo rollout; the old "
                "desired root trajectory is not restored"
            ),
            "continuity": continuity,
            "promoted_to_stage_a": classification == "A",
        },
        "classification": classification,
        "classification_basis": basis,
    }

    artifact_path = output / "direct_contact_dynamics_start_000.npz"
    diagnostic_data = direct_diagnostic.data
    np.savez_compressed(
        artifact_path,
        through95_trajectory=through95_np,
        analytical_through140_trajectory=through140_np,
        realized_dynamic_trajectory=realized,
        optimized_joint_target=direct.best.target.astype(np.float32),
        target_residual=direct.best.correction.astype(np.float32),
        optimizer_parameters=direct.parameters.astype(np.float32),
        actual_root_position_m=diagnostic_data["root_position_m"].astype(np.float32),
        actual_root_quaternion_wxyz=diagnostic_data["root_quaternion_wxyz"].astype(
            np.float32
        ),
        actual_root_linear_velocity_m_s=diagnostic_data[
            "root_linear_velocity_m_s"
        ].astype(np.float32),
        actual_root_angular_velocity_rad_s=diagnostic_data[
            "root_angular_velocity_rad_s"
        ].astype(np.float32),
        actual_joint_position_rad=diagnostic_data["actuator_position_rad"].astype(
            np.float32
        ),
        actual_joint_velocity_rad_s=diagnostic_data["actuator_velocity_rad_s"].astype(
            np.float32
        ),
        actual_com_position_m=diagnostic_data["com_position_m"].astype(np.float32),
        actual_com_velocity_m_s=diagnostic_data["com_velocity_m_s"].astype(np.float32),
        actual_centroidal_angular_momentum=diagnostic_data[
            "centroidal_angular_momentum_kg_m2_s"
        ].astype(np.float32),
        actual_foot_contact=diagnostic_data["foot_contact"],
        actual_foot_normal_force_n=diagnostic_data["foot_normal_force_n"].astype(
            np.float32
        ),
        actual_foot_force_world_n=diagnostic_data["foot_force_world_n"].astype(
            np.float32
        ),
        actual_foot_slip_velocity_m_s=diagnostic_data["foot_slip_velocity_m_s"].astype(
            np.float32
        ),
        actual_actuator_torque_nm=diagnostic_data["actuator_torque_nm"].astype(
            np.float32
        ),
        actual_normalized_torque_usage=diagnostic_data[
            "normalized_torque_usage"
        ].astype(np.float32),
        actual_joint_tracking_error_rad=diagnostic_data[
            "actuator_position_error_rad"
        ].astype(np.float32),
        object_position_m=direct_rollout.object_pos.astype(np.float32),
        object_quaternion_wxyz=direct_rollout.object_quat_wxyz.astype(np.float32),
        anchor_frame=np.asarray([config.anchor_frame], dtype=np.int32),
        optimize_end_frame=np.asarray([config.optimize_end_frame], dtype=np.int32),
        evaluation_end_frame=np.asarray([config.evaluation_end_frame], dtype=np.int32),
        dt=np.asarray([0.02], dtype=np.float32),
        physics_dt=np.asarray([0.005], dtype=np.float32),
    )
    report["artifact"] = str(artifact_path)
    report_path = output / "report_start_000.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=545)
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--knot-frames", type=int, nargs="+")
    parser.add_argument("--warm-start-artifact", type=Path)
    parser.add_argument(
        "--through95-artifact",
        type=Path,
        default=Path(
            "logs/extended_centroidal_reference_repair/"
            "extended_centroidal_repaired_reference_start_000.npz"
        ),
    )
    parser.add_argument(
        "--through140-artifact",
        type=Path,
        default=Path(
            "logs/centroidal_reference_repair_through140/"
            "centroidal_repaired_reference_through140_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/direct_contact_dynamics_refinement"),
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
