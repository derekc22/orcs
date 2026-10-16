"""Test proximal-arm momentum freedom on a fixed five-knot balance solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_contact_dynamics import (
    _comparison_row,
    _first_loss,
    _first_wrist,
    _serializable,
)
from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.contact_dynamics_refinement import (
    ContactDynamicsConfig,
    ContactDynamicsResult,
    refine_contact_dynamics,
)
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.support_diagnostics import (
    SupportDiagnosticResult,
    instrument_support_rollout,
)

_PROXIMAL_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


def _actuator_indexes(
    diagnostic: SupportDiagnosticResult, names: tuple[str, ...]
) -> list[int]:
    actuator_names = list(diagnostic.data["actuator_names"])
    return [actuator_names.index(name) for name in names]


def _arm_comparison_row(
    diagnostic: SupportDiagnosticResult,
    engine: _DynamicRollout,
) -> dict[str, Any]:
    data = diagnostic.data
    invalid = diagnostic.rollout.first_invalid_frame
    last_frame = 299 if invalid is None else invalid
    last_sample = min(last_frame * 4, len(data["time_s"]) - 1)
    arm_indexes = _actuator_indexes(diagnostic, _PROXIMAL_ARM_JOINTS)
    balance_names = tuple(engine.joint_names[index] for index in engine.balance_indexes)
    balance_indexes = _actuator_indexes(diagnostic, balance_names)
    frame125 = 125 * 4
    row = _comparison_row(diagnostic.rollout, diagnostic)
    row.update(
        {
            "left_foot_slip_frame125_m_s": float(
                data["foot_slip_velocity_m_s"][frame125, 0]
            ),
            "left_foot_normal_force_frame125_n": float(
                data["foot_normal_force_n"][frame125, 0]
            ),
            "maximum_arm_joint_usage_before_invalidation": float(
                np.max(data["normalized_torque_usage"][: last_sample + 1, arm_indexes])
            ),
            "maximum_balance_joint_usage_before_invalidation": float(
                np.max(
                    data["normalized_torque_usage"][: last_sample + 1, balance_indexes]
                )
            ),
            "maximum_arm_tracking_error_before_invalidation_rad": float(
                np.max(
                    np.abs(
                        data["actuator_position_error_rad"][
                            : last_sample + 1, arm_indexes
                        ]
                    )
                )
            ),
            "hand_object_contact_left_right_s": [
                diagnostic.rollout.metrics["left_hand_contact_duration_s"],
                diagnostic.rollout.metrics["right_hand_contact_duration_s"],
            ],
            "other_robot_object_contact_duration_s": diagnostic.rollout.metrics[
                "other_robot_object_contact_duration_s"
            ],
            "nonfoot_terrain_contact_duration_s": diagnostic.rollout.metrics[
                "nonfoot_terrain_contact_duration_s"
            ],
        }
    )
    return row


def _support_timeline(result: SupportDiagnosticResult) -> dict[str, Any]:
    data = result.data
    values: dict[str, Any] = {}
    for frame in (100, 105, 110, 115, 120, 125, 128, 130, 134, 140):
        sample = frame * 4
        values[str(frame)] = {
            "foot_contact_left_right": data["foot_contact"][sample].tolist(),
            "foot_normal_force_left_right_n": data["foot_normal_force_n"][
                sample
            ].tolist(),
            "foot_slip_left_right_m_s": data["foot_slip_velocity_m_s"][sample].tolist(),
            "root_height_m": float(data["root_position_m"][sample, 2]),
            "vertical_momentum_kg_m_s": float(data["vertical_momentum_kg_m_s"][sample]),
            "torso_tilt_deg": float(np.rad2deg(result.rollout.torso_tilt_rad[frame])),
            "centroidal_angular_momentum_kg_m2_s": data[
                "centroidal_angular_momentum_kg_m2_s"
            ][sample].tolist(),
            "root_angular_velocity_rad_s": data["root_angular_velocity_rad_s"][
                sample
            ].tolist(),
            "torso_roll_pitch_rad": data["torso_roll_pitch_rad"][sample].tolist(),
        }
    return values


def _momentum_comparison(
    baseline: SupportDiagnosticResult,
    candidate: SupportDiagnosticResult,
) -> dict[str, Any]:
    first = baseline.data
    second = candidate.data
    sample = slice(110 * 4, 131 * 4)
    total_delta = (
        second["centroidal_angular_momentum_kg_m2_s"][sample]
        - first["centroidal_angular_momentum_kg_m2_s"][sample]
    )
    arm_delta = (
        second["arm_subtree_angular_momentum_kg_m2_s"][sample]
        - first["arm_subtree_angular_momentum_kg_m2_s"][sample]
    )
    root_omega_delta = (
        second["root_angular_velocity_rad_s"][sample]
        - first["root_angular_velocity_rad_s"][sample]
    )
    torso_delta = (
        second["torso_roll_pitch_rad"][sample] - first["torso_roll_pitch_rad"][sample]
    )
    selected: dict[str, Any] = {}
    for frame in (110, 120, 125, 130):
        index = frame * 4
        selected[str(frame)] = {
            "baseline_total_h": first["centroidal_angular_momentum_kg_m2_s"][
                index
            ].tolist(),
            "candidate_total_h": second["centroidal_angular_momentum_kg_m2_s"][
                index
            ].tolist(),
            "baseline_arm_subtree_h_left_right": first[
                "arm_subtree_angular_momentum_kg_m2_s"
            ][index].tolist(),
            "candidate_arm_subtree_h_left_right": second[
                "arm_subtree_angular_momentum_kg_m2_s"
            ][index].tolist(),
            "baseline_root_angular_velocity": first["root_angular_velocity_rad_s"][
                index
            ].tolist(),
            "candidate_root_angular_velocity": second["root_angular_velocity_rad_s"][
                index
            ].tolist(),
            "baseline_torso_roll_pitch": first["torso_roll_pitch_rad"][index].tolist(),
            "candidate_torso_roll_pitch": second["torso_roll_pitch_rad"][
                index
            ].tolist(),
        }
    return {
        "frames110_130_rms_total_h_delta_kg_m2_s": float(
            np.sqrt(np.mean(np.square(total_delta)))
        ),
        "frames110_130_rms_roll_pitch_h_delta_kg_m2_s": float(
            np.sqrt(np.mean(np.square(total_delta[:, :2])))
        ),
        "frames110_130_rms_root_angular_velocity_delta_rad_s": float(
            np.sqrt(np.mean(np.square(root_omega_delta)))
        ),
        "frames110_130_rms_torso_roll_pitch_delta_rad": float(
            np.sqrt(np.mean(np.square(torso_delta)))
        ),
        "frames110_130_rms_arm_subtree_h_delta_left_right": [
            float(np.sqrt(np.mean(np.square(arm_delta[:, side])))) for side in range(2)
        ],
        "arm_subtree_note": (
            "approximation using MuJoCo subtree_angmom at each shoulder-pitch "
            "body; each is expressed about that arm subtree's COM"
        ),
        "selected_frames": selected,
    }


def _arm_strategy(
    result: ContactDynamicsResult,
    diagnostic: SupportDiagnosticResult,
) -> dict[str, Any]:
    correction = result.best.correction
    knot_values = {
        str(frame): {
            name: float(correction[frame, index])
            for index, name in enumerate(_PROXIMAL_ARM_JOINTS)
        }
        for frame in (95, 110, 125, 140)
    }
    rms = np.sqrt(np.mean(np.square(correction[85:161]), axis=0))
    arm_indexes = _actuator_indexes(diagnostic, _PROXIMAL_ARM_JOINTS)
    window = slice(100 * 4, 131 * 4)
    velocity = diagnostic.data["actuator_velocity_rad_s"][window][:, arm_indexes]
    usage = diagnostic.data["normalized_torque_usage"][window][:, arm_indexes]
    error = np.abs(
        diagnostic.data["actuator_position_error_rad"][window][:, arm_indexes]
    )
    order = np.argsort(rms)[::-1]
    return {
        "residuals_at_free_knots_rad": knot_values,
        "residual_rms_by_joint_rad": {
            _PROXIMAL_ARM_JOINTS[index]: float(rms[index]) for index in order
        },
        "maximum_abs_velocity_frames100_130_rad_s": {
            name: float(np.max(np.abs(velocity[:, index])))
            for index, name in enumerate(_PROXIMAL_ARM_JOINTS)
        },
        "maximum_usage_frames100_130": {
            name: float(np.max(usage[:, index]))
            for index, name in enumerate(_PROXIMAL_ARM_JOINTS)
        },
        "maximum_tracking_error_frames100_130_rad": {
            name: float(np.max(error[:, index]))
            for index, name in enumerate(_PROXIMAL_ARM_JOINTS)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.baseline_artifact.resolve()) as artifact:
        trajectory_np = artifact["analytical_through140_trajectory"].copy()
        baseline_target = artifact["optimized_joint_target"].astype(np.float64)

    device = torch.device(args.device)
    trajectory = torch.as_tensor(trajectory_np, device=device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        trajectory
    )
    model = _build_model()
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    config = ContactDynamicsConfig(
        knot_frames=(85, 95, 110, 125, 140, 160),
        residual_bound_rad=0.10,
        population=args.population,
        elite_count=min(args.elite_count, args.population),
        generations=args.generations,
        seed=args.seed,
        decision_joint_names=_PROXIMAL_ARM_JOINTS,
    )
    result = refine_contact_dynamics(
        model,
        reference,
        baseline_target,
        baseline_target,
        dynamics,
        config,
    )
    engine = _DynamicRollout(model, reference, dynamics)
    if not np.array_equal(
        result.best.target[:, engine.balance_indexes],
        baseline_target[:, engine.balance_indexes],
    ):
        raise AssertionError("proximal-arm search changed a balance target")
    wrist_names = tuple(name for name in engine.joint_names if "wrist" in name)
    wrist_indexes = np.asarray([engine.joint_names.index(name) for name in wrist_names])
    if not np.array_equal(
        result.best.target[:, wrist_indexes], baseline_target[:, wrist_indexes]
    ):
        raise AssertionError("proximal-arm search changed a wrist target")

    baseline_rollout = engine.rollout(baseline_target)
    candidate_rollout = engine.rollout(result.best.target)
    baseline_diagnostic = instrument_support_rollout(
        model,
        reference,
        baseline_target,
        label="fixed-arm five-knot balance baseline",
        config=dynamics,
        end_frame=299,
    )
    candidate_diagnostic = instrument_support_rollout(
        model,
        reference,
        result.best.target,
        label="proximal-arm residual refinement",
        config=dynamics,
        end_frame=299,
    )
    comparison = {
        "fixed_arms": _arm_comparison_row(baseline_diagnostic, engine),
        "proximal_arm_refinement": _arm_comparison_row(candidate_diagnostic, engine),
    }
    failure = candidate_rollout.first_invalid_frame
    baseline_failure = baseline_rollout.first_invalid_frame
    baseline_left_loss = _first_loss(baseline_diagnostic, 0)
    candidate_left_loss = _first_loss(candidate_diagnostic, 0)
    if (
        result.best.valid
        and result.best.recoverable_terminal
        and (failure is None or failure > config.evaluation_end_frame)
    ):
        classification = "A"
        basis = "proximal-arm candidate survives frame 160 and exits recoverably"
    elif (
        failure is None
        or (
            baseline_failure is not None
            and failure is not None
            and failure >= baseline_failure + 5
        )
        or (
            baseline_left_loss is not None
            and candidate_left_loss is not None
            and candidate_left_loss >= baseline_left_loss + 5
        )
    ):
        classification = "B"
        basis = (
            "proximal-arm freedom materially delays failure or left-foot support loss"
        )
    elif (
        failure is not None
        and baseline_failure is not None
        and failure < baseline_failure - 2
    ):
        classification = "D"
        basis = f"proximal-arm search regresses failure {baseline_failure} to {failure}"
    else:
        classification = "C"
        basis = (
            f"failure remains near the fixed-arm result ({baseline_failure} versus "
            f"{failure}) with the same support-loss mechanism"
        )

    report: dict[str, Any] = {
        "experiment": "fixed_balance_proximal_arm_contact_dynamics_refinement",
        "parameterization": {
            "resolved_joint_names": list(_PROXIMAL_ARM_JOINTS),
            "fixed_zero_frames": [85, 160],
            "free_knot_frames": [95, 110, 125, 140],
            "dimensions": 32,
            "residual_bound_rad": 0.10,
            "lower_body_and_waist_bit_exact": True,
            "wrist_targets_bit_exact": True,
            "optimizer": result.optimizer,
            "optimization_seconds": result.optimization_seconds,
            "generation_history": result.history,
            "zero_arm_residual_baseline": result.zero.metrics,
            "best_candidate": result.best.metrics,
        },
        "comparison": comparison,
        "support_timeline": {
            "fixed_arms": _support_timeline(baseline_diagnostic),
            "proximal_arm_refinement": _support_timeline(candidate_diagnostic),
        },
        "momentum_effect": _momentum_comparison(
            baseline_diagnostic, candidate_diagnostic
        ),
        "arm_strategy": _arm_strategy(result, candidate_diagnostic),
        "contact_validity": {
            "baseline_first_wrist_terrain_contact_frame": _first_wrist(
                baseline_diagnostic
            ),
            "candidate_first_wrist_terrain_contact_frame": _first_wrist(
                candidate_diagnostic
            ),
            "candidate_segment_arm_object_contact_duration_s": result.best.metrics[
                "arm_object_contact_duration_s"
            ],
            "new_sustained_arm_object_contact_rejected": True,
        },
        "classification": classification,
        "classification_basis": basis,
        "promoted_to_stage_a": classification == "A",
    }

    artifact_path = output / "proximal_arm_contact_dynamics_start_000.npz"
    data = candidate_diagnostic.data
    np.savez_compressed(
        artifact_path,
        reference_trajectory=trajectory_np,
        fixed_balance_target=baseline_target.astype(np.float32),
        proximal_arm_target=result.best.target.astype(np.float32),
        proximal_arm_residual=result.best.correction.astype(np.float32),
        optimizer_parameters=result.parameters.astype(np.float32),
        actual_root_position_m=data["root_position_m"].astype(np.float32),
        actual_root_linear_velocity_m_s=data["root_linear_velocity_m_s"].astype(
            np.float32
        ),
        actual_root_angular_velocity_rad_s=data["root_angular_velocity_rad_s"].astype(
            np.float32
        ),
        actual_com_position_m=data["com_position_m"].astype(np.float32),
        actual_com_velocity_m_s=data["com_velocity_m_s"].astype(np.float32),
        actual_centroidal_angular_momentum=data[
            "centroidal_angular_momentum_kg_m2_s"
        ].astype(np.float32),
        approximate_arm_subtree_angular_momentum=data[
            "arm_subtree_angular_momentum_kg_m2_s"
        ].astype(np.float32),
        actual_torso_roll_pitch_rad=data["torso_roll_pitch_rad"].astype(np.float32),
        actual_joint_position_rad=data["actuator_position_rad"].astype(np.float32),
        actual_joint_velocity_rad_s=data["actuator_velocity_rad_s"].astype(np.float32),
        actual_foot_contact=data["foot_contact"],
        actual_foot_normal_force_n=data["foot_normal_force_n"].astype(np.float32),
        actual_foot_slip_velocity_m_s=data["foot_slip_velocity_m_s"].astype(np.float32),
        actual_actuator_torque_nm=data["actuator_torque_nm"].astype(np.float32),
        actual_normalized_torque_usage=data["normalized_torque_usage"].astype(
            np.float32
        ),
        actual_joint_tracking_error_rad=data["actuator_position_error_rad"].astype(
            np.float32
        ),
        physics_dt=np.asarray([0.005], dtype=np.float32),
        control_dt=np.asarray([0.02], dtype=np.float32),
    )
    report["artifact"] = str(artifact_path)
    report_path = output / "report_start_000.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=646)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=Path(
            "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/proximal_arm_contact_dynamics_refinement"),
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
