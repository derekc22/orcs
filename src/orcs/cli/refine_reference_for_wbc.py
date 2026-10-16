"""Refine a compact nominal reference under the frozen compliant-contact WBC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_contact_dynamics import _serializable
from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.wbc_reference_refinement import (
    WBCReferenceEvaluation,
    WBCReferenceRefinementConfig,
    refine_reference_for_wbc,
)


def _comparison(evaluation: WBCReferenceEvaluation) -> dict[str, Any]:
    keys = (
        "first_invalid_frame",
        "invalid_reason",
        "first_left_foot_sustained_loss_frame",
        "first_right_foot_sustained_loss_frame",
        "both_feet_off_duration_before_invalidation_s",
        "root_height_frame100_m",
        "root_height_frame110_m",
        "vertical_momentum_frame110_kg_m_s",
        "torso_tilt_frame110_deg",
        "maximum_joint_position_error_before_invalidation_rad",
        "maximum_joint_velocity_error_before_invalidation_rad_s",
        "maximum_balance_torque_usage_before_invalidation",
        "solver_failure_count",
        "maximum_hard_constraint_residual",
        "arm_terrain_contact_duration_full_horizon_s",
        "arm_object_contact_duration_full_horizon_s",
        "root_position_rms_error_frames85_140_m",
        "root_height_rms_error_frames85_140_m",
        "com_position_rms_error_frames85_140_m",
        "com_velocity_rms_error_frames85_140_m_s",
        "torque_limit_activity_fraction",
    )
    return {key: evaluation.metrics[key] for key in keys}


def _first_recovery(contact: np.ndarray, side: int, after: int) -> int | None:
    found = np.flatnonzero(contact[after:, side])
    return None if not len(found) else int(after + found[0])


def _timeline(
    evaluation: WBCReferenceEvaluation,
    actuator_names: list[str],
    config: WBCReferenceRefinementConfig,
) -> dict[str, Any]:
    data = evaluation.rollout.data
    result: dict[str, Any] = {}
    for frame in (85, 87, 90, 95, 100, 105, 110, 115, 117, 120, 130, 140, 150, 160):
        sample = (frame - config.wbc.anchor_frame) * 4
        torque = data["desired_torque_nm"][sample]
        order = np.argsort(np.abs(torque))[::-1][:5]
        result[str(frame)] = {
            "foot_contact_left_right": data["foot_contact"][sample].tolist(),
            "active_contact_left_right": data["active_contact"][sample].tolist(),
            "foot_normal_force_left_right_n": data["foot_normal_force_n"][sample].tolist(),
            "foot_slip_left_right_m_s": data["foot_slip_m_s"][sample].tolist(),
            "root_height_m": float(data["root_position"][sample, 2]),
            "com_height_m": float(data["actual_com"][sample, 2]),
            "com_vertical_velocity_m_s": float(data["actual_com_velocity"][sample, 2]),
            "torso_tilt_deg": float(np.rad2deg(data["torso_tilt_rad"][sample])),
            "dominant_requested_torques_nm": {
                actuator_names[index]: float(torque[index]) for index in order
            },
            "maximum_torque_usage": float(np.max(data["torque_usage"][sample])),
            "solver_success": bool(data["solver_success"][sample]),
            "hard_constraint_residual": float(data["hard_constraint_residual"][sample]),
        }
    return result


def _reference_change(
    evaluation: WBCReferenceEvaluation,
    engine: _DynamicRollout,
    config: WBCReferenceRefinementConfig,
) -> dict[str, Any]:
    residual = evaluation.metrics["reference_residual"]
    velocity = np.gradient(residual, engine.config.physics_dt, axis=0, edge_order=2)
    names = [engine.joint_names[index] for index in engine.balance_indexes]
    knot_values: dict[str, Any] = {}
    for frame in config.knot_frames:
        sample = (frame - config.wbc.anchor_frame) * engine.config.substeps
        joint = residual[sample, 3:]
        order = np.argsort(np.abs(joint))[::-1][:5]
        knot_values[str(frame)] = {
            "root_z_m": float(residual[sample, 0]),
            "root_roll_rad": float(residual[sample, 1]),
            "root_pitch_rad": float(residual[sample, 2]),
            "dominant_balance_joint_residuals_rad": {
                names[index]: float(joint[index]) for index in order
            },
        }
    joint_rms = np.sqrt(np.mean(np.square(residual[:, 3:]), axis=0))
    root_order = np.argmax(np.abs(residual[:, :3]), axis=0)
    return {
        "knots": knot_values,
        "root_z_rms_m": float(np.sqrt(np.mean(np.square(residual[:, 0])))),
        "root_z_max_m": float(np.max(np.abs(residual[:, 0]))),
        "root_roll_pitch_rms_rad": float(
            np.sqrt(np.mean(np.square(residual[:, 1:3])))
        ),
        "root_roll_pitch_max_rad": float(np.max(np.abs(residual[:, 1:3]))),
        "balance_joint_rms_rad": float(
            np.sqrt(np.mean(np.square(residual[:, 3:])))
        ),
        "balance_joint_max_rad": float(np.max(np.abs(residual[:, 3:]))),
        "maximum_residual_velocity": float(np.max(np.abs(velocity))),
        "dominant_balance_joint_rms_rad": {
            names[index]: float(joint_rms[index])
            for index in np.argsort(joint_rms)[::-1][:8]
        },
        "root_component_peak_frames": {
            name: config.wbc.anchor_frame + float(root_order[index]) / 4.0
            for index, name in enumerate(("z", "roll", "pitch"))
        },
    }


def _classification(
    baseline: WBCReferenceEvaluation, best: WBCReferenceEvaluation
) -> tuple[str, str, str]:
    arm_cheat = (
        best.metrics["arm_terrain_contact_duration_full_horizon_s"] >= 0.1
        or best.metrics["arm_object_contact_duration_full_horizon_s"] >= 0.1
    )
    if best.rollout.valid and best.rollout.recoverable_terminal and not arm_cheat:
        return (
            "A",
            "valid and recoverable through frame 160 without sustained arm support",
            "Generalize WBC-aware simulator-in-the-loop reference repair into an automatic rolling procedure across the remaining horizon.",
        )
    base_failure = baseline.metrics["first_invalid_frame"] or 161
    best_failure = best.metrics["first_invalid_frame"] or 161
    left_gain = (
        (best.metrics["first_left_foot_sustained_loss_frame"] or 161)
        - (baseline.metrics["first_left_foot_sustained_loss_frame"] or 161)
    )
    right_gain = (
        (best.metrics["first_right_foot_sustained_loss_frame"] or 161)
        - (baseline.metrics["first_right_foot_sustained_loss_frame"] or 161)
    )
    if best_failure >= base_failure + 10 or max(left_gain, right_gain) >= 10:
        return (
            "B",
            "failure/support retention moved materially later under the frozen WBC",
            "Expand the WBC-aware reference space once by adding compact root horizontal-velocity corrections.",
        )
    if (
        best.metrics["maximum_hard_constraint_residual"] > 1.0e-3
        or best.metrics["torque_limit_activity_fraction"] > 0.25
    ):
        return (
            "D",
            "the best candidates repeatedly encounter fixed-WBC contact or torque infeasibility",
            "Expose WBC torque/contact infeasibility margins as explicit constraints in reference generation.",
        )
    return (
        "C",
        "the compact nearby-reference search did not materially improve frozen-WBC execution",
        "Reassess the low-level control formulation before adding further reference knots.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact.resolve()
    with np.load(artifact_path) as loaded:
        source = {key: loaded[key].copy() for key in loaded.files}
    trajectory = source["analytical_through140_trajectory"].copy()
    nominal_target = source["optimized_joint_target"].astype(np.float64)
    device = torch.device(args.device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        torch.as_tensor(trajectory, device=device)
    )
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    engine = _DynamicRollout(_build_model(), reference, dynamics)
    config = WBCReferenceRefinementConfig(
        population=args.population,
        elite_count=args.elites,
        generations=args.generations,
        seed=args.seed,
    )

    def progress(item: dict[str, Any]) -> None:
        print("CEM " + json.dumps(_serializable(item)), flush=True)

    result = refine_reference_for_wbc(
        engine, nominal_target, source, config, progress=progress
    )
    classification, basis, next_action = _classification(result.baseline, result.best)
    actuator_names = [
        engine.joint_names[index] for index in engine.actuator_joint_indexes
    ]
    nominal_path = output / "candidate_nominal_reference_start_000.npz"
    residual = result.best.metrics["reference_residual"]
    start = config.wbc.anchor_frame * dynamics.substeps
    end = config.wbc.evaluation_end_frame * dynamics.substeps + 1
    np.savez_compressed(
        nominal_path,
        parameters=result.best.parameters,
        reference_residual=residual,
        root_position_m=result.best.artifact["actual_root_position_m"][start:end],
        root_quaternion_wxyz=result.best.artifact[
            "actual_root_quaternion_wxyz"
        ][start:end],
        root_linear_velocity_m_s=result.best.artifact[
            "actual_root_linear_velocity_m_s"
        ][start:end],
        root_angular_velocity_rad_s=result.best.artifact[
            "actual_root_angular_velocity_rad_s"
        ][start:end],
        joint_position_rad=result.best.artifact["actual_joint_position_rad"][start:end],
        joint_velocity_rad_s=result.best.artifact["actual_joint_velocity_rad_s"][start:end],
        com_position_m=result.best.artifact["actual_com_position_m"][start:end],
        com_velocity_m_s=result.best.artifact["actual_com_velocity_m_s"][start:end],
        centroidal_angular_momentum=result.best.artifact[
            "actual_centroidal_angular_momentum"
        ][start:end],
        knot_frames=np.asarray(config.knot_frames, dtype=np.int32),
        physics_dt=np.asarray([dynamics.physics_dt]),
    )
    realized_path = output / "actual_mujoco_trajectory_start_000.npz"
    np.savez_compressed(
        realized_path,
        **{
            key: value
            for key, value in result.best.rollout.data.items()
            if isinstance(value, np.ndarray)
        },
        anchor_frame=np.asarray([config.wbc.anchor_frame]),
        evaluation_end_frame=np.asarray([config.wbc.evaluation_end_frame]),
        physics_dt=np.asarray([dynamics.physics_dt]),
    )
    report: dict[str, Any] = {
        "experiment": "wbc_aware_simulator_in_loop_reference_refinement",
        "source_artifact": str(artifact_path),
        "wbc_frozen": True,
        "parameterization": {
            "variables_per_free_knot": [
                "root_z reference",
                "root_roll reference",
                "root_pitch reference",
                "12 leg joint references",
                "3 waist joint references",
            ],
            "fixed_boundary_frames": [85, 160],
            "free_knot_frames": list(config.knot_frames[1:-1]),
            "dimensions": len(result.best.parameters),
            "interpolation": "quintic smootherstep with analytic finite-difference-consistent qdot references",
            "bounds": {
                "root_z_m": config.root_height_bound_m,
                "root_roll_pitch_rad": config.root_roll_pitch_bound_rad,
                "balance_joints_rad": config.balance_joint_bound_rad,
            },
            "arms": "fixed to existing nominal reference",
            "physical_root": "uncommanded; emerges only from MuJoCo dynamics",
        },
        "optimization": {
            "method": "diagonal CEM",
            "population": config.population,
            "elites": config.elite_count,
            "generations": config.generations,
            "seed": config.seed,
            "physical_rollouts": result.evaluations,
            "runtime_s": result.optimization_seconds,
            "failure_frame_progression": result.history,
            "zero_correction_explicitly_in_generation0": True,
        },
        "comparison": {
            "frozen_reference_wbc": _comparison(result.baseline),
            "wbc_aware_refined": _comparison(result.best),
        },
        "physical_support_timeline": _timeline(result.best, actuator_names, config),
        "reference_change": _reference_change(result.best, engine, config),
        "realized_reference_distinction": {
            "candidate_nominal_reference": str(nominal_path),
            "actual_mujoco_trajectory": str(realized_path),
            "actual_dynamics_define_physical_outcome": True,
        },
        "classification": classification,
        "classification_basis": basis,
        "next_action": next_action,
    }
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
        default=Path("logs/wbc_aware_reference_refinement"),
    )
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=782)
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
