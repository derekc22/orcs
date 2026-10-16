"""Extend the start-0 centroidal repair through the frame-80 failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.balance_mpc import BalanceMPCConfig, refine_balance_with_mpc
from orcs.tasks.uolm.centroidal_reference_repair import (
    CentroidalRepairConfig,
    CentroidalRepairMetrics,
    analyze_centroidal_reference,
    extend_centroidal_reference,
)
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    RolloutResult,
    _DynamicRollout,
    refine_generated_reference,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
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


def _failure_diagnosis(result: SupportDiagnosticResult) -> dict[str, Any]:
    data = result.data
    start = 40 * 4
    root_error = data["reference_root_position_m"][:, 2] - data["root_position_m"][:, 2]
    velocity_error = (
        data["reference_com_velocity_m_s"][:, 2] - data["com_velocity_m_s"][:, 2]
    )
    force_deficit = (
        data["required_vertical_force_n"] - data["total_support_force_world_n"][:, 2]
    )
    supported = np.any(data["foot_contact"], axis=1)
    criteria = {
        "root_height_deficit_gt_10mm": _first_sustained(root_error > 0.010, start),
        "actual_downward_velocity_excess_gt_0p10m_s": _first_sustained(
            velocity_error > 0.10, start
        ),
        "support_force_deficit_gt_10pct": _first_sustained(
            supported
            & (force_deficit > 0.10 * np.maximum(data["required_vertical_force_n"], 1.0)),
            start,
        ),
        "loss_of_all_foot_support": _first_sustained(~supported, start),
        "wrist_contact": _first_sustained(data["wrist_terrain_contact"], start),
        "nonfoot_contact": _first_sustained(data["nonfoot_terrain_contact"], start),
        "actuator_usage_gt_80pct": _first_sustained(
            np.max(data["normalized_torque_usage"], axis=1) > 0.80, start
        ),
    }
    frames = {
        name: None if sample is None else int(data["control_frame"][sample])
        for name, sample in criteria.items()
    }
    selected = {}
    for frame in (40, 46, 50, 55, 56, 60, 70, 75, 80, 90):
        sample = frame * 4
        selected[str(frame)] = {
            "desired_root_height_m": float(data["reference_root_position_m"][sample, 2]),
            "actual_root_height_m": float(data["root_position_m"][sample, 2]),
            "root_height_error_m": float(root_error[sample]),
            "desired_com_vertical_velocity_m_s": float(
                data["reference_com_velocity_m_s"][sample, 2]
            ),
            "actual_com_vertical_velocity_m_s": float(data["com_velocity_m_s"][sample, 2]),
            "vertical_momentum_kg_m_s": float(data["vertical_momentum_kg_m_s"][sample]),
            "required_vertical_force_n": float(data["required_vertical_force_n"][sample]),
            "actual_vertical_grf_n": float(data["total_support_force_world_n"][sample, 2]),
            "required_cop_margin_m": (
                None
                if not np.isfinite(data["support_polygon_margin_reference_cop_m"][sample])
                else float(data["support_polygon_margin_reference_cop_m"][sample])
            ),
            "foot_contact_left_right": data["foot_contact"][sample].tolist(),
            "wrist_contact": bool(data["wrist_terrain_contact"][sample]),
            "nonfoot_contact": bool(data["nonfoot_terrain_contact"][sample]),
            "maximum_actuator_usage": float(
                np.max(data["normalized_torque_usage"][sample])
            ),
        }
    return {
        "first_invalid_frame": result.rollout.first_invalid_frame,
        "invalid_reason": result.rollout.invalid_reason,
        "first_sustained_post40_events": frames,
        "selected_frames": selected,
        "contact_sequence": (
            "right_wrist_yaw_link at frames 56-58; right wrist again at 85-86; "
            "right_hip_roll_link from frame 87 in the continued rollout"
        ),
    }


def _first_wrist_frame(result: SupportDiagnosticResult) -> int | None:
    indexes = np.flatnonzero(result.data["wrist_terrain_contact"])
    return None if not len(indexes) else int(result.data["control_frame"][indexes[0]])


def _dynamic_row(
    rollout: RolloutResult,
    diagnostic: SupportDiagnosticResult,
    analytical: CentroidalRepairMetrics,
) -> dict[str, Any]:
    last_frame = rollout.first_invalid_frame
    if last_frame is None:
        last_frame = len(rollout.root_pos) - 1
    last_frame = min(last_frame, len(rollout.root_pos) - 1)
    last_sample = min(last_frame * 4, len(diagnostic.data["time_s"]) - 1)
    return {
        "first_invalid_frame": rollout.first_invalid_frame,
        "invalid_reason": rollout.invalid_reason,
        "frame0_vertical_momentum_kg_m_s": analytical.vertical_momentum_frame0_kg_m_s,
        "first_wrist_contact_frame": _first_wrist_frame(diagnostic),
        "minimum_required_cop_margin_m": analytical.minimum_cop_margin_m,
        "minimum_friction_margin_n": analytical.minimum_friction_margin_n,
        "minimum_root_height_before_invalidation_m": float(
            np.min(rollout.root_pos[: last_frame + 1, 2])
        ),
        "maximum_torso_tilt_before_invalidation_deg": float(
            np.rad2deg(np.max(rollout.torso_tilt_rad[: last_frame + 1]))
        ),
        "maximum_actuator_usage_before_invalidation": float(
            np.max(diagnostic.data["normalized_torque_usage"][: last_sample + 1])
        ),
    }


def _tracking(result: SupportDiagnosticResult, start: int = 35, end: int = 95) -> dict[str, float]:
    data = result.data
    samples = slice(start * 4, end * 4 + 1)
    desired = data["robot_mass_kg"][0] * data["reference_com_velocity_m_s"][samples, 2]
    actual = data["vertical_momentum_kg_m_s"][samples]
    error = actual - desired
    frame80 = 80 * 4
    return {
        "rms_vertical_momentum_error_kg_m_s": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_abs_vertical_momentum_error_kg_m_s": float(np.max(np.abs(error))),
        "vertical_momentum_error_frame80_kg_m_s": float(
            data["vertical_momentum_kg_m_s"][frame80]
            - data["robot_mass_kg"][0] * data["reference_com_velocity_m_s"][frame80, 2]
        ),
        "root_height_error_frame80_m": float(
            data["reference_root_position_m"][frame80, 2]
            - data["root_position_m"][frame80, 2]
        ),
    }


def _state_at_failure(result: SupportDiagnosticResult) -> dict[str, Any] | None:
    frame = result.rollout.first_invalid_frame
    if frame is None:
        return None
    data = result.data
    sample = min(frame * 4, len(data["time_s"]) - 1)
    desired_momentum = float(
        data["robot_mass_kg"][0] * data["reference_com_velocity_m_s"][sample, 2]
    )
    actual_momentum = float(data["vertical_momentum_kg_m_s"][sample])
    return {
        "frame": frame,
        "desired_root_height_m": float(data["reference_root_position_m"][sample, 2]),
        "actual_root_height_m": float(data["root_position_m"][sample, 2]),
        "root_height_error_m": float(
            data["reference_root_position_m"][sample, 2]
            - data["root_position_m"][sample, 2]
        ),
        "desired_vertical_momentum_kg_m_s": desired_momentum,
        "actual_vertical_momentum_kg_m_s": actual_momentum,
        "vertical_momentum_error_kg_m_s": actual_momentum - desired_momentum,
        "foot_contact_left_right": data["foot_contact"][sample].tolist(),
        "wrist_contact": bool(data["wrist_terrain_contact"][sample]),
        "nonfoot_contact": bool(data["nonfoot_terrain_contact"][sample]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.reference_artifact.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as artifact:
        raw_np = artifact["generated_trajectory"].copy()
        kinematic_np = artifact["kinematic_repaired_trajectory"].copy()
        baseline_np = artifact["centroidal_repaired_trajectory"].copy()
        baseline_target = artifact["balance_joint_target"].astype(np.float64)
    with np.load(source.parent.parent / "early_reference_repair" / "early_repaired_reference_start_000.npz") as artifact:
        kinematic_target = artifact["balance_joint_target"].astype(np.float64)

    device = torch.device(args.device)
    raw = torch.as_tensor(raw_np, device=device)
    kinematic = torch.as_tensor(kinematic_np, device=device)
    baseline = torch.as_tensor(baseline_np, device=device)
    adapter = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)
    raw_reference = adapter(raw)
    kinematic_reference = adapter(kinematic)
    baseline_reference = adapter(baseline)
    model = _build_model()
    dynamics = DynamicsRefinementConfig(
        balance_population=24,
        balance_elite_count=6,
        balance_generations=14,
        run_stage_b=False,
        seed=42,
    )
    mpc = BalanceMPCConfig(
        horizon_frames=12,
        replan_interval_frames=2,
        population=12,
        elite_count=3,
        generations=2,
    )

    baseline_diagnostic = instrument_support_rollout(
        model,
        baseline_reference,
        baseline_target,
        label="first centroidal repair + saved MPC",
        config=dynamics,
        end_frame=95,
    )
    diagnosis = _failure_diagnosis(baseline_diagnostic)

    config = CentroidalRepairConfig(
        dynamics_end_frame=95,
        blend_end_frame=110,
        knot_frames=(35, 55, 75, 95, 110),
        repair_start_frame=35,
        optimize_first_knot=False,
        initialize_from_baseline=True,
        overlap_preservation_end_frame=40,
        overlap_preservation_weight=200.0,
        horizontal_projection_iterations=1,
        final_horizontal_projection_iterations=2,
        population=args.population,
        elite_count=min(args.elite_count, args.population),
        generations=args.generations,
        seed=args.seed,
    )
    repair = extend_centroidal_reference(
        model,
        raw,
        raw_reference,
        baseline,
        baseline_reference,
        config,
    )
    extended_reference = adapter(repair.trajectory)
    final_metrics = analyze_centroidal_reference(
        model,
        raw,
        raw_reference,
        baseline,
        baseline_reference,
        extended_reference,
        config,
    )
    final_metrics.correction_rms = repair.repaired.correction_rms
    final_metrics.correction_max = repair.repaired.correction_max
    final_metrics.prior_objective = repair.repaired.prior_objective

    base_root = baseline_reference.root_pos_local[0].detach().cpu().numpy()
    new_root = extended_reference.root_pos_local[0].detach().cpu().numpy()
    base_joint = baseline_reference.joint_pos[0].detach().cpu().numpy()
    new_joint = extended_reference.joint_pos[0].detach().cpu().numpy()
    continuity = {
        "maximum_root_position_change_frames0_35_m": float(
            np.max(np.abs(new_root[:36] - base_root[:36]))
        ),
        "maximum_joint_change_frames0_35_rad": float(
            np.max(np.abs(new_joint[:36] - base_joint[:36]))
        ),
        "maximum_overlap_root_position_change_frames35_40_m": float(
            np.max(np.abs(new_root[35:41] - base_root[35:41]))
        ),
        "maximum_overlap_joint_change_frames35_40_rad": float(
            np.max(np.abs(new_joint[35:41] - base_joint[35:41]))
        ),
        "root_velocity_discrepancy_at_frame35_m_s": float(
            np.max(np.abs((new_root[36] - new_root[34]) - (base_root[36] - base_root[34])))
            / 0.04
        ),
        "joint_velocity_discrepancy_at_frame35_rad_s": float(
            np.max(np.abs((new_joint[36] - new_joint[34]) - (base_joint[36] - base_joint[34])))
            / 0.04
        ),
    }

    report: dict[str, Any] = {
        "experiment": "extended_centroidal_support_reference_repair",
        "source_artifact": str(source),
        "new_failure_diagnosis": diagnosis,
        "repair": {
            "constrained_frames_inclusive": [0, 95],
            "second_window_frames_inclusive": [35, 95],
            "blend_end_frame": 110,
            "optimizer": repair.optimizer,
            "history": repair.history,
            "continuity": continuity,
            "support_model": {
                "friction_coefficient": config.friction_coefficient,
                "vertical_force_upper_body_weights": config.maximum_vertical_force_weight,
                "wrist_clearance_margin_m": config.wrist_clearance_margin_m,
                "maximum_foot_penetration_m": config.maximum_foot_penetration_m,
                "cop_numerical_tolerance_m": config.cop_numerical_tolerance_m,
            },
        },
        "analytical": {
            "first_centroidal": repair.kinematic.as_dict(),
            "extended_centroidal": final_metrics.as_dict(),
        },
        "dynamic_validation_performed": False,
        "dynamic": None,
        "classification": "D",
        "classification_basis": "no analytically feasible fixed-arm second repair",
        "artifact": None,
    }

    if final_metrics.feasible:
        raw_stage = refine_generated_reference(model, raw_reference, dynamics)
        raw_mpc = refine_balance_with_mpc(model, raw_reference, raw_stage, dynamics, mpc)
        kinematic_rollout = _DynamicRollout(model, kinematic_reference, dynamics).rollout(
            kinematic_target
        )
        baseline_rollout = _DynamicRollout(model, baseline_reference, dynamics).rollout(
            baseline_target
        )
        extended_engine = _DynamicRollout(model, extended_reference, dynamics)
        baseline_joint = baseline_reference.joint_pos[0].detach().cpu().numpy()
        extended_joint = extended_reference.joint_pos[0].detach().cpu().numpy()
        translated_feedforward = extended_joint.copy()
        translated_feedforward[:, extended_engine.balance_indexes] += (
            baseline_target[:, extended_engine.balance_indexes]
            - baseline_joint[:, extended_engine.balance_indexes]
        )
        translated_feedforward = np.clip(
            translated_feedforward,
            extended_engine.joint_limits[:, 0],
            extended_engine.joint_limits[:, 1],
        )
        extended_mpc = refine_balance_with_mpc(
            model,
            extended_reference,
            SimpleNamespace(
                manipulation=None,
                balance_joint_target=translated_feedforward,
            ),
            dynamics,
            mpc,
        )
        cases = {
            "raw": (raw_reference, raw_mpc.joint_target, raw_mpc.rollout, repair.raw),
            "kinematic": (
                kinematic_reference,
                kinematic_target,
                kinematic_rollout,
                analyze_centroidal_reference(
                    model, raw, raw_reference, baseline, baseline_reference, kinematic_reference, config
                ),
            ),
            "first_centroidal": (
                baseline_reference,
                baseline_target,
                baseline_rollout,
                repair.kinematic,
            ),
            "extended_centroidal": (
                extended_reference,
                extended_mpc.joint_target,
                extended_mpc.rollout,
                final_metrics,
            ),
        }
        diagnostics: dict[str, SupportDiagnosticResult] = {}
        rows: dict[str, Any] = {}
        tracking: dict[str, Any] = {}
        for name, (reference, target, rollout, analytical) in cases.items():
            diagnostic = instrument_support_rollout(
                model,
                reference,
                target,
                label=name,
                config=dynamics,
                end_frame=299,
            )
            diagnostics[name] = diagnostic
            rows[name] = _dynamic_row(rollout, diagnostic, analytical)
            tracking[name] = _tracking(diagnostic)
        report["dynamic_validation_performed"] = True
        report["dynamic"] = {
            "controller_changed": False,
            "feedforward_handoff": (
                "saved first-centroidal balance correction translated onto the "
                "extended reference, joint-limit clipped, then unchanged MPC"
            ),
            "comparison": rows,
            "tracking_frames35_95": tracking,
            "extended_state_at_new_failure": _state_at_failure(
                diagnostics["extended_centroidal"]
            ),
            "extended_failure_inside_repaired_interval": (
                extended_mpc.rollout.first_invalid_frame is not None
                and extended_mpc.rollout.first_invalid_frame <= 95
            ),
        }
        failure = extended_mpc.rollout.first_invalid_frame
        if extended_mpc.rollout.valid:
            report["classification"] = "A"
            report["classification_basis"] = "full 300-frame stability achieved"
        elif failure is not None and failure > 95:
            report["classification"] = "B"
            report["classification_basis"] = (
                f"second repair remained valid through frame 95; next failure is frame {failure}"
            )
        else:
            report["classification"] = "C"
            report["classification_basis"] = (
                "analytical gate passed but physical failure remained inside frames 0-95"
            )

        artifact_path = output / "extended_centroidal_repaired_reference_start_000.npz"
        values: dict[str, np.ndarray] = {
            "generated_trajectory": raw_np,
            "kinematic_repaired_trajectory": kinematic_np,
            "first_centroidal_repaired_trajectory": baseline_np,
            "extended_centroidal_repaired_trajectory": repair.trajectory.detach().cpu().numpy(),
            "repair_parameters": repair.parameters.astype(np.float32),
            "repair_correction": repair.correction.astype(np.float32),
            "balance_joint_target": extended_mpc.joint_target.astype(np.float32),
            "feedback_residual": extended_mpc.feedback_residual.astype(np.float32),
            "simulated_root_pos": extended_mpc.rollout.root_pos.astype(np.float32),
            "repair_frames": np.asarray([35, 95, 110], dtype=np.int32),
            "dt": np.asarray([0.02], dtype=np.float32),
        }
        values.update(
            {f"analytical__{name}": value for name, value in repair.analytical_data.items()}
        )
        np.savez_compressed(artifact_path, **values)
        report["artifact"] = str(artifact_path)

    report_path = output / "report_start_000.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=343)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path(
            "logs/centroidal_reference_repair/centroidal_repaired_reference_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/extended_centroidal_reference_repair")
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
