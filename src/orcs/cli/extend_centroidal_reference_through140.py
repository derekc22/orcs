"""Repair the third support-infeasible segment of the start-0 reference."""

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
from orcs.tasks.uolm.generated_reference import (
    GeneratedReference,
    GeneratedReferenceAdapter,
)
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
    if len(mask) - start < samples:
        return None
    runs = np.convolve(mask[start:].astype(np.int32), np.ones(samples), "valid")
    found = np.flatnonzero(runs == samples)
    return None if not len(found) else int(start + found[0])


def _frame(data: dict[str, np.ndarray], sample: int | None) -> int | None:
    return None if sample is None else int(data["control_frame"][sample])


def _diagnose_failure(result: SupportDiagnosticResult) -> dict[str, Any]:
    data = result.data
    start = 90 * 4
    mass = float(data["robot_mass_kg"][0])
    desired_momentum = mass * data["reference_com_velocity_m_s"][:, 2]
    actual_momentum = data["vertical_momentum_kg_m_s"]
    desired_root_vz = data["reference_root_velocity_m_s"][:, 2]
    actual_root_vz = data["root_linear_velocity_m_s"][:, 2]
    supported = np.any(data["foot_contact"], axis=1)
    force = data["required_vertical_force_n"]
    force_deficit = force - data["total_support_force_world_n"][:, 2]
    cop_margin = data["support_polygon_margin_reference_cop_m"]
    torso_error = np.linalg.norm(
        data["torso_angular_velocity_rad_s"][:, :2]
        - data["reference_torso_angular_velocity_rad_s"][:, :2],
        axis=1,
    )
    angular_momentum_error = np.linalg.norm(
        data["centroidal_angular_momentum_kg_m2_s"][:, :2]
        - data["reference_angular_momentum_kg_m2_s"][:, :2],
        axis=1,
    )
    events = {
        "root_vertical_velocity_error_gt_0p10m_s": _first_sustained(
            np.abs(actual_root_vz - desired_root_vz) > 0.10, start
        ),
        "com_vertical_momentum_error_gt_3kgm_s": _first_sustained(
            np.abs(actual_momentum - desired_momentum) > 3.0, start
        ),
        "required_cop_outside_actual_support": _first_sustained(
            np.isfinite(cop_margin) & (cop_margin < 0.0), start
        ),
        "vertical_force_deficit_gt_10pct": _first_sustained(
            supported & (force_deficit > 0.10 * np.maximum(force, 1.0)), start
        ),
        "left_foot_loss": _first_sustained(~data["foot_contact"][:, 0], start),
        "right_foot_loss": _first_sustained(~data["foot_contact"][:, 1], start),
        "all_support_loss": _first_sustained(~supported, start),
        "torso_roll_pitch_rate_error_gt_0p5rad_s": _first_sustained(
            torso_error > 0.5, start
        ),
        "centroidal_roll_pitch_momentum_error_gt_2kgm2_s": _first_sustained(
            angular_momentum_error > 2.0, start
        ),
        "wrist_contact": _first_sustained(data["wrist_terrain_contact"], start),
        "other_nonfoot_contact": _first_sustained(
            data["nonfoot_terrain_contact"] & ~data["wrist_terrain_contact"], start
        ),
    }
    selected: dict[str, Any] = {}
    for control_frame in (90, 95, 100, 104, 105, 110, 115, 118, 120, 122, 125, 130):
        sample = min(control_frame * 4, len(data["time_s"]) - 1)
        selected[str(control_frame)] = {
            "desired_root_height_m": float(
                data["reference_root_position_m"][sample, 2]
            ),
            "actual_root_height_m": float(data["root_position_m"][sample, 2]),
            "desired_vertical_momentum_kg_m_s": float(desired_momentum[sample]),
            "actual_vertical_momentum_kg_m_s": float(actual_momentum[sample]),
            "required_vertical_force_n": float(force[sample]),
            "actual_vertical_grf_n": float(
                data["total_support_force_world_n"][sample, 2]
            ),
            "required_cop_margin_m": (
                None
                if not np.isfinite(cop_margin[sample])
                else float(cop_margin[sample])
            ),
            "actual_foot_support_left_right": data["foot_contact"][sample].tolist(),
            "torso_roll_pitch_rate_error_rad_s": float(torso_error[sample]),
            "centroidal_roll_pitch_momentum_error_kg_m2_s": float(
                angular_momentum_error[sample]
            ),
            "maximum_actuator_usage": float(
                np.max(data["normalized_torque_usage"][sample])
            ),
        }
    return {
        "first_invalid_frame": result.rollout.first_invalid_frame,
        "invalid_reason": result.rollout.invalid_reason,
        "first_sustained_events": {
            name: _frame(data, sample) for name, sample in events.items()
        },
        "selected_frames": selected,
    }


def _first_contact_or_loss(
    data: dict[str, np.ndarray], side: int, *, start_frame: int = 85
) -> int | None:
    mask = ~data["foot_contact"][:, side]
    sample = _first_sustained(mask, start_frame * 4)
    return _frame(data, sample)


def _first_wrist_contact(data: dict[str, np.ndarray]) -> int | None:
    samples = np.flatnonzero(data["wrist_terrain_contact"])
    return None if not len(samples) else _frame(data, int(samples[0]))


def _dynamic_row(
    rollout: RolloutResult,
    diagnostic: SupportDiagnosticResult,
    analytical: CentroidalRepairMetrics,
    *,
    tracking_start: int = 85,
    tracking_end: int = 140,
) -> dict[str, Any]:
    data = diagnostic.data
    last_frame = rollout.first_invalid_frame
    if last_frame is None:
        last_frame = len(rollout.root_pos) - 1
    last_frame = min(last_frame, len(rollout.root_pos) - 1)
    last_sample = min(last_frame * 4, len(data["time_s"]) - 1)
    samples = slice(tracking_start * 4, tracking_end * 4 + 1)
    desired = data["robot_mass_kg"][0] * data["reference_com_velocity_m_s"][samples, 2]
    actual = data["vertical_momentum_kg_m_s"][samples]
    return {
        "first_invalid_frame": rollout.first_invalid_frame,
        "invalid_reason": rollout.invalid_reason,
        "first_left_foot_loss_frame_after85": _first_contact_or_loss(data, 0),
        "first_right_foot_loss_frame_after85": _first_contact_or_loss(data, 1),
        "first_wrist_contact_frame": _first_wrist_contact(data),
        "minimum_required_cop_margin_m": analytical.minimum_cop_margin_m,
        "minimum_friction_margin_n": analytical.minimum_friction_margin_n,
        "minimum_roll_pitch_moment_margin_nm": (
            analytical.minimum_roll_pitch_moment_margin_nm
        ),
        "minimum_root_height_before_invalidation_m": float(
            np.min(rollout.root_pos[: last_frame + 1, 2])
        ),
        "maximum_torso_tilt_before_invalidation_deg": float(
            np.rad2deg(np.max(rollout.torso_tilt_rad[: last_frame + 1]))
        ),
        "com_vertical_momentum_rms_error_frames85_140_kg_m_s": float(
            np.sqrt(np.mean(np.square(actual - desired)))
        ),
        "maximum_actuator_usage_before_invalidation": float(
            np.max(data["normalized_torque_usage"][: last_sample + 1])
        ),
    }


def _state_samples(result: SupportDiagnosticResult) -> dict[str, Any]:
    data = result.data
    mass = float(data["robot_mass_kg"][0])
    states: dict[str, Any] = {}
    for frame in (100, 105, 110, 115, 120, 122, 125, 130):
        sample = frame * 4
        states[str(frame)] = {
            "desired_root_height_m": float(
                data["reference_root_position_m"][sample, 2]
            ),
            "actual_root_height_m": float(data["root_position_m"][sample, 2]),
            "desired_vertical_momentum_kg_m_s": float(
                mass * data["reference_com_velocity_m_s"][sample, 2]
            ),
            "actual_vertical_momentum_kg_m_s": float(
                data["vertical_momentum_kg_m_s"][sample]
            ),
            "desired_torso_roll_pitch_rate_rad_s": data[
                "reference_torso_angular_velocity_rad_s"
            ][sample, :2].tolist(),
            "actual_torso_roll_pitch_rate_rad_s": data["torso_angular_velocity_rad_s"][
                sample, :2
            ].tolist(),
            "actual_foot_support_left_right": data["foot_contact"][sample].tolist(),
        }
    return states


def _ranges(states: list[str], start: int, end: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    lower = start
    active = states[start]
    for frame in range(start + 1, end + 1):
        if states[frame] == active:
            continue
        result.append({"frames": [lower, frame - 1], "support": active})
        lower, active = frame, states[frame]
    result.append({"frames": [lower, end], "support": active})
    return result


def _analyze(
    model: Any,
    raw: torch.Tensor,
    raw_reference: GeneratedReference,
    baseline: torch.Tensor,
    baseline_reference: GeneratedReference,
    reference: GeneratedReference,
    config: CentroidalRepairConfig,
) -> CentroidalRepairMetrics:
    return analyze_centroidal_reference(
        model,
        raw,
        raw_reference,
        baseline,
        baseline_reference,
        reference,
        config,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.reference_artifact.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as artifact:
        raw_np = artifact["generated_trajectory"].copy()
        first_np = artifact["first_centroidal_repaired_trajectory"].copy()
        baseline_np = artifact["extended_centroidal_repaired_trajectory"].copy()
        baseline_target = artifact["balance_joint_target"].astype(np.float64)
    with np.load(args.first_centroidal_artifact.resolve()) as artifact:
        first_target = artifact["balance_joint_target"].astype(np.float64)

    device = torch.device(args.device)
    raw = torch.as_tensor(raw_np, device=device)
    first = torch.as_tensor(first_np, device=device)
    baseline = torch.as_tensor(baseline_np, device=device)
    adapter = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)
    raw_reference = adapter(raw)
    first_reference = adapter(first)
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
        label="through-95 reference + saved MPC",
        config=dynamics,
        end_frame=140,
    )
    diagnosis = _diagnose_failure(baseline_diagnostic)
    diagnosis_path = output / "failure122_diagnosis.json"
    diagnosis_path.write_text(json.dumps(_serializable(diagnosis), indent=2) + "\n")

    config = CentroidalRepairConfig(
        dynamics_end_frame=140,
        blend_end_frame=160,
        knot_frames=(85, 100, 120, 140, 160),
        repair_start_frame=85,
        optimize_first_knot=False,
        initialize_from_baseline=True,
        overlap_preservation_end_frame=95,
        overlap_preservation_weight=200.0,
        horizontal_projection_iterations=1,
        final_horizontal_projection_iterations=3,
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
    repaired_reference = adapter(repair.trajectory)
    final_metrics = _analyze(
        model,
        raw,
        raw_reference,
        baseline,
        baseline_reference,
        repaired_reference,
        config,
    )
    final_metrics.correction_rms = repair.repaired.correction_rms
    final_metrics.correction_max = repair.repaired.correction_max
    final_metrics.prior_objective = repair.repaired.prior_objective

    baseline_root = baseline_reference.root_pos_local[0].detach().cpu().numpy()
    repaired_root = repaired_reference.root_pos_local[0].detach().cpu().numpy()
    baseline_joint = baseline_reference.joint_pos[0].detach().cpu().numpy()
    repaired_joint = repaired_reference.joint_pos[0].detach().cpu().numpy()
    continuity = {
        "frames0_85_bit_exact": bool(
            np.array_equal(repair.trajectory[:, :86].cpu().numpy(), baseline_np[:, :86])
        ),
        "maximum_root_change_frames0_85_m": float(
            np.max(np.abs(repaired_root[:86] - baseline_root[:86]))
        ),
        "maximum_joint_change_frames0_85_rad": float(
            np.max(np.abs(repaired_joint[:86] - baseline_joint[:86]))
        ),
        "maximum_overlap_root_change_frames85_95_m": float(
            np.max(np.abs(repaired_root[85:96] - baseline_root[85:96]))
        ),
        "maximum_overlap_joint_change_frames85_95_rad": float(
            np.max(np.abs(repaired_joint[85:96] - baseline_joint[85:96]))
        ),
        "root_velocity_discrepancy_at_frame85_m_s": float(
            np.max(
                np.abs(
                    (repaired_root[86] - repaired_root[84])
                    - (baseline_root[86] - baseline_root[84])
                )
            )
            / 0.04
        ),
        "joint_velocity_discrepancy_at_frame85_rad_s": float(
            np.max(
                np.abs(
                    (repaired_joint[86] - repaired_joint[84])
                    - (baseline_joint[86] - baseline_joint[84])
                )
            )
            / 0.04
        ),
    }
    analytical = {
        "support_schedule_frames0_140": _ranges(
            final_metrics.support_state_by_frame, 0, 140
        ),
        "unsupported_frames": final_metrics.unsupported_frames,
        "unsupported_force_violation_frames": (
            final_metrics.unsupported_force_violation_frames
        ),
        "required_vertical_force_range_n": [
            final_metrics.minimum_required_vertical_force_n,
            final_metrics.maximum_required_vertical_force_n,
        ],
        "minimum_cop_margin_m": final_metrics.minimum_cop_margin_m,
        "cop_outside_time_s": final_metrics.cop_outside_time_s,
        "minimum_friction_margin_n": final_metrics.minimum_friction_margin_n,
        "friction_violation_time_s": final_metrics.friction_violation_time_s,
        "minimum_roll_pitch_moment_margin_nm": (
            final_metrics.minimum_roll_pitch_moment_margin_nm
        ),
        "maximum_required_roll_pitch_moment_nm": (
            final_metrics.maximum_required_roll_pitch_moment_nm
        ),
        "wrist_clearance_left_right_m": [
            final_metrics.minimum_left_wrist_clearance_m,
            final_metrics.minimum_right_wrist_clearance_m,
        ],
        "maximum_foot_penetration_m": final_metrics.maximum_foot_penetration_m,
        "first_forbidden_contact_frame": (
            final_metrics.first_forbidden_terrain_contact_frame
        ),
        "maximum_joint_limit_violation_rad": (
            final_metrics.maximum_joint_limit_violation_rad
        ),
        "frame0_vertical_momentum_kg_m_s": (
            final_metrics.vertical_momentum_frame0_kg_m_s
        ),
        "feasible": final_metrics.feasible,
    }

    report: dict[str, Any] = {
        "experiment": "centroidal_support_reference_repair_through140",
        "source_artifact": str(source),
        "failure_diagnosis": diagnosis,
        "repair": {
            "analytical_constraint_frames_inclusive": [0, 140],
            "new_repair_anchor_frame": 85,
            "strong_overlap_frames_inclusive": [85, 95],
            "blend_end_frame": 160,
            "optimizer": repair.optimizer,
            "history": repair.history,
            "continuity": continuity,
            "angular_momentum_treatment": (
                "MuJoCo centroidal H is differentiated; required roll/pitch support "
                "moment is Hdot + COM cross required force and is constrained by "
                "the active support polygon through the COP margin"
            ),
        },
        "analytical": analytical,
        "dynamic_validation_performed": False,
        "dynamic": None,
        "classification": "D",
        "classification_basis": "no analytically feasible fixed-arm third repair",
        "artifact": None,
    }

    if final_metrics.feasible:
        raw_stage = refine_generated_reference(model, raw_reference, dynamics)
        raw_mpc = refine_balance_with_mpc(
            model, raw_reference, raw_stage, dynamics, mpc
        )
        first_rollout = _DynamicRollout(model, first_reference, dynamics).rollout(
            first_target
        )
        baseline_rollout = _DynamicRollout(model, baseline_reference, dynamics).rollout(
            baseline_target
        )
        repaired_engine = _DynamicRollout(model, repaired_reference, dynamics)
        translated_feedforward = repaired_joint.copy()
        translated_feedforward[:, repaired_engine.balance_indexes] += (
            baseline_target[:, repaired_engine.balance_indexes]
            - baseline_joint[:, repaired_engine.balance_indexes]
        )
        translated_feedforward = np.clip(
            translated_feedforward,
            repaired_engine.joint_limits[:, 0],
            repaired_engine.joint_limits[:, 1],
        )
        repaired_mpc = refine_balance_with_mpc(
            model,
            repaired_reference,
            SimpleNamespace(
                manipulation=None,
                balance_joint_target=translated_feedforward,
            ),
            dynamics,
            mpc,
        )
        analytical_cases = {
            "raw": _analyze(
                model,
                raw,
                raw_reference,
                baseline,
                baseline_reference,
                raw_reference,
                config,
            ),
            "first_centroidal": _analyze(
                model,
                raw,
                raw_reference,
                baseline,
                baseline_reference,
                first_reference,
                config,
            ),
            "through95": repair.kinematic,
            "through140": final_metrics,
        }
        cases = {
            "raw": (raw_reference, raw_mpc.joint_target, raw_mpc.rollout),
            "first_centroidal": (first_reference, first_target, first_rollout),
            "through95": (baseline_reference, baseline_target, baseline_rollout),
            "through140": (
                repaired_reference,
                repaired_mpc.joint_target,
                repaired_mpc.rollout,
            ),
        }
        diagnostics: dict[str, SupportDiagnosticResult] = {}
        rows: dict[str, Any] = {}
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
            rows[name] = _dynamic_row(rollout, diagnostic, analytical_cases[name])
        failure = repaired_mpc.rollout.first_invalid_frame
        report["dynamic_validation_performed"] = True
        report["dynamic"] = {
            "controller_changed": False,
            "feedforward_handoff": (
                "saved through-95 balance correction translated onto the repaired "
                "reference, joint-limit clipped, then unchanged 12-frame MPC"
            ),
            "comparison": rows,
            "states_frames100_130": {
                "through95": _state_samples(diagnostics["through95"]),
                "through140": _state_samples(diagnostics["through140"]),
            },
            "new_failure_inside_repaired_interval": (
                failure is not None and failure <= config.dynamics_end_frame
            ),
        }
        if repaired_mpc.rollout.valid:
            report["classification"] = "A"
            report["classification_basis"] = "full 300-frame stability achieved"
        elif failure is not None and failure > config.dynamics_end_frame:
            report["classification"] = "B"
            report["classification_basis"] = (
                "third repair remained dynamically valid through frame 140; "
                f"next first invalid frame is {failure}"
            )
        else:
            report["classification"] = "C"
            report["classification_basis"] = (
                "analytical gate passed but dynamic execution still failed in "
                "the frames 95-140 region"
            )

        artifact_path = (
            output / "centroidal_repaired_reference_through140_start_000.npz"
        )
        values: dict[str, np.ndarray] = {
            "generated_trajectory": raw_np,
            "first_centroidal_repaired_trajectory": first_np,
            "through95_centroidal_repaired_trajectory": baseline_np,
            "through140_centroidal_repaired_trajectory": (
                repair.trajectory.detach().cpu().numpy()
            ),
            "repair_parameters": repair.parameters.astype(np.float32),
            "repair_correction": repair.correction.astype(np.float32),
            "balance_joint_target": repaired_mpc.joint_target.astype(np.float32),
            "feedback_residual": repaired_mpc.feedback_residual.astype(np.float32),
            "simulated_root_pos": repaired_mpc.rollout.root_pos.astype(np.float32),
            "repair_frames": np.asarray([85, 140, 160], dtype=np.int32),
            "dt": np.asarray([0.02], dtype=np.float32),
        }
        values.update(
            {
                f"analytical__{name}": value
                for name, value in repair.analytical_data.items()
            }
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
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path(
            "logs/extended_centroidal_reference_repair/"
            "extended_centroidal_repaired_reference_start_000.npz"
        ),
    )
    parser.add_argument(
        "--first-centroidal-artifact",
        type=Path,
        default=Path(
            "logs/centroidal_reference_repair/"
            "centroidal_repaired_reference_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/centroidal_reference_repair_through140"),
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
