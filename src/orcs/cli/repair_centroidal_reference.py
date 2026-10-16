"""Repair start-0 early support motion with explicit centroidal constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.balance_mpc import BalanceMPCConfig, refine_balance_with_mpc
from orcs.tasks.uolm.centroidal_reference_repair import (
    CentroidalRepairConfig,
    CentroidalRepairMetrics,
    analyze_centroidal_reference,
    repair_centroidal_reference,
)
from orcs.tasks.uolm.dynamics_refinement import (
    _LEG_WAIST_JOINTS,
    DynamicsRefinementConfig,
    RolloutResult,
    _DynamicRollout,
    refine_generated_reference,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.support_diagnostics import instrument_support_rollout


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


def _first_wrist_contact_frame(summary: dict[str, Any]) -> int | None:
    value = summary["first_wrist_terrain_contact_time_s"]
    return None if value is None else int(round(float(value) / 0.02))


def _dynamic_row(
    rollout: RolloutResult,
    reference: Any,
    analytical: CentroidalRepairMetrics,
    diagnostic_summary: dict[str, Any],
) -> dict[str, Any]:
    frame = min(34, len(rollout.root_pos) - 1)
    return {
        "first_invalid_frame": rollout.first_invalid_frame,
        "invalid_reason": rollout.invalid_reason,
        "frame0_vertical_momentum_kg_m_s": (analytical.vertical_momentum_frame0_kg_m_s),
        "minimum_root_height_m": rollout.metrics["minimum_root_height_m"],
        "root_height_error_frame34_m": float(
            reference.root_pos_local[0, frame, 2].item() - rollout.root_pos[frame, 2]
        ),
        "maximum_torso_tilt_deg": rollout.metrics["maximum_torso_tilt_deg"],
        "first_wrist_contact_frame": _first_wrist_contact_frame(diagnostic_summary),
        "nonfoot_terrain_contact_duration_s": rollout.metrics[
            "nonfoot_terrain_contact_duration_s"
        ],
        "minimum_required_cop_margin_m": analytical.minimum_cop_margin_m,
        "max_actuator_usage": rollout.metrics["max_actuator_usage"],
    }


def _momentum_tracking(result: Any) -> dict[str, float]:
    data = result.data
    end = 40 * 4 + 1
    desired = data["robot_mass_kg"][0] * data["reference_com_velocity_m_s"][:end, 2]
    actual = data["vertical_momentum_kg_m_s"][:end]
    error = actual - desired
    return {
        "rms_error_kg_m_s": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_abs_error_kg_m_s": float(np.max(np.abs(error))),
        "error_frame20_kg_m_s": float(error[20 * 4]),
        "error_frame34_kg_m_s": float(error[34 * 4]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.reference_artifact.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as artifact:
        raw_np = artifact["generated_trajectory"].copy()
        kinematic_np = artifact["repaired_trajectory"].copy()
        kinematic_target = artifact["balance_joint_target"].astype(np.float64)

    device = torch.device(args.device)
    raw_trajectory = torch.as_tensor(raw_np, device=device)
    kinematic_trajectory = torch.as_tensor(kinematic_np, device=device)
    model = _build_model()
    adapter = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)
    raw_reference = adapter(raw_trajectory)
    kinematic_reference = adapter(kinematic_trajectory)
    config = CentroidalRepairConfig(
        population=args.population,
        elite_count=min(args.elite_count, args.population),
        generations=args.generations,
        seed=args.seed,
    )
    repair = repair_centroidal_reference(
        model,
        raw_trajectory,
        raw_reference,
        kinematic_trajectory,
        kinematic_reference,
        config,
    )
    centroidal_reference = None
    final_metrics = repair.repaired
    if repair.repaired.feasible:
        centroidal_reference = adapter(repair.trajectory)
        final_metrics = analyze_centroidal_reference(
            model,
            raw_trajectory,
            raw_reference,
            kinematic_trajectory,
            kinematic_reference,
            centroidal_reference,
            config,
        )
        final_metrics.correction_rms = repair.repaired.correction_rms
        final_metrics.correction_max = repair.repaired.correction_max
        final_metrics.prior_objective = repair.repaired.prior_objective
    report: dict[str, Any] = {
        "experiment": "centroidal_support_feasible_reference_repair",
        "source_artifact": str(source),
        "dataset": "sub3_largebox_003",
        "conditioning_start": 0,
        "repair": {
            "dynamics_frames_inclusive": [0, config.dynamics_end_frame],
            "blend_end_frame": config.blend_end_frame,
            "optimizer": repair.optimizer,
            "history": repair.history,
            "support_model": {
                "contact_distance_m": config.contact_distance_m,
                "maximum_foot_penetration_m": config.maximum_foot_penetration_m,
                "cop_target_margin_m": config.cop_target_margin_m,
                "cop_numerical_tolerance_m": config.cop_numerical_tolerance_m,
                "friction_coefficient": config.friction_coefficient,
                "wrench_approximation": (
                    "net force friction cone plus COP inside convex hull of active "
                    "MuJoCo sole collision capsules; no independent torsional-yaw "
                    "moment constraint"
                ),
            },
            "arms_unchanged": bool(
                torch.equal(
                    repair.trajectory[..., 9:38][
                        ...,
                        [
                            index
                            for index, name in enumerate(repair.joint_names)
                            if name not in _LEG_WAIST_JOINTS
                        ],
                    ],
                    raw_trajectory[..., 9:38][
                        ...,
                        [
                            index
                            for index, name in enumerate(repair.joint_names)
                            if name not in _LEG_WAIST_JOINTS
                        ],
                    ],
                )
            ),
            "object_unchanged": bool(
                torch.equal(repair.trajectory[..., 38:47], raw_trajectory[..., 38:47])
            ),
        },
        "analytical": {
            "raw": repair.raw.as_dict(),
            "kinematic_repair": repair.kinematic.as_dict(),
            "retimed_initialization": repair.initial.as_dict(),
            "centroidal_repair": final_metrics.as_dict(),
        },
        "dynamic_validation_performed": False,
        "dynamic": None,
        "classification": "D",
        "classification_basis": (
            "no centroidal-feasible fixed-arm reference found within bounds"
        ),
        "artifact": None,
    }

    # The analytical gate is deliberately before every physical rollout of the
    # newly repaired candidate.
    if final_metrics.feasible:
        assert centroidal_reference is not None
        dynamics_config = DynamicsRefinementConfig(
            balance_population=24,
            balance_elite_count=6,
            balance_generations=14,
            run_stage_b=False,
            seed=42,
        )
        mpc_config = BalanceMPCConfig(
            horizon_frames=12,
            replan_interval_frames=2,
            population=12,
            elite_count=3,
            generations=2,
        )
        raw_stage = refine_generated_reference(model, raw_reference, dynamics_config)
        raw_mpc = refine_balance_with_mpc(
            model, raw_reference, raw_stage, dynamics_config, mpc_config
        )
        kinematic_rollout = _DynamicRollout(
            model, kinematic_reference, dynamics_config
        ).rollout(kinematic_target)
        centroidal_stage = refine_generated_reference(
            model, centroidal_reference, dynamics_config
        )
        centroidal_mpc = refine_balance_with_mpc(
            model,
            centroidal_reference,
            centroidal_stage,
            dynamics_config,
            mpc_config,
        )
        raw_diagnostic = instrument_support_rollout(
            model,
            raw_reference,
            raw_mpc.joint_target,
            label="raw reference + MPC",
            config=dynamics_config,
            end_frame=299,
        )
        kinematic_diagnostic = instrument_support_rollout(
            model,
            kinematic_reference,
            kinematic_target,
            label="kinematic repair + MPC",
            config=dynamics_config,
            end_frame=299,
        )
        centroidal_diagnostic = instrument_support_rollout(
            model,
            centroidal_reference,
            centroidal_mpc.joint_target,
            label="centroidal repair + MPC",
            config=dynamics_config,
            end_frame=299,
        )
        report["dynamic_validation_performed"] = True
        report["dynamic"] = {
            "controller_changed": False,
            "raw": _dynamic_row(
                raw_mpc.rollout,
                raw_reference,
                repair.raw,
                raw_diagnostic.summary,
            ),
            "kinematic_repair": _dynamic_row(
                kinematic_rollout,
                kinematic_reference,
                repair.kinematic,
                kinematic_diagnostic.summary,
            ),
            "centroidal_repair": _dynamic_row(
                centroidal_mpc.rollout,
                centroidal_reference,
                final_metrics,
                centroidal_diagnostic.summary,
            ),
            "com_vertical_momentum_tracking": {
                "raw": _momentum_tracking(raw_diagnostic),
                "kinematic_repair": _momentum_tracking(kinematic_diagnostic),
                "centroidal_repair": _momentum_tracking(centroidal_diagnostic),
            },
        }
        new_failure = centroidal_mpc.rollout.first_invalid_frame
        if centroidal_mpc.rollout.valid:
            report["classification"] = "A"
            report["classification_basis"] = (
                "centroidal-feasible repair remained valid for all 300 frames"
            )
        elif new_failure is not None and new_failure > 40:
            report["classification"] = "B"
            report["classification_basis"] = (
                "centroidal repair removed the early failure and failed after frame 40"
            )
        else:
            report["classification"] = "C"
            report["classification_basis"] = (
                "analytical centroidal gate passed but physical execution still "
                "failed in frames 0-40"
            )

        artifact_path = output / "centroidal_repaired_reference_start_000.npz"
        artifact_values: dict[str, np.ndarray] = {
            "generated_trajectory": raw_np,
            "kinematic_repaired_trajectory": kinematic_np,
            "centroidal_repaired_trajectory": repair.trajectory.detach().cpu().numpy(),
            "repair_correction": repair.correction.astype(np.float32),
            "repair_parameters": repair.parameters.astype(np.float32),
            "balance_joint_target": centroidal_mpc.joint_target.astype(np.float32),
            "feedback_residual": centroidal_mpc.feedback_residual.astype(np.float32),
            "simulated_root_pos": centroidal_mpc.rollout.root_pos.astype(np.float32),
            "repair_frames": np.asarray([0, 40, 60], dtype=np.int32),
            "dt": np.asarray([0.02], dtype=np.float32),
        }
        artifact_values.update(
            {
                f"analytical__{name}": value
                for name, value in repair.analytical_data.items()
            }
        )
        np.savez_compressed(artifact_path, **artifact_values)
        report["artifact"] = str(artifact_path)

    report_path = output / "report_start_000.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path(
            "logs/early_reference_repair/early_repaired_reference_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/centroidal_reference_repair")
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
