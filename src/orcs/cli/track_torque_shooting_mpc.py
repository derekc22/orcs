"""Execute the frozen reference with short-horizon MuJoCo torque shooting MPC."""

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
from orcs.tasks.uolm.torque_shooting_mpc import (
    TorqueShootingMPCConfig,
    TorqueShootingMPCResult,
    track_with_torque_shooting_mpc,
)

_OPEN_LOOP = {
    "first_invalid_frame": 134,
    "invalid_reason": "torso_tilt",
    "first_right_foot_sustained_loss_frame": 103,
    "first_left_foot_sustained_loss_frame": 128,
    "both_feet_off_duration_s": 0.170,
    "root_height_frame110_m": None,
    "root_height_frame120_m": 0.616,
    "vertical_momentum_frame120_kg_m_s": -5.81,
    "torso_tilt_frame120_deg": 29.4,
    "maximum_joint_position_error_rad": 0.652,
    "maximum_joint_velocity_error_rad_s": 31.22,
    "maximum_balance_torque_usage": 0.412,
    "torque_clipping_duration_s": None,
    "arm_terrain_contact_duration_s": None,
}

_COMPLIANT_WBC = {
    "first_invalid_frame": 117,
    "invalid_reason": "root_height",
    "first_right_foot_sustained_loss_frame": 96,
    "first_left_foot_sustained_loss_frame": 87,
    "both_feet_off_duration_s": 0.255,
    "root_height_frame110_m": 0.628,
    "root_height_frame120_m": 0.379,
    "vertical_momentum_frame120_kg_m_s": -18.91,
    "torso_tilt_frame120_deg": 41.5,
    "maximum_joint_position_error_rad": 1.111,
    "maximum_joint_velocity_error_rad_s": 18.99,
    "maximum_balance_torque_usage": 1.0,
    "torque_clipping_duration_s": None,
    "arm_terrain_contact_duration_s": 0.850,
}


def _first_sustained_loss(contact: np.ndarray, side: int, count: int = 4) -> int | None:
    absent = ~contact[:, side]
    runs = np.convolve(absent.astype(np.int32), np.ones(count, dtype=np.int32), "valid")
    found = np.flatnonzero(runs == count)
    return None if not len(found) else int(found[0])


def _global_frame(sample: int | None, config: TorqueShootingMPCConfig) -> int | None:
    return None if sample is None else config.anchor_frame + int(np.ceil(sample / 4))


def _comparison(
    result: TorqueShootingMPCResult,
    engine: _DynamicRollout,
    config: TorqueShootingMPCConfig,
) -> dict[str, Any]:
    data = result.data
    stop = len(data["foot_contact"]) if result.first_invalid_sample is None else result.first_invalid_sample + 1
    left_loss = _first_sustained_loss(data["foot_contact"], 0)
    right_loss = _first_sustained_loss(data["foot_contact"], 1)
    both_off = ~np.any(data["foot_contact"][:stop], axis=1)
    root_body = int(engine.model.jnt_bodyid[engine.robot_free_joint])
    mass = float(engine.model.body_subtreemass[root_body])
    frame110 = (110 - config.anchor_frame) * engine.config.substeps
    frame120 = (120 - config.anchor_frame) * engine.config.substeps
    return {
        "first_invalid_frame": result.first_invalid_frame,
        "invalid_reason": result.invalid_reason,
        "first_right_foot_sustained_loss_frame": _global_frame(right_loss, config),
        "first_left_foot_sustained_loss_frame": _global_frame(left_loss, config),
        "both_feet_off_duration_s": float(np.sum(both_off) * engine.config.physics_dt),
        "root_height_frame110_m": float(data["root_position"][frame110, 2]),
        "root_height_frame120_m": float(data["root_position"][frame120, 2]),
        "vertical_momentum_frame120_kg_m_s": float(
            mass * data["actual_com_velocity"][frame120, 2]
        ),
        "torso_tilt_frame120_deg": float(np.rad2deg(data["torso_tilt_rad"][frame120])),
        "maximum_joint_position_error_rad": result.summary[
            "maximum_joint_position_error_rad"
        ],
        "maximum_joint_velocity_error_rad_s": result.summary[
            "maximum_joint_velocity_error_rad_s"
        ],
        "maximum_balance_torque_usage": result.summary[
            "maximum_balance_torque_usage"
        ],
        "torque_clipping_duration_s": result.summary["torque_clipping_duration_s"],
        "torque_clipping_fraction": result.summary["torque_clipping_fraction"],
        "arm_terrain_contact_duration_s": result.summary[
            "arm_terrain_contact_duration_s"
        ],
        "arm_terrain_contact_before_invalidation_s": result.summary[
            "arm_terrain_contact_before_invalidation_s"
        ],
        "arm_object_contact_duration_s": result.summary[
            "arm_object_contact_duration_s"
        ],
        "new_arm_object_contact_duration_s": result.summary[
            "new_arm_object_contact_duration_s"
        ],
        "root_position_rms_error_m": result.summary["root_position_rms_error_m"],
        "com_position_rms_error_m": result.summary["com_position_rms_error_m"],
        "com_velocity_rms_error_m_s": result.summary["com_velocity_rms_error_m_s"],
    }


def _timeline(
    result: TorqueShootingMPCResult,
    engine: _DynamicRollout,
    config: TorqueShootingMPCConfig,
) -> dict[str, Any]:
    data = result.data
    names = [engine.joint_names[index] for index in engine.actuator_joint_indexes]
    timeline: dict[str, Any] = {}
    for frame in (85, 90, 95, 100, 103, 105, 110, 115, 120, 125, 128, 130, 135, 140, 150, 160):
        sample = (frame - config.anchor_frame) * engine.config.substeps
        residual = data["torque_residual_nm"][sample]
        order = np.argsort(np.abs(residual))[::-1][:5]
        timeline[str(frame)] = {
            "foot_contact_left_right": data["foot_contact"][sample].tolist(),
            "foot_normal_force_left_right_n": data["foot_normal_force_n"][sample].tolist(),
            "foot_slip_left_right_m_s": data["foot_slip_m_s"][sample].tolist(),
            "foot_height_left_right_m": data["foot_height_m"][sample].tolist(),
            "root_height_m": float(data["root_position"][sample, 2]),
            "com_vertical_velocity_m_s": float(data["actual_com_velocity"][sample, 2]),
            "vertical_momentum_kg_m_s": float(
                engine.model.body_subtreemass[
                    int(engine.model.jnt_bodyid[engine.robot_free_joint])
                ]
                * data["actual_com_velocity"][sample, 2]
            ),
            "torso_tilt_deg": float(np.rad2deg(data["torso_tilt_rad"][sample])),
            "dominant_torque_channels": {
                names[index]: {
                    "feedforward_nm": float(data["feedforward_torque_nm"][sample, index]),
                    "residual_nm": float(residual[index]),
                    "requested_nm": float(data["requested_torque_nm"][sample, index]),
                    "actual_nm": float(data["actual_torque_nm"][sample, index]),
                    "usage": float(data["torque_usage"][sample, index]),
                }
                for index in order
            },
            "any_torque_clipping": bool(np.any(data["torque_clipped"][sample])),
            "arm_terrain_contact": bool(data["arm_terrain_contact"][sample]),
            "arm_object_contact": bool(data["arm_object_contact"][sample]),
            "new_arm_object_contact": bool(
                data["new_arm_object_contact"][sample]
            ),
        }
    return timeline


def _dominant_actuation(
    result: TorqueShootingMPCResult, engine: _DynamicRollout
) -> dict[str, Any]:
    data = result.data
    names = [engine.joint_names[index] for index in engine.actuator_joint_indexes]
    balance = np.flatnonzero(
        np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
    )
    residual_rms = np.sqrt(np.mean(np.square(data["torque_residual_nm"][:, balance]), axis=0))
    residual_max = np.max(np.abs(data["torque_residual_nm"][:, balance]), axis=0)
    feedforward_rms = np.sqrt(np.nanmean(np.square(data["feedforward_torque_nm"][:, balance]), axis=0))
    requested_rms = np.sqrt(np.nanmean(np.square(data["requested_torque_nm"][:, balance]), axis=0))
    order = np.argsort(residual_rms)[::-1]
    return {
        "dominant_balance_channels": {
            names[balance[index]]: {
                "feedforward_rms_nm": float(feedforward_rms[index]),
                "residual_rms_nm": float(residual_rms[index]),
                "residual_max_nm": float(residual_max[index]),
                "requested_rms_nm": float(requested_rms[index]),
            }
            for index in order[:10]
        },
        "overall_residual_rms_nm": float(
            np.sqrt(np.mean(np.square(data["torque_residual_nm"][:, balance])))
        ),
        "overall_residual_max_nm": float(
            np.max(np.abs(data["torque_residual_nm"][:, balance]))
        ),
    }


def _classification(
    result: TorqueShootingMPCResult,
    comparison: dict[str, Any],
) -> tuple[str, str, str]:
    arm_cheat = (
        comparison["arm_terrain_contact_duration_s"] >= 0.1
        or comparison["new_arm_object_contact_duration_s"] >= 0.1
    )
    if result.valid and result.recoverable_terminal and not arm_cheat:
        return (
            "A",
            "valid and recoverable through frame 160 without sustained arm support",
            "Use this MuJoCo shooting MPC as the execution/evaluation layer for automatic rolling dynamics-aware reference refinement.",
        )
    right = comparison["first_right_foot_sustained_loss_frame"] or 161
    left = comparison["first_left_foot_sustained_loss_frame"] or 161
    failure = comparison["first_invalid_frame"] or 161
    if failure >= 145 or (right >= 113 and left >= 138 and comparison["both_feet_off_duration_s"] < 0.12):
        return (
            "B",
            "failure/support loss moved materially later under predictive torque feedback",
            "Extend the torque-MPC prediction horizon once to cover the measured remaining support transition.",
        )
    if result.summary["candidate_rollouts"] > 0 and result.summary["runtime_s"] > 3600.0:
        return (
            "D",
            "bounded physical CEM was computationally ineffective for this horizon",
            "Replace CEM with a more sample-efficient derivative-free shooting optimizer while retaining MuJoCo rollout physics.",
        )
    return (
        "C",
        "failure remains near frame 134 with the same physical foot-loss mechanism",
        "Diagnose whether the frozen reference needs actuation outside the 15-joint balance torque space or must be changed further.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact.resolve()
    with np.load(artifact_path) as loaded:
        artifact = {key: loaded[key].copy() for key in loaded.files}
    trajectory = artifact["analytical_through140_trajectory"].copy()
    nominal_target = artifact["optimized_joint_target"].astype(np.float64)
    device = torch.device(args.device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        torch.as_tensor(trajectory, device=device)
    )
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    engine = _DynamicRollout(_build_model(), reference, dynamics)
    config = TorqueShootingMPCConfig(
        population=args.population,
        elite_count=args.elites,
        generations=args.generations,
        seed=args.seed,
    )
    result = track_with_torque_shooting_mpc(engine, nominal_target, artifact, config)
    comparison = _comparison(result, engine, config)
    classification, basis, next_action = _classification(result, comparison)
    reproducibility: dict[str, Any] = {"run": False, "reason": "success gate not met"}
    if classification == "A":
        repeat = track_with_torque_shooting_mpc(engine, nominal_target, artifact, config)
        reproducibility = {
            "run": True,
            "first_invalid_frame": repeat.first_invalid_frame,
            "recoverable_terminal": repeat.recoverable_terminal,
            "maximum_root_position_difference_m": float(
                np.max(np.abs(result.data["root_position"] - repeat.data["root_position"]))
            ),
            "identical_foot_contact": bool(
                np.array_equal(result.data["foot_contact"], repeat.data["foot_contact"])
            ),
        }
    artifact_output = output / "torque_shooting_mpc_start_000.npz"
    np.savez_compressed(
        artifact_output,
        **result.data,
        physics_dt=np.asarray([dynamics.physics_dt]),
        anchor_frame=np.asarray([config.anchor_frame]),
        evaluation_end_frame=np.asarray([config.evaluation_end_frame]),
    )
    report: dict[str, Any] = {
        "experiment": "mujoco_shooting_torque_residual_mpc",
        "source_artifact": str(artifact_path),
        "frozen_reference": True,
        "analytical_contact_constraints": False,
        "formulation": {
            "prediction_horizon_frames": config.horizon_frames,
            "prediction_horizon_s": config.horizon_frames * 0.02,
            "replan_interval_frames": config.replan_interval_frames,
            "replan_interval_s": config.replan_interval_frames * 0.02,
            "future_residual_knots": config.future_knots,
            "controlled_joints": [
                engine.joint_names[index] for index in engine.balance_indexes
            ],
            "dimensions_per_solve": config.future_knots * len(engine.balance_indexes),
            "residual_limit_fraction_of_torque_limit": config.residual_limit_fraction,
            "actuator_interface": "tau feedforward + residual converted online through qtarget=q+(tau+Kd*qdot)/Kp",
            "arms": "zero torque residual; saved feedforward torque only",
            "contact_model": "unmodified MuJoCo forward simulation only",
        },
        "cem": {
            "population": config.population,
            "elites": config.elite_count,
            "generations": config.generations,
            "seed": config.seed,
            "zero_residual_candidate_in_every_generation": True,
        },
        "compute": {
            key: result.summary[key]
            for key in (
                "replans",
                "candidate_rollouts",
                "candidate_simulated_steps",
                "applied_simulated_steps",
                "total_mujoco_steps",
                "runtime_s",
            )
        },
        "anchor": result.prefix,
        "comparison": {
            "open_loop_position_targets": _OPEN_LOOP,
            "compliant_contact_wbc": _COMPLIANT_WBC,
            "torque_shooting_mpc": comparison,
        },
        "support_actuation_timeline": _timeline(result, engine, config),
        "dominant_actuation": _dominant_actuation(result, engine),
        "replans": result.replans,
        "reproducibility": reproducibility,
        "classification": classification,
        "classification_basis": basis,
        "next_action": next_action,
        "artifact": str(artifact_output),
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
        "--output", type=Path, default=Path("logs/torque_shooting_mpc")
    )
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=884)
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
