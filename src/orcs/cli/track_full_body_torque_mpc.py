"""Diagnose all-actuator authority in the frozen-reference MuJoCo shooting MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_contact_dynamics import _serializable
from orcs.cli.refine_diffusion_reference import _build_model
from orcs.cli.track_torque_shooting_mpc import _first_sustained_loss, _global_frame
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


def _anchor_difference(
    current: dict[str, Any], prior: dict[str, Any]
) -> dict[str, float]:
    keys = (
        "root_position_m",
        "root_velocity",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "foot_normal_force_left_right_n",
    )
    return {
        key: float(
            np.max(
                np.abs(
                    np.asarray(current[key], dtype=np.float64)
                    - np.asarray(prior[key], dtype=np.float64)
                )
            )
        )
        for key in keys
    }


def _metrics(
    result: TorqueShootingMPCResult,
    engine: _DynamicRollout,
    config: TorqueShootingMPCConfig,
    arm_actuators: np.ndarray,
) -> dict[str, Any]:
    data = result.data
    stop = (
        len(data["foot_contact"])
        if result.first_invalid_sample is None
        else result.first_invalid_sample + 1
    )
    left_loss = _first_sustained_loss(data["foot_contact"], 0)
    right_loss = _first_sustained_loss(data["foot_contact"], 1)
    both_off = ~np.any(data["foot_contact"][:stop], axis=1)
    root_body = int(engine.model.jnt_bodyid[engine.robot_free_joint])
    mass = float(engine.model.body_subtreemass[root_body])

    def sample(frame: int) -> int:
        return (frame - config.anchor_frame) * engine.config.substeps

    frame110, frame120, frame125 = sample(110), sample(120), sample(125)
    arm_usage = (
        float(np.nanmax(data["torque_usage"][:stop, arm_actuators]))
        if len(arm_actuators)
        else 0.0
    )
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
        "left_foot_force_frame125_n": float(data["foot_normal_force_n"][frame125, 0]),
        "left_foot_slip_frame125_m_s": float(data["foot_slip_m_s"][frame125, 0]),
        "maximum_balance_torque_usage": result.summary["maximum_balance_torque_usage"],
        "maximum_arm_torque_usage": arm_usage,
        "torque_clipping_duration_s": result.summary["torque_clipping_duration_s"],
        "forbidden_arm_support_duration_s": result.summary[
            "arm_terrain_contact_before_invalidation_s"
        ],
        "new_arm_object_contact_duration_s": result.summary[
            "new_arm_object_contact_duration_s"
        ],
        "new_arm_object_contact_before_invalidation_s": float(
            np.sum(data["new_arm_object_contact"][:stop])
            * engine.config.physics_dt
        ),
        "maximum_joint_position_error_rad": result.summary[
            "maximum_joint_position_error_rad"
        ],
        "maximum_joint_velocity_error_rad_s": result.summary[
            "maximum_joint_velocity_error_rad_s"
        ],
    }


def _upper_body_diagnostics(
    result: TorqueShootingMPCResult,
    prior_data: dict[str, np.ndarray],
    engine: _DynamicRollout,
    config: TorqueShootingMPCConfig,
    arm_actuators: np.ndarray,
) -> dict[str, Any]:
    data = result.data
    names = [engine.joint_names[index] for index in engine.actuator_joint_indexes]
    start = (95 - config.anchor_frame) * engine.config.substeps
    end = (135 - config.anchor_frame) * engine.config.substeps + 1
    arm_residual = data["torque_residual_nm"][start:end, arm_actuators]
    velocity_delta = (
        data["joint_velocity"][start:end, arm_actuators]
        - prior_data["joint_velocity"][start:end, arm_actuators]
    )
    channels: dict[str, Any] = {}
    for local, actuator in enumerate(arm_actuators):
        channels[names[actuator]] = {
            "residual_rms_nm": float(
                np.sqrt(np.mean(np.square(arm_residual[:, local])))
            ),
            "residual_max_nm": float(np.max(np.abs(arm_residual[:, local]))),
            "joint_velocity_delta_rms_rad_s": float(
                np.sqrt(np.mean(np.square(velocity_delta[:, local])))
            ),
            "joint_velocity_delta_max_rad_s": float(
                np.max(np.abs(velocity_delta[:, local]))
            ),
        }
    groups: dict[str, Any] = {}
    for group in ("shoulder", "elbow", "wrist"):
        selected = np.asarray(
            [
                index
                for index, actuator in enumerate(arm_actuators)
                if group in names[actuator]
            ],
            dtype=np.int32,
        )
        values = arm_residual[:, selected]
        groups[group] = {
            "residual_rms_nm": float(np.sqrt(np.mean(np.square(values)))),
            "residual_max_nm": float(np.max(np.abs(values))),
            "joint_velocity_delta_rms_rad_s": float(
                np.sqrt(np.mean(np.square(velocity_delta[:, selected])))
            ),
        }
    momentum_delta = (
        data["actual_angular_momentum"][start:end]
        - prior_data["actual_angular_momentum"][start:end]
    )
    return {
        "window_frames": [95, 135],
        "per_joint": channels,
        "by_group": groups,
        "whole_body_angular_momentum_delta_rms_kg_m2_s": np.sqrt(
            np.mean(np.square(momentum_delta), axis=0)
        ).tolist(),
        "roll_pitch_momentum_delta_rms_kg_m2_s": float(
            np.sqrt(np.mean(np.square(momentum_delta[:, :2])))
        ),
        "note": "Momentum delta is the measured whole-body outcome, not a per-limb decomposition.",
    }


def _critical_timeline(
    result: TorqueShootingMPCResult,
    prior_data: dict[str, np.ndarray],
    engine: _DynamicRollout,
    config: TorqueShootingMPCConfig,
    balance_actuators: np.ndarray,
    arm_actuators: np.ndarray,
) -> dict[str, Any]:
    data = result.data
    mass = float(
        engine.model.body_subtreemass[
            int(engine.model.jnt_bodyid[engine.robot_free_joint])
        ]
    )
    timeline: dict[str, Any] = {}
    for frame in (98, 100, 102, 103, 104, 106, 108, 110, 120, 125, 130, 135):
        sample = (frame - config.anchor_frame) * engine.config.substeps
        new_residual = data["torque_residual_nm"][sample]
        timeline[str(frame)] = {
            "right_contact_15_29": [
                bool(prior_data["foot_contact"][sample, 1]),
                bool(data["foot_contact"][sample, 1]),
            ],
            "right_force_15_29_n": [
                float(prior_data["foot_normal_force_n"][sample, 1]),
                float(data["foot_normal_force_n"][sample, 1]),
            ],
            "right_height_15_29_m": [
                float(prior_data["foot_height_m"][sample, 1]),
                float(data["foot_height_m"][sample, 1]),
            ],
            "right_slip_15_29_m_s": [
                float(prior_data["foot_slip_m_s"][sample, 1]),
                float(data["foot_slip_m_s"][sample, 1]),
            ],
            "vertical_momentum_15_29_kg_m_s": [
                float(mass * prior_data["actual_com_velocity"][sample, 2]),
                float(mass * data["actual_com_velocity"][sample, 2]),
            ],
            "torso_tilt_15_29_deg": [
                float(np.rad2deg(prior_data["torso_tilt_rad"][sample])),
                float(np.rad2deg(data["torso_tilt_rad"][sample])),
            ],
            "balance_residual_rms_nm": float(
                np.sqrt(np.mean(np.square(new_residual[balance_actuators])))
            ),
            "upper_residual_rms_nm": float(
                np.sqrt(np.mean(np.square(new_residual[arm_actuators])))
            ),
            "new_arm_terrain_contact": bool(data["arm_terrain_contact"][sample]),
            "new_arm_object_contact": bool(data["new_arm_object_contact"][sample]),
        }
    return timeline


def _classification(
    result: TorqueShootingMPCResult,
    metrics: dict[str, Any],
    replans: list[dict[str, Any]],
) -> tuple[str, str, str]:
    arm_cheat = (
        metrics["forbidden_arm_support_duration_s"] >= 0.1
        or metrics["new_arm_object_contact_duration_s"] >= 0.1
    )
    if result.valid and result.recoverable_terminal and not arm_cheat:
        return (
            "A",
            "full-body torque authority remained valid and recoverable through frame 160",
            "Use whole-body torque-MPC as the execution layer for renewed dynamics-aware reference refinement.",
        )
    failure = metrics["first_invalid_frame"] or 161
    right = metrics["first_right_foot_sustained_loss_frame"] or 161
    if failure >= 145 or right >= 113:
        return (
            "B",
            "full-body torque authority materially delayed failure or right-foot loss",
            "Use whole-body torque-MPC as the execution layer for renewed dynamics-aware reference refinement.",
        )
    baseline_wins = [
        generation["baseline_candidate_rank"] == 0
        for replan in replans
        for generation in replan["generations"]
    ]
    if baseline_wins and np.mean(baseline_wins) > 0.9 and failure < 130:
        return (
            "D",
            "the embedded 15-joint candidate dominated over 90% of generations and search quality collapsed",
            "Reduce the whole-body torque search to a lower-dimensional momentum-control basis.",
        )
    return (
        "C",
        "failure remains near frames 130-136 with the same sequential foot-loss mechanism",
        "Stop controller-space expansion and refine the reference around physical right-foot support retention near frame 103 using MuJoCo rollout feasibility.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = args.artifact.resolve()
    prior_path = args.prior_mpc_artifact.resolve()
    prior_report_path = args.prior_mpc_report.resolve()
    with np.load(source_path) as loaded:
        artifact = {key: loaded[key].copy() for key in loaded.files}
    with np.load(prior_path) as loaded:
        prior_data = {key: loaded[key].copy() for key in loaded.files}
    prior_report = json.loads(prior_report_path.read_text())

    trajectory = artifact["analytical_through140_trajectory"].copy()
    nominal_target = artifact["optimized_joint_target"].astype(np.float64)
    device = torch.device(args.device)
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        torch.as_tensor(trajectory, device=device)
    )
    dynamics = DynamicsRefinementConfig(run_stage_b=False, seed=42)
    engine = _DynamicRollout(_build_model(), reference, dynamics)
    config = TorqueShootingMPCConfig(
        control_all_actuators=True,
        population=args.population,
        elite_count=args.elites,
        generations=args.generations,
        seed=args.seed,
    )
    result = track_with_torque_shooting_mpc(
        engine,
        nominal_target,
        artifact,
        config,
        warm_start_residual=prior_data["torque_residual_nm"],
    )
    balance = np.flatnonzero(
        np.isin(engine.actuator_joint_indexes, engine.balance_indexes)
    )
    arms = np.setdiff1d(np.arange(engine.model.nu), balance)
    names = [engine.joint_names[index] for index in engine.actuator_joint_indexes]
    metrics = _metrics(result, engine, config, arms)
    classification, basis, next_action = _classification(
        result, metrics, result.replans
    )
    anchor_difference = _anchor_difference(result.prefix, prior_report["anchor"])
    if max(anchor_difference.values()) > 1.0e-12:
        raise RuntimeError(f"frame-85 anchor mismatch: {anchor_difference}")

    torque_limit = np.max(np.abs(engine.model.actuator_forcerange), axis=1)
    prior_metrics = dict(prior_report["comparison"]["torque_shooting_mpc"])
    prior_frame125 = (125 - config.anchor_frame) * engine.config.substeps
    prior_metrics.update(
        {
            "left_foot_force_frame125_n": float(
                prior_data["foot_normal_force_n"][prior_frame125, 0]
            ),
            "left_foot_slip_frame125_m_s": float(
                prior_data["foot_slip_m_s"][prior_frame125, 0]
            ),
        }
    )
    bounds = {
        names[index]: {
            "torque_limit_nm": float(torque_limit[index]),
            "residual_fraction": 0.25 if index in balance else 0.15,
            "residual_bound_nm": float(
                torque_limit[index] * (0.25 if index in balance else 0.15)
            ),
        }
        for index in range(engine.model.nu)
    }
    output_artifact = output / "full_body_torque_shooting_mpc_start_000.npz"
    np.savez_compressed(
        output_artifact,
        **result.data,
        physics_dt=np.asarray([dynamics.physics_dt]),
        anchor_frame=np.asarray([config.anchor_frame]),
        evaluation_end_frame=np.asarray([config.evaluation_end_frame]),
    )
    report: dict[str, Any] = {
        "experiment": "all_29_actuator_torque_shooting_mpc_authority_diagnostic",
        "frozen_reference": True,
        "source_artifact": str(source_path),
        "prior_15_joint_artifact": str(prior_path),
        "anchor_identical": True,
        "anchor_maximum_absolute_differences": anchor_difference,
        "control_space": {
            "all_29_joints": names,
            "balance_15_joints": [names[index] for index in balance],
            "additional_14_arm_wrist_joints": [names[index] for index in arms],
            "future_knots": config.future_knots,
            "dimensions": config.future_knots * engine.model.nu,
            "residual_bounds": bounds,
        },
        "mpc": {
            "horizon_frames": config.horizon_frames,
            "horizon_s": config.horizon_frames * 0.02,
            "replan_frames": config.replan_interval_frames,
            "replan_s": config.replan_interval_frames * 0.02,
            "population": config.population,
            "elites": config.elite_count,
            "generations": config.generations,
            "seed": config.seed,
            "prior_15_joint_candidate_explicitly_included": True,
            "new_arm_residuals_initialized_zero": True,
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
        "comparison": {
            "15_joint_mpc": prior_metrics,
            "29_joint_mpc": metrics,
        },
        "upper_body_contribution": _upper_body_diagnostics(
            result, prior_data, engine, config, arms
        ),
        "critical_right_foot_timeline": _critical_timeline(
            result, prior_data, engine, config, balance, arms
        ),
        "replans": result.replans,
        "classification": classification,
        "classification_basis": basis,
        "next_action": next_action,
        "artifact": str(output_artifact),
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
        "--prior-mpc-artifact",
        type=Path,
        default=Path("logs/torque_shooting_mpc/torque_shooting_mpc_start_000.npz"),
    )
    parser.add_argument(
        "--prior-mpc-report",
        type=Path,
        default=Path("logs/torque_shooting_mpc/report_start_000.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/full_body_torque_shooting_mpc")
    )
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=884)
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
