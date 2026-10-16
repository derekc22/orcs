"""Diagnose early support dynamics for raw and repaired Stage-A references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.balance_mpc import BalanceMPCConfig, refine_balance_with_mpc
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    refine_generated_reference,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.support_diagnostics import (
    SupportDiagnosticResult,
    instrument_support_rollout,
)

plt.switch_backend("Agg")


_REPORT_FRAMES = (0, 5, 10, 15, 20, 25, 30, 34, 36, 40)


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


def _frame_table(
    result: SupportDiagnosticResult, substeps: int
) -> list[dict[str, Any]]:
    data = result.data
    rows = []
    for frame in _REPORT_FRAMES:
        sample = frame * substeps
        rows.append(
            {
                "frame": frame,
                "time_s": float(data["time_s"][sample]),
                "reference_root_height_m": float(
                    data["reference_root_position_m"][sample, 2]
                ),
                "actual_root_height_m": float(data["root_position_m"][sample, 2]),
                "reference_com_height_m": float(
                    data["reference_com_position_m"][sample, 2]
                ),
                "actual_com_height_m": float(data["com_position_m"][sample, 2]),
                "reference_com_vertical_velocity_m_s": float(
                    data["reference_com_velocity_m_s"][sample, 2]
                ),
                "actual_com_vertical_velocity_m_s": float(
                    data["com_velocity_m_s"][sample, 2]
                ),
                "required_vertical_force_n": float(
                    data["required_vertical_force_n"][sample]
                ),
                "actual_foot_vertical_grf_n": float(
                    data["total_support_force_world_n"][sample, 2]
                ),
                "all_robot_vertical_grf_n": float(
                    data["total_robot_ground_force_world_n"][sample, 2]
                ),
                "vertical_momentum_kg_m_s": float(
                    data["vertical_momentum_kg_m_s"][sample]
                ),
                "com_support_margin_m": (
                    None
                    if not np.isfinite(data["support_polygon_margin_com_m"][sample])
                    else float(data["support_polygon_margin_com_m"][sample])
                ),
                "cop_support_margin_m": (
                    None
                    if not np.isfinite(data["support_polygon_margin_cop_m"][sample])
                    else float(data["support_polygon_margin_cop_m"][sample])
                ),
                "reference_com_support_margin_m": (
                    None
                    if not np.isfinite(
                        data["support_polygon_margin_reference_com_m"][sample]
                    )
                    else float(data["support_polygon_margin_reference_com_m"][sample])
                ),
                "reference_required_cop_support_margin_m": (
                    None
                    if not np.isfinite(
                        data["support_polygon_margin_reference_cop_m"][sample]
                    )
                    else float(data["support_polygon_margin_reference_cop_m"][sample])
                ),
                "cop_xy_m": [
                    None if not np.isfinite(value) else float(value)
                    for value in data["cop_xy_m"][sample]
                ],
                "reference_required_cop_xy_m": [
                    None if not np.isfinite(value) else float(value)
                    for value in data["reference_required_cop_xy_m"][sample]
                ],
                "left_foot_force_world_n": data["foot_force_world_n"][
                    sample, 0
                ].tolist(),
                "right_foot_force_world_n": data["foot_force_world_n"][
                    sample, 1
                ].tolist(),
                "left_foot_moment_world_origin_nm": data["foot_moment_world_origin_nm"][
                    sample, 0
                ].tolist(),
                "right_foot_moment_world_origin_nm": data[
                    "foot_moment_world_origin_nm"
                ][sample, 1].tolist(),
                "left_right_slip_velocity_m_s": data["foot_slip_velocity_m_s"][
                    sample
                ].tolist(),
                "support_contact_count": int(data["support_contact_count"][sample]),
            }
        )
    return rows


def _save_npz(
    path: Path,
    raw: SupportDiagnosticResult,
    repaired: SupportDiagnosticResult,
) -> None:
    values: dict[str, np.ndarray] = {}
    for prefix, result in (("raw", raw), ("repaired", repaired)):
        for name, value in result.data.items():
            values[f"{prefix}__{name}"] = value
    np.savez_compressed(path, **values)


def _plot_heights(
    raw: SupportDiagnosticResult,
    repaired: SupportDiagnosticResult,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axis, result in zip(axes, (raw, repaired), strict=True):
        data, time = result.data, result.data["time_s"]
        axis.plot(
            time, data["reference_root_position_m"][:, 2], "k--", label="root desired"
        )
        axis.plot(time, data["root_position_m"][:, 2], label="root actual")
        axis.plot(
            time, data["reference_com_position_m"][:, 2], "--", label="COM desired"
        )
        axis.plot(time, data["com_position_m"][:, 2], label="COM actual")
        axis.set_title(result.label)
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("height (m)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_vertical_velocity(
    raw: SupportDiagnosticResult,
    repaired: SupportDiagnosticResult,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axis, result in zip(axes, (raw, repaired), strict=True):
        data, time = result.data, result.data["time_s"]
        axis.plot(
            time, data["reference_com_velocity_m_s"][:, 2], "k--", label="COM desired"
        )
        axis.plot(time, data["com_velocity_m_s"][:, 2], label="COM actual")
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set_title(result.label)
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("vertical velocity (m/s)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_vertical_force(
    raw: SupportDiagnosticResult,
    repaired: SupportDiagnosticResult,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axis, result in zip(axes, (raw, repaired), strict=True):
        data, time = result.data, result.data["time_s"]
        axis.plot(time, data["required_vertical_force_n"], "k--", label="required")
        axis.plot(time, data["total_support_force_world_n"][:, 2], label="feet GRF")
        axis.plot(
            time,
            data["total_robot_ground_force_world_n"][:, 2],
            ":",
            label="all robot GRF",
        )
        axis.set_title(result.label)
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("vertical force (N)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_support(
    raw: SupportDiagnosticResult,
    repaired: SupportDiagnosticResult,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for column, result in enumerate((raw, repaired)):
        data, time = result.data, result.data["time_s"]
        for row, coordinate in enumerate((0, 1)):
            axis = axes[row, column]
            lower = data["support_bounds_xy_m"][:, coordinate * 2]
            upper = data["support_bounds_xy_m"][:, coordinate * 2 + 1]
            axis.fill_between(time, lower, upper, alpha=0.2, label="contact bounds")
            axis.plot(time, data["com_position_m"][:, coordinate], label="COM")
            axis.plot(
                time,
                data["reference_com_position_m"][:, coordinate],
                "--",
                label="COM desired",
            )
            axis.plot(time, data["cop_xy_m"][:, coordinate], label="COP")
            axis.plot(
                time,
                data["reference_required_cop_xy_m"][:, coordinate],
                ":",
                label="required COP",
            )
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(result.label)
            if column == 0:
                axis.set_ylabel(f"{'x' if coordinate == 0 else 'y'} (m)")
            if row == 1:
                axis.set_xlabel("time (s)")
    axes[0, 1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _dominant_actuators(result: SupportDiagnosticResult, count: int = 6) -> list[int]:
    names = result.data["actuator_names"].tolist()
    balance = set(result.data["balance_joint_names"].tolist())
    indexes = [index for index, name in enumerate(names) if name in balance]
    peak = np.max(result.data["normalized_torque_usage"][:, indexes], axis=0)
    return [indexes[index] for index in np.argsort(peak)[::-1][:count]]


def _plot_actuation(
    raw: SupportDiagnosticResult,
    repaired: SupportDiagnosticResult,
    path: Path,
    *,
    field: str,
    ylabel: str,
) -> None:
    dominant = _dominant_actuators(repaired)
    names = repaired.data["actuator_names"].tolist()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, result in zip(axes, (raw, repaired), strict=True):
        for index in dominant:
            axis.plot(
                result.data["time_s"], result.data[field][:, index], label=names[index]
            )
        axis.set_title(result.label)
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _rank_actuation(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(name=name, **values) for name, values in summary["actuation"].items()]
    return sorted(rows, key=lambda row: row["peak_normalized_usage"], reverse=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = args.reference_artifact.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(artifact_path) as artifact:
        raw_trajectory_np = artifact["generated_trajectory"]
        repaired_trajectory_np = artifact["repaired_trajectory"]
        repaired_target = artifact["balance_joint_target"].astype(np.float64)

    device = torch.device(args.device)
    model = _build_model()
    adapter = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)
    raw_trajectory = torch.as_tensor(raw_trajectory_np, device=device)
    repaired_trajectory = torch.as_tensor(repaired_trajectory_np, device=device)
    raw_reference = adapter(raw_trajectory)
    repaired_reference = adapter(repaired_trajectory)
    dynamics_cfg = DynamicsRefinementConfig(
        balance_population=24,
        balance_elite_count=6,
        balance_generations=14,
        run_stage_b=False,
        seed=42,
    )
    mpc_cfg = BalanceMPCConfig(
        horizon_frames=12,
        replan_interval_frames=2,
        population=12,
        elite_count=3,
        generations=2,
    )
    raw_stage = refine_generated_reference(model, raw_reference, dynamics_cfg)
    raw_mpc = refine_balance_with_mpc(
        model, raw_reference, raw_stage, dynamics_cfg, mpc_cfg
    )
    raw = instrument_support_rollout(
        model,
        raw_reference,
        raw_mpc.joint_target,
        label="raw reference + MPC",
        config=dynamics_cfg,
    )
    repaired = instrument_support_rollout(
        model,
        repaired_reference,
        repaired_target,
        label="repaired reference + MPC",
        config=dynamics_cfg,
    )

    numeric_path = output / "support_diagnostics_start_000.npz"
    _save_npz(numeric_path, raw, repaired)
    plot_paths = {
        "height": output / "01_root_com_height.png",
        "vertical_velocity": output / "02_vertical_velocity.png",
        "vertical_force": output / "03_vertical_force.png",
        "support_geometry": output / "04_com_cop_support.png",
        "torque_usage": output / "05_dominant_torque_usage.png",
        "tracking_error": output / "06_dominant_tracking_error.png",
    }
    _plot_heights(raw, repaired, plot_paths["height"])
    _plot_vertical_velocity(raw, repaired, plot_paths["vertical_velocity"])
    _plot_vertical_force(raw, repaired, plot_paths["vertical_force"])
    _plot_support(raw, repaired, plot_paths["support_geometry"])
    _plot_actuation(
        raw,
        repaired,
        plot_paths["torque_usage"],
        field="normalized_torque_usage",
        ylabel="normalized torque usage",
    )
    _plot_actuation(
        raw,
        repaired,
        plot_paths["tracking_error"],
        field="actuator_position_error_rad",
        ylabel="target - position (rad)",
    )

    report: dict[str, Any] = {
        "experiment": "early_support_dynamics_diagnosis",
        "source_artifact": str(artifact_path),
        "constraints": {
            "controller_changed": False,
            "reference_changed": False,
            "physics_changed": False,
            "sampling_hz": 200,
            "reference_rate_hz": 50,
            "frames_inclusive": [0, 40],
        },
        "raw": {
            "summary": raw.summary,
            "selected_frames": _frame_table(raw, dynamics_cfg.substeps),
            "dominant_actuation": _rank_actuation(raw.summary),
        },
        "repaired": {
            "summary": repaired.summary,
            "selected_frames": _frame_table(repaired, dynamics_cfg.substeps),
            "dominant_actuation": _rank_actuation(repaired.summary),
        },
        "artifacts": {
            "numeric": str(numeric_path),
            "plots": {key: str(path) for key, path in plot_paths.items()},
        },
    }
    report_path = output / "support_diagnostics_start_000.json"
    report_path.write_text(json.dumps(_serializable(report), indent=2) + "\n")
    report["artifacts"]["report"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=Path(
            "logs/early_reference_repair/early_repaired_reference_start_000.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/support_dynamics_diagnosis")
    )
    args = parser.parse_args()
    print(json.dumps(_serializable(run(args)), indent=2))


if __name__ == "__main__":
    main()
