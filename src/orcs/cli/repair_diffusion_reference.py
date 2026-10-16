"""Repair and dynamically validate the early HF-BPS balance reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import (
    _build_model,
    _condition_from_sample,
    _contact_intervals,
    _rollout_report,
)
from orcs.tasks.uolm.balance_mpc import BalanceMPCConfig, refine_balance_with_mpc
from orcs.tasks.uolm.diffusion_runtime import (
    BatchedDiffusionGenerator,
    DiffusionGeneratorCfg,
)
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    RolloutResult,
    refine_generated_reference,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.reference_repair import (
    EarlyReferenceRepairConfig,
    repair_early_reference,
)


def _compact_rollout(result: RolloutResult) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "first_invalid_frame": result.first_invalid_frame,
        "invalid_reason": result.invalid_reason,
        "minimum_root_height_m": metrics["minimum_root_height_m"],
        "maximum_torso_tilt_deg": metrics["maximum_torso_tilt_deg"],
        "minimum_left_wrist_clearance_m": metrics["minimum_left_wrist_clearance_m"],
        "minimum_right_wrist_clearance_m": metrics["minimum_right_wrist_clearance_m"],
        "nonfoot_terrain_contact_duration_s": metrics[
            "nonfoot_terrain_contact_duration_s"
        ],
        "max_actuator_usage": metrics["max_actuator_usage"],
    }


def _generate(args: argparse.Namespace) -> tuple[torch.Tensor, Any, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the HF-BPS diffusion generator requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    condition = _condition_from_sample(args.sample, args.start, 300, device)
    generator = BatchedDiffusionGenerator(
        DiffusionGeneratorCfg(
            source_path=str(args.diffusion_source),
            checkpoint_path=str(args.checkpoint),
            sampler="ddim",
            num_inference_steps=50,
            eta=0.0,
            generation_batch_size=1,
            horizon=300,
            precision="fp32",
        ),
        device,
    )
    trajectory = generator.generate(condition).clone()
    model = _build_model()
    adapter = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)
    # Match the existing integration: generated hinge targets are clipped once.
    import mujoco

    hinge_ids = sorted(
        (
            index
            for index in range(model.njnt)
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
            and (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or ""
            ).startswith("robot/")
        ),
        key=lambda index: int(model.jnt_qposadr[index]),
    )
    limits = torch.as_tensor(
        model.jnt_range[hinge_ids], dtype=torch.float32, device=device
    )
    trajectory[..., 9:38].clamp_(limits[:, 0], limits[:, 1])
    return trajectory, model, adapter


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.diffusion_source = args.diffusion_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.sample = args.sample.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory, model, adapter = _generate(args)
    raw_reference = adapter(trajectory)

    repair_cfg = EarlyReferenceRepairConfig(
        population=args.repair_population,
        elite_count=min(args.repair_elite_count, args.repair_population),
        generations=args.repair_generations,
        seed=args.seed + 100,
    )
    repair = repair_early_reference(model, trajectory, raw_reference, repair_cfg)
    report: dict[str, Any] = {
        "experiment": "early_support_feasible_reference_repair",
        "dataset": "sub3_largebox_003",
        "conditioning_start": args.start,
        "diffusion": {
            "checkpoint": str(args.checkpoint),
            "sampler": "ddim",
            "steps": 50,
            "eta": 0.0,
            "precision": "fp32",
            "seed": args.seed,
        },
        "repair": {
            "frames_inclusive": [repair_cfg.start_frame, repair_cfg.end_frame],
            "optimized_variables": (
                "root xyz + root roll/pitch + 12 leg + 3 waist at two "
                "free interior controls"
            ),
            "arms_unchanged": bool(
                torch.equal(
                    repair.trajectory[..., 9:38][
                        ...,
                        [
                            repair.joint_names.index(name)
                            for name in (
                                "left_shoulder_pitch_joint",
                                "left_shoulder_roll_joint",
                                "left_shoulder_yaw_joint",
                                "left_elbow_joint",
                                "left_wrist_roll_joint",
                                "left_wrist_pitch_joint",
                                "left_wrist_yaw_joint",
                                "right_shoulder_pitch_joint",
                                "right_shoulder_roll_joint",
                                "right_shoulder_yaw_joint",
                                "right_elbow_joint",
                                "right_wrist_roll_joint",
                                "right_wrist_pitch_joint",
                                "right_wrist_yaw_joint",
                            )
                        ],
                    ],
                    trajectory[..., 9:38][
                        ...,
                        [
                            repair.joint_names.index(name)
                            for name in (
                                "left_shoulder_pitch_joint",
                                "left_shoulder_roll_joint",
                                "left_shoulder_yaw_joint",
                                "left_elbow_joint",
                                "left_wrist_roll_joint",
                                "left_wrist_pitch_joint",
                                "left_wrist_yaw_joint",
                                "right_shoulder_pitch_joint",
                                "right_shoulder_roll_joint",
                                "right_shoulder_yaw_joint",
                                "right_elbow_joint",
                                "right_wrist_roll_joint",
                                "right_wrist_pitch_joint",
                                "right_wrist_yaw_joint",
                            )
                        ],
                    ],
                )
            ),
            "object_unchanged": bool(
                torch.equal(repair.trajectory[..., 38:47], trajectory[..., 38:47])
            ),
            "wrist_clearance_margin_m": repair_cfg.wrist_clearance_margin_m,
            "support_foot_distance_m": repair_cfg.support_foot_distance_m,
            "maximum_foot_penetration_m": (repair_cfg.maximum_foot_penetration_m),
            "support_projection_margin_m": (repair_cfg.support_projection_margin_m),
            "optimizer": repair.optimizer,
            "history": repair.history,
        },
        "kinematic": {
            "raw": repair.raw.as_dict(),
            "repaired": repair.repaired.as_dict(),
        },
        "dynamic_validation_performed": False,
        "dynamic": None,
        "classification": "D",
        "classification_basis": (
            "no feasible early kinematic repair satisfying clearance/support gates"
        ),
        "artifact": None,
    }

    if repair.repaired.feasible:
        repaired_reference = adapter(repair.trajectory)
        dynamics_cfg = DynamicsRefinementConfig(
            balance_population=args.balance_population,
            balance_elite_count=min(args.balance_elite_count, args.balance_population),
            balance_generations=args.balance_generations,
            run_stage_b=False,
            seed=args.seed,
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
        repaired_stage = refine_generated_reference(
            model, repaired_reference, dynamics_cfg
        )
        repaired_mpc = refine_balance_with_mpc(
            model, repaired_reference, repaired_stage, dynamics_cfg, mpc_cfg
        )
        report["dynamic_validation_performed"] = True
        report["dynamic"] = {
            "controller": {
                "open_loop": raw_stage.optimizer["balance"],
                "mpc": {
                    "horizon_frames": mpc_cfg.horizon_frames,
                    "replan_interval_frames": mpc_cfg.replan_interval_frames,
                    "population": mpc_cfg.population,
                    "elite_count": mpc_cfg.elite_count,
                    "generations": mpc_cfg.generations,
                    "settings_changed_from_previous_experiment": False,
                },
            },
            "raw": _compact_rollout(raw_stage.raw),
            "open_loop_raw_reference": _compact_rollout(raw_stage.balance),
            "mpc_raw_reference": _compact_rollout(raw_mpc.rollout),
            "repaired_reference": _compact_rollout(repaired_mpc.rollout),
            "repaired_reference_diagnostics": {
                "root_height_m": {
                    str(frame): {
                        "reference": float(
                            repaired_reference.root_pos_local[0, frame, 2]
                        ),
                        "physical": float(repaired_mpc.rollout.root_pos[frame, 2]),
                    }
                    for frame in (20, 26, 30, 34, 36, 40)
                },
                "foot_contact_intervals_frames": _contact_intervals(
                    repaired_mpc.rollout.foot_contact
                ),
            },
            "repaired_open_loop": _rollout_report(repaired_stage.balance),
            "repaired_mpc": _rollout_report(repaired_mpc.rollout),
        }
        raw_mpc_frame = raw_mpc.rollout.first_invalid_frame or 0
        repaired_frame = repaired_mpc.rollout.first_invalid_frame or 300
        if repaired_mpc.rollout.valid:
            classification = "A"
            basis = "kinematic repair succeeded and MPC remained valid for 300 frames"
        elif repaired_frame >= raw_mpc_frame + 20:
            classification = "B"
            basis = (
                "kinematic repair succeeded and dynamic failure moved at least "
                "20 frames later"
            )
        elif repaired_frame <= 40:
            classification = "C"
            basis = (
                "kinematic repair succeeded but dynamic failure remained in "
                "frames 20-40"
            )
        else:
            classification = "E"
            basis = "kinematic repair succeeded with a changed dynamic failure"
        report["classification"] = classification
        report["classification_basis"] = basis

        artifact = args.output / f"early_repaired_reference_start_{args.start:03d}.npz"
        np.savez_compressed(
            artifact,
            generated_trajectory=trajectory.detach().cpu().numpy(),
            repaired_trajectory=repair.trajectory.detach().cpu().numpy(),
            repair_correction=repair.correction.astype(np.float32),
            balance_joint_target=repaired_mpc.joint_target.astype(np.float32),
            feedback_residual=repaired_mpc.feedback_residual.astype(np.float32),
            simulated_root_pos=repaired_mpc.rollout.root_pos.astype(np.float32),
            repair_frames=np.asarray([0, repair_cfg.end_frame], dtype=np.int32),
            dt=np.asarray([0.02], dtype=np.float32),
        )
        report["artifact"] = str(artifact)

    report_path = args.output / f"report_start_{args.start:03d}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["report"] = str(report_path)
    return report


def main() -> None:
    root = Path("/home/learning/Documents/g1-diffusion")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-source", type=Path, default=root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / (
            "logs/object_goal_single_stage_init_goal_hf_bps_e10000_b16_lr5e-06_"
            "ts1000_w300_s10_transformer_2026Aug05_21-21-42/checkpoints/"
            "object_goal_single_stage_init_goal_epoch_000999.pt"
        ),
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=root
        / "data/hf_dataset/data/unitree_g1/omomo/sub3_largebox_003/sample1",
    )
    parser.add_argument("--start", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repair-population", type=int, default=48)
    parser.add_argument("--repair-elite-count", type=int, default=8)
    parser.add_argument("--repair-generations", type=int, default=20)
    parser.add_argument("--balance-population", type=int, default=24)
    parser.add_argument("--balance-elite-count", type=int, default=6)
    parser.add_argument("--balance-generations", type=int, default=14)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/early_reference_repair")
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
