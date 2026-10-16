"""Run staged balance/manipulation refinement for one HF-BPS reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mocke.sonic import profile

from orcs.assets.g1 import get_g1_flat_hand_cfg
from orcs.assets.objects import omni_object_entity_cfg
from orcs.tasks.uolm.balance_mpc import (
    BalanceMPCConfig,
    refine_balance_with_mpc,
)
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


def _rotation_columns6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    matrix = matrix_from_quat(quat_wxyz)
    return matrix[..., :, :2].transpose(-1, -2).reshape(*quat_wxyz.shape[:-1], 6)


def _physical_object_frame(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    # Match DiffusionObjectMotionCommand's checkpoint-compatibility convention.
    return torch.cat([pos, _rotation_columns6d(quat_wxyz.roll(1, dims=-1))])


def _condition_from_sample(
    sample: Path, start: int, horizon: int, device: torch.device
) -> torch.Tensor:
    with np.load(sample / "motion.npz", allow_pickle=True) as motion:
        body_names = [str(name) for name in motion["body_names"]]
        pelvis = body_names.index("pelvis")
        end = start + horizon - 1
        if end >= len(motion["joint_pos"]):
            raise ValueError(
                f"conditioning start {start} with horizon {horizon} exceeds "
                f"the {len(motion['joint_pos'])}-frame source clip"
            )
        root_pos = torch.as_tensor(
            motion["body_pos_w"][start, pelvis], dtype=torch.float32, device=device
        )
        root_quat = torch.as_tensor(
            motion["body_quat_w"][start, pelvis], dtype=torch.float32, device=device
        )
        joint_pos = torch.as_tensor(
            motion["joint_pos"][start, 7:36], dtype=torch.float32, device=device
        )
    with np.load(sample / "object_motion.npz") as object_motion:
        initial_pos = torch.as_tensor(
            object_motion["obj_pos_w"][start], dtype=torch.float32, device=device
        )
        initial_quat = torch.as_tensor(
            object_motion["obj_quat_w"][start], dtype=torch.float32, device=device
        )
        goal_pos = torch.as_tensor(
            object_motion["obj_pos_w"][end], dtype=torch.float32, device=device
        )
        goal_quat = torch.as_tensor(
            object_motion["obj_quat_w"][end], dtype=torch.float32, device=device
        )
    robot_initial = torch.cat([root_pos, _rotation_columns6d(root_quat), joint_pos])
    return torch.cat(
        [
            _physical_object_frame(goal_pos, goal_quat),
            robot_initial,
            _physical_object_frame(initial_pos, initial_quat),
        ]
    ).unsqueeze(0)


def _build_model() -> mujoco.MjModel:
    scene = Scene(
        SceneCfg(
            num_envs=1,
            env_spacing=0.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": profile.robot_cfg(base=get_g1_flat_hand_cfg()),
                "object": omni_object_entity_cfg(("largebox",)),
            },
        ),
        device="cuda:0",
    )
    return scene.compile()


def _contact_intervals(contact: np.ndarray) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for side, index in (("left", 0), ("right", 1)):
        active = np.asarray(contact[:, index], dtype=bool)
        padded = np.r_[False, active, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        result[side] = [
            [int(changes[i]), int(changes[i + 1] - 1)]
            for i in range(0, len(changes), 2)
        ]
    return result


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def _behavior_summary(result: RolloutResult) -> dict[str, Any]:
    position = result.object_pos
    lift = position[:, 2] - position[0, 2]
    displacement = np.linalg.norm(position - position[0], axis=1)
    return {
        "object_initial_position": position[0].tolist(),
        "object_final_position": position[-1].tolist(),
        "max_lift_frame": int(np.argmax(lift)),
        "max_displacement_frame": int(np.argmax(displacement)),
        "minimum_root_height_frame": int(np.argmin(result.root_pos[:, 2])),
    }


def _rollout_report(result: RolloutResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "valid": result.valid,
        "invalid_reason": result.invalid_reason,
        "first_invalid_frame": result.first_invalid_frame,
        "objective": result.objective,
        "objective_terms": result.objective_terms,
        "metrics": _public_metrics(result.metrics),
        "hand_contact_intervals_frames": _contact_intervals(result.hand_contact),
        "physical_behavior": _behavior_summary(result),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.diffusion_source.resolve()
    checkpoint = args.checkpoint.resolve()
    sample = args.sample.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the existing HF-BPS generator requires a CUDA device")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    condition = _condition_from_sample(sample, args.start, 300, device)
    generator_cfg = DiffusionGeneratorCfg(
        source_path=str(source),
        checkpoint_path=str(checkpoint),
        sampler="ddim",
        num_inference_steps=50,
        eta=0.0,
        generation_batch_size=1,
        horizon=300,
        precision="fp32",
    )
    generator = BatchedDiffusionGenerator(generator_cfg, device)
    trajectory = generator.generate(condition).clone()

    model = _build_model()
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
    reference = GeneratedReferenceAdapter(device=device, dt=0.02, fk_batch_size=256)(
        trajectory
    )

    refinement_cfg = DynamicsRefinementConfig(
        balance_population=args.balance_population,
        balance_elite_count=min(args.balance_elite_count, args.balance_population),
        balance_generations=args.balance_generations,
        arm_population=args.arm_population,
        arm_elite_count=min(args.arm_elite_count, args.arm_population),
        arm_generations=args.arm_generations,
        run_stage_b=not (args.stage_a_only or args.mpc_stage_a_only),
        seed=args.seed,
    )
    result = refine_generated_reference(model, reference, refinement_cfg)
    mpc_config = BalanceMPCConfig(
        horizon_frames=args.mpc_horizon,
        replan_interval_frames=args.mpc_replan_interval,
        population=args.mpc_population,
        elite_count=min(args.mpc_elite_count, args.mpc_population),
        generations=args.mpc_generations,
    )
    mpc_result = None
    if args.mpc_stage_a_only:
        mpc_result = refine_balance_with_mpc(
            model, reference, result, refinement_cfg, mpc_config
        )

    classification = result.status
    classification_basis = None
    if args.stage_a_only:
        raw_frame = result.raw.first_invalid_frame or 0
        balance_frame = result.balance.first_invalid_frame or 0
        if result.stage_a_passed:
            classification = "A"
            classification_basis = "full-horizon stable baseline"
        elif balance_frame >= raw_frame + 20:
            classification = "B"
            classification_basis = "first failure delayed by at least 20 frames"
        elif balance_frame <= 10 and (
            result.balance.invalid_reason != result.raw.invalid_reason
        ):
            classification = "D"
            classification_basis = "different immediate failure"
        else:
            classification = "C"
            classification_basis = (
                "failure remained near the raw timing without a valid rollout"
            )
    elif args.mpc_stage_a_only:
        assert mpc_result is not None
        mpc_frame = mpc_result.rollout.first_invalid_frame or 0
        if mpc_result.rollout.valid:
            classification = "A"
            classification_basis = "full-horizon feedback-stable baseline"
        elif mpc_frame >= 100:
            classification = "B"
            classification_basis = "feedback delayed failure well into the horizon"
        else:
            classification = "C"
            classification_basis = "feedback remained in the early failure region"

    initial_root_height = float(result.raw.root_pos[0, 2])
    report: dict[str, Any] = {
        "experiment": "staged_dynamics_aware_reference_refinement_v2",
        "dataset": "sub3_largebox_003",
        "sample": str(sample),
        "conditioning_start": args.start,
        "diffusion": {
            "checkpoint": str(checkpoint),
            "sampler": "ddim",
            "num_inference_steps": 50,
            "eta": 0.0,
            "precision": "fp32",
            "seed": args.seed,
            "shape": list(trajectory.shape),
        },
        "simulation": {
            "physics_dt": refinement_cfg.physics_dt,
            "control_dt": refinement_cfg.control_dt,
            "substeps": refinement_cfg.substeps,
            "full_horizon_frames": 300,
            "object_writes_after_initialization": 0,
            "robot_morphology": "ORCS flat-hand G1",
            "object": "ORCS largebox convex decomposition",
        },
        "pickup_window_inclusive": list(result.window),
        "stage_a": {
            "passed": result.stage_a_passed,
            "joint_names": list(result.balance_joint_names),
            "num_joints": len(result.balance_joint_names),
            "num_knots": refinement_cfg.balance_num_knots,
            "interior_knots": refinement_cfg.balance_num_knots - 2,
            "knot_frames": result.optimizer["balance"]["knot_frames"],
            "initial_knot_optimized": (
                refinement_cfg.balance_optimize_initial_knot
            ),
            "initial_knot_bound_rad": (
                refinement_cfg.balance_initial_knot_bound_rad
            ),
            "later_knot_bound_rad": (
                refinement_cfg.balance_correction_bound_rad
            ),
            "final_knot_fixed_zero": True,
            "free_parameters": result.optimizer["balance"]["dimensions"],
            "interpolation": (
                "Catmull-Rom cubic; optimized first knot, zero final knot"
            ),
            "invalid_candidate_ranking": (
                "lexicographic: full-horizon valid first; otherwise latest "
                "first-invalid frame; then soft objective"
            ),
            "optimizer": result.optimizer["balance"],
        },
        "hard_balance_gates": {
            "minimum_root_height_m": refinement_cfg.root_height_fraction
            * initial_root_height,
            "minimum_root_height_fraction_of_initial": (
                refinement_cfg.root_height_fraction
            ),
            "maximum_torso_tilt_deg": refinement_cfg.max_torso_tilt_deg,
            "allowed_terrain_contact_bodies": [
                "robot/left_ankle_roll_link",
                "robot/right_ankle_roll_link",
            ],
            "immediate_invalid_terrain_contact": (
                "pelvis, torso, head, hip, knee, or ankle-pitch body"
            ),
            "maximum_sustained_other_nonfoot_terrain_contact_s": (
                refinement_cfg.sustained_nonfoot_terrain_contact_s
            ),
            "maximum_penetration_m": refinement_cfg.hard_penetration_m,
            "maximum_target_joint_limit_violation_rad": (
                refinement_cfg.joint_limit_margin_rad
            ),
            "sustained_saturation": {
                "usage": refinement_cfg.severe_saturation_usage,
                "actuator_fraction": (
                    refinement_cfg.severe_saturation_actuator_fraction
                ),
                "duration_s": refinement_cfg.severe_saturation_duration_s,
            },
        },
        "raw_diffusion_dynamic_rollout": _rollout_report(result.raw),
        "balance_refined_dynamic_rollout": _rollout_report(result.balance),
        "manipulation_refined_dynamic_rollout": _rollout_report(
            result.manipulation
        ),
        "prior_diagnostic": {
            "performed": result.stage_a_passed,
            "normal_arm_prior": refinement_cfg.manipulation_weights.arm_prior,
            "loose_arm_prior": refinement_cfg.loose_arm_prior,
            "normal": _rollout_report(result.normal_prior_manipulation),
            "loose": _rollout_report(result.loose_prior_manipulation),
            "selected_arm_prior": result.selected_arm_prior,
        },
        "manipulation_optimizer": result.optimizer["manipulation"],
        "mpc": None if mpc_result is None else {
            "horizon_frames": mpc_config.horizon_frames,
            "horizon_seconds": mpc_config.horizon_frames
            * refinement_cfg.control_dt,
            "replan_interval_frames": mpc_config.replan_interval_frames,
            "variables": mpc_config.num_future_knots
            * len(result.balance_joint_names),
            "num_future_knots": mpc_config.num_future_knots,
            "population": mpc_config.population,
            "elite_count": mpc_config.elite_count,
            "generations": mpc_config.generations,
            "residual_bound_rad": mpc_config.residual_bound_rad,
            "residual_rate_limit_rad_per_frame": (
                mpc_config.residual_rate_limit_rad_per_frame
            ),
            "warm_start": "shifted previous local solution",
            "weights": mpc_config.weights.as_dict(),
            "terminal_multiplier": mpc_config.weights.terminal_multiplier,
            "evaluations": mpc_result.evaluations,
            "optimization_seconds": mpc_result.optimization_seconds,
            "actual_first_invalid_frame": (
                mpc_result.actual_first_invalid_frame
            ),
            "actual_invalid_reason": mpc_result.actual_invalid_reason,
            "replans": mpc_result.replans,
            "rollout": _rollout_report(mpc_result.rollout),
            "state_trace": {
                "root_height_m": mpc_result.rollout.root_pos[:, 2].tolist(),
                "torso_tilt_deg": np.rad2deg(
                    mpc_result.rollout.torso_tilt_rad
                ).tolist(),
                "foot_contact": mpc_result.rollout.foot_contact.tolist(),
                "foot_contact_intervals_frames": _contact_intervals(
                    mpc_result.rollout.foot_contact
                ),
            },
        },
        "classification": classification,
        "classification_basis": classification_basis,
        "artifact": None,
    }

    artifact = output / f"refined_reference_start_{args.start:03d}.npz"
    balance_artifact = output / f"balance_baseline_start_{args.start:03d}.npz"
    if args.mpc_stage_a_only and mpc_result is not None and mpc_result.rollout.valid:
        reference_joint = reference.joint_pos[0].cpu().numpy()
        np.savez_compressed(
            balance_artifact,
            joint_target=mpc_result.joint_target.astype(np.float32),
            joint_correction=(
                mpc_result.joint_target - reference_joint
            ).astype(np.float32),
            feedback_residual=mpc_result.feedback_residual.astype(np.float32),
            simulated_root_pos=mpc_result.rollout.root_pos.astype(np.float32),
            dt=np.asarray([reference.dt], dtype=np.float32),
        )
        report["artifact"] = str(balance_artifact)
    elif args.mpc_stage_a_only and balance_artifact.exists():
        balance_artifact.unlink()
    if args.stage_a_only and result.stage_a_passed:
        reference_joint = reference.joint_pos[0].cpu().numpy()
        np.savez_compressed(
            balance_artifact,
            joint_target=result.balance_joint_target.astype(np.float32),
            joint_correction=(
                result.balance_joint_target - reference_joint
            ).astype(np.float32),
            simulated_root_pos=result.balance.root_pos.astype(np.float32),
            dt=np.asarray([reference.dt], dtype=np.float32),
        )
        report["artifact"] = str(balance_artifact)
    elif balance_artifact.exists():
        balance_artifact.unlink()
    if result.status in ("A", "B"):
        assert result.candidate_joint_target is not None
        assert result.refined_object_pos is not None
        assert result.refined_object_quat_wxyz is not None
        assert result.manipulation is not None
        np.savez_compressed(
            artifact,
            root_pos_local=reference.root_pos_local[0].cpu().numpy(),
            root_quat_wxyz=reference.root_quat_wxyz[0].cpu().numpy(),
            joint_pos=result.candidate_joint_target.astype(np.float32),
            simulated_root_pos=result.manipulation.root_pos.astype(np.float32),
            object_pos_local=result.refined_object_pos.astype(np.float32),
            object_quat_wxyz=result.refined_object_quat_wxyz.astype(np.float32),
            physical_object_frames=np.asarray([0, 299], dtype=np.int32),
            pickup_window=np.asarray(result.window, dtype=np.int32),
            dt=np.asarray([reference.dt], dtype=np.float32),
        )
        report["artifact"] = str(artifact)
    elif artifact.exists():
        artifact.unlink()

    report_path = output / f"report_start_{args.start:03d}.json"
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
    parser.add_argument("--start", type=int, choices=(0, 10, 20), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--balance-population", type=int, default=24)
    parser.add_argument("--balance-elite-count", type=int, default=6)
    parser.add_argument("--balance-generations", type=int, default=14)
    parser.add_argument("--arm-population", type=int, default=24)
    parser.add_argument("--arm-elite-count", type=int, default=6)
    parser.add_argument("--arm-generations", type=int, default=12)
    parser.add_argument(
        "--stage-a-only",
        action="store_true",
        help="stop after balance refinement even if the full-horizon gate passes",
    )
    parser.add_argument(
        "--mpc-stage-a-only",
        action="store_true",
        help="run feedback balance MPC around Stage-A feedforward and stop",
    )
    parser.add_argument("--mpc-horizon", type=int, default=12)
    parser.add_argument("--mpc-replan-interval", type=int, default=2)
    parser.add_argument("--mpc-population", type=int, default=12)
    parser.add_argument("--mpc-elite-count", type=int, default=3)
    parser.add_argument("--mpc-generations", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/dynamics_refinement")
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
