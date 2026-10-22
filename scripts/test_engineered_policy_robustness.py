#!/usr/bin/env python3
"""Deterministic perturbation suite for the engineered-task learned policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from engineered_policy_model import EngineeredTrackingPolicy
from evaluate_engineered_tracking_policy import run_policy, strict_success
from finalize_engineered_expert import replay


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/engineering_policy"
MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
EXPERT_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz"
CHECKPOINT_PATH = LOG / "engineered_tracking_policy_v1.pt"
TASK_STEPS = 161 * 4


PROFILES = {
    "mild": dict(joint_qpos=0.002, joint_qvel=0.010, root_xy=0.001, root_yaw=0.003, object_xy=0.001, object_yaw=0.004),
    "medium": dict(joint_qpos=0.005, joint_qvel=0.025, root_xy=0.003, root_yaw=0.008, object_xy=0.003, object_yaw=0.010),
}


def rotate_quaternion_wxyz(quat: np.ndarray, yaw: float) -> np.ndarray:
    rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
    changed = Rotation.from_rotvec([0.0, 0.0, yaw]) * rotation
    xyzw = changed.as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def perturbed_state(model: mujoco.MjModel, source: np.lib.npyio.NpzFile, checkpoint: dict, seed: int, profile: dict):
    rng = np.random.default_rng(seed)
    qpos = source["initial_qpos"].copy()
    qvel = source["initial_qvel"].copy()
    actuator_qpos = np.asarray(checkpoint["actuator_qpos"], dtype=int)
    actuator_dof = np.asarray(checkpoint["actuator_dof"], dtype=int)
    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    qpos[actuator_qpos] += rng.normal(0.0, profile["joint_qpos"], len(actuator_qpos))
    qvel[actuator_dof] += rng.normal(0.0, profile["joint_qvel"], len(actuator_dof))
    qpos[robot_qpos:robot_qpos + 2] += rng.normal(0.0, profile["root_xy"], 2)
    qpos[robot_qpos + 3:robot_qpos + 7] = rotate_quaternion_wxyz(
        qpos[robot_qpos + 3:robot_qpos + 7], rng.normal(0.0, profile["root_yaw"])
    )
    qpos[object_qpos:object_qpos + 2] += rng.normal(0.0, profile["object_xy"], 2)
    qpos[object_qpos + 3:object_qpos + 7] = rotate_quaternion_wxyz(
        qpos[object_qpos + 3:object_qpos + 7], rng.normal(0.0, profile["object_yaw"])
    )
    return qpos, qvel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--profiles", nargs="+", default=["mild", "medium"], choices=sorted(PROFILES))
    parser.add_argument("--feedback-scales", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = 0.005
    source = np.load(EXPERT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    policy = EngineeredTrackingPolicy(
        checkpoint["horizon_steps"], checkpoint["error_dim"], checkpoint["action_dim"]
    ).float()
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    goal = np.asarray(checkpoint["goal_position"], dtype=np.float64)
    goal_quat = np.asarray(checkpoint["goal_quaternion"], dtype=np.float64)
    trials = []
    for profile_index, profile_name in enumerate(args.profiles):
        profile = PROFILES[profile_name]
        for trial in range(args.trials):
            seed = 20261021 + profile_index * 1000 + trial
            qpos, qvel = perturbed_state(model, source, checkpoint, seed, profile)
            for feedback_scale in args.feedback_scales:
                controls, _, _, errors = run_policy(
                    model, policy, checkpoint, qpos, qvel, feedback_scale
                )
                metrics, _ = replay(model, qpos, qvel, controls[:TASK_STEPS], goal, goal_quat, True)
                continuation, _ = replay(model, qpos, qvel, controls, goal, goal_quat, False)
                result = {
                    "profile": profile_name,
                    "trial": trial,
                    "seed": seed,
                    "feedback_scale": feedback_scale,
                    "success": strict_success(metrics, continuation),
                    "position_error_m": metrics["final_position_error_m"],
                    "orientation_error_rad": metrics["final_orientation_error_rad"],
                    "maximum_lift_m": metrics["maximum_lift_m"],
                    "supported_lift_frames": metrics["longest_hand_supported_lift_frames"],
                    "carried_displacement_m": metrics["carried_displacement_m"],
                    "non_ee_impulse_ns": metrics["non_end_effector_robot_object_impulse_ns"],
                    "valid": metrics["full_horizon_valid"],
                    "continuation_valid": continuation["full_horizon_valid"],
                    "final_object_speed": metrics["final_object_speed"],
                    "maximum_tracking_error": float(np.max(np.abs(errors))),
                }
                trials.append(result)
                print(json.dumps(result), flush=True)
    summaries = []
    for profile_name in args.profiles:
        for feedback_scale in args.feedback_scales:
            group = [r for r in trials if r["profile"] == profile_name and r["feedback_scale"] == feedback_scale]
            summaries.append({
                "profile": profile_name,
                "perturbation_standard_deviations": PROFILES[profile_name],
                "feedback_scale": feedback_scale,
                "successes": sum(r["success"] for r in group),
                "trials": len(group),
                "success_rate": sum(r["success"] for r in group) / len(group),
                "maximum_position_error_m": max(r["position_error_m"] for r in group),
                "minimum_lift_m": min(r["maximum_lift_m"] for r in group),
                "minimum_supported_lift_frames": min(r["supported_lift_frames"] for r in group),
            })
    report = {
        "policy": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "task_frozen": True,
        "purpose": "closed-loop feedback ablation under deterministic initial-state perturbations",
        "summaries": summaries,
        "trials": trials,
    }
    output = LOG / f"robustness_report_{args.suffix}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT", output)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
