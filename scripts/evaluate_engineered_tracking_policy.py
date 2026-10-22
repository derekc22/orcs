#!/usr/bin/env python3
"""Run and strictly evaluate the learned engineered-task tracking policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from engineered_policy_model import EngineeredTrackingPolicy
from finalize_engineered_expert import replay


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/engineering_policy"
MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
EXPERT_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_release_retraction_v1.npz"
CHECKPOINT_PATH = LOG / "engineered_tracking_policy_v1.pt"
ROLLOUT_PATH = LOG / "engineered_tracking_policy_rollout_v1.npz"
REPORT_PATH = LOG / "evaluation_report_v1.json"
DT = 0.005
TASK_STEPS = 161 * 4


def quat_error(current_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    current = Rotation.from_quat(np.r_[current_wxyz[1:], current_wxyz[0]])
    reference = Rotation.from_quat(np.r_[reference_wxyz[1:], reference_wxyz[0]])
    return (reference.inv() * current).as_rotvec()


def state_error(
    data: mujoco.MjData,
    reference_qpos: np.ndarray,
    reference_qvel: np.ndarray,
    actuator_qpos: np.ndarray,
    actuator_dof: np.ndarray,
    robot_qpos: int,
    robot_dof: int,
    object_qpos: int,
    object_dof: int,
) -> np.ndarray:
    return np.concatenate((
        data.qpos[actuator_qpos] - reference_qpos[actuator_qpos],
        data.qvel[actuator_dof] - reference_qvel[actuator_dof],
        data.qpos[robot_qpos:robot_qpos + 3] - reference_qpos[robot_qpos:robot_qpos + 3],
        quat_error(data.qpos[robot_qpos + 3:robot_qpos + 7], reference_qpos[robot_qpos + 3:robot_qpos + 7]),
        data.qvel[robot_dof:robot_dof + 6] - reference_qvel[robot_dof:robot_dof + 6],
        data.qpos[object_qpos:object_qpos + 3] - reference_qpos[object_qpos:object_qpos + 3],
        quat_error(data.qpos[object_qpos + 3:object_qpos + 7], reference_qpos[object_qpos + 3:object_qpos + 7]),
        data.qvel[object_dof:object_dof + 6] - reference_qvel[object_dof:object_dof + 6],
    ))


def run_policy(
    model: mujoco.MjModel,
    policy: EngineeredTrackingPolicy,
    checkpoint: dict,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    feedback_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    robot_dof = int(model.jnt_dofadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    references_qpos = np.asarray(checkpoint["reference_qpos"], dtype=np.float64)
    references_qvel = np.asarray(checkpoint["reference_qvel"], dtype=np.float64)
    actuator_qpos = np.asarray(checkpoint["actuator_qpos"], dtype=int)
    actuator_dof = np.asarray(checkpoint["actuator_dof"], dtype=int)
    controls, qpos, qvel, errors = [], [], [], []
    lower = model.actuator_ctrlrange[:, 0]
    upper = model.actuator_ctrlrange[:, 1]
    dtype = next(policy.parameters()).dtype
    for step in range(int(checkpoint["horizon_steps"])):
        error = state_error(
            data, references_qpos[step], references_qvel[step], actuator_qpos, actuator_dof,
            robot_qpos, robot_dof, object_qpos, object_dof,
        )
        with torch.inference_mode():
            phase_control = policy.feedforward(policy.phase_features(torch.tensor([step], dtype=dtype)))
            correction = policy.feedback(torch.as_tensor(error[None], dtype=dtype)) * feedback_scale
            control = (phase_control + correction).cpu().numpy()[0].astype(np.float64)
        control = np.clip(control, lower, upper)
        controls.append(control)
        errors.append(error)
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
    return np.asarray(controls), np.asarray(qpos), np.asarray(qvel), np.asarray(errors)


def strict_success(metrics: dict, continuation: dict) -> bool:
    return bool(
        metrics["full_horizon_valid"]
        and continuation["full_horizon_valid"]
        and metrics["maximum_lift_m"] > 0.05
        and metrics["longest_hand_supported_lift_frames"] >= 25
        and metrics["carried_displacement_m"] > 0.5
        and max(metrics["foot_translation_m"]) > 0.5
        and metrics["non_end_effector_robot_object_impulse_ns"] < 0.1
        and metrics["final_position_error_m"] < 0.15
        and metrics["final_orientation_error_rad"] < 0.35
        and metrics["final_object_speed"] < 0.25
        and not any(metrics["final_hand_contact"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--suffix", default="v1")
    args = parser.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    source = np.load(EXPERT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    policy = EngineeredTrackingPolicy(
        checkpoint["horizon_steps"], checkpoint["error_dim"], checkpoint["action_dim"]
    ).float()
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    controls, online_qpos, online_qvel, errors = run_policy(
        model, policy, checkpoint, initial_qpos, initial_qvel, args.feedback_scale
    )
    goal = np.asarray(checkpoint["goal_position"], dtype=np.float64)
    goal_quat = np.asarray(checkpoint["goal_quaternion"], dtype=np.float64)
    metrics, trace = replay(
        model, initial_qpos, initial_qvel, controls[:TASK_STEPS], goal, goal_quat, True
    )
    continuation, continuation_trace = replay(
        model, initial_qpos, initial_qvel, controls, goal, goal_quat, True
    )
    replay_qpos_error = float(np.max(np.abs(continuation_trace["qpos"] - online_qpos[3::4])))
    replay_qvel_error = float(np.max(np.abs(continuation_trace["qvel"] - online_qvel[3::4])))
    source_controls = source["ctrl_physics"]
    report = {
        "success": strict_success(metrics, continuation),
        "policy": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "source_expert": str(EXPERT_PATH.relative_to(ROOT)),
        "method": "200 Hz learned spectral feedforward plus learned state-error feedback",
        "feedback_scale": args.feedback_scale,
        "closed_loop_observation_dim": int(checkpoint["error_dim"]),
        "action_dim": int(checkpoint["action_dim"]),
        "physics_steps": len(controls),
        "maximum_action_difference_from_expert": float(np.max(np.abs(controls - source_controls))),
        "rms_action_difference_from_expert": float(np.sqrt(np.mean(np.square(controls - source_controls)))),
        "maximum_tracking_error": float(np.max(np.abs(errors))),
        "independent_replay_max_qpos_error": replay_qpos_error,
        "independent_replay_max_qvel_error": replay_qvel_error,
        **metrics,
        "continuation_full_horizon_valid": continuation["full_horizon_valid"],
        "continuation_first_invalid": continuation["first_invalid"],
        "continuation_final_root_height_m": continuation["final_root_height_m"],
    }
    output = LOG / f"engineered_tracking_policy_rollout_{args.suffix}.npz"
    np.savez_compressed(
        output,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        ctrl_physics=controls,
        task_ctrl_physics=controls[:TASK_STEPS],
        online_qpos=online_qpos,
        online_qvel=online_qvel,
        state_error=errors,
        goal_position=goal,
        goal_quaternion=goal_quat,
        **{f"task_{key}": value for key, value in trace.items()},
    )
    report_path = LOG / f"evaluation_report_{args.suffix}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
