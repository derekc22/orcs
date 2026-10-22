#!/usr/bin/env python3
"""Extract state/action demonstrations from a compact ORCS feedback expert."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from search_strict_orcs_expert import _make_env, _raw_action
from search_strict_orcs_feedback_expert import _group_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--feedback-start", type=int, default=0)
    parser.add_argument("--root-feedback-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    saved = np.load(args.feedback)
    target = saved["target_rad"].astype(np.float32)
    parameters = saved["parameters"].astype(np.float32)
    env = _make_env(
        args.start,
        1,
        reference,
        args.device,
        task="Orcs-Uolm-Realized-Strict-TaRa",
    )
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_box_minus

    command = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    robot, obj = command.robot, command.object
    groups = torch.as_tensor(_group_ids(list(robot.joint_names)), device=env.device)
    controlled = groups >= 0
    group_index = groups.clamp_min(0)
    kp = torch.as_tensor(parameters[:4], device=env.device)
    kd = torch.as_tensor(parameters[4:], device=env.device)
    targets = torch.as_tensor(target, device=env.device)
    target_vel = torch.zeros_like(targets)
    target_vel[1:-1] = (targets[2:] - targets[:-2]) * 25.0
    target_vel[0] = (targets[1] - targets[0]) * 50.0
    target_vel[-1] = (targets[-1] - targets[-2]) * 50.0
    torso = robot.body_names.index("torso_link")
    motion_torso = command.cfg.body_names.index("torso_link")
    initial_rotation = matrix_from_quat(robot.data.body_link_quat_w[:, torso])
    local_up = (
        initial_rotation.transpose(-1, -2)
        @ torch.tensor([0.0, 0.0, 1.0], device=env.device)[None, :, None]
    ).squeeze(-1)
    observations, actions, states = [], [], []
    minimum_root, maximum_tilt = float("inf"), 0.0
    first_invalid = invalid_reason = None
    initial_object = obj.data.root_link_pos_w[0].clone()

    for frame in range(300):
        observation = env.observation_manager.compute(update_history=False)["policy"]
        nominal = targets[frame : frame + 1]
        q = robot.data.joint_pos[:, action_term.target_ids]
        qd = robot.data.joint_vel[:, action_term.target_ids]
        correction = kp[group_index] * (nominal - q) + kd[group_index] * (
            target_vel[frame : frame + 1] - qd
        )
        if len(parameters) >= 10 and frame >= args.root_feedback_start:
            desired_root_z = command.motion.body_pos_w[
                command.time_steps, command.motion_anchor_body_index, 2
            ]
            root_error = desired_root_z - robot.data.root_link_pos_w[:, 2]
            knee_ids = torch.as_tensor(
                [i for i, name in enumerate(robot.joint_names) if "knee" in name],
                device=env.device,
            )
            hip_pitch_ids = torch.as_tensor(
                [
                    i
                    for i, name in enumerate(robot.joint_names)
                    if "hip_pitch" in name
                ],
                device=env.device,
            )
            if len(parameters) == 12:
                gains = torch.as_tensor(parameters, device=env.device)
                correction[:, knee_ids] += gains[8:10] * root_error[:, None]
                correction[:, hip_pitch_ids] += gains[10:12] * root_error[:, None]
            else:
                correction[:, knee_ids] += parameters[8] * root_error[:, None]
                correction[:, hip_pitch_ids] += parameters[9] * root_error[:, None]
        if len(parameters) >= 26 and frame >= args.root_feedback_start:
            gains = torch.as_tensor(parameters, device=env.device)
            desired_torso_quat = command.motion.body_quat_w[
                command.time_steps, motion_torso
            ]
            actual_torso_quat = robot.data.body_link_quat_w[:, torso]
            torso_error = quat_box_minus(desired_torso_quat, actual_torso_quat)
            desired_torso_omega = command.motion.body_ang_vel_w[
                command.time_steps, motion_torso
            ]
            actual_torso_omega = robot.data.body_link_ang_vel_w[:, torso]
            torso_velocity_error = desired_torso_omega - actual_torso_omega
            names = list(robot.joint_names)
            channel_specs = (
                (("left_hip_roll_joint",), 0),
                (("right_hip_roll_joint",), 0),
                (("left_ankle_roll_joint",), 0),
                (("right_ankle_roll_joint",), 0),
                (("waist_roll_joint",), 0),
                (("left_hip_pitch_joint", "right_hip_pitch_joint"), 1),
                (("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 1),
                (("waist_pitch_joint",), 1),
            )
            for channel, (joint_names, axis) in enumerate(channel_specs):
                ids = torch.as_tensor(
                    [names.index(name) for name in joint_names], device=env.device
                )
                value = (
                    gains[10 + 2 * channel] * torso_error[:, axis]
                    + gains[11 + 2 * channel] * torso_velocity_error[:, axis]
                )
                correction[:, ids] += value[:, None]
        correction[:, ~controlled] = 0.0
        if frame < args.feedback_start:
            correction.zero_()
        applied_target = nominal + correction.clamp(-0.25, 0.25)
        raw = _raw_action(action_term, applied_target)
        observations.append(observation[0].detach().cpu().numpy())
        actions.append(raw[0].detach().cpu().numpy())
        states.append(
            np.concatenate(
                [
                    robot.data.root_link_pos_w[0].cpu().numpy(),
                    robot.data.root_link_quat_w[0].cpu().numpy(),
                    robot.data.root_link_lin_vel_w[0].cpu().numpy(),
                    robot.data.root_link_ang_vel_w[0].cpu().numpy(),
                    robot.data.joint_pos[0].cpu().numpy(),
                    robot.data.joint_vel[0].cpu().numpy(),
                    obj.data.root_link_pos_w[0].cpu().numpy(),
                    obj.data.root_link_quat_w[0].cpu().numpy(),
                    obj.data.root_link_lin_vel_w[0].cpu().numpy(),
                    obj.data.root_link_ang_vel_w[0].cpu().numpy(),
                ]
            )
        )
        _, _, terminated, truncated, _ = env.step(raw)
        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        up = (
            matrix_from_quat(robot.data.body_link_quat_w[:, torso])
            @ local_up.unsqueeze(-1)
        ).squeeze(-1)
        tilt = math.degrees(math.acos(float(up[0, 2].clamp(-1.0, 1.0).item())))
        minimum_root, maximum_tilt = min(minimum_root, root_height), max(maximum_tilt, tilt)
        active = [
            name
            for name in env.termination_manager.active_terms
            if bool(env.termination_manager.get_term(name)[0].item())
        ]
        if "illegal_contact" in active:
            invalid_reason = "forbidden_nonfoot_terrain_contact"
        elif root_height < 0.439:
            invalid_reason = "root_height"
        elif tilt > 50.0:
            invalid_reason = "torso_tilt"
        elif bool((terminated | truncated)[0].item()) and not set(active) <= {
            "time_out",
            "exceeded_motion",
        }:
            invalid_reason = "+".join(active) or "termination"
        if invalid_reason is not None:
            first_invalid = frame + 1
            break

    goal = command.object_goal_pos_env[0] + env.scene.env_origins[0]
    pos_error = float(torch.linalg.norm(obj.data.root_link_pos_w[0] - goal).item())
    dot = torch.sum(obj.data.root_link_quat_w[0] * command.object_goal_quat[0]).abs()
    ori_error = float((2.0 * torch.acos(dot.clamp(max=1.0))).item())
    report = {
        "start": args.start,
        "frames": len(actions),
        "first_invalid_frame": first_invalid,
        "invalid_reason": invalid_reason,
        "robot_valid": first_invalid is None and len(actions) == 300,
        "object_at_goal": pos_error < 0.15 and ori_error < 0.35,
        "final_success": first_invalid is None
        and len(actions) == 300
        and pos_error < 0.15
        and ori_error < 0.35,
        "final_position_error_m": pos_error,
        "final_orientation_error_rad": ori_error,
        "object_displacement_m": float(
            torch.linalg.norm(obj.data.root_link_pos_w[0] - initial_object).item()
        ),
        "minimum_root_height_m": minimum_root,
        "maximum_torso_tilt_deg": maximum_tilt,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"expert_start_{args.start:03d}.npz",
        policy_observation=np.asarray(observations, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        state=np.asarray(states, dtype=np.float32),
        frame=np.arange(len(actions), dtype=np.int64),
        start_id=np.full(len(actions), args.start, dtype=np.int64),
        target_rad=target[: len(actions)],
        feedback_parameters=parameters,
    )
    (args.output / f"expert_start_{args.start:03d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
