#!/usr/bin/env python3
"""Extract exact-ORCS expert actions and observations under strict validity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from search_strict_orcs_expert import _make_env, _raw_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="Orcs-Uolm-Realized-Expert-TaRa")
    args = parser.parse_args()
    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    env = _make_env(args.start, 1, reference, args.device, task=args.task)
    from mjlab.utils.lab_api.math import matrix_from_quat

    target = np.load(args.target)["target_rad"].astype(np.float32)
    command = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    robot = command.robot
    obj = command.object
    torso_index = robot.body_names.index("torso_link")
    initial_rotation = matrix_from_quat(robot.data.body_link_quat_w[:, torso_index])
    local_up = (
        initial_rotation.transpose(-1, -2)
        @ torch.tensor([0.0, 0.0, 1.0], device=env.device)[None, :, None]
    ).squeeze(-1)
    observations, actions, states = [], [], []
    first_invalid = None
    invalid_reason = None
    minimum_root = float("inf")
    maximum_tilt = 0.0
    initial_object = obj.data.root_link_pos_w[0].clone()
    for frame in range(300):
        observation = env.observation_manager.compute(update_history=False)["policy"]
        raw = _raw_action(
            action_term,
            torch.as_tensor(target[frame : frame + 1], device=env.device),
        )
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
            matrix_from_quat(robot.data.body_link_quat_w[:, torso_index])
            @ local_up.unsqueeze(-1)
        ).squeeze(-1)
        tilt = math.degrees(math.acos(float(up[0, 2].clamp(-1.0, 1.0).item())))
        minimum_root = min(minimum_root, root_height)
        maximum_tilt = max(maximum_tilt, tilt)
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
    goal_pos = command.object_goal_pos_env[0] + env.scene.env_origins[0]
    pos_error = float(torch.linalg.norm(obj.data.root_link_pos_w[0] - goal_pos).item())
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
    )
    (args.output / f"expert_start_{args.start:03d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
