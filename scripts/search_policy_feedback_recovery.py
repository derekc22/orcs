#!/usr/bin/env python3
"""Search a compact state-feedback residual for late learned-policy recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from search_policy_late_recovery import _make


def _evaluate(raw, env, actor, parameters: np.ndarray, start_frame: int):
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_box_minus

    obs, _ = env.reset()
    count = len(parameters)
    p = torch.as_tensor(parameters, device=raw.device, dtype=torch.float32)
    robot = raw.scene["robot"]
    command = raw.command_manager.get_term("motion")
    obj = command.object
    names = list(robot.joint_names)
    torso = robot.body_names.index("torso_link")
    motion_torso = command.cfg.body_names.index("torso_link")
    initial_rotation = matrix_from_quat(robot.data.body_link_quat_w[:, torso])
    local_up = (
        initial_rotation.transpose(-1, -2)
        @ torch.tensor([0.0, 0.0, 1.0], device=raw.device).expand(count, 3).unsqueeze(-1)
    ).squeeze(-1)
    knee = torch.as_tensor([names.index("left_knee_joint"), names.index("right_knee_joint")], device=raw.device)
    hip_pitch = torch.as_tensor([names.index("left_hip_pitch_joint"), names.index("right_hip_pitch_joint")], device=raw.device)
    channels = (
        (("left_hip_roll_joint",), 0),
        (("right_hip_roll_joint",), 0),
        (("left_ankle_roll_joint",), 0),
        (("right_ankle_roll_joint",), 0),
        (("waist_roll_joint",), 0),
        (("left_hip_pitch_joint", "right_hip_pitch_joint"), 1),
        (("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 1),
        (("waist_pitch_joint",), 1),
    )
    channel_ids = [torch.as_tensor([names.index(name) for name in joint_names], device=raw.device) for joint_names, _ in channels]
    alive = torch.ones(count, dtype=torch.bool, device=raw.device)
    first_invalid = torch.full((count,), 300, dtype=torch.long, device=raw.device)
    minimum_root = torch.full((count,), float("inf"), device=raw.device)
    maximum_tilt = torch.zeros(count, device=raw.device)
    maximum_correction = torch.zeros(count, device=raw.device)
    soft = torch.zeros(count, device=raw.device)
    reason = np.full(count, "horizon", dtype=object)
    for frame in range(300):
        with torch.inference_mode():
            action = actor(obs)
            correction = torch.zeros_like(action)
            if frame >= start_frame:
                step = command.time_steps
                desired_z = command.motion.body_pos_w[step, command.motion_anchor_body_index, 2]
                desired_vz = command.motion.body_lin_vel_w[step, command.motion_anchor_body_index, 2]
                root_error = desired_z - robot.data.root_link_pos_w[:, 2]
                velocity_error = desired_vz - robot.data.root_link_lin_vel_w[:, 2]
                correction[:, knee] += (
                    p[:, 0] * root_error + p[:, 1] * velocity_error
                )[:, None]
                correction[:, hip_pitch] += (
                    p[:, 2] * root_error + p[:, 3] * velocity_error
                )[:, None]
                torso_error = quat_box_minus(
                    command.motion.body_quat_w[step, motion_torso],
                    robot.data.body_link_quat_w[:, torso],
                )
                omega_error = (
                    command.motion.body_ang_vel_w[step, motion_torso]
                    - robot.data.body_link_ang_vel_w[:, torso]
                )
                for channel, ((_, axis), ids) in enumerate(zip(channels, channel_ids)):
                    value = (
                        p[:, 4 + 2 * channel] * torso_error[:, axis]
                        + p[:, 5 + 2 * channel] * omega_error[:, axis]
                    )
                    correction[:, ids] += value[:, None]
                correction.clamp_(-0.35, 0.35)
            maximum_correction = torch.maximum(maximum_correction, correction.abs().amax(dim=1))
        obs, _, done, _ = env.step(action + correction)
        root = robot.data.root_link_pos_w[:, 2]
        up = (matrix_from_quat(robot.data.body_link_quat_w[:, torso]) @ local_up.unsqueeze(-1)).squeeze(-1)
        tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
        minimum_root = torch.minimum(minimum_root, root)
        maximum_tilt = torch.maximum(maximum_tilt, tilt)
        soft += alive * (
            200.0 * torch.relu(0.58 - root).square()
            + 0.05 * tilt.square()
            + 0.2 * robot.data.root_link_lin_vel_w[:, 2].square()
        )
        invalid = done.bool() | (root < 0.439) | (tilt > 50.0)
        newly = alive & invalid
        first_invalid[newly] = frame + 1
        for index in newly.nonzero().flatten().tolist():
            active = [name for name in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(name)[index].item())]
            if "illegal_contact" in active:
                reason[index] = "forbidden_nonfoot_terrain_contact"
            elif root[index] < 0.439:
                reason[index] = "root_height"
            elif tilt[index] > 50.0:
                reason[index] = "torso_tilt"
            else:
                reason[index] = "+".join(active) or "termination"
        alive &= ~invalid
        reset_ids = invalid.nonzero().flatten()
        if len(reset_ids):
            raw.reset(env_ids=reset_ids)
    goal = command.object_goal_pos_env + raw.scene.env_origins
    pos = torch.linalg.norm(obj.data.root_link_pos_w - goal, dim=-1)
    dot = torch.sum(obj.data.root_link_quat_w * command.object_goal_quat, dim=-1)
    ori = 2.0 * torch.acos(dot.abs().clamp(max=1.0))
    return {
        "alive": alive.cpu().numpy(), "first_invalid": first_invalid.cpu().numpy(),
        "reason": reason, "soft": soft.cpu().numpy(), "min_root": minimum_root.cpu().numpy(),
        "max_tilt": maximum_tilt.cpu().numpy(), "max_correction": maximum_correction.cpu().numpy(),
        "pos": pos.detach().cpu().numpy(), "ori": ori.detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="Orcs-Uolm-Realized-Strict-ExpertDualSidecar")
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--feedback-start", type=int, default=100)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, env, actor = _make(args.task, args.checkpoint.resolve(), args.start, args.population * args.replicas, args.device)
    rng = np.random.default_rng(20261019)
    dimension = 20
    mean = np.zeros(dimension)
    std = np.asarray([0.8, 0.15, 0.8, 0.15] + [0.6, 0.08] * 8)
    lower = np.asarray([-4.0, -1.0, -4.0, -1.0] + [-4.0, -0.5] * 8)
    upper = -lower
    best = mean.copy()
    best_score = float("inf")
    history = []
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, dimension)), lower, upper)
        population[0], population[1], population[2] = 0.0, mean, best
        expanded = np.repeat(population, args.replicas, axis=0)
        metric = _evaluate(raw, env, actor, expanded, args.feedback_start)
        regularization = np.repeat(20.0 * np.square(population).mean(axis=1), args.replicas)
        score_expanded = (300 - metric["first_invalid"]) * 1.0e8 + metric["soft"] + regularization
        task = (metric["pos"] / 0.15) ** 2 + (metric["ori"] / 0.35) ** 2
        score_expanded[metric["alive"]] = 1.0e4 * task[metric["alive"]] + metric["soft"][metric["alive"]] + regularization[metric["alive"]]
        replica_score = score_expanded.reshape(args.population, args.replicas)
        score = replica_score.max(axis=1)
        order = np.argsort(score)
        winner = int(order[0]); worst_replica = int(replica_score[winner].argmax()); index = winner * args.replicas + worst_replica
        if score[winner] < best_score:
            best_score = float(score[winner]); best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(np.asarray([0.02, 0.005, 0.02, 0.005] + [0.02, 0.003] * 8), np.sqrt(0.2 * std**2 + 0.8 * elite.var(axis=0)))
        row = {
            "generation": generation, "full_valid": bool(metric["alive"][index]),
            "first_invalid_frame": int(metric["first_invalid"][index]), "invalid_reason": str(metric["reason"][index]),
            "position_error_m": float(metric["pos"][index]), "orientation_error_rad": float(metric["ori"][index]),
            "minimum_root_height_m": float(metric["min_root"][index]), "maximum_torso_tilt_deg": float(metric["max_tilt"][index]),
            "maximum_correction": float(metric["max_correction"][index]),
        }
        history.append(row); print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, parameters=best, feedback_start=args.feedback_start)
    args.output.with_suffix(".json").write_text(json.dumps({"history": history}, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
