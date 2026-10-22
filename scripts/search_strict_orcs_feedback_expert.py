#!/usr/bin/env python3
"""Search a compact state-feedback recovery expert in exact ORCS physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from search_strict_orcs_expert import (
    MAXIMUM_TORSO_TILT_DEG,
    MINIMUM_ROOT_HEIGHT_M,
    _make_env,
    _raw_action,
    _rank,
)


GROUP_TOKENS = (("hip_",), ("knee",), ("ankle_",), ("waist_",))


def _group_ids(names: list[str]) -> np.ndarray:
    ids = np.full(len(names), -1, dtype=np.int64)
    for joint, name in enumerate(names):
        for group, tokens in enumerate(GROUP_TOKENS):
            if any(token in name for token in tokens):
                ids[joint] = group
                break
    return ids


def _evaluate_feedback(
    env,
    target: np.ndarray,
    parameters: np.ndarray,
    feedback_start: int = 0,
    root_feedback_start: int = 0,
):
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_box_minus

    env.reset()
    command = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    robot, obj = command.robot, command.object
    count = len(parameters)
    horizon = target.shape[-2]
    names = list(robot.joint_names)
    groups = torch.as_tensor(_group_ids(names), device=env.device)
    controlled = groups >= 0
    group_index = groups.clamp_min(0)
    params = torch.as_tensor(parameters, device=env.device, dtype=torch.float32)
    kp = params[:, :4]
    kd = params[:, 4:]
    targets = torch.as_tensor(target, device=env.device, dtype=torch.float32)
    target_vel = torch.zeros_like(targets)
    if targets.ndim == 2:
        target_vel[1:-1] = (targets[2:] - targets[:-2]) * 25.0
        target_vel[0] = (targets[1] - targets[0]) * 50.0
        target_vel[-1] = (targets[-1] - targets[-2]) * 50.0
    else:
        target_vel[:, 1:-1] = (targets[:, 2:] - targets[:, :-2]) * 25.0
        target_vel[:, 0] = (targets[:, 1] - targets[:, 0]) * 50.0
        target_vel[:, -1] = (targets[:, -1] - targets[:, -2]) * 50.0

    torso = robot.body_names.index("torso_link")
    motion_torso = command.cfg.body_names.index("torso_link")
    initial_rotation = matrix_from_quat(robot.data.body_link_quat_w[:, torso])
    local_up = (
        initial_rotation.transpose(-1, -2)
        @ torch.tensor([0.0, 0.0, 1.0], device=env.device)
        .expand(count, 3)
        .unsqueeze(-1)
    ).squeeze(-1)
    alive = torch.ones(count, dtype=torch.bool, device=env.device)
    first_invalid = torch.full((count,), horizon, dtype=torch.long, device=env.device)
    soft_cost = torch.zeros(count, device=env.device)
    invalid_reason = np.full(count, "horizon", dtype=object)
    min_root = torch.full((count,), float("inf"), device=env.device)
    max_tilt = torch.zeros(count, device=env.device)
    max_correction = torch.zeros(count, device=env.device)

    for frame in range(horizon):
        nominal = (
            targets[frame].expand(count, -1)
            if targets.ndim == 2
            else targets[:, frame]
        )
        desired_velocity = (
            target_vel[frame].expand(count, -1)
            if target_vel.ndim == 2
            else target_vel[:, frame]
        )
        actual_q = robot.data.joint_pos[:, action_term.target_ids]
        actual_qd = robot.data.joint_vel[:, action_term.target_ids]
        correction = (
            kp[:, group_index] * (nominal - actual_q)
            + kd[:, group_index] * (desired_velocity - actual_qd)
        )
        if params.shape[1] >= 10 and frame >= root_feedback_start:
            desired_root_z = command.motion.body_pos_w[
                command.time_steps, command.motion_anchor_body_index, 2
            ]
            root_error = desired_root_z - robot.data.root_link_pos_w[:, 2]
            knee_ids = torch.as_tensor(
                [i for i, name in enumerate(names) if "knee" in name],
                device=env.device,
            )
            hip_pitch_ids = torch.as_tensor(
                [i for i, name in enumerate(names) if "hip_pitch" in name],
                device=env.device,
            )
            if params.shape[1] == 12:
                correction[:, knee_ids] += params[:, 8:10] * root_error[:, None]
                correction[:, hip_pitch_ids] += params[:, 10:12] * root_error[:, None]
            else:
                correction[:, knee_ids] += params[:, 8:9] * root_error[:, None]
                correction[:, hip_pitch_ids] += params[:, 9:10] * root_error[:, None]
        if params.shape[1] >= 26 and frame >= root_feedback_start:
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
                joint_ids = torch.as_tensor(
                    [names.index(name) for name in joint_names], device=env.device
                )
                gain = params[:, 10 + 2 * channel]
                damping = params[:, 11 + 2 * channel]
                value = gain * torso_error[:, axis] + damping * torso_velocity_error[:, axis]
                correction[:, joint_ids] += value[:, None]
        correction[:, ~controlled] = 0.0
        if frame < feedback_start:
            correction.zero_()
        correction = correction.clamp(-0.25, 0.25)
        max_correction = torch.maximum(max_correction, correction.abs().amax(dim=1))
        _, _, terminated, truncated, _ = env.step(
            _raw_action(action_term, nominal + correction)
        )
        root_height = robot.data.root_link_pos_w[:, 2]
        torso_up = (
            matrix_from_quat(robot.data.body_link_quat_w[:, torso])
            @ local_up.unsqueeze(-1)
        ).squeeze(-1)
        tilt = torch.rad2deg(torch.acos(torso_up[:, 2].clamp(-1.0, 1.0)))
        min_root = torch.minimum(min_root, root_height)
        max_tilt = torch.maximum(max_tilt, tilt)
        soft_cost += alive * (
            50.0 * torch.relu(0.58 - root_height).square()
            + 0.02 * tilt.square()
            + 0.05 * robot.data.root_link_lin_vel_w[:, 2].square()
        )
        invalid = terminated | (root_height < MINIMUM_ROOT_HEIGHT_M) | (
            tilt > MAXIMUM_TORSO_TILT_DEG
        )
        newly = alive & invalid
        first_invalid[newly] = frame + 1
        for index in newly.nonzero().flatten().tolist():
            active = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[index].item())
            ]
            if "illegal_contact" in active:
                invalid_reason[index] = "forbidden_nonfoot_terrain_contact"
            elif root_height[index] < MINIMUM_ROOT_HEIGHT_M:
                invalid_reason[index] = "root_height"
            elif tilt[index] > MAXIMUM_TORSO_TILT_DEG:
                invalid_reason[index] = "torso_tilt"
            else:
                invalid_reason[index] = "+".join(active) or "termination"
        alive &= ~invalid
        reset_ids = (invalid | truncated).nonzero().flatten()
        if len(reset_ids):
            env.reset(env_ids=reset_ids)

    goal = command.object_goal_pos_env + env.scene.env_origins
    pos = torch.linalg.norm(obj.data.root_link_pos_w - goal, dim=-1)
    dot = torch.sum(obj.data.root_link_quat_w * command.object_goal_quat, dim=-1)
    ori = 2.0 * torch.acos(dot.abs().clamp(max=1.0))
    return {
        "alive": alive.cpu().numpy(),
        "first_invalid": first_invalid.cpu().numpy(),
        "invalid_reason": invalid_reason,
        "soft_cost": soft_cost.cpu().numpy(),
        "position_error": pos.detach().cpu().numpy(),
        "orientation_error": ori.detach().cpu().numpy(),
        "minimum_root_height": min_root.cpu().numpy(),
        "maximum_torso_tilt": max_tilt.cpu().numpy(),
        "maximum_correction": max_correction.cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Evaluate each gain vector in consecutive simulator worlds and rank its worst rollout.",
    )
    parser.add_argument("--feedback-start", type=int, default=0)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--root-feedback", action="store_true")
    parser.add_argument("--torso-feedback", action="store_true")
    parser.add_argument("--root-feedback-start", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    target = np.load(args.target)["target_rad"].astype(np.float32)
    env = _make_env(
        args.start,
        args.population * args.replicas,
        reference,
        args.device,
        task="Orcs-Uolm-Realized-Strict-TaRa",
    )
    rng = np.random.default_rng(14000 + args.start)
    # Position-error multiplier and velocity-error time constant for
    # hip/knee/ankle/waist groups.
    mean = np.asarray([0.15, 0.15, 0.15, 0.10, 0.01, 0.01, 0.01, 0.01])
    if args.initial is not None:
        mean = np.load(args.initial)["parameters"].astype(np.float64)
    if args.root_feedback:
        if len(mean) == 8:
            mean = np.concatenate([mean, np.zeros(2)])
        std = np.asarray([0.12] * 4 + [0.008] * 4 + [0.75, 0.75])
        lower = np.asarray([0.0] * 8 + [-3.0, -3.0])
        upper = np.asarray([0.8] * 4 + [0.06] * 4 + [3.0, 3.0])
    else:
        std = np.asarray([0.12] * 4 + [0.008] * 4)
        lower = np.zeros(8)
        upper = np.asarray([0.8] * 4 + [0.06] * 4)
    if args.torso_feedback:
        if len(mean) < 10:
            mean = np.concatenate([mean, np.zeros(10 - len(mean))])
            std = np.concatenate([std, np.asarray([0.75, 0.75])])
            lower = np.concatenate([lower, np.asarray([-3.0, -3.0])])
            upper = np.concatenate([upper, np.asarray([3.0, 3.0])])
        mean = np.concatenate([mean, np.zeros(16)])
        std = np.concatenate([std, np.tile(np.asarray([0.5, 0.05]), 8)])
        lower = np.concatenate([lower, np.tile(np.asarray([-3.0, -0.3]), 8)])
        upper = np.concatenate([upper, np.tile(np.asarray([3.0, 0.3]), 8)])
    best, best_score = mean.copy(), float("inf")
    history = []
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, (args.population, len(mean))), lower, upper
        )
        population[0], population[1] = mean, best
        population[2] = 0.0
        expanded = np.repeat(population, args.replicas, axis=0)
        metrics = _evaluate_feedback(
            env,
            target,
            expanded,
            feedback_start=args.feedback_start,
            root_feedback_start=args.root_feedback_start,
        )
        regularization = 10.0 * np.square(population).mean(axis=1)
        expanded_regularization = np.repeat(regularization, args.replicas)
        expanded_score = _rank(metrics, expanded_regularization)
        replica_score = expanded_score.reshape(args.population, args.replicas)
        score = replica_score.max(axis=1)
        order = np.argsort(score)
        winner = int(order[0])
        winner_replica = int(replica_score[winner].argmax())
        metric_index = winner * args.replicas + winner_replica
        if score[winner] < best_score:
            best_score, best = float(score[winner]), population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(
            np.asarray(
                [0.005] * 4
                + [0.0005] * 4
                + ([0.02, 0.02] if args.root_feedback else [])
                + ([0.02, 0.002] * 8 if args.torso_feedback else [])
            ),
            np.sqrt(0.2 * std**2 + 0.8 * elite.var(axis=0)),
        )
        row = {
            "generation": generation,
            "full_valid": bool(metrics["alive"][metric_index]),
            "first_invalid_frame": int(metrics["first_invalid"][metric_index]),
            "invalid_reason": str(metrics["invalid_reason"][metric_index]),
            "position_error_m": float(metrics["position_error"][metric_index]),
            "orientation_error_rad": float(metrics["orientation_error"][metric_index]),
            "max_correction_rad": float(metrics["maximum_correction"][metric_index]),
            "parameters": population[winner].tolist(),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        parameters=best,
        target_rad=target,
        group_names=np.asarray(["hip", "knee", "ankle", "waist"]),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "start": args.start,
                "feedback_start": args.feedback_start,
                "replicas": args.replicas,
                "root_feedback": args.root_feedback,
                "torso_feedback": args.torso_feedback,
                "root_feedback_start": args.root_feedback_start,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
