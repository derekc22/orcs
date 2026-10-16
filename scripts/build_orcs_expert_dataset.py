#!/usr/bin/env python3
"""Fit ORCS/MJWarp expert strikes and extract deployable imitation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _quat_error(actual: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(actual * desired, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _spline(parameters: np.ndarray, joint_ids: np.ndarray, horizon: int = 300) -> np.ndarray:
    knot_frames = np.asarray([0, 4, 10, 20, 32], dtype=np.float64)
    values = np.concatenate(
        [parameters.reshape(4, len(joint_ids)), np.zeros((1, len(joint_ids)))], axis=0
    )
    residual = np.zeros((horizon, 29), dtype=np.float32)
    for local, joint_id in enumerate(joint_ids):
        residual[:, joint_id] = np.interp(
            np.arange(horizon), knot_frames, values[:, local], right=0.0
        )
    return residual


def _make_env(start: int, num_envs: int, device: str):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import orcs.tasks.uolm  # noqa: F401

    root = Path(__file__).resolve().parents[1]
    reference = root / "logs/unified_physical_reference" / f"final_start_{start:03d}" / "physical_reference.npz"
    cfg = load_env_cfg("Orcs-Uolm-Realized-Expert-TaRa", play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    cfg.auto_reset = False
    motion = cfg.commands["motion"]
    motion.realized_reference_paths = (str(reference),)
    motion.realized_reference_start_ids = (start,)
    motion.debug_vis = False
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    env.reset()
    return env


def _raw_action(action_term, target: torch.Tensor) -> torch.Tensor:
    bias = action_term._entity.data.encoder_bias[:, action_term.target_ids]
    return (target + bias - action_term.offset) / action_term.scale


def _roll_population(env, residuals: np.ndarray, frames: int) -> dict[str, np.ndarray]:
    env.reset()
    cmd = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    obj = cmd.object
    device = env.device
    residual = torch.as_tensor(residuals, device=device)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    first_invalid = torch.full(
        (env.num_envs,), frames, dtype=torch.long, device=device
    )
    for frame in range(frames):
        target = cmd.realized_control_target + residual[:, frame]
        _, _, terminated, truncated, _ = env.step(_raw_action(action_term, target))
        newly = alive & (terminated | truncated)
        first_invalid[newly] = frame + 1
        alive &= ~(terminated | truncated)
        done_ids = (terminated | truncated).nonzero().flatten()
        if len(done_ids):
            # Invalid candidates are already dominated in the ranking. Reset
            # only so ManagerBasedRlEnv permits the surviving batch to keep
            # advancing; their post-reset terminal metrics are ignored.
            env.reset(env_ids=done_ids)
    goal_pos = cmd.object_goal_pos_env + env.scene.env_origins
    pos_error = torch.linalg.norm(obj.data.root_link_pos_w - goal_pos, dim=-1)
    ori_error = _quat_error(obj.data.root_link_quat_w, cmd.object_goal_quat)
    speed = torch.linalg.norm(obj.data.root_link_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.norm(obj.data.root_link_ang_vel_w, dim=-1)
    cost = (
        (pos_error / 0.15).square()
        + (ori_error / 0.35).square()
        + 0.15 * speed.square()
        + 0.03 * angular_speed.square()
        + (~alive).float() * (1.0e4 + (frames - first_invalid).float() * 50.0)
    )
    return {
        "cost": cost.detach().cpu().numpy(),
        "position_error": pos_error.detach().cpu().numpy(),
        "orientation_error": ori_error.detach().cpu().numpy(),
        "first_invalid": first_invalid.detach().cpu().numpy(),
        "alive": alive.detach().cpu().numpy(),
    }


def _optimize(start: int, args) -> tuple[np.ndarray, list[dict[str, float]]]:
    env = _make_env(start, args.population * args.replicates, args.device)
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    joint_ids = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name.startswith("right_")
            and any(token in name for token in ("hip_", "knee", "ankle_"))
        ],
        dtype=np.int64,
    )
    print("expert joints", [names[index] for index in joint_ids], flush=True)
    dimension = 4 * len(joint_ids)
    rng = np.random.default_rng(1000 + start)
    mean = np.zeros(dimension, dtype=np.float64)
    if args.initial_expert is not None:
        initial = np.load(args.initial_expert)["residual_target_rad"]
        mean = initial[np.asarray([0, 4, 10, 20])][:, joint_ids].reshape(-1)
    std = np.full(dimension, args.initial_std, dtype=np.float64)
    best = mean.copy()
    best_cost = np.inf
    ledger: list[dict[str, float]] = []
    for generation in range(args.generations):
        population = rng.normal(mean, std, size=(args.population, dimension))
        population = np.clip(population, -args.bound, args.bound)
        population[0] = mean
        population[1] = mean
        paths = np.stack([_spline(item, joint_ids) for item in population])
        replicated_paths = np.repeat(paths, args.replicates, axis=0)
        metrics = _roll_population(env, replicated_paths, args.search_frames)
        replica_cost = metrics["cost"].reshape(args.population, args.replicates)
        robust_cost = replica_cost.mean(axis=1) + args.worst_case_weight * replica_cost.max(axis=1)
        order = np.argsort(robust_cost)
        elite = population[order[: args.elites]]
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(0.015, np.sqrt(0.25 * std**2 + 0.75 * elite.var(axis=0)))
        index = int(order[0])
        if robust_cost[index] < best_cost:
            best_cost = float(robust_cost[index])
            best = population[index].copy()
        replica_index = index * args.replicates
        row = {
            "generation": generation,
            "cost": float(robust_cost[index]),
            "position_error_m": float(
                metrics["position_error"].reshape(args.population, args.replicates)[index].mean()
            ),
            "orientation_error_rad": float(
                metrics["orientation_error"].reshape(args.population, args.replicates)[index].mean()
            ),
            "worst_position_error_m": float(
                metrics["position_error"].reshape(args.population, args.replicates)[index].max()
            ),
            "worst_orientation_error_rad": float(
                metrics["orientation_error"].reshape(args.population, args.replicates)[index].max()
            ),
            "first_invalid_frame": int(
                metrics["first_invalid"][
                    replica_index : replica_index + args.replicates
                ].min()
            ),
        }
        ledger.append(row)
        print(json.dumps({"start": start, **row}), flush=True)
    env.close()
    return _spline(best, joint_ids), ledger


def _extract(start: int, residual: np.ndarray, output: Path, device: str) -> dict[str, object]:
    env = _make_env(start, 1, device)
    cmd = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    obj = cmd.object
    initial_obj = obj.data.root_link_pos_w[0].clone()
    observations = []
    actions = []
    states = []
    contacts = []
    first_invalid = None
    termination = "horizon"
    residual_t = torch.as_tensor(residual, device=env.device)
    for frame in range(300):
        observation = env.observation_manager.compute(update_history=False)
        target = cmd.realized_control_target + residual_t[frame : frame + 1]
        raw = _raw_action(action_term, target)
        observations.append(observation["policy"][0].detach().cpu().numpy())
        actions.append(raw[0].detach().cpu().numpy())
        robot = env.scene["robot"]
        states.append(
            np.concatenate(
                [
                    robot.data.root_link_pos_w[0].detach().cpu().numpy(),
                    robot.data.root_link_quat_w[0].detach().cpu().numpy(),
                    robot.data.root_link_lin_vel_w[0].detach().cpu().numpy(),
                    robot.data.root_link_ang_vel_w[0].detach().cpu().numpy(),
                    robot.data.joint_pos[0].detach().cpu().numpy(),
                    robot.data.joint_vel[0].detach().cpu().numpy(),
                    obj.data.root_link_pos_w[0].detach().cpu().numpy(),
                    obj.data.root_link_quat_w[0].detach().cpu().numpy(),
                    obj.data.root_link_lin_vel_w[0].detach().cpu().numpy(),
                    obj.data.root_link_ang_vel_w[0].detach().cpu().numpy(),
                ]
            )
        )
        contacts.append(np.zeros(1, dtype=np.float32))
        _, _, terminated, truncated, _ = env.step(raw)
        if bool((terminated | truncated)[0].item()) and first_invalid is None:
            first_invalid = frame + 1
            active = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[0].item())
            ]
            termination = "+".join(active) if active else "unknown"
            break
    goal_pos = cmd.object_goal_pos_env[0] + env.scene.env_origins[0]
    pos_error = float(torch.linalg.norm(obj.data.root_link_pos_w[0] - goal_pos).item())
    ori_error = float(_quat_error(obj.data.root_link_quat_w, cmd.object_goal_quat)[0].item())
    at_goal = pos_error < 0.15 and ori_error < 0.35
    report = {
        "start": start,
        "frames": len(actions),
        "first_invalid_frame": first_invalid,
        "termination": termination,
        "at_goal": at_goal,
        "final_position_error_m": pos_error,
        "final_orientation_error_rad": ori_error,
        "object_displacement_m": float(
            torch.linalg.norm(obj.data.root_link_pos_w[0] - initial_obj).item()
        ),
    }
    np.savez_compressed(
        output / f"expert_start_{start:03d}.npz",
        policy_observation=np.asarray(observations, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        state=np.asarray(states, dtype=np.float32),
        contact=np.asarray(contacts, dtype=np.float32),
        frame=np.arange(len(actions), dtype=np.int64),
        start_id=np.full(len(actions), start, dtype=np.int64),
        residual_target_rad=residual[: len(actions)],
    )
    env.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, nargs="+", default=(0, 10, 20))
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--worst-case-weight", type=float, default=0.5)
    parser.add_argument("--initial-expert", type=Path)
    parser.add_argument("--elites", type=int, default=24)
    parser.add_argument("--generations", type=int, default=18)
    parser.add_argument("--search-frames", type=int, default=100)
    parser.add_argument("--initial-std", type=float, default=0.15)
    parser.add_argument("--bound", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("logs/realized_expert_dataset")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = []
    reports = []
    for start in args.starts:
        residual, progress = _optimize(start, args)
        report = _extract(start, residual, args.output, args.device)
        ledger.append({"start": start, "progress": progress, "report": report})
        reports.append(report)
    (args.output / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
