#!/usr/bin/env python3
"""Search a late balance residual after a frozen learned strike.

The frozen actor drives every candidate through the task-critical prefix.  A
compact lower-body/waist residual is enabled only in the late recovery phase,
and every candidate is ranked by its uninterrupted MuJoCo rollout.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def _residual_paths(
    parameters: np.ndarray,
    knot_frames: np.ndarray,
    joint_ids: np.ndarray,
    horizon: int,
    switch_frame: int,
) -> np.ndarray:
    count = len(parameters)
    knots = parameters.reshape(count, len(knot_frames), len(joint_ids))
    paths = np.zeros((count, horizon, 29), dtype=np.float32)
    frames = np.arange(switch_frame, horizon)
    for candidate in range(count):
        for local, joint in enumerate(joint_ids):
            paths[candidate, frames, joint] = np.interp(
                frames,
                np.concatenate(([switch_frame], knot_frames)),
                np.concatenate(([0.0], knots[candidate, :, local])),
            )
    return paths


def _make(task: str, checkpoint: Path, start: int, count: int, device: str):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import orcs.tasks.uolm  # noqa: F401

    root = Path(__file__).resolve().parents[1]
    reference = root / "logs/unified_physical_reference" / f"final_start_{start:03d}" / "physical_reference.npz"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = count
    cfg.seed = 42
    cfg.auto_reset = False
    motion = cfg.commands["motion"]
    motion.realized_reference_paths = (str(reference),)
    motion.realized_reference_start_ids = (start,)
    motion.debug_vis = False
    raw = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    agent_cfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        map_location=device,
    )
    return raw, env, runner.get_inference_policy(device=device)


def _evaluate(raw, env, actor, residual: np.ndarray) -> dict[str, np.ndarray]:
    from mjlab.utils.lab_api.math import matrix_from_quat

    obs, _ = env.reset()
    count, horizon = residual.shape[:2]
    correction = torch.as_tensor(residual, device=raw.device)
    robot = raw.scene["robot"]
    command = raw.command_manager.get_term("motion")
    obj = command.object
    torso = robot.body_names.index("torso_link")
    initial_rotation = matrix_from_quat(robot.data.body_link_quat_w[:, torso])
    local_up = (
        initial_rotation.transpose(-1, -2)
        @ torch.tensor([0.0, 0.0, 1.0], device=raw.device).expand(count, 3).unsqueeze(-1)
    ).squeeze(-1)
    alive = torch.ones(count, dtype=torch.bool, device=raw.device)
    first_invalid = torch.full((count,), horizon, dtype=torch.long, device=raw.device)
    min_root = torch.full((count,), float("inf"), device=raw.device)
    max_tilt = torch.zeros(count, device=raw.device)
    soft = torch.zeros(count, device=raw.device)
    reason = np.full(count, "horizon", dtype=object)
    for frame in range(horizon):
        with torch.inference_mode():
            action = actor(obs) + correction[:, frame]
        obs, _, done, _ = env.step(action)
        root = robot.data.root_link_pos_w[:, 2]
        up = (matrix_from_quat(robot.data.body_link_quat_w[:, torso]) @ local_up.unsqueeze(-1)).squeeze(-1)
        tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
        min_root = torch.minimum(min_root, root)
        max_tilt = torch.maximum(max_tilt, tilt)
        soft += alive * (
            200.0 * torch.relu(0.58 - root).square()
            + 0.05 * tilt.square()
            + 0.2 * robot.data.root_link_lin_vel_w[:, 2].square()
            + 0.02 * robot.data.root_link_ang_vel_w.square().sum(dim=-1)
        )
        invalid = done.bool() | (root < 0.439) | (tilt > 50.0)
        newly = alive & invalid
        first_invalid[newly] = frame + 1
        for index in newly.nonzero().flatten().tolist():
            active = [
                name
                for name in raw.termination_manager.active_terms
                if bool(raw.termination_manager.get_term(name)[index].item())
            ]
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
        "alive": alive.cpu().numpy(),
        "first_invalid": first_invalid.cpu().numpy(),
        "reason": reason,
        "soft": soft.cpu().numpy(),
        "min_root": min_root.cpu().numpy(),
        "max_tilt": max_tilt.cpu().numpy(),
        "pos": pos.detach().cpu().numpy(),
        "ori": ori.detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="Orcs-Uolm-Realized-Strict-ExpertDualSidecar")
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--switch-frame", type=int, default=180)
    parser.add_argument("--knot-frames", default="205,235,265,295")
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--bound", type=float, default=0.35)
    parser.add_argument("--initial-std", type=float, default=0.10)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    knot_frames = np.asarray([int(x) for x in args.knot_frames.split(",")], dtype=np.int64)
    raw, env, actor = _make(
        args.task,
        args.checkpoint.resolve(),
        args.start,
        args.population * args.replicas,
        args.device,
    )
    names = list(raw.scene["robot"].joint_names)
    joint_ids = np.asarray(
        [i for i, name in enumerate(names) if any(token in name for token in ("hip_", "knee", "ankle_", "waist_"))],
        dtype=np.int64,
    )
    dimension = len(knot_frames) * len(joint_ids)
    rng = np.random.default_rng(20261018)
    mean = (
        np.load(args.initial)["parameters"].astype(np.float64)
        if args.initial is not None
        else np.zeros(dimension)
    )
    std = np.full(dimension, args.initial_std)
    best = mean.copy()
    best_score = float("inf")
    history = []
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, dimension)), -args.bound, args.bound)
        population[0] = 0.0
        population[1] = mean
        population[2] = best
        expanded = np.repeat(population, args.replicas, axis=0)
        paths = _residual_paths(expanded, knot_frames, joint_ids, 300, args.switch_frame)
        metric = _evaluate(raw, env, actor, paths)
        expanded_score = (300 - metric["first_invalid"]) * 1.0e8 + metric["soft"]
        task = (metric["pos"] / 0.15) ** 2 + (metric["ori"] / 0.35) ** 2
        expanded_score[metric["alive"]] = (
            1.0e4 * task[metric["alive"]] + metric["soft"][metric["alive"]]
        )
        expanded_score += np.repeat(
            50.0 * np.square(population).mean(axis=1), args.replicas
        )
        replica_score = expanded_score.reshape(args.population, args.replicas)
        score = replica_score.max(axis=1)
        order = np.argsort(score)
        winner = int(order[0])
        winner_replica = int(replica_score[winner].argmax())
        metric_index = winner * args.replicas + winner_replica
        if score[winner] < best_score:
            best_score = float(score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(0.005, np.sqrt(0.2 * std**2 + 0.8 * elite.var(axis=0)))
        row = {
            "generation": generation,
            "full_valid": bool(metric["alive"][metric_index]),
            "first_invalid_frame": int(metric["first_invalid"][metric_index]),
            "invalid_reason": str(metric["reason"][metric_index]),
            "position_error_m": float(metric["pos"][metric_index]),
            "orientation_error_rad": float(metric["ori"][metric_index]),
            "minimum_root_height_m": float(metric["min_root"][metric_index]),
            "maximum_torso_tilt_deg": float(metric["max_tilt"][metric_index]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        parameters=best,
        residual_action=_residual_paths(best[None], knot_frames, joint_ids, 300, args.switch_frame)[0],
        knot_frames=knot_frames,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[i] for i in joint_ids]),
        switch_frame=args.switch_frame,
    )
    args.output.with_suffix(".json").write_text(json.dumps({"history": history}, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
