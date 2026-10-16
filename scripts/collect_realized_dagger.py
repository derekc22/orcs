#!/usr/bin/env python3
"""Collect learner-visited UOLM states and label them with expert controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def _make_env(task: str, start: int, num_envs: int, device: str, seed: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import orcs.tasks.uolm  # noqa: F401

    root = Path(__file__).resolve().parents[1]
    reference = (
        root
        / "logs/unified_physical_reference"
        / f"final_start_{start:03d}"
        / "physical_reference.npz"
    )
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    motion = cfg.commands["motion"]
    motion.realized_reference_paths = (str(reference),)
    motion.realized_reference_start_ids = (start,)
    motion.debug_vis = False
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    agent_cfg = load_rl_cfg(task)
    agent_cfg.upload_model = False
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    return env, runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=Path("logs/realized_expert_dataset_broad"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Orcs-Uolm-Realized-Expert-TaRa")
    parser.add_argument("--starts", default="0,10,20")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--action-noise", type=float, default=0.02)
    parser.add_argument("--teacher-mixture", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    chunks: dict[str, list[np.ndarray]] = {
        "policy_observation": [],
        "action": [],
        "frame": [],
        "start_id": [],
    }
    reports = []
    starts = tuple(int(item) for item in args.starts.split(","))
    for start in starts:
        env, runner = _make_env(
            args.task, start, args.num_envs, args.device, args.seed + start
        )
        runner.load(str(args.checkpoint), map_location=args.device)
        actor = runner.get_inference_policy(device=args.device)
        expert = torch.as_tensor(
            np.load(args.expert_root / f"expert_start_{start:03d}.npz")["action"],
            device=args.device,
        )
        generator = torch.Generator(device=args.device).manual_seed(args.seed + start)
        obs = env.get_observations()
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        collected = 0
        for frame in range(300):
            chunks["policy_observation"].append(
                obs["policy"].detach().cpu().numpy()
            )
            chunks["action"].append(
                expert[frame].expand(args.num_envs, -1).detach().cpu().numpy()
            )
            chunks["frame"].append(
                np.full(args.num_envs, frame, dtype=np.int64)
            )
            chunks["start_id"].append(
                np.full(args.num_envs, start, dtype=np.int64)
            )
            with torch.inference_mode():
                learner_action = actor(obs)
            teacher_action = expert[frame].expand_as(learner_action)
            action = (
                (1.0 - args.teacher_mixture) * learner_action
                + args.teacher_mixture * teacher_action
            )
            action += args.action_noise * torch.randn(
                action.shape, generator=generator, device=action.device
            )
            obs, _, done, _ = env.step(action)
            done = done.bool()
            alive &= ~done
            collected += args.num_envs
            if not bool(alive.any().item()):
                break
        reports.append(
            {
                "start": start,
                "frames": frame + 1,
                "samples": collected,
                "survivors": int(alive.sum().item()),
            }
        )
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{
            key: np.concatenate(value, axis=0)
            for key, value in chunks.items()
        },
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
