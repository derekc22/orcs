#!/usr/bin/env python3
"""Zero-assistance evaluation for the realized-reference UOLM policy.

Each rollout uses one validated physical reference, nominal play-time physics,
and no virtual object wrench.  The environment is deliberately configured with
``auto_reset=False`` so terminal object/robot state can be measured before any
reset overwrites it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def _quat_error(q1: torch.Tensor, q2: torch.Tensor) -> float:
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
    return float((2.0 * torch.acos(dot)).item())


def _evaluate_one(
    *, task: str, checkpoint: Path, reference: Path, start: int, seed: int, device: str
) -> dict[str, object]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    # Importing the task package performs its registry declarations.
    import orcs.tasks.uolm  # noqa: F401
    from orcs.assets import OBJECT_BODY_NAME

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    motion_cfg = cfg.commands["motion"]
    motion_cfg.realized_reference_paths = (str(reference),)
    motion_cfg.realized_reference_start_ids = (start,)
    motion_cfg.debug_vis = False

    agent_cfg = load_rl_cfg(task)
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd = raw_env.command_manager.get_term("motion")
    obj = raw_env.scene[OBJECT_BODY_NAME]
    initial_pos = obj.data.root_link_pos_w[0].clone()
    episode_return = 0.0
    steps = 0
    termination = "horizon"
    at_goal_ever = False

    obs = env.get_observations()
    for step in range(300):
        with torch.inference_mode():
            action = policy(obs)
        obs, reward, done, _ = env.step(action)
        episode_return += float(reward[0].item())
        steps = step + 1
        at_goal_ever |= bool(cmd.metrics["at_goal"][0].item() > 0.5)
        if bool(done[0].item()):
            active = [
                name
                for name in raw_env.termination_manager.active_terms
                if bool(raw_env.termination_manager.get_term(name)[0].item())
            ]
            termination = "+".join(active) if active else "unknown"
            break

    final_pos = obj.data.root_link_pos_w[0]
    final_quat = obj.data.root_link_quat_w[0]
    goal_pos = cmd.object_goal_pos_env[0] + raw_env.scene.env_origins[0]
    goal_quat = cmd.object_goal_quat[0]
    pos_error = float(torch.linalg.norm(final_pos - goal_pos).item())
    ori_error = _quat_error(final_quat, goal_quat)
    displacement = float(torch.linalg.norm(final_pos - initial_pos).item())
    at_goal = pos_error < 0.15 and ori_error < 0.35
    illegal = termination not in ("horizon", "time_out", "exceeded_motion")
    result = {
        "start": start,
        "seed": seed,
        "success": at_goal,
        "at_goal_ever": at_goal_ever,
        "final_position_error_m": pos_error,
        "final_orientation_error_rad": ori_error,
        "object_displacement_m": displacement,
        "episode_length": steps,
        "return": episode_return,
        "termination": termination,
        "fall_or_illegal": illegal,
        "virtual_object_force": 0.0,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="Orcs-Uolm-Realized-AdaptSonic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--starts", default="0,10,20")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("logs/unified_physical_reference"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(int(item) for item in args.seeds.split(","))
    starts = tuple(int(item) for item in args.starts.split(","))
    checkpoint = args.checkpoint.expanduser().resolve()
    rows: list[dict[str, object]] = []
    for start in starts:
        reference = (
            args.reference_root
            / f"final_start_{start:03d}"
            / "physical_reference.npz"
        ).resolve()
        for seed in seeds:
            row = _evaluate_one(
                task=args.task,
                checkpoint=checkpoint,
                reference=reference,
                start=start,
                seed=seed,
                device=args.device,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    per_start: dict[str, dict[str, float]] = {}
    for start in starts:
        group = [row for row in rows if row["start"] == start]
        per_start[str(start)] = {
            "success_rate": float(np.mean([row["success"] for row in group])),
            "final_position_error_m": float(
                np.mean([row["final_position_error_m"] for row in group])
            ),
            "final_orientation_error_rad": float(
                np.mean([row["final_orientation_error_rad"] for row in group])
            ),
            "object_displacement_m": float(
                np.mean([row["object_displacement_m"] for row in group])
            ),
            "episode_length": float(np.mean([row["episode_length"] for row in group])),
            "fall_or_illegal_rate": float(
                np.mean([row["fall_or_illegal"] for row in group])
            ),
        }
    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "seeds": seeds,
        "zero_assistance": True,
        "num_rollouts": len(rows),
        "per_start": per_start,
        "overall": {
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_episode_length": float(
                np.mean([row["episode_length"] for row in rows])
            ),
        },
        "rollouts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
