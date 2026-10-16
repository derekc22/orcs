#!/usr/bin/env python3
"""Replay saved physical-reference actuator targets through the UOLM action API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, choices=(0, 10, 20), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import orcs.tasks.uolm  # noqa: F401
    from orcs.assets import OBJECT_BODY_NAME

    cfg = load_env_cfg("Orcs-Uolm-Realized-AdaptSonic", play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    path = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/"
        "physical_reference.npz"
    ).resolve()
    mc = cfg.commands["motion"]
    mc.realized_reference_paths = (str(path),)
    mc.realized_reference_start_ids = (args.start,)
    mc.debug_vis = False
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env.reset()
    cmd = env.command_manager.get_term("motion")
    action = env.action_manager.get_term("joint_pos")
    obj = env.scene[OBJECT_BODY_NAME]
    initial = obj.data.root_link_pos_w[0].clone()
    termination = "horizon"
    frames = 0
    for _frame in range(300):
        bias = action._entity.data.encoder_bias[:, action.target_ids]
        raw = (cmd.realized_control_target + bias - action.offset) / action.scale
        _, _, terminated, truncated, _ = env.step(raw)
        frames = _frame + 1
        if bool((terminated | truncated)[0].item()):
            names = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[0].item())
            ]
            termination = "+".join(names)
            break
    goal = cmd.object_goal_pos_env[0] + env.scene.env_origins[0]
    pos_error = torch.linalg.norm(obj.data.root_link_pos_w[0] - goal)
    dot = torch.sum(obj.data.root_link_quat_w[0] * cmd.object_goal_quat[0]).abs()
    ori_error = 2.0 * torch.acos(dot.clamp(max=1.0))
    print(
        json.dumps(
            {
                "start": args.start,
                "frames": frames,
                "termination": termination,
                "position_error_m": float(pos_error.item()),
                "orientation_error_rad": float(ori_error.item()),
                "object_displacement_m": float(
                    torch.linalg.norm(obj.data.root_link_pos_w[0] - initial).item()
                ),
            },
            sort_keys=True,
        )
    )
    env.close()


if __name__ == "__main__":
    main()
