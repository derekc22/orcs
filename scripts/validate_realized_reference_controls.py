#!/usr/bin/env python3
"""Replay saved physical-reference actuator targets through the UOLM action API."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, choices=(0, 10, 20), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--task", default="Orcs-Uolm-Realized-Expert-TaRa"
    )
    parser.add_argument("--high-rate", action="store_true")
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import orcs.tasks.uolm  # noqa: F401
    from orcs.assets import OBJECT_BODY_NAME
    from orcs.core.sensors import (
        GROUND_CONTACT_SENSOR_NAME,
        STRICT_KILL_BODIES,
        ground_contact_sensor,
    )
    from mjlab.utils.lab_api.math import matrix_from_quat

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    root = "unified_physical_reference_200hz" if args.high_rate else "unified_physical_reference"
    path = Path(
        f"logs/{root}/final_start_{args.start:03d}/"
        "physical_reference.npz"
    ).resolve()
    mc = cfg.commands["motion"]
    mc.realized_reference_paths = (str(path),)
    mc.realized_reference_start_ids = (args.start,)
    mc.debug_vis = False
    strict = ground_contact_sensor(STRICT_KILL_BODIES, exclude=(".*foot.*",))
    cfg.scene.sensors = tuple(
        strict if sensor.name == GROUND_CONTACT_SENSOR_NAME else sensor
        for sensor in cfg.scene.sensors
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env.reset()
    cmd = env.command_manager.get_term("motion")
    action = env.action_manager.get_term("joint_pos")
    obj = env.scene[OBJECT_BODY_NAME]
    initial = obj.data.root_link_pos_w[0].clone()
    robot = env.scene["robot"]
    torso_index = robot.body_names.index("torso_link")
    initial_torso = matrix_from_quat(robot.data.body_link_quat_w[0, torso_index])
    local_up = initial_torso.transpose(-1, -2) @ torch.tensor(
        [0.0, 0.0, 1.0], device=env.device
    )
    termination = "horizon"
    frames = 0
    first_invalid = None
    invalid_reason = None
    horizon = 1200 if args.high_rate else 300
    for _frame in range(horizon):
        bias = action._entity.data.encoder_bias[:, action.target_ids]
        raw = (cmd.realized_control_target + bias - action.offset) / action.scale
        _, _, terminated, truncated, _ = env.step(raw)
        frames = _frame + 1
        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        up = matrix_from_quat(robot.data.body_link_quat_w[0, torso_index]) @ local_up
        tilt = math.degrees(math.acos(float(up[2].clamp(-1.0, 1.0).item())))
        active = [
            name
            for name in env.termination_manager.active_terms
            if bool(env.termination_manager.get_term(name)[0].item())
        ]
        if "illegal_contact" in active:
            first_invalid, invalid_reason = frames, "forbidden_nonfoot_terrain_contact"
        elif root_height < 0.439:
            first_invalid, invalid_reason = frames, "root_height"
        elif tilt > 50.0:
            first_invalid, invalid_reason = frames, "torso_tilt"
        if first_invalid is not None:
            termination = invalid_reason
            break
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
                "first_invalid_step": first_invalid,
                "full_valid": first_invalid is None and frames == horizon,
                "position_error_m": float(pos_error.item()),
                "orientation_error_rad": float(ori_error.item()),
                "at_goal": bool(pos_error < 0.15 and ori_error < 0.35),
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
