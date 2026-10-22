#!/usr/bin/env python3
"""Search a robot-valid 50 Hz expert in the exact ORCS execution path.

The validated standalone artifact was generated with 200 Hz target changes.
This search converts it into controls that are executable through ORCS's
50 Hz action manager and MJWarp physics.  Robot validity is lexicographically
dominant; object-goal error only ranks full-horizon survivors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


MINIMUM_ROOT_HEIGHT_M = 0.439
MAXIMUM_TORSO_TILT_DEG = 50.0


def _make_env(
    start: int,
    count: int,
    reference: Path,
    device: str,
    task: str = "Orcs-Uolm-Realized-Expert-TaRa",
):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import orcs.tasks.uolm  # noqa: F401
    from orcs.core.sensors import (
        GROUND_CONTACT_SENSOR_NAME,
        STRICT_KILL_BODIES,
        ground_contact_sensor,
    )

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = count
    cfg.seed = 42
    cfg.auto_reset = False
    strict = ground_contact_sensor(STRICT_KILL_BODIES, exclude=(".*foot.*",))
    cfg.scene.sensors = tuple(
        strict if sensor.name == GROUND_CONTACT_SENSOR_NAME else sensor
        for sensor in cfg.scene.sensors
    )
    motion = cfg.commands["motion"]
    motion.realized_reference_paths = (str(reference),)
    motion.realized_reference_start_ids = (start,)
    motion.debug_vis = False
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    env.reset()
    return env


def _raw_action(term, target: torch.Tensor) -> torch.Tensor:
    bias = term._entity.data.encoder_bias[:, term.target_ids]
    return (target + bias - term.offset) / term.scale


def _targets(
    nominal: np.ndarray,
    parameters: np.ndarray,
    joint_ids: np.ndarray,
    switch_frame: int,
    ramp_frames: int,
) -> np.ndarray:
    """Blend from the physical prior into a searched static recovery pose."""
    count = len(parameters)
    targets = np.repeat(nominal[None], count, axis=0)
    hold = nominal[-1].copy()[None].repeat(count, axis=0)
    hold[:, joint_ids] += parameters
    for frame in range(switch_frame, len(nominal)):
        phase = np.clip((frame - switch_frame + 1) / ramp_frames, 0.0, 1.0)
        phase = phase * phase * (3.0 - 2.0 * phase)
        targets[:, frame] = (1.0 - phase) * nominal[frame] + phase * hold
    return targets.astype(np.float32)


def _evaluate(env, target_paths: np.ndarray) -> dict[str, np.ndarray]:
    from mjlab.utils.lab_api.math import matrix_from_quat

    env.reset()
    command = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    robot = command.robot
    obj = command.object
    count, horizon = target_paths.shape[:2]
    paths = torch.as_tensor(target_paths, device=env.device)
    torso_index = robot.body_names.index("torso_link")
    initial_rotation = matrix_from_quat(robot.data.body_link_quat_w[:, torso_index])
    local_up = (
        initial_rotation.transpose(-1, -2)
        @ torch.tensor([0.0, 0.0, 1.0], device=env.device)
        .expand(count, 3)
        .unsqueeze(-1)
    ).squeeze(-1)
    alive = torch.ones(count, dtype=torch.bool, device=env.device)
    first_invalid = torch.full(
        (count,), horizon, dtype=torch.long, device=env.device
    )
    soft_cost = torch.zeros(count, device=env.device)
    invalid_reason = np.full(count, "horizon", dtype=object)
    min_root = torch.full((count,), float("inf"), device=env.device)
    max_tilt = torch.zeros(count, device=env.device)

    for frame in range(horizon):
        _, _, terminated, truncated, _ = env.step(
            _raw_action(action_term, paths[:, frame])
        )
        root_height = robot.data.root_link_pos_w[:, 2]
        torso_up = (
            matrix_from_quat(robot.data.body_link_quat_w[:, torso_index])
            @ local_up.unsqueeze(-1)
        ).squeeze(-1)
        tilt = torch.rad2deg(torch.acos(torso_up[:, 2].clamp(-1.0, 1.0)))
        min_root = torch.minimum(min_root, root_height)
        max_tilt = torch.maximum(max_tilt, tilt)
        # Dense shaping only breaks ties between candidates with equal survival.
        soft_cost += alive * (
            50.0 * torch.relu(0.58 - root_height).square()
            + 0.02 * tilt.square()
            + 0.05 * robot.data.root_link_lin_vel_w[:, 2].square()
            + 0.005 * robot.data.root_link_ang_vel_w.square().sum(dim=-1)
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

    goal_pos = command.object_goal_pos_env + env.scene.env_origins
    position_error = torch.linalg.norm(obj.data.root_link_pos_w - goal_pos, dim=-1)
    dot = torch.sum(obj.data.root_link_quat_w * command.object_goal_quat, dim=-1)
    orientation_error = 2.0 * torch.acos(dot.abs().clamp(max=1.0))
    return {
        "alive": alive.cpu().numpy(),
        "first_invalid": first_invalid.cpu().numpy(),
        "invalid_reason": invalid_reason,
        "soft_cost": soft_cost.cpu().numpy(),
        "position_error": position_error.detach().cpu().numpy(),
        "orientation_error": orientation_error.detach().cpu().numpy(),
        "minimum_root_height": min_root.cpu().numpy(),
        "maximum_torso_tilt": max_tilt.cpu().numpy(),
    }


def _rank(metrics: dict[str, np.ndarray], regularization: np.ndarray) -> np.ndarray:
    valid = metrics["alive"]
    failure = metrics["first_invalid"]
    # One frame of survival dominates all soft costs.  Full survivors are then
    # ordered by the actual task goal, not tracking similarity.
    score = (300 - failure) * 1.0e7 + metrics["soft_cost"] + regularization
    task = (
        (metrics["position_error"] / 0.15) ** 2
        + (metrics["orientation_error"] / 0.35) ** 2
    )
    score[valid] = task[valid] * 1.0e4 + metrics["soft_cost"][valid] + regularization[valid]
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--switch-frame", type=int, default=8)
    parser.add_argument("--ramp-frames", type=int, default=12)
    parser.add_argument("--bound", type=float, default=0.8)
    parser.add_argument("--initial-std", type=float, default=0.25)
    parser.add_argument("--initial-target", type=Path)
    parser.add_argument("--base-target", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("logs/strict_orcs_expert_search")
    )
    args = parser.parse_args()

    from orcs.tasks.uolm.realized_reference import load_realized_reference

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    artifact = load_realized_reference(reference)
    nominal = (
        np.load(args.base_target)["target_rad"]
        if args.base_target is not None
        else artifact.control
    )
    env = _make_env(args.start, args.population, reference, args.device)
    names = list(env.scene["robot"].joint_names)
    joint_ids = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if any(token in name for token in ("hip_", "knee", "ankle_", "waist_"))
        ],
        dtype=np.int64,
    )
    dimension = len(joint_ids)
    rng = np.random.default_rng(4200 + args.start)
    mean = np.zeros(dimension)
    if args.initial_target is not None:
        seed_target = np.load(args.initial_target)["target_rad"]
        mean = seed_target[-1, joint_ids] - nominal[-1, joint_ids]
    std = np.full(dimension, args.initial_std)
    best = mean.copy()
    best_score = float("inf")
    history = []
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimension)),
            -args.bound,
            args.bound,
        )
        population[0] = mean
        population[1] = best
        population[2] = 0.0
        paths = _targets(
            nominal,
            population,
            joint_ids,
            args.switch_frame,
            args.ramp_frames,
        )
        metrics = _evaluate(env, paths)
        regularization = 10.0 * np.square(population).mean(axis=1)
        score = _rank(metrics, regularization)
        order = np.argsort(score)
        winner = int(order[0])
        if score[winner] < best_score:
            best_score = float(score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(0.02, np.sqrt(0.25 * std**2 + 0.75 * elite.var(axis=0)))
        row = {
            "generation": generation,
            "score": float(score[winner]),
            "full_valid": bool(metrics["alive"][winner]),
            "first_invalid_frame": int(metrics["first_invalid"][winner]),
            "invalid_reason": str(metrics["invalid_reason"][winner]),
            "position_error_m": float(metrics["position_error"][winner]),
            "orientation_error_rad": float(metrics["orientation_error"][winner]),
            "minimum_root_height_m": float(metrics["minimum_root_height"][winner]),
            "maximum_torso_tilt_deg": float(metrics["maximum_torso_tilt"][winner]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    best_path = _targets(
        nominal,
        best[None],
        joint_ids,
        args.switch_frame,
        args.ramp_frames,
    )[0]
    np.savez_compressed(
        args.output / f"search_start_{args.start:03d}.npz",
        target_rad=best_path,
        parameters=best,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[index] for index in joint_ids]),
    )
    (args.output / f"search_start_{args.start:03d}.json").write_text(
        json.dumps(
            {
                "start": args.start,
                "dimension": dimension,
                "switch_frame": args.switch_frame,
                "ramp_frames": args.ramp_frames,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
