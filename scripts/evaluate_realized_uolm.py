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
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

MINIMUM_ROOT_HEIGHT_M = 0.439
MAXIMUM_TORSO_TILT_DEG = 50.0
GOAL_PERSISTENCE_FRAMES = 20


def _quat_error(q1: torch.Tensor, q2: torch.Tensor) -> float:
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
    return float((2.0 * torch.acos(dot)).item())


def _overlay_diagnostics(
    frame: np.ndarray,
    *,
    start: int,
    seed: int,
    metrics: dict[str, object],
) -> np.ndarray:
    """Add a readable diagnostic band without obscuring the simulation."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    band_height = 74
    canvas = Image.new("RGB", (image.width, image.height + band_height), "black")
    canvas.paste(image, (0, band_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    valid = bool(metrics["robot_valid_so_far"])
    forbidden = bool(metrics["forbidden_contact"])
    final_status = metrics.get("final_success")
    lines = (
        f"Start {start:03d} | Seed {seed} | Frame {int(metrics['frame']):03d}/300",
        f"Robot valid: {'YES' if valid else 'NO '} | Forbidden: {'YES' if forbidden else 'NO'}",
        f"Box pos err: {float(metrics['box_position_error_m']):.3f} m",
        f"Box ori err: {float(metrics['box_orientation_error_rad']):.3f} rad"
        + (
            f" | FINAL: {'PASS' if final_status else 'FAIL'}"
            if final_status is not None
            else ""
        ),
    )
    colors = (
        "white",
        "#6CFF6C" if valid and not forbidden else "#FF6464",
        "white",
        "#6CFF6C"
        if final_status
        else ("#FF6464" if final_status is False else "white"),
    )
    for index, (line, color) in enumerate(zip(lines, colors, strict=True)):
        draw.text((5, 2 + 17 * index), line, fill=color, font=font)
    return np.asarray(canvas)


def _evaluate_one(
    *,
    task: str,
    checkpoint: Path,
    reference: Path,
    start: int,
    seed: int,
    device: str,
    late_checkpoint: Path | None = None,
    switch_frame: int | None = None,
    override_target: Path | None = None,
    override_frames: int = 0,
    action_residual: Path | None = None,
    feedback_residual: Path | None = None,
    video_path: Path | None = None,
    video_fps: int = 25,
    overlay_metrics: bool = False,
    frame_metrics_path: Path | None = None,
) -> dict[str, object]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_box_minus

    # Importing the task package performs its registry declarations.
    import orcs.tasks.uolm  # noqa: F401
    from orcs.assets import OBJECT_BODY_NAME
    from orcs.core.sensors import (
        GROUND_CONTACT_SENSOR_NAME,
        STRICT_KILL_BODIES,
        ground_contact_sensor,
    )

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    # Evaluation is deliberately stricter than the historical UOLM training
    # task.  The latter only terminated pelvis contact, which let a prone
    # robot fling the object and be counted as successful.  A learned-policy
    # success may use only the feet for terrain support.
    strict_ground_sensor = ground_contact_sensor(
        STRICT_KILL_BODIES, exclude=(".*foot.*",)
    )
    cfg.scene.sensors = tuple(
        strict_ground_sensor if sensor.name == GROUND_CONTACT_SENSOR_NAME else sensor
        for sensor in cfg.scene.sensors
    )
    # Keep simulating after a robot-invalid state so object-at-goal can be
    # reported independently from robot validity.  This is exactly the case
    # the old evaluator got wrong: a collapsed robot could fling the box into
    # the goal later in the rollout.  Strict validity is measured below from
    # the live state/contact sensor; only the 300-frame timeout remains an env
    # termination so no automatic/manual reset interrupts the diagnostic.
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    motion_cfg = cfg.commands["motion"]
    motion_cfg.realized_reference_paths = (str(reference),)
    motion_cfg.realized_reference_start_ids = (start,)
    motion_cfg.debug_vis = False

    agent_cfg = load_rl_cfg(task)
    raw_env = ManagerBasedRlEnv(
        cfg=cfg,
        device=device,
        render_mode="rgb_array" if video_path is not None else None,
    )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    override = None
    residual = None
    feedback_parameters = None
    feedback_start = 0
    action_term = raw_env.action_manager.get_term("joint_pos")
    if override_target is not None:
        from search_strict_orcs_expert import _raw_action

        target = np.load(override_target)["target_rad"].astype(np.float32)
        override = _raw_action(action_term, torch.as_tensor(target, device=device))
    if action_residual is not None:
        residual = torch.as_tensor(
            np.load(action_residual)["residual_action"],
            device=device,
            dtype=torch.float32,
        )
    if feedback_residual is not None:
        feedback_data = np.load(feedback_residual)
        feedback_parameters = torch.as_tensor(
            feedback_data["parameters"], device=device, dtype=torch.float32
        )
        feedback_start = int(feedback_data["feedback_start"])
    late_policy = None
    if late_checkpoint is not None:
        late_runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
        late_runner.load(
            str(late_checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            map_location=device,
        )
        late_policy = late_runner.get_inference_policy(device=device)

    cmd = raw_env.command_manager.get_term("motion")
    obj = raw_env.scene[OBJECT_BODY_NAME]
    robot = raw_env.scene["robot"]
    torso_index = robot.body_names.index("torso_link")
    forbidden_contact_sensor = raw_env.scene[GROUND_CONTACT_SENSOR_NAME]
    motion_torso_index = cmd.cfg.body_names.index("torso_link")
    joint_names = list(robot.joint_names)
    knee_ids = torch.as_tensor(
        [joint_names.index("left_knee_joint"), joint_names.index("right_knee_joint")],
        device=device,
    )
    hip_pitch_ids = torch.as_tensor(
        [
            joint_names.index("left_hip_pitch_joint"),
            joint_names.index("right_hip_pitch_joint"),
        ],
        device=device,
    )
    feedback_channels = (
        (("left_hip_roll_joint",), 0),
        (("right_hip_roll_joint",), 0),
        (("left_ankle_roll_joint",), 0),
        (("right_ankle_roll_joint",), 0),
        (("waist_roll_joint",), 0),
        (("left_hip_pitch_joint", "right_hip_pitch_joint"), 1),
        (("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 1),
        (("waist_pitch_joint",), 1),
    )
    initial_torso_rotation = matrix_from_quat(
        robot.data.body_link_quat_w[0, torso_index]
    )
    torso_local_up = initial_torso_rotation.transpose(-1, -2) @ torch.tensor(
        [0.0, 0.0, 1.0], device=device
    )
    initial_pos = obj.data.root_link_pos_w[0].clone()
    goal_pos = cmd.object_goal_pos_env[0] + raw_env.scene.env_origins[0]
    goal_quat = cmd.object_goal_quat[0]
    episode_return = 0.0
    steps = 0
    termination = "horizon"
    at_goal_history: list[bool] = []
    first_invalid_frame: int | None = None
    invalid_reason: str | None = None
    invalid_root_height_m: float | None = None
    invalid_torso_tilt_deg: float | None = None
    minimum_root_height_m = float("inf")
    maximum_torso_tilt_deg = 0.0
    no_forbidden_support = True
    frame_metrics: list[dict[str, object]] = []

    def current_metrics(frame_number: int) -> dict[str, object]:
        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        torso_rotation = matrix_from_quat(robot.data.body_link_quat_w[0, torso_index])
        torso_up = torso_rotation @ torso_local_up
        torso_tilt_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, float(torso_up[2].item()))))
        )
        found = forbidden_contact_sensor.data.found
        if found is None:
            raise RuntimeError("strict ground-contact sensor has no found field")
        forbidden_contact = bool(found[0].any().item())
        frame_pos_error = float(
            torch.linalg.norm(obj.data.root_link_pos_w[0] - goal_pos).item()
        )
        frame_ori_error = _quat_error(obj.data.root_link_quat_w[0], goal_quat)
        current_reason = None
        if forbidden_contact:
            current_reason = "forbidden_nonfoot_terrain_contact"
        elif root_height < MINIMUM_ROOT_HEIGHT_M:
            current_reason = "root_height"
        elif torso_tilt_deg > MAXIMUM_TORSO_TILT_DEG:
            current_reason = "torso_tilt"
        valid_so_far = first_invalid_frame is None and current_reason is None
        snapshot: dict[str, object] = {
            "frame": frame_number,
            "robot_valid_so_far": valid_so_far,
            "forbidden_contact": forbidden_contact,
            "root_height_m": root_height,
            "torso_tilt_deg": torso_tilt_deg,
            "box_position_error_m": frame_pos_error,
            "box_orientation_error_rad": frame_ori_error,
            "physical_reason": current_reason,
            "final_success": None,
        }
        if frame_number == 300:
            snapshot["final_success"] = bool(
                valid_so_far
                and not forbidden_contact
                and frame_pos_error < 0.15
                and frame_ori_error < 0.35
            )
        return snapshot

    video_writer = None
    if video_path is not None:
        import imageio.v2 as imageio

        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=video_fps)
        frame = raw_env.render()
        if frame is None:
            raise RuntimeError("rgb_array renderer returned no initial frame")
        initial_metrics = current_metrics(0)
        frame_metrics.append(initial_metrics)
        if overlay_metrics:
            frame = _overlay_diagnostics(
                frame, start=start, seed=seed, metrics=initial_metrics
            )
        video_writer.append_data(frame)
    elif frame_metrics_path is not None:
        frame_metrics.append(current_metrics(0))

    obs = env.get_observations()
    for step in range(300):
        with torch.inference_mode():
            active_policy = (
                late_policy
                if late_policy is not None
                and switch_frame is not None
                and step >= switch_frame
                else policy
            )
            action = active_policy(obs)
            if override is not None and step < override_frames:
                action = override[step : step + 1]
            if residual is not None:
                action = action + residual[step : step + 1]
            if feedback_parameters is not None and step >= feedback_start:
                correction = torch.zeros_like(action)
                motion_step = cmd.time_steps
                desired_z = cmd.motion.body_pos_w[
                    motion_step, cmd.motion_anchor_body_index, 2
                ]
                desired_vz = cmd.motion.body_lin_vel_w[
                    motion_step, cmd.motion_anchor_body_index, 2
                ]
                root_error = desired_z - robot.data.root_link_pos_w[:, 2]
                velocity_error = desired_vz - robot.data.root_link_lin_vel_w[:, 2]
                correction[:, knee_ids] += (
                    feedback_parameters[0] * root_error
                    + feedback_parameters[1] * velocity_error
                )[:, None]
                correction[:, hip_pitch_ids] += (
                    feedback_parameters[2] * root_error
                    + feedback_parameters[3] * velocity_error
                )[:, None]
                torso_error = quat_box_minus(
                    cmd.motion.body_quat_w[motion_step, motion_torso_index],
                    robot.data.body_link_quat_w[:, torso_index],
                )
                omega_error = (
                    cmd.motion.body_ang_vel_w[motion_step, motion_torso_index]
                    - robot.data.body_link_ang_vel_w[:, torso_index]
                )
                for channel, (names, axis) in enumerate(feedback_channels):
                    ids = torch.as_tensor(
                        [joint_names.index(name) for name in names], device=device
                    )
                    value = (
                        feedback_parameters[4 + 2 * channel] * torso_error[:, axis]
                        + feedback_parameters[5 + 2 * channel] * omega_error[:, axis]
                    )
                    correction[:, ids] += value[:, None]
                action = action + correction.clamp(-0.35, 0.35)
        obs, reward, done, _ = env.step(action)
        episode_return += float(reward[0].item())
        steps = step + 1
        at_goal_history.append(bool(cmd.metrics["at_goal"][0].item() > 0.5))
        root_height = float(robot.data.root_link_pos_w[0, 2].item())
        torso_rotation = matrix_from_quat(robot.data.body_link_quat_w[0, torso_index])
        torso_up = torso_rotation @ torso_local_up
        torso_tilt_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, float(torso_up[2].item()))))
        )
        minimum_root_height_m = min(minimum_root_height_m, root_height)
        maximum_torso_tilt_deg = max(maximum_torso_tilt_deg, torso_tilt_deg)
        found = forbidden_contact_sensor.data.found
        if found is None:
            raise RuntimeError("strict ground-contact sensor has no found field")
        forbidden_support = bool(found[0].any().item())
        no_forbidden_support &= not forbidden_support
        physical_reason = None
        if forbidden_support:
            physical_reason = "forbidden_nonfoot_terrain_contact"
        elif root_height < MINIMUM_ROOT_HEIGHT_M:
            physical_reason = "root_height"
        elif torso_tilt_deg > MAXIMUM_TORSO_TILT_DEG:
            physical_reason = "torso_tilt"
        if physical_reason is not None and first_invalid_frame is None:
            first_invalid_frame = steps
            invalid_reason = physical_reason
            invalid_root_height_m = root_height
            invalid_torso_tilt_deg = torso_tilt_deg
            termination = physical_reason
        step_metrics = current_metrics(steps)
        # Preserve the cumulative validity decision made immediately above.
        step_metrics["robot_valid_so_far"] = first_invalid_frame is None
        frame_metrics.append(step_metrics)
        if video_writer is not None:
            frame = raw_env.render()
            if frame is None:
                raise RuntimeError("rgb_array renderer returned no rollout frame")
            if overlay_metrics:
                frame = _overlay_diagnostics(
                    frame, start=start, seed=seed, metrics=step_metrics
                )
            video_writer.append_data(frame)
        if bool(done[0].item()):
            if first_invalid_frame is None:
                termination = "horizon" if steps == 300 else "time_out"
            break

    final_pos = obj.data.root_link_pos_w[0]
    final_quat = obj.data.root_link_quat_w[0]
    pos_error = float(torch.linalg.norm(final_pos - goal_pos).item())
    ori_error = _quat_error(final_quat, goal_quat)
    displacement = float(torch.linalg.norm(final_pos - initial_pos).item())
    object_at_goal = pos_error < 0.15 and ori_error < 0.35
    full_horizon_valid = steps == 300 and first_invalid_frame is None
    robot_valid = first_invalid_frame is None
    goal_persistent = len(at_goal_history) >= GOAL_PERSISTENCE_FRAMES and all(
        at_goal_history[-GOAL_PERSISTENCE_FRAMES:]
    )
    final_success = (
        object_at_goal and robot_valid and no_forbidden_support and full_horizon_valid
    )
    result = {
        "start": start,
        "seed": seed,
        "success": final_success,
        "object_at_goal": object_at_goal,
        "object_at_goal_ever": any(at_goal_history),
        "object_at_goal_final_window": goal_persistent,
        "robot_valid": robot_valid,
        "no_forbidden_support": no_forbidden_support,
        "full_horizon_valid": full_horizon_valid,
        "final_success": final_success,
        "first_invalid_frame": first_invalid_frame,
        "invalid_reason": invalid_reason,
        "invalid_root_height_m": invalid_root_height_m,
        "invalid_torso_tilt_deg": invalid_torso_tilt_deg,
        "minimum_root_height_m": minimum_root_height_m,
        "maximum_torso_tilt_deg": maximum_torso_tilt_deg,
        "final_position_error_m": pos_error,
        "final_orientation_error_rad": ori_error,
        "object_displacement_m": displacement,
        "episode_length": steps,
        "return": episode_return,
        "termination": termination,
        "fall_or_illegal": not robot_valid or not no_forbidden_support,
        "virtual_object_force": 0.0,
        "late_checkpoint": str(late_checkpoint) if late_checkpoint else None,
        "policy_switch_frame": switch_frame,
        "override_target": str(override_target) if override_target else None,
        "override_frames": override_frames,
        "action_residual": str(action_residual) if action_residual else None,
        "feedback_residual": str(feedback_residual) if feedback_residual else None,
        "video_path": str(video_path.resolve()) if video_path else None,
        "frame_metrics_path": (
            str(frame_metrics_path.resolve()) if frame_metrics_path else None
        ),
    }
    if video_writer is not None:
        video_writer.close()
    if frame_metrics_path is not None:
        frame_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        frame_metrics_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "reference": str(reference),
                    "start": start,
                    "seed": seed,
                    "zero_assistance": True,
                    "late_checkpoint": None,
                    "override_target": None,
                    "action_residual": None,
                    "feedback_residual": None,
                    "frames": frame_metrics,
                    "result": result,
                },
                indent=2,
            )
            + "\n"
        )
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--late-checkpoint", type=Path)
    parser.add_argument("--switch-frame", type=int)
    parser.add_argument("--override-target", type=Path)
    parser.add_argument("--override-frames", type=int, default=0)
    parser.add_argument("--action-residual", type=Path)
    parser.add_argument("--feedback-residual", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument(
        "--overlay-metrics",
        action="store_true",
        help="Burn strict per-frame validity and box-error metrics into videos.",
    )
    parser.add_argument(
        "--frame-metrics-dir",
        type=Path,
        help="Write one machine-readable per-frame metrics JSON per rollout.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=25,
        help="Encoding rate; 25 fps produces a review-friendly 0.5x-speed video.",
    )
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
            args.reference_root / f"final_start_{start:03d}" / "physical_reference.npz"
        ).resolve()
        for seed in seeds:
            row = _evaluate_one(
                task=args.task,
                checkpoint=checkpoint,
                reference=reference,
                start=start,
                seed=seed,
                device=args.device,
                late_checkpoint=(
                    args.late_checkpoint.expanduser().resolve()
                    if args.late_checkpoint
                    else None
                ),
                switch_frame=args.switch_frame,
                override_target=(
                    args.override_target.expanduser().resolve()
                    if args.override_target
                    else None
                ),
                override_frames=args.override_frames,
                action_residual=(
                    args.action_residual.expanduser().resolve()
                    if args.action_residual
                    else None
                ),
                feedback_residual=(
                    args.feedback_residual.expanduser().resolve()
                    if args.feedback_residual
                    else None
                ),
                video_path=(
                    args.video_dir / f"start_{start:03d}_seed_{seed}_strict.mp4"
                    if args.video_dir
                    else None
                ),
                video_fps=args.video_fps,
                overlay_metrics=args.overlay_metrics,
                frame_metrics_path=(
                    args.frame_metrics_dir
                    / f"start_{start:03d}_seed_{seed}_frames.json"
                    if args.frame_metrics_dir
                    else None
                ),
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    per_start: dict[str, dict[str, float]] = {}
    for start in starts:
        group = [row for row in rows if row["start"] == start]
        per_start[str(start)] = {
            "success_rate": float(np.mean([row["final_success"] for row in group])),
            "robot_valid_rate": float(np.mean([row["robot_valid"] for row in group])),
            "object_at_goal_rate": float(
                np.mean([row["object_at_goal"] for row in group])
            ),
            "no_forbidden_support_rate": float(
                np.mean([row["no_forbidden_support"] for row in group])
            ),
            "full_horizon_valid_rate": float(
                np.mean([row["full_horizon_valid"] for row in group])
            ),
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
            "success_rate": float(np.mean([row["final_success"] for row in rows])),
            "robot_valid_rate": float(np.mean([row["robot_valid"] for row in rows])),
            "object_at_goal_rate": float(
                np.mean([row["object_at_goal"] for row in rows])
            ),
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
