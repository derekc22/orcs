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
    return raw_env, env, runner


def _load_actor_on_env(env, task: str, checkpoint: Path, device: str):
    """Load a second frozen actor without constructing another simulator."""
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg

    agent_cfg = load_rl_cfg(task)
    agent_cfg.upload_model = False
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
    return runner.get_inference_policy(device=device)


def _feedback_teacher(
    raw_env,
    target: torch.Tensor,
    target_velocity: torch.Tensor,
    parameters: torch.Tensor,
    frame: int,
    feedback_start: int,
    root_feedback_start: int,
) -> torch.Tensor:
    """Query the compact state-feedback expert on the states actually visited.

    The original DAgger prototype repeated the nominal demonstration action at
    every off-nominal state.  That is not DAgger: it supplies no recovery
    feedback and teaches the actor to ignore the very state error that caused
    the rollout to diverge.  This is the same legal joint-target feedback law
    used to generate the validated strict expert, evaluated on each current
    MuJoCo world independently.
    """
    from mjlab.utils.lab_api.math import quat_box_minus
    from search_strict_orcs_expert import _raw_action
    from search_strict_orcs_feedback_expert import _group_ids

    robot = raw_env.scene["robot"]
    action_term = raw_env.action_manager.get_term("joint_pos")
    groups = torch.as_tensor(
        _group_ids(list(robot.joint_names)), device=raw_env.device
    )
    controlled = groups >= 0
    group_index = groups.clamp_min(0)
    kp = parameters[:4]
    kd = parameters[4:8]
    q = robot.data.joint_pos[:, action_term.target_ids]
    qd = robot.data.joint_vel[:, action_term.target_ids]
    nominal = target[frame].expand(q.shape[0], -1)
    desired_velocity = target_velocity[frame].expand(q.shape[0], -1)
    correction = kp[group_index] * (nominal - q) + kd[group_index] * (
        desired_velocity - qd
    )
    command = raw_env.command_manager.get_term("motion")
    names = list(robot.joint_names)
    if len(parameters) >= 10 and frame >= root_feedback_start:
        desired_root_z = command.motion.body_pos_w[
            command.time_steps, command.motion_anchor_body_index, 2
        ]
        root_error = desired_root_z - robot.data.root_link_pos_w[:, 2]
        knee_ids = torch.as_tensor(
            [i for i, name in enumerate(names) if "knee" in name],
            device=raw_env.device,
        )
        hip_pitch_ids = torch.as_tensor(
            [i for i, name in enumerate(names) if "hip_pitch" in name],
            device=raw_env.device,
        )
        if len(parameters) == 12:
            correction[:, knee_ids] += parameters[8:10] * root_error[:, None]
            correction[:, hip_pitch_ids] += parameters[10:12] * root_error[:, None]
        else:
            correction[:, knee_ids] += parameters[8] * root_error[:, None]
            correction[:, hip_pitch_ids] += parameters[9] * root_error[:, None]
    if len(parameters) >= 26 and frame >= root_feedback_start:
        torso = robot.body_names.index("torso_link")
        motion_torso = command.cfg.body_names.index("torso_link")
        torso_error = quat_box_minus(
            command.motion.body_quat_w[command.time_steps, motion_torso],
            robot.data.body_link_quat_w[:, torso],
        )
        torso_velocity_error = (
            command.motion.body_ang_vel_w[command.time_steps, motion_torso]
            - robot.data.body_link_ang_vel_w[:, torso]
        )
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
            ids = torch.as_tensor(
                [names.index(name) for name in joint_names], device=raw_env.device
            )
            value = (
                parameters[10 + 2 * channel] * torso_error[:, axis]
                + parameters[11 + 2 * channel] * torso_velocity_error[:, axis]
            )
            correction[:, ids] += value[:, None]
    correction[:, ~controlled] = 0.0
    if frame < feedback_start:
        correction.zero_()
    applied_target = nominal + correction.clamp(-0.25, 0.25)
    return _raw_action(action_term, applied_target)


def _policy_feedback_correction(
    raw_env,
    parameters: torch.Tensor,
    frame: int,
    feedback_start: int,
) -> torch.Tensor:
    """Action-space recovery used by the late learned-policy expert."""
    from mjlab.utils.lab_api.math import quat_box_minus

    robot = raw_env.scene["robot"]
    command = raw_env.command_manager.get_term("motion")
    names = list(robot.joint_names)
    correction = torch.zeros(
        (raw_env.num_envs, len(names)), device=raw_env.device
    )
    if frame < feedback_start:
        return correction
    step = command.time_steps
    desired_z = command.motion.body_pos_w[
        step, command.motion_anchor_body_index, 2
    ]
    desired_vz = command.motion.body_lin_vel_w[
        step, command.motion_anchor_body_index, 2
    ]
    root_error = desired_z - robot.data.root_link_pos_w[:, 2]
    velocity_error = desired_vz - robot.data.root_link_lin_vel_w[:, 2]
    knee = torch.as_tensor(
        [names.index("left_knee_joint"), names.index("right_knee_joint")],
        device=raw_env.device,
    )
    hip_pitch = torch.as_tensor(
        [names.index("left_hip_pitch_joint"), names.index("right_hip_pitch_joint")],
        device=raw_env.device,
    )
    correction[:, knee] += (
        parameters[0] * root_error + parameters[1] * velocity_error
    )[:, None]
    correction[:, hip_pitch] += (
        parameters[2] * root_error + parameters[3] * velocity_error
    )[:, None]
    torso = robot.body_names.index("torso_link")
    motion_torso = command.cfg.body_names.index("torso_link")
    torso_error = quat_box_minus(
        command.motion.body_quat_w[step, motion_torso],
        robot.data.body_link_quat_w[:, torso],
    )
    omega_error = (
        command.motion.body_ang_vel_w[step, motion_torso]
        - robot.data.body_link_ang_vel_w[:, torso]
    )
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
    for channel, (joint_names, axis) in enumerate(channels):
        ids = torch.as_tensor(
            [names.index(name) for name in joint_names], device=raw_env.device
        )
        value = (
            parameters[4 + 2 * channel] * torso_error[:, axis]
            + parameters[5 + 2 * channel] * omega_error[:, axis]
        )
        correction[:, ids] += value[:, None]
    return correction.clamp(-0.35, 0.35)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=Path("logs/realized_expert_dataset_broad"),
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        help=(
            "Validated compact feedback-expert artifact. When supplied, label "
            "every learner-visited state by querying this feedback law instead "
            "of replaying the nominal demonstration action."
        ),
    )
    parser.add_argument(
        "--policy-feedback",
        type=Path,
        help="Late action-space feedback artifact layered on a frozen teacher actor.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Frozen actor used with --policy-feedback (defaults to learner checkpoint).",
    )
    parser.add_argument("--feedback-start", type=int, default=0)
    parser.add_argument("--root-feedback-start", type=int, default=0)
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
        raw_env, env, runner = _make_env(
            args.task, start, args.num_envs, args.device, args.seed + start
        )
        runner.load(
            str(args.checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            map_location=args.device,
        )
        actor = runner.get_inference_policy(device=args.device)
        teacher_actor = None
        if args.policy_feedback is not None:
            feedback = np.load(args.policy_feedback)
            parameters = torch.as_tensor(
                feedback["parameters"], device=args.device, dtype=torch.float32
            )
            policy_feedback_start = int(feedback["feedback_start"])
            teacher_actor = _load_actor_on_env(
                env,
                args.task,
                args.teacher_checkpoint or args.checkpoint,
                args.device,
            )
            max_frames = 300
            expert = None
        elif args.feedback is not None:
            feedback = np.load(args.feedback)
            target = torch.as_tensor(
                feedback["target_rad"], device=args.device, dtype=torch.float32
            )
            parameters = torch.as_tensor(
                feedback["parameters"], device=args.device, dtype=torch.float32
            )
            target_velocity = torch.zeros_like(target)
            target_velocity[1:-1] = (target[2:] - target[:-2]) * 25.0
            target_velocity[0] = (target[1] - target[0]) * 50.0
            target_velocity[-1] = (target[-1] - target[-2]) * 50.0
            max_frames = min(300, int(target.shape[0]))
            expert = None
        else:
            expert = torch.as_tensor(
                np.load(args.expert_root / f"expert_start_{start:03d}.npz")["action"],
                device=args.device,
            )
            max_frames = min(300, int(expert.shape[0]))
        generator = torch.Generator(device=args.device).manual_seed(args.seed + start)
        obs = env.get_observations()
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        collected = 0
        from mjlab.utils.lab_api.math import matrix_from_quat

        robot = raw_env.scene["robot"]
        torso_index = robot.body_names.index("torso_link")
        initial_rotation = matrix_from_quat(
            robot.data.body_link_quat_w[:, torso_index]
        )
        torso_local_up = (
            initial_rotation.transpose(-1, -2)
            @ torch.tensor([0.0, 0.0, 1.0], device=args.device)
            .expand(args.num_envs, 3)
            .unsqueeze(-1)
        ).squeeze(-1)
        # Never manufacture labels beyond the demonstrated teacher horizon.
        # Some recovery experts are intentionally partial and end at their
        # first unsupported state; clamping their final action would silently
        # turn that terminal control into a fictitious long-horizon teacher.
        for frame in range(max_frames):
            if teacher_actor is not None:
                with torch.inference_mode():
                    teacher_action = teacher_actor(obs) + _policy_feedback_correction(
                        raw_env,
                        parameters,
                        frame,
                        policy_feedback_start,
                    )
            else:
                teacher_action = (
                    _feedback_teacher(
                    raw_env,
                    target,
                    target_velocity,
                    parameters,
                    frame,
                    args.feedback_start,
                    args.root_feedback_start,
                )
                    if args.feedback is not None
                    else expert[frame].expand(args.num_envs, -1)
                )
            chunks["policy_observation"].append(
                obs["policy"].detach().cpu().numpy()
            )
            chunks["action"].append(
                teacher_action.detach().cpu().numpy()
            )
            chunks["frame"].append(
                np.full(args.num_envs, frame, dtype=np.int64)
            )
            chunks["start_id"].append(
                np.full(args.num_envs, start, dtype=np.int64)
            )
            with torch.inference_mode():
                learner_action = actor(obs)
            action = (
                (1.0 - args.teacher_mixture) * learner_action
                + args.teacher_mixture * teacher_action
            )
            action += args.action_noise * torch.randn(
                action.shape, generator=generator, device=action.device
            )
            obs, _, done, _ = env.step(action)
            root_height = robot.data.root_link_pos_w[:, 2]
            torso_rotation = matrix_from_quat(
                robot.data.body_link_quat_w[:, torso_index]
            )
            torso_up = (torso_rotation @ torso_local_up.unsqueeze(-1)).squeeze(-1)
            torso_tilt = torch.rad2deg(
                torch.acos(torso_up[:, 2].clamp(-1.0, 1.0))
            )
            physical_invalid = (root_height < 0.439) | (torso_tilt > 50.0)
            done = done.bool() | physical_invalid
            alive &= ~done
            collected += args.num_envs
            # With auto_reset disabled MJLab requires every terminated world
            # to be reset before another vector step.  Stop at the first
            # learner failure; these are precisely the boundary states this
            # DAgger batch is intended to label.
            if bool(done.any().item()):
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
