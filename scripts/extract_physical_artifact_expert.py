#!/usr/bin/env python3
"""Extract deployable observations/actions from validated physical artifacts.

The unified physical-reference artifacts store the actual MuJoCo state and the
legal position targets that generated it.  Replaying those targets open loop is
contact-fragile, so it is the wrong way to reconstruct teacher states.  This
script instead writes each independently validated physical state into the
normal ORCS environment, computes the exact deployable policy observation, and
pairs it with the corresponding action-manager command.  State writes happen
only while building an offline imitation dataset; learned-policy evaluation
remains an uninterrupted physical rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from search_strict_orcs_expert import _make_env, _raw_action


def _set_physical_state(env, command, reference, frame: int) -> None:
    origin = env.scene.env_origins[0:1]
    robot_root = torch.cat(
        [
            reference.root_pos_local[:, frame] + origin,
            reference.root_quat_wxyz[:, frame],
            reference.root_lin_vel_w[:, frame],
            reference.root_ang_vel_w[:, frame],
        ],
        dim=-1,
    )
    object_root = torch.cat(
        [
            reference.object_pos_local[:, frame] + origin,
            reference.object_quat_wxyz[:, frame],
            reference.object_lin_vel_w[:, frame],
            reference.object_ang_vel_w[:, frame],
        ],
        dim=-1,
    )
    command.robot.write_root_state_to_sim(robot_root)
    command.robot.write_joint_state_to_sim(
        reference.joint_pos[:, frame], reference.joint_vel[:, frame]
    )
    command.object.write_root_state_to_sim(object_root)

    # All command observations index the flattened reference store through
    # time_steps.  Keep the goal/start selection from reset, changing only the
    # phase represented by this offline sample.
    command.time_steps[:] = command.motion.clip_offsets + frame
    env.scene.write_data_to_sim()
    env.sim.forward()
    # Entity data are cached from the simulator.  A zero-dt update is a no-op
    # in MJLab and would silently yield one repeated observation per start.
    env.scene.update(dt=env.physics_dt)
    env.sim.sense()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, choices=(0, 10, 20))
    parser.add_argument(
        "--output", type=Path, default=Path("logs/physical_artifact_expert_dataset")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from orcs.tasks.uolm.realized_reference import load_realized_reference

    artifact_path = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/"
        "physical_reference.npz"
    ).resolve()
    artifact = load_realized_reference(artifact_path)
    if not artifact.is_task_success:
        raise RuntimeError(f"artifact is not validated successful: {artifact_path}")
    reference = artifact.to_generated_reference(
        device=args.device, dt=0.02, fk_batch_size=300
    )
    env = _make_env(args.start, 1, artifact_path, args.device)
    command = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    controls = torch.as_tensor(
        artifact.control, dtype=torch.float32, device=env.device
    )
    actions = _raw_action(action_term, controls)

    observations: list[np.ndarray] = []
    states: list[np.ndarray] = []
    previous_action = torch.zeros_like(actions[0:1])
    for frame in range(len(artifact.qpos)):
        _set_physical_state(env, command, reference, frame)
        # The deployed observation contains the previous raw actor action.
        # Reconstruct that history exactly without advancing physics.
        env.action_manager._action[:] = previous_action
        env.action_manager._prev_action[:] = previous_action
        env.action_manager._prev_prev_action[:] = previous_action
        # Offline state injection does not advance env.step(), so explicitly
        # invalidate ObservationManager's per-control-step memoization.
        env.observation_manager._obs_buffer = None
        observation = env.observation_manager.compute(update_history=False)["policy"]
        observations.append(observation[0].detach().cpu().numpy())
        states.append(
            np.concatenate(
                [artifact.qpos[frame], artifact.qvel[frame]], dtype=np.float64
            ).astype(np.float32)
        )
        previous_action = actions[frame : frame + 1]

    raw_actions = actions.detach().cpu().numpy().astype(np.float32)
    observation_array = np.asarray(observations, dtype=np.float32)
    if not np.isfinite(observation_array).all() or not np.isfinite(raw_actions).all():
        raise RuntimeError("non-finite artifact expert sample")

    report = {
        "start": args.start,
        "samples": len(observation_array),
        "observation_dim": observation_array.shape[1],
        "action_dim": raw_actions.shape[1],
        "action_min": float(raw_actions.min()),
        "action_max": float(raw_actions.max()),
        "action_clip_fraction": float((np.abs(raw_actions) > 1.0).mean()),
        "source_artifact": str(artifact_path),
        "source_full_valid": bool(artifact.validation.get("full_valid")),
        "source_at_goal": bool(artifact.validation.get("at_goal")),
        "offline_state_injection_only": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"expert_start_{args.start:03d}.npz",
        policy_observation=observation_array,
        action=raw_actions,
        state=np.asarray(states, dtype=np.float32),
        frame=np.arange(len(observation_array), dtype=np.int64),
        start_id=np.full(len(observation_array), args.start, dtype=np.int64),
        target_rad=artifact.control.astype(np.float32),
    )
    (args.output / f"expert_start_{args.start:03d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
