#!/usr/bin/env python3
"""Distill ORCS-native largebox expert rollouts into a deployable TaRa actor.

The demonstrations contain the exact normalized actions accepted by the normal
UOLM action manager and observations from the same runtime policy stream.  This
script only trains the neural actor; the saved checkpoint therefore has no
expert controller or action playback path at inference time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


def _load_dataset(
    root: Path, extras: tuple[Path, ...], starts: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    arrays: dict[str, list[np.ndarray]] = {
        "observation": [],
        "action": [],
        "frame": [],
        "start": [],
    }
    for start in starts:
        data = np.load(root / f"expert_start_{start:03d}.npz")
        arrays["observation"].append(data["policy_observation"])
        arrays["action"].append(data["action"])
        arrays["frame"].append(data["frame"])
        arrays["start"].append(data["start_id"])
    for extra in extras:
        data = np.load(extra)
        arrays["observation"].append(data["policy_observation"])
        arrays["action"].append(data["action"])
        arrays["frame"].append(data["frame"])
        arrays["start"].append(data["start_id"])
    return {
        key: torch.from_numpy(np.concatenate(value, axis=0))
        for key, value in arrays.items()
    }


def _sampling_weight(frame: torch.Tensor) -> torch.Tensor:
    """Keep the short contact event competitive with easy settling frames."""
    weight = torch.ones_like(frame, dtype=torch.float32)
    weight[(frame >= 0) & (frame <= 12)] = 12.0  # strike setup and impulse
    weight[(frame > 12) & (frame <= 40)] = 8.0  # contact result and arrest
    weight[(frame > 40) & (frame <= 100)] = 3.0  # support recovery
    return weight


def _make_runner(task: str, device: str):
    # mjlab must be imported before ORCS task registration (entry-point scan).
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import orcs.tasks.uolm  # noqa: F401

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    agent_cfg = load_rl_cfg(task)
    agent_cfg.upload_model = False
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    return env, runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("logs/realized_expert_dataset_broad")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/realized_expert_bc")
    )
    parser.add_argument("--extra-dataset", type=Path, nargs="*", default=())
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", default="Orcs-Uolm-Realized-Expert-TaRa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--starts", default="0,10,20")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--teacher-frame-max",
        type=int,
        help=(
            "If set, apply expert imitation only through this frame and anchor "
            "later outputs to the initial checkpoint."
        ),
    )
    parser.add_argument(
        "--late-anchor-weight",
        type=float,
        default=1.0,
        help="Output-distillation weight after --teacher-frame-max.",
    )
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="Use the complete weighted dataset each update for exact interpolation.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    starts = tuple(int(item) for item in args.starts.split(","))
    data = _load_dataset(args.dataset, tuple(args.extra_dataset), starts)
    device = torch.device(args.device)
    observation = data["observation"].to(device=device, dtype=torch.float32)
    action = data["action"].to(device=device, dtype=torch.float32)
    frame = data["frame"].to(device=device)
    start = data["start"].to(device=device)
    weight = _sampling_weight(frame)

    env, runner = _make_runner(args.task, args.device)
    if args.checkpoint is not None:
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
    actor = runner.alg.get_policy()
    actor.train()
    full_obs = TensorDict({"policy": observation}, batch_size=[len(observation)])
    # Freeze exact dataset normalization statistics before optimization.  PPO
    # checkpoints retain these buffers, so evaluation uses the same transform.
    if args.checkpoint is None:
        actor.update_normalization(full_obs)
    actor.obs_normalizer.eval()
    with torch.inference_mode():
        anchor_action = actor(full_obs).detach().clone()
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    probabilities = weight / weight.sum()
    right_leg_ids = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
    ledger: list[dict[str, float | int]] = []
    best_strike_mse = float("inf")
    best_path = args.output / "model_best.pt"
    for update in range(1, args.updates + 1):
        indices = (
            torch.arange(len(observation), device=device)
            if args.full_batch
            else torch.multinomial(probabilities, args.batch_size, replacement=True)
        )
        batch_obs = TensorDict(
            {"policy": observation[indices]}, batch_size=[len(indices)]
        )
        prediction = actor(batch_obs)
        error = prediction - action[indices]
        # The right leg creates the physical strike.  Its modest auxiliary
        # weight prevents its six channels from being diluted by all 29 DOFs.
        if args.teacher_frame_max is not None:
            teacher_mask = frame[indices] <= args.teacher_frame_max
            late_mask = ~teacher_mask
            if not bool(teacher_mask.any().item()):
                raise RuntimeError("batch contains no samples inside teacher window")
            teacher_error = error[teacher_mask]
            loss_all = teacher_error.square().mean()
            loss_right_leg = teacher_error[:, right_leg_ids].square().mean()
            late_anchor = (
                (prediction[late_mask] - anchor_action[indices][late_mask])
                .square()
                .mean()
                if bool(late_mask.any().item())
                else prediction.new_zeros(())
            )
        elif args.full_batch:
            batch_weight = weight[indices]
            batch_weight = batch_weight / batch_weight.sum()
            loss_all = (error.square().mean(dim=-1) * batch_weight).sum()
            loss_right_leg = (
                error[:, right_leg_ids].square().mean(dim=-1) * batch_weight
            ).sum()
        else:
            loss_all = error.square().mean()
            loss_right_leg = error[:, right_leg_ids].square().mean()
        if args.teacher_frame_max is None:
            late_anchor = prediction.new_zeros(())
        loss = (
            loss_all
            + 0.5 * loss_right_leg
            + args.late_anchor_weight * late_anchor
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()

        if update == 1 or update % 100 == 0 or update == args.updates:
            actor.eval()
            with torch.inference_mode():
                prediction_all = actor(full_obs)
                squared = (prediction_all - action).square()
                strike = frame <= 40
                metrics = {
                    "update": update,
                    "loss": float(loss.item()),
                    "action_mse": float(squared.mean().item()),
                    "strike_action_mse": float(squared[strike].mean().item()),
                    "right_leg_mse": float(
                        squared[:, right_leg_ids].mean().item()
                    ),
                    "strike_right_leg_mse": float(
                        squared[strike][:, right_leg_ids].mean().item()
                    ),
                    "max_abs_error": float(
                        (prediction_all - action).abs().max().item()
                    ),
                }
                for start_id in starts:
                    mask = start == start_id
                    metrics[f"start_{start_id}_mse"] = float(
                        squared[mask].mean().item()
                    )
            actor.train()
            ledger.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if metrics["strike_action_mse"] < best_strike_mse:
                best_strike_mse = float(metrics["strike_action_mse"])
                runner.current_learning_iteration = update
                runner.save(
                    str(best_path),
                    infos={"training": "phase_balanced_behavior_cloning"},
                )
        if update % args.save_every == 0 or update == args.updates:
            runner.current_learning_iteration = update
            runner.save(
                str(args.output / f"model_{update}.pt"),
                infos={"training": "phase_balanced_behavior_cloning"},
            )

    (args.output / "training_ledger.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "dataset": str(args.dataset.resolve()),
                "extra_datasets": [str(path.resolve()) for path in args.extra_dataset],
                "initial_checkpoint": (
                    str(args.checkpoint.resolve()) if args.checkpoint else None
                ),
                "num_demonstrations": len(starts),
                "num_samples": len(observation),
                "phase_sampling": {
                    "frames_0_12": 12.0,
                    "frames_13_40": 8.0,
                    "frames_41_100": 3.0,
                    "frames_101_299": 1.0,
                },
                "updates": args.updates,
                "batch_size": args.batch_size,
                "full_batch": args.full_batch,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "teacher_frame_max": args.teacher_frame_max,
                "late_anchor_weight": args.late_anchor_weight,
                "metrics": ledger,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
