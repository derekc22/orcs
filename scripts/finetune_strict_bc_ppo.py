#!/usr/bin/env python3
"""Fine-tune a distilled realized-reference actor with strict physical PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Orcs-Uolm-Realized-Strict-TaRa")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--starts",
        default=None,
        help="Optional comma-separated realized-reference starts for this run.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--freeze-action-std", action="store_true")
    parser.add_argument("--freeze-actor-normalizer", action="store_true")
    parser.add_argument(
        "--non-strict-load",
        action="store_true",
        help="Allow a checkpoint to initialize a compatible expanded actor.",
    )
    parser.add_argument("--bc-dataset", type=Path)
    parser.add_argument("--bc-updates-per-iteration", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=3.0e-5)
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import orcs.tasks.uolm  # noqa: F401

    # Start from the exact nominal evaluation domain.  Robustness can be added
    # after the corrected task gate is met; random pushes before that obscure
    # whether the distilled physical strategy itself has been learned.
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.starts is not None:
        starts = tuple(int(item) for item in args.starts.split(","))
        root = Path(__file__).resolve().parents[1]
        motion = cfg.commands["motion"]
        motion.realized_reference_paths = tuple(
            str(
                root
                / "logs/unified_physical_reference"
                / f"final_start_{start:03d}"
                / "physical_reference.npz"
            )
            for start in starts
        )
        motion.realized_reference_start_ids = starts
    agent_cfg = load_rl_cfg(args.task)
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = args.save_interval
    agent_cfg.upload_model = False
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.algorithm.entropy_coef = 0.001
    args.output.mkdir(parents=True, exist_ok=True)
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
        env, asdict(agent_cfg), str(args.output), args.device
    )
    # The imitation checkpoints carry a critic from the data-collection task.
    # Its privileged-observation width can differ from this strict task even
    # though the deployable actor interface is identical.  Transfer only the
    # actor (including its observation normalizer); PPO learns a fresh critic.
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
            strict=not args.non_strict_load,
            map_location=args.device,
        )
    distribution = runner.alg._raw_actor.distribution
    if hasattr(distribution, "std_param"):
        distribution.std_param.data.fill_(args.action_std)
    elif hasattr(distribution, "log_std_param"):
        import math

        distribution.log_std_param.data.fill_(math.log(args.action_std))
    else:
        raise RuntimeError("Unsupported actor distribution: cannot set exploration std")
    if args.freeze_action_std:
        for name in ("std_param", "log_std_param"):
            parameter = getattr(distribution, name, None)
            if parameter is not None:
                parameter.requires_grad_(False)
    if args.freeze_actor_normalizer:
        # The imitation checkpoint's weights were trained against these exact
        # statistics.  Updating them before the actor has learned an invariant
        # representation changes every action despite unchanged network
        # weights, which is catastrophic for the contact-sensitive strike.
        runner.alg._raw_actor.update_normalization = lambda obs: None
    if args.bc_dataset is not None and args.bc_updates_per_iteration > 0:
        import numpy as np
        import torch
        from tensordict import TensorDict

        demonstration = np.load(args.bc_dataset)
        bc_obs = torch.as_tensor(
            demonstration["policy_observation"],
            device=args.device,
            dtype=torch.float32,
        )
        bc_action = torch.as_tensor(
            demonstration["action"], device=args.device, dtype=torch.float32
        )
        bc_frame = torch.as_tensor(
            demonstration["frame"], device=args.device, dtype=torch.long
        )
        bc_weight = torch.ones(len(bc_frame), device=args.device)
        bc_weight[bc_frame <= 12] = 12.0
        bc_weight[(bc_frame > 12) & (bc_frame <= 40)] = 8.0
        bc_weight[(bc_frame > 40) & (bc_frame <= 100)] = 3.0
        bc_probability = bc_weight / bc_weight.sum()
        bc_optimizer = torch.optim.AdamW(
            runner.alg._raw_actor.parameters(),
            lr=args.bc_learning_rate,
            weight_decay=1.0e-6,
        )
        ppo_update = runner.alg.update

        def update_with_bc():
            update_result = ppo_update()
            if isinstance(update_result, tuple):
                losses, update_infos = update_result
            else:
                losses, update_infos = update_result, None
            actor = runner.alg._raw_actor
            actor.train()
            bc_loss_value = 0.0
            for _ in range(args.bc_updates_per_iteration):
                indices = torch.multinomial(
                    bc_probability, args.bc_batch_size, replacement=True
                )
                batch = TensorDict(
                    {"policy": bc_obs[indices]}, batch_size=[len(indices)]
                )
                error = actor(batch) - bc_action[indices]
                # The right leg contains the brief task-critical strike.
                loss = error.square().mean() + 0.5 * error[:, :6].square().mean()
                bc_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                bc_optimizer.step()
                bc_loss_value += float(loss.detach().item())
            losses["bc_anchor"] = bc_loss_value / args.bc_updates_per_iteration
            return (
                (losses, update_infos)
                if update_infos is not None
                else losses
            )

        runner.alg.update = update_with_bc
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    main()
