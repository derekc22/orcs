#!/usr/bin/env python3
"""Generate the deterministic HF-BPS diffusion priors for starts 10 and 20."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from orcs.cli.refine_diffusion_reference import _build_model, _condition_from_sample
from orcs.tasks.uolm.diffusion_runtime import BatchedDiffusionGenerator, DiffusionGeneratorCfg


def main() -> None:
    root = Path("/home/learning/Documents/g1-diffusion")
    checkpoint = root / "logs/object_goal_single_stage_init_goal_hf_bps_e10000_b16_lr5e-06_ts1000_w300_s10_transformer_2026Aug05_21-21-42/checkpoints/object_goal_single_stage_init_goal_epoch_000999.pt"
    sample = root / "data/hf_dataset/data/unitree_g1/omomo/sub3_largebox_003/sample1"
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    generator = BatchedDiffusionGenerator(
        DiffusionGeneratorCfg(
            source_path=str(root),
            checkpoint_path=str(checkpoint),
            sampler="ddim",
            num_inference_steps=50,
            eta=0.0,
            generation_batch_size=1,
            horizon=300,
            precision="fp32",
        ),
        device,
    )
    model = _build_model()
    hinge_ids = sorted(
        [
            index
            for index in range(model.njnt)
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or "").startswith("robot/")
        ],
        key=lambda index: int(model.jnt_qposadr[index]),
    )
    limits = torch.as_tensor(model.jnt_range[hinge_ids], dtype=torch.float32, device=device)
    output = Path("logs/unified_physical_reference")
    for start in (10, 20):
        # Re-seeding each deterministic DDIM call matches the existing start-0 protocol.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        condition = _condition_from_sample(sample, start, 300, device)
        trajectory = generator.generate(condition).clone()
        trajectory[..., 9:38].clamp_(limits[:, 0], limits[:, 1])
        np.savez_compressed(
            output / f"diffusion_start_{start:03d}.npz",
            trajectory=trajectory.cpu().numpy(),
            condition=condition.cpu().numpy(),
            start=np.asarray(start),
        )
        print(start, tuple(trajectory.shape))


if __name__ == "__main__":
    main()
