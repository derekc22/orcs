"""Thin runtime bridge to the active g1-diffusion inference pipeline.

ORCS owns conditioning and reference semantics.  This bridge only loads the
configured diffusion checkout/checkpoint once and exposes batched physical
``[B,T,47]`` generation.  No training or dataset code is imported by ORCS.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Literal

import torch

__all__ = ["DiffusionGeneratorCfg", "BatchedDiffusionGenerator"]


@dataclass(kw_only=True)
class DiffusionGeneratorCfg:
    source_path: str
    checkpoint_path: str
    sampler: Literal["ddim", "ddpm"] = "ddim"
    num_inference_steps: int = 50
    eta: float = 0.0
    generation_batch_size: int = 16
    horizon: int = 300
    # This checkpoint is numerically unstable under pure FP16 DDIM on RTX 20
    # series hardware (the denoising state becomes non-finite).  Match the
    # checkpoint's active sampling configuration and default to FP32.
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class BatchedDiffusionGenerator:
    """Load one active checkpoint and generate reset batches in chunks."""

    def __init__(self, cfg: DiffusionGeneratorCfg, device: str | torch.device):
        self.cfg = cfg
        self.device = torch.device(device)
        if not cfg.source_path:
            raise ValueError(
                "diffusion source path is required; set diffusion_source_path "
                "or ORCS_G1_DIFFUSION_SOURCE"
            )
        if not cfg.checkpoint_path:
            raise ValueError(
                "diffusion checkpoint path is required; set "
                "diffusion_checkpoint_path or ORCS_G1_DIFFUSION_CHECKPOINT"
            )
        source = Path(cfg.source_path).expanduser().resolve()
        checkpoint = Path(cfg.checkpoint_path).expanduser().resolve()
        if not (source / "scripts" / "sample_object_goal_single_stage_init_goal_hf_bps.py").is_file():
            raise FileNotFoundError(f"not a g1-diffusion checkout: {source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"diffusion checkpoint not found: {checkpoint}")
        if cfg.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")
        if cfg.horizon < 2:
            raise ValueError("diffusion horizon must be at least two frames")

        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        module = importlib.import_module(
            "scripts.sample_object_goal_single_stage_init_goal_hf_bps"
        )
        module_path = Path(module.__file__).resolve()
        if source not in module_path.parents:
            raise ImportError(
                f"diffusion sampler resolved to {module_path}, expected under {source}"
            )

        inference = module.InferenceConfig()
        inference.use_torch_compile = False
        inference.precision = module.PrecisionMode(cfg.precision)
        inference.sampler = module.SamplerType(cfg.sampler)
        inference.num_inference_steps = int(cfg.num_inference_steps)
        inference.ddim_eta = float(cfg.eta)

        self.pipeline = module.SingleStageInitGoalPipeline(
            str(checkpoint), inference, self.device
        )
        self.checkpoint_path = str(checkpoint)
        self.source_path = source_text

    @torch.inference_mode()
    def generate(self, physical_global_condition: torch.Tensor) -> torch.Tensor:
        """Generate denormalized physical trajectories for ``[B,56]`` input."""
        if physical_global_condition.ndim != 2 or physical_global_condition.shape[1] != 56:
            raise ValueError(
                "physical_global_condition must have shape [B,56], got "
                f"{tuple(physical_global_condition.shape)}"
            )
        chunks: list[torch.Tensor] = []
        for condition in physical_global_condition.split(
            self.cfg.generation_batch_size, dim=0
        ):
            chunks.append(
                self.pipeline.generate_batch(condition, self.cfg.horizon).to(
                    device=self.device, dtype=torch.float32
                )
            )
        result = torch.cat(chunks, dim=0)
        return result
