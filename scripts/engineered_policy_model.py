#!/usr/bin/env python3
"""Compact learned policy used by engineered locomomanipulation distillation."""

from __future__ import annotations

import math

import torch
from torch import nn


class EngineeredTrackingPolicy(nn.Module):
    """Spectral feedforward actor plus learned state-error feedback.

    The model stores learned spectral coefficients, not an action trajectory.
    At inference it evaluates a continuous phase basis and corrects the output
    from the current robot/object tracking error.
    """

    def __init__(self, horizon_steps: int, error_dim: int, action_dim: int = 29):
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.error_dim = int(error_dim)
        self.action_dim = int(action_dim)
        self.maximum_frequency = self.horizon_steps // 2
        feature_dim = self.horizon_steps
        self.feedforward = nn.Linear(feature_dim, action_dim, bias=False)
        self.feedback = nn.Linear(error_dim, action_dim, bias=False)

    def phase_features(self, step: torch.Tensor) -> torch.Tensor:
        step = step.to(dtype=self.feedforward.weight.dtype).reshape(-1, 1)
        n = float(self.horizon_steps)
        pieces = [torch.ones_like(step)]
        # For even N, the real DFT basis contains cos/sin pairs through
        # N/2-1 and one Nyquist cosine, totaling exactly N features.
        for frequency in range(1, self.maximum_frequency):
            angle = (2.0 * math.pi * frequency / n) * step
            pieces.extend((torch.cos(angle), torch.sin(angle)))
        pieces.append(torch.cos(math.pi * step))
        return torch.cat(pieces, dim=-1)

    def forward(self, step: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        return self.feedforward(self.phase_features(step)) + self.feedback(error)

