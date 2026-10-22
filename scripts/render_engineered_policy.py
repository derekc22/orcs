#!/usr/bin/env python3
"""Render the frozen learned hierarchical policy rollout with audit overlays."""

from pathlib import Path

import render_engineered_locomomanipulation as renderer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_policy"
renderer.MODEL_PATH = ROOT / "logs/engineering_locomomanipulation/engineered_locomomanipulation_expert_start0.mjb"
renderer.EXPERT_PATH = OUT / "engineered_hierarchical_policy_success.npz"
renderer.VIDEO_PATH = OUT / "engineered_hierarchical_policy_success.mp4"
renderer.MONTAGE_PATH = OUT / "engineered_hierarchical_policy_success_montage.png"
renderer.DENSE_PATH = OUT / "engineered_hierarchical_policy_success_dense_sheet.jpg"


if __name__ == "__main__":
    renderer.main()
