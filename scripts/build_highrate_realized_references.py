#!/usr/bin/env python3
"""Derive immutable 200 Hz ORCS references from validated MuJoCo traces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs" / "unified_physical_reference"
OUTPUT = ROOT / "logs" / "unified_physical_reference_200hz"
HORIZON = 1200
NQ = 43
NV = 41
NU = 29


def _pad(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) >= length:
        return array[:length].copy()
    return np.concatenate(
        [array, np.repeat(array[-1:], length - len(array), axis=0)], axis=0
    )


def main() -> None:
    for start in (0, 10, 20):
        source_dir = SOURCE / f"final_start_{start:03d}"
        destination = OUTPUT / f"final_start_{start:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        with np.load(source_dir / "physical_reference.npz") as data:
            trace = data["simulation_trace"]
            # A trace row is the state *after* the row's control was applied.
            # The reference state at action sample zero is therefore the saved
            # initial state followed by all but the final post-step state.
            qpos = _pad(
                np.concatenate([data["initial_qpos"][None], trace[:, :NQ]]),
                HORIZON,
            )
            qvel = _pad(
                np.concatenate(
                    [data["initial_qvel"][None], trace[:, NQ : NQ + NV]]
                ),
                HORIZON,
            )
            control = _pad(trace[:, NQ + NV : NQ + NV + NU], HORIZON)
            prior = np.repeat(data["diffusion_prior"], 4, axis=1)[:, :HORIZON]
            np.savez_compressed(
                destination / "physical_reference.npz",
                initial_qpos=data["initial_qpos"],
                initial_qvel=data["initial_qvel"],
                qpos=qpos,
                qvel=qvel,
                ctrl=control,
                simulation_trace=trace,
                diffusion_prior=prior,
                goal_position=data["goal_position"],
                goal_quaternion=data["goal_quaternion"],
            )
        validation = json.loads((source_dir / "validation.json").read_text())
        validation.update(
            {
                "derived_control_rate_hz": 200,
                "derived_from": str(source_dir / "physical_reference.npz"),
                "post_reset_object_pose_writes": 0,
            }
        )
        (destination / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
