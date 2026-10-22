#!/usr/bin/env python3
"""Optimize the short object strike on top of a robot-valid ORCS recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from search_strict_orcs_expert import _evaluate, _make_env, _rank


def _strike_paths(
    base: np.ndarray,
    parameters: np.ndarray,
    joint_ids: np.ndarray,
) -> np.ndarray:
    knot_frames = np.asarray([0, 3, 7, 12, 20], dtype=np.float64)
    values = np.concatenate(
        [
            parameters.reshape(len(parameters), 4, len(joint_ids)),
            np.zeros((len(parameters), 1, len(joint_ids))),
        ],
        axis=1,
    )
    result = np.repeat(base[None], len(parameters), axis=0)
    frames = np.arange(len(base), dtype=np.float64)
    for candidate in range(len(parameters)):
        for local, joint_id in enumerate(joint_ids):
            result[candidate, :, joint_id] += np.interp(
                frames,
                knot_frames,
                values[candidate, :, local],
                left=values[candidate, 0, local],
                right=0.0,
            )
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=18)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input", type=Path, default=Path("logs/strict_orcs_expert_search")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/strict_orcs_expert_strike")
    )
    args = parser.parse_args()

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    base = np.load(args.input / f"search_start_{args.start:03d}.npz")["target_rad"]
    env = _make_env(args.start, args.population, reference, args.device)
    names = list(env.scene["robot"].joint_names)
    joint_ids = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name.startswith("right_")
            and any(token in name for token in ("hip_", "knee", "ankle_"))
        ],
        dtype=np.int64,
    )
    dimension = 4 * len(joint_ids)
    rng = np.random.default_rng(5200 + args.start)
    mean = np.zeros(dimension)
    std = np.full(dimension, 0.25)
    best = mean.copy()
    best_score = float("inf")
    history = []
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimension)),
            -args.bound,
            args.bound,
        )
        population[0] = mean
        population[1] = best
        population[2] = 0.0
        paths = _strike_paths(base, population, joint_ids)
        metrics = _evaluate(env, paths)
        # Keep strike residuals compact and avoid high-frequency knot changes.
        knots = population.reshape(args.population, 4, len(joint_ids))
        regularization = (
            1.0 * np.square(population).mean(axis=1)
            + 2.0 * np.square(np.diff(knots, axis=1)).mean(axis=(1, 2))
        )
        score = _rank(metrics, regularization)
        order = np.argsort(score)
        winner = int(order[0])
        if score[winner] < best_score:
            best_score = float(score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(0.015, np.sqrt(0.2 * std**2 + 0.8 * elite.var(axis=0)))
        row = {
            "generation": generation,
            "score": float(score[winner]),
            "full_valid": bool(metrics["alive"][winner]),
            "first_invalid_frame": int(metrics["first_invalid"][winner]),
            "invalid_reason": str(metrics["invalid_reason"][winner]),
            "position_error_m": float(metrics["position_error"][winner]),
            "orientation_error_rad": float(metrics["orientation_error"][winner]),
            "minimum_root_height_m": float(metrics["minimum_root_height"][winner]),
            "maximum_torso_tilt_deg": float(metrics["maximum_torso_tilt"][winner]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    target = _strike_paths(base, best[None], joint_ids)[0]
    np.savez_compressed(
        args.output / f"expert_target_start_{args.start:03d}.npz",
        target_rad=target,
        strike_parameters=best,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[index] for index in joint_ids]),
    )
    (args.output / f"expert_target_start_{args.start:03d}.json").write_text(
        json.dumps(
            {
                "start": args.start,
                "dimension": dimension,
                "knot_frames": [0, 3, 7, 12, 20],
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
