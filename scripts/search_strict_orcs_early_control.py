#!/usr/bin/env python3
"""Find a short ORCS-native recovery prefix before a known stable target.

The three realized references share the same robot pose but have different
initial root/object momentum.  This search keeps a previously validated stable
300-frame target as the tail and changes only a compact early lower-body/waist
prefix.  Every candidate is ranked by uninterrupted ORCS/MJWarp survival.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from search_strict_orcs_expert import _evaluate, _make_env, _rank

def _paths(
    base: np.ndarray,
    parameters: np.ndarray,
    joint_ids: np.ndarray,
    knot_frames: np.ndarray,
    end_frame: int,
) -> np.ndarray:
    count = len(parameters)
    knots = parameters.reshape(count, len(knot_frames), len(joint_ids))
    result = np.repeat(base[None], count, axis=0)
    frame_grid = np.arange(int(knot_frames[0]), end_frame + 1)
    for candidate in range(count):
        for local_joint, joint_id in enumerate(joint_ids):
            correction = np.interp(
                frame_grid,
                np.append(knot_frames, end_frame),
                np.append(knots[candidate, :, local_joint], 0.0),
            )
            result[candidate, frame_grid, joint_id] += correction
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--base-target", type=Path, required=True)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--initial-std", type=float, default=0.30)
    parser.add_argument("--knot-frames", default="0,2,5,9,14")
    parser.add_argument("--end-frame", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("logs/strict_orcs_early_control")
    )
    args = parser.parse_args()
    knot_frames = np.asarray(
        [int(item) for item in args.knot_frames.split(",")], dtype=np.int64
    )
    if len(knot_frames) == 0 or np.any(np.diff(knot_frames) <= 0):
        raise ValueError("--knot-frames must be a non-empty increasing sequence")
    if args.end_frame <= int(knot_frames[-1]):
        raise ValueError("--end-frame must be greater than the final free knot")

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/"
        "physical_reference.npz"
    ).resolve()
    base = np.load(args.base_target)["target_rad"].astype(np.float32)
    env = _make_env(args.start, args.population, reference, args.device)
    names = list(env.scene["robot"].joint_names)
    joint_ids = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if any(token in name for token in ("hip_", "knee", "ankle_", "waist_"))
        ],
        dtype=np.int64,
    )
    dimension = len(knot_frames) * len(joint_ids)
    rng = np.random.default_rng(8700 + args.start)
    mean = np.zeros(dimension, dtype=np.float64)
    std = np.full(dimension, args.initial_std, dtype=np.float64)
    best = mean.copy()
    best_score = float("inf")
    history: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / f"search_start_{args.start:03d}.json"

    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimension)),
            -args.bound,
            args.bound,
        )
        population[0] = mean
        population[1] = best
        population[2] = 0.0
        metrics = _evaluate(
            env,
            _paths(base, population, joint_ids, knot_frames, args.end_frame),
        )
        regularization = 2.0 * np.square(population).mean(axis=1)
        score = _rank(metrics, regularization)
        order = np.argsort(score)
        winner = int(order[0])
        if score[winner] < best_score:
            best_score = float(score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.20 * mean + 0.80 * elite.mean(axis=0)
        std = np.maximum(
            0.015, np.sqrt(0.20 * std**2 + 0.80 * elite.var(axis=0))
        )
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
        progress_path.write_text(json.dumps({"history": history}, indent=2) + "\n")
        if bool(metrics["alive"][winner]) and generation >= 4:
            break

    target = _paths(base, best[None], joint_ids, knot_frames, args.end_frame)[0]
    validation = _evaluate(env, np.repeat(target[None], args.population, axis=0))
    winner = 0
    report = {
        "start": args.start,
        "dimension": dimension,
        "knot_frames": knot_frames.tolist(),
        "end_frame": args.end_frame,
        "history": history,
        "full_validation": {
            "full_valid": bool(validation["alive"][winner]),
            "first_invalid_frame": int(validation["first_invalid"][winner]),
            "invalid_reason": str(validation["invalid_reason"][winner]),
            "position_error_m": float(validation["position_error"][winner]),
            "orientation_error_rad": float(validation["orientation_error"][winner]),
            "minimum_root_height_m": float(validation["minimum_root_height"][winner]),
            "maximum_torso_tilt_deg": float(validation["maximum_torso_tilt"][winner]),
        },
    }
    np.savez_compressed(
        args.output / f"target_start_{args.start:03d}.npz",
        target_rad=target,
        parameters=best,
        knot_frames=knot_frames,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[index] for index in joint_ids]),
    )
    progress_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"full_validation": report["full_validation"]}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
