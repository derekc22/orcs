#!/usr/bin/env python3
"""Tune the one-contact right-foot impulse atop a valid ORCS recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from search_strict_orcs_expert import _evaluate, _make_env, _rank


def _paths(base: np.ndarray, parameters: np.ndarray, joint_ids: np.ndarray) -> np.ndarray:
    result = np.repeat(base[None], len(parameters), axis=0)
    # The right ankle contacts the box only during the first 20 ms control
    # interval.  Concentrating the search here avoids diluting six meaningful
    # variables among later knots that cannot alter the settled object pose.
    if parameters.shape[1] == len(joint_ids):
        result[:, 0, joint_ids] += parameters
        result[:, 1, joint_ids] += 0.5 * parameters
    elif parameters.shape[1] == 2 * len(joint_ids):
        frame_parameters = parameters.reshape(len(parameters), 2, len(joint_ids))
        result[:, 0, joint_ids] += frame_parameters[:, 0]
        result[:, 1, joint_ids] += frame_parameters[:, 1]
    else:
        raise ValueError("impulse parameters must contain one or two target frames")
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elites", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--search-frames", type=int, default=100)
    parser.add_argument("--validate-top", type=int, default=None)
    parser.add_argument("--bound", type=float, default=2.0)
    parser.add_argument("--initial-std", type=float, default=0.6)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--independent-two-frame", action="store_true")
    parser.add_argument("--base-target", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input", type=Path, default=Path("logs/strict_orcs_expert_search")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/strict_orcs_expert_impulse")
    )
    args = parser.parse_args()
    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    base_path = (
        args.base_target
        if args.base_target is not None
        else args.input / f"search_start_{args.start:03d}.npz"
    )
    base = np.load(base_path)["target_rad"]
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
    rng = np.random.default_rng(6200 + args.start)
    dimension = len(joint_ids) * (2 if args.independent_two_frame else 1)
    mean = np.zeros(dimension)
    if args.initial is not None:
        seed = np.load(args.initial)["impulse_parameters"].astype(np.float64)
        if args.independent_two_frame and len(seed) == len(joint_ids):
            seed = np.concatenate([seed, 0.5 * seed])
        mean = seed
    std = np.full(dimension, args.initial_std)
    best = mean.copy()
    best_score = float("inf")
    history = []
    archive_parameters: list[np.ndarray] = []
    archive_scores: list[np.ndarray] = []
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / f"search_progress_start_{args.start:03d}.json"
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimension)),
            -args.bound,
            args.bound,
        )
        population[0] = mean
        population[1] = best
        population[2] = 0.0
        # These six parameters affect only frames 0--1.  Search through the
        # object-settling/initial-recovery interval, then strictly validate an
        # archive over all 300 frames.  This preserves the success contract
        # without spending 3x the simulation work on every exploratory draw.
        metrics = _evaluate(
            env, _paths(base, population, joint_ids)[:, : args.search_frames]
        )
        score = _rank(metrics, 0.05 * np.square(population).mean(axis=1))
        archive_parameters.append(population.copy())
        archive_scores.append(score.copy())
        order = np.argsort(score)
        winner = int(order[0])
        if score[winner] < best_score:
            best_score = float(score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.15 * mean + 0.85 * elite.mean(axis=0)
        std = np.maximum(0.01, np.sqrt(0.15 * std**2 + 0.85 * elite.var(axis=0)))
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
            "parameters": population[winner].tolist(),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        progress_path.write_text(json.dumps(history, indent=2) + "\n")

    archived = np.concatenate(archive_parameters, axis=0)
    archived_score = np.concatenate(archive_scores, axis=0)
    validate_top = args.validate_top or args.population
    if validate_top != args.population:
        raise ValueError("--validate-top must equal --population for the vectorized env")
    shortlisted = archived[np.argsort(archived_score)[:validate_top]]
    full_metrics = _evaluate(env, _paths(base, shortlisted, joint_ids))
    full_score = _rank(
        full_metrics, 0.05 * np.square(shortlisted).mean(axis=1)
    )
    full_order = np.argsort(full_score)
    best = shortlisted[int(full_order[0])].copy()
    target = _paths(base, best[None], joint_ids)[0]
    validation = {
        "full_valid": bool(full_metrics["alive"][full_order[0]]),
        "first_invalid_frame": int(full_metrics["first_invalid"][full_order[0]]),
        "invalid_reason": str(full_metrics["invalid_reason"][full_order[0]]),
        "position_error_m": float(full_metrics["position_error"][full_order[0]]),
        "orientation_error_rad": float(
            full_metrics["orientation_error"][full_order[0]]
        ),
        "minimum_root_height_m": float(
            full_metrics["minimum_root_height"][full_order[0]]
        ),
        "maximum_torso_tilt_deg": float(
            full_metrics["maximum_torso_tilt"][full_order[0]]
        ),
    }
    print(json.dumps({"full_validation": validation}), flush=True)
    np.savez_compressed(
        args.output / f"expert_target_start_{args.start:03d}.npz",
        target_rad=target,
        impulse_parameters=best,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[index] for index in joint_ids]),
    )
    (args.output / f"expert_target_start_{args.start:03d}.json").write_text(
        json.dumps(
            {
                "start": args.start,
                "dimension": dimension,
                "search_frames": args.search_frames,
                "history": history,
                "full_validation": validation,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
