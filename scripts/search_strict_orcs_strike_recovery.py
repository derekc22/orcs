#!/usr/bin/env python3
"""Search strict post-strike recovery targets in the exact ORCS environment.

The existing physical-reference controls plus their early right-leg strike
residual already move the box to the goal, but their later open-loop execution
is not robot-valid in MJWarp.  This search preserves that short strike and
smoothly blends all joints toward the known-valid start-0 recovery posture,
while optimizing only the 15 leg/waist coordinates for each start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from search_strict_orcs_expert import _evaluate, _make_env, _rank


def _paths(
    strike: np.ndarray,
    seed_hold: np.ndarray,
    parameters: np.ndarray,
    joint_ids: np.ndarray,
    switch_frame: int,
    ramp_frames: int,
) -> np.ndarray:
    paths = np.repeat(strike[None], len(parameters), axis=0)
    holds = np.repeat(seed_hold[None], len(parameters), axis=0)
    holds[:, joint_ids] += parameters
    for frame in range(switch_frame, len(strike)):
        phase = np.clip((frame - switch_frame + 1) / ramp_frames, 0.0, 1.0)
        phase = phase * phase * (3.0 - 2.0 * phase)
        paths[:, frame] = (1.0 - phase) * strike[frame] + phase * holds
    return paths.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, choices=(10, 20))
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=18)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--switch-frame", type=int, default=20)
    parser.add_argument("--ramp-frames", type=int, default=24)
    parser.add_argument("--bound", type=float, default=0.55)
    parser.add_argument("--initial-std", type=float, default=0.18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--strike-dataset", type=Path, default=Path("logs/realized_expert_dataset_broad")
    )
    parser.add_argument(
        "--stable-target",
        type=Path,
        default=Path("logs/strict_orcs_expert_impulse/expert_target_start_000.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/strict_orcs_strike_recovery")
    )
    args = parser.parse_args()

    from orcs.tasks.uolm.realized_reference import load_realized_reference

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    artifact = load_realized_reference(reference)
    residual = np.load(
        args.strike_dataset / f"expert_start_{args.start:03d}.npz"
    )["residual_target_rad"]
    strike = artifact.control.copy()
    strike[: len(residual)] += residual
    stable = np.load(args.stable_target)["target_rad"][-1].astype(np.float64)

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
    rng = np.random.default_rng(7300 + args.start)
    mean = np.zeros(len(joint_ids), dtype=np.float64)
    std = np.full(len(joint_ids), args.initial_std, dtype=np.float64)
    best = mean.copy()
    best_score = float("inf")
    history: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, len(joint_ids))),
            -args.bound,
            args.bound,
        )
        population[0] = mean
        population[1] = best
        population[2] = 0.0
        targets = _paths(
            strike,
            stable,
            population,
            joint_ids,
            args.switch_frame,
            args.ramp_frames,
        )
        metrics = _evaluate(env, targets)
        score = _rank(metrics, 5.0 * np.square(population).mean(axis=1))
        order = np.argsort(score)
        winner = int(order[0])
        if score[winner] < best_score:
            best_score = float(score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(0.015, np.sqrt(0.2 * std**2 + 0.8 * elite.var(axis=0)))
        row: dict[str, object] = {
            "generation": generation,
            "first_invalid_frame": int(metrics["first_invalid"][winner]),
            "invalid_reason": str(metrics["invalid_reason"][winner]),
            "full_valid": bool(metrics["alive"][winner]),
            "position_error_m": float(metrics["position_error"][winner]),
            "orientation_error_rad": float(metrics["orientation_error"][winner]),
            "minimum_root_height_m": float(metrics["minimum_root_height"][winner]),
            "maximum_torso_tilt_deg": float(metrics["maximum_torso_tilt"][winner]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["full_valid"] and row["position_error_m"] < 0.15 and row["orientation_error_rad"] < 0.35:
            break

    target = _paths(
        strike,
        stable,
        best[None],
        joint_ids,
        args.switch_frame,
        args.ramp_frames,
    )[0]
    final = _evaluate(env, np.repeat(target[None], args.population, axis=0))
    report = {
        "start": args.start,
        "switch_frame": args.switch_frame,
        "ramp_frames": args.ramp_frames,
        "dimension": len(joint_ids),
        "history": history,
        "validation": {
            "full_valid": bool(final["alive"][0]),
            "first_invalid_frame": int(final["first_invalid"][0]),
            "invalid_reason": str(final["invalid_reason"][0]),
            "position_error_m": float(final["position_error"][0]),
            "orientation_error_rad": float(final["orientation_error"][0]),
            "minimum_root_height_m": float(final["minimum_root_height"][0]),
            "maximum_torso_tilt_deg": float(final["maximum_torso_tilt"][0]),
        },
    }
    np.savez_compressed(
        args.output / f"expert_target_start_{args.start:03d}.npz",
        target_rad=target,
        parameters=best,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[i] for i in joint_ids]),
    )
    (args.output / f"expert_target_start_{args.start:03d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["validation"]), flush=True)
    env.close()


if __name__ == "__main__":
    main()
