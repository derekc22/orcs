#!/usr/bin/env python3
"""Tune a two-frame strike while a fixed robust feedback recovery executes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from search_strict_orcs_expert import _make_env, _rank
from search_strict_orcs_feedback_expert import _evaluate_feedback


def _targets(
    base: np.ndarray, population: np.ndarray, joint_ids: np.ndarray
) -> np.ndarray:
    targets = np.repeat(base[None], len(population), axis=0)
    residual = population.reshape(len(population), 2, len(joint_ids))
    targets[:, :2, joint_ids] += residual
    return targets.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--base-target", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--feedback-start", type=int, default=40)
    parser.add_argument("--root-feedback-start", type=int, default=40)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--search-frames", type=int, default=300)
    parser.add_argument("--bound", type=float, default=0.35)
    parser.add_argument("--initial-std", type=float, default=0.08)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    reference = Path(
        f"logs/unified_physical_reference/final_start_{args.start:03d}/physical_reference.npz"
    ).resolve()
    base = np.load(args.base_target)["target_rad"].astype(np.float32)
    feedback = np.load(args.feedback)["parameters"].astype(np.float32)
    env = _make_env(
        args.start,
        args.population * args.replicas,
        reference,
        args.device,
        task="Orcs-Uolm-Realized-Strict-TaRa",
    )
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
    dimension = 2 * len(joint_ids)
    rng = np.random.default_rng(26000 + args.start)
    mean = np.zeros(dimension)
    if args.initial is not None:
        mean = np.load(args.initial)["impulse_parameters"].astype(np.float64)
    std = np.full(dimension, args.initial_std)
    best = mean.copy()
    best_score = float("inf")
    history: list[dict[str, object]] = []
    archive_population: list[np.ndarray] = []
    archive_scores: list[np.ndarray] = []
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, (args.population, dimension)),
            -args.bound,
            args.bound,
        )
        population[0] = mean
        population[1] = best
        population[2] = 0.0
        expanded_population = np.repeat(population, args.replicas, axis=0)
        targets = _targets(base, expanded_population, joint_ids)
        parameters = np.repeat(
            feedback[None], args.population * args.replicas, axis=0
        )
        metrics = _evaluate_feedback(
            env,
            targets[:, : args.search_frames],
            parameters,
            feedback_start=args.feedback_start,
            root_feedback_start=args.root_feedback_start,
        )
        regularization = 0.5 * np.square(population).mean(axis=1)
        scores = _rank(metrics, np.repeat(regularization, args.replicas)).reshape(
            args.population, args.replicas
        )
        robust_score = scores.max(axis=1)
        archive_population.append(population.copy())
        archive_scores.append(robust_score.copy())
        order = np.argsort(robust_score)
        winner = int(order[0])
        worst_replica = int(scores[winner].argmax())
        metric_index = winner * args.replicas + worst_replica
        if robust_score[winner] < best_score:
            best_score = float(robust_score[winner])
            best = population[winner].copy()
        elite = population[order[: args.elites]]
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(
            0.002,
            np.sqrt(0.2 * std**2 + 0.8 * elite.var(axis=0)),
        )
        row = {
            "generation": generation,
            "all_replicas_valid": bool(
                metrics["alive"]
                .reshape(args.population, args.replicas)[winner]
                .all()
            ),
            "worst_first_invalid_frame": int(
                metrics["first_invalid"][metric_index]
            ),
            "worst_position_error_m": float(
                metrics["position_error"][metric_index]
            ),
            "worst_orientation_error_rad": float(
                metrics["orientation_error"][metric_index]
            ),
            "parameters": population[winner].tolist(),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    if args.search_frames < base.shape[0]:
        archived = np.concatenate(archive_population, axis=0)
        archived_scores = np.concatenate(archive_scores, axis=0)
        shortlisted = archived[np.argsort(archived_scores)[: args.population]]
        expanded_shortlist = np.repeat(shortlisted, args.replicas, axis=0)
        full_targets = _targets(base, expanded_shortlist, joint_ids)
        full_parameters = np.repeat(
            feedback[None], args.population * args.replicas, axis=0
        )
        full_metrics = _evaluate_feedback(
            env,
            full_targets,
            full_parameters,
            feedback_start=args.feedback_start,
            root_feedback_start=args.root_feedback_start,
        )
        full_regularization = 0.5 * np.square(shortlisted).mean(axis=1)
        full_scores = _rank(
            full_metrics, np.repeat(full_regularization, args.replicas)
        ).reshape(args.population, args.replicas)
        best = shortlisted[int(np.argmin(full_scores.max(axis=1)))].copy()
        winner = int(np.argmin(full_scores.max(axis=1)))
        worst = int(full_scores[winner].argmax())
        index = winner * args.replicas + worst
        print(
            json.dumps(
                {
                    "full_validation": {
                        "all_replicas_valid": bool(
                            full_metrics["alive"]
                            .reshape(args.population, args.replicas)[winner]
                            .all()
                        ),
                        "worst_first_invalid_frame": int(
                            full_metrics["first_invalid"][index]
                        ),
                        "worst_position_error_m": float(
                            full_metrics["position_error"][index]
                        ),
                        "worst_orientation_error_rad": float(
                            full_metrics["orientation_error"][index]
                        ),
                    }
                }
            ),
            flush=True,
        )
    target = _targets(base, best[None], joint_ids)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target_rad=target,
        parameters=feedback,
        impulse_parameters=best,
        joint_ids=joint_ids,
        joint_names=np.asarray([names[index] for index in joint_ids]),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "start": args.start,
                "dimension": dimension,
                "replicas": args.replicas,
                "feedback_start": args.feedback_start,
                "search_frames": args.search_frames,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
