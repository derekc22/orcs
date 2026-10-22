#!/usr/bin/env python3
"""Optimize a dynamically stable lower-body hold for the flat-hand pregrasp."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import WholeBodyHandIK, build

TOKENS = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=72)
    parser.add_argument("--generations", type=int, default=28)
    parser.add_argument("--robot-clearance", type=float, default=0.60)
    parser.add_argument("--ik-key", default="distance_0.270")
    parser.add_argument("--horizon-frames", type=int, default=120)
    parser.add_argument(
        "--temporal-knots",
        type=int,
        default=1,
        help="Number of smooth full-horizon balance residual knots.",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Existing stance artifact whose balance parameters seed CEM.",
    )
    args = parser.parse_args()

    engine, generator = build(args.robot_clearance)
    model = engine.model
    ik = WholeBodyHandIK(engine, generator)
    values = np.load(args.ik)[args.ik_key]
    ik.set_state(values)
    initial_qpos = ik.data.qpos.copy()
    initial_qvel = np.zeros(model.nv, dtype=np.float64)
    joint_position = initial_qpos[engine.actuator_qpos].copy()
    actuator_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    selected = np.asarray(
        [
            int(np.flatnonzero([token in name for name in actuator_names])[0])
            for token in TOKENS
        ]
    )
    initial_root = initial_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    initial_rotation = ik.data.xmat[generator.root_body].reshape(3, 3).copy()
    horizon = args.horizon_frames * engine.config.substeps
    object_geoms = set(engine.object_geoms)
    hand_geoms = set(engine.hand_geoms)

    knot_frames = np.linspace(0.0, float(args.horizon_frames - 1), args.temporal_knots)

    def residual_at(parameters: np.ndarray, frame: float) -> np.ndarray:
        knots = parameters.reshape(args.temporal_knots, len(selected))
        if args.temporal_knots == 1:
            return knots[0]
        if frame <= knot_frames[0]:
            return knots[0]
        if frame >= knot_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(knot_frames, frame, side="right") - 1)
        amount = float((frame - knot_frames[segment]) / (knot_frames[segment + 1] - knot_frames[segment]))
        amount = amount * amount * (3.0 - 2.0 * amount)
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def evaluate(parameters: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        target = joint_position.copy()
        target[selected] += residual_at(parameters, 0.0)
        target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = target
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        cost = 0.0
        nonhand_object_impulse = 0.0
        trace = []
        for step in range(horizon):
            frame = step / engine.config.substeps
            target = joint_position.copy()
            target[selected] += residual_at(parameters, frame)
            target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                pair = {first, second}
                if not pair & object_geoms:
                    continue
                other = second if first in object_geoms else first
                if (
                    int(model.geom_bodyid[other]) in engine.robot_bodies
                    and other not in hand_geoms
                ):
                    force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    nonhand_object_impulse += (
                        float(np.linalg.norm(force[:3])) * model.opt.timestep
                    )
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            rotation = data.xmat[generator.root_body].reshape(3, 3)
            rotation_error = np.arccos(
                np.clip(
                    (np.trace(rotation @ initial_rotation.T) - 1.0) / 2.0, -1.0, 1.0
                )
            )
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            support = int(np.sum(physical["foot_contact"]))
            torso = data.xmat[engine.torso_body].reshape(3, 3)
            torso_tilt = np.arccos(np.clip(torso[2, 2], -1.0, 1.0))
            cost += (
                6000.0 * (root[2] - initial_root[2]) ** 2
                + 10000.0 * float(np.square(root[:2] - initial_root[:2]).sum())
                + 1200.0 * rotation_error**2
                + 1200.0 * torso_tilt**2
                + 80.0 * float(np.square(velocity).sum())
                + 3000.0 * float(support == 0)
                + 60.0 * float(support == 1)
                + 5.0 * float(np.square(physical["foot_slip"]).sum())
                + 5000.0 * float(bool(physical["forbidden"]))
                + 40.0 * nonhand_object_impulse
                + 2.0 * float(np.square(parameters).mean())
            )
            invalid = generator._invalid_reason(data, physical)
            # The source object starts above its settled height and can brush a
            # nearby leg while dropping.  Record that impulse, but reserve the
            # hard rejection for material body propulsion (kick-scale impulses
            # are orders of magnitude larger than this threshold).
            if invalid is None and nonhand_object_impulse > 10.0:
                invalid = "non_hand_object_contact"
            if invalid is not None and first_invalid is None:
                first_invalid = step
                reason = invalid
            if record:
                trace.append(
                    np.r_[
                        data.qpos.copy(),
                        data.qvel.copy(),
                        data.ctrl.copy(),
                        data.actuator_force.copy(),
                        physical["foot_contact"],
                    ]
                )
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        cost /= max(step + 1, 1)
        rank = (0, cost) if first_invalid is None else (1, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": (
                None
                if first_invalid is None
                else first_invalid / engine.config.substeps
            ),
            "invalid_reason": reason,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(
                np.rad2deg(generator._physical(data)["tilt"])
            ),
            "final_foot_contact": generator._physical(data)["foot_contact"].tolist(),
            "maximum_actuator_usage": float(
                np.max(np.abs(data.actuator_force) / engine.force_scale)
            ),
            "nonhand_object_impulse_ns": float(nonhand_object_impulse),
            "cost": float(cost),
        }
        return rank, metrics, np.asarray(trace), target

    dimensions = args.temporal_knots * len(selected)
    lower = np.full(dimensions, -0.65)
    upper = np.full(dimensions, 0.65)
    mean = np.zeros(dimensions)
    if args.warm_start is not None:
        warm = np.load(args.warm_start)["parameters"].astype(float)
        if warm.shape == (len(selected),) and args.temporal_knots > 1:
            warm = np.tile(warm, args.temporal_knots)
        if warm.shape != mean.shape:
            raise ValueError(f"warm parameters {warm.shape} do not match {mean.shape}")
        mean = np.clip(warm, lower, upper)
    std = np.full(dimensions, 0.18)
    rng = np.random.default_rng(8917)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimensions)), lower, upper
        )
        population[0] = best_parameters
        if len(population) > 1:
            population[1] = 0.0
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:10]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace, target = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        target=target,
        trace=trace,
        selected_actuators=selected,
    )
    report = {
        "method": "MuJoCo CEM stable pregrasp hold",
        "balance_joints": actuator_names[selected].tolist(),
        "dimensions": dimensions,
        "temporal_knots": args.temporal_knots,
        "knot_frames": knot_frames.tolist(),
        "population": args.population,
        "generations": args.generations,
        "evaluations": 1 + args.population * args.generations,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "parameters": best_parameters.tolist(),
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
