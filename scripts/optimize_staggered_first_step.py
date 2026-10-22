#!/usr/bin/env python3
"""Optimize one legal forward step into a stopped staggered support stance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from probe_flat_hand_grasp import build, smoothstep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elites", type=int, default=8)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    base = source["base_target"].copy()
    balance = source["balance_actuators"].astype(int)
    poses = source["pose_controls"].copy()
    substeps = engine.config.substeps
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    nominal_frames = np.asarray([90.0, 110.0, 140.0, 155.0, 175.0])
    nominal_poses = np.vstack([base[balance], poses[:4]])
    residual_frames = np.asarray([0.0, 25.0, 50.0, 85.0, 135.0])
    dimensions = len(residual_frames) * len(balance)

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    root0 = anchor.qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    feet0 = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    object_position = anchor.qpos[engine.object_qpos : engine.object_qpos + 3]
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_position[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])

    def interpolate(values: np.ndarray, frames: np.ndarray, frame: float) -> np.ndarray:
        if frame <= frames[0]:
            return values[0]
        if frame >= frames[-1]:
            return values[-1]
        segment = int(np.searchsorted(frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - frames[segment]) / (frames[segment + 1] - frames[segment])
        )
        return (1.0 - amount) * values[segment] + amount * values[segment + 1]

    def evaluate(parameters: np.ndarray, record: bool = False):
        residuals = parameters.reshape(len(residual_frames), len(balance))
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        invalid_reason = None
        nonhand = 0.0
        longest_ready = 0
        ready_run = 0
        cost = 0.0
        ctrl_physics = []
        trace = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "actuator_force",
                "foot_contact",
                "root_height",
                "torso_tilt",
                "nonhand_impulse",
                "foot_position",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            held = base.copy()
            if frame >= nominal_frames[0]:
                nominal = interpolate(nominal_poses, nominal_frames, frame)
                residual = interpolate(
                    residuals, residual_frames, frame - nominal_frames[0]
                )
                held[balance] = nominal + residual
            held = np.clip(
                held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            data.ctrl[:] = held
            ctrl_physics.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if (
                    int(model.geom_bodyid[other]) in engine.robot_bodies
                    and other not in hand_geoms
                ):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    nonhand += float(np.linalg.norm(force[:3])) * model.opt.timestep
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            foot_progress = (feet - feet0) @ forward
            root_progress = float(np.dot(root - root0, forward))
            target_progress = 0.08 * smoothstep((frame - 100.0) / 75.0)
            if frame >= 90.0:
                cost += (
                    2400.0 * (root_progress - target_progress) ** 2
                    + 900.0 * float(np.square(velocity[:3]).sum())
                    + 300.0 * float(np.square(velocity[3:]).sum())
                    + 1800.0 * physical["tilt"] ** 2
                    + 900.0 * max(0.0, 0.57 - float(root[2])) ** 2
                    + 300.0 * float(np.sum(physical["foot_contact"]) == 0)
                )
            stopped_stagger = bool(
                frame >= 175.0
                and root_progress > 0.045
                and max(foot_progress) > 0.075
                and root[2] > 0.56
                and np.rad2deg(physical["tilt"]) < 32.0
                and np.linalg.norm(velocity[:3]) < 0.28
                and int(np.sum(physical["foot_contact"])) == 2
                and nonhand <= 0.10
            )
            if step % substeps == substeps - 1:
                ready_run = ready_run + 1 if stopped_stagger else 0
                longest_ready = max(longest_ready, ready_run)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["root_height"].append(float(root[2]))
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["nonhand_impulse"].append(nonhand)
                    trace["foot_position"].append(feet.copy())
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if invalid is not None:
                first_invalid = step
                invalid_reason = invalid
                break
        physical = generator._physical(data)
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        foot_progress = (feet - feet0) @ forward
        root_progress = float(np.dot(root - root0, forward))
        lateral_drift = float(abs(np.dot(root - root0, lateral)))
        valid = first_invalid is None
        cost /= max(step - 90 * substeps + 1, 1)
        if valid and longest_ready >= 30:
            rank = (0, -longest_ready, cost)
        elif valid:
            rank = (1, -root_progress, -float(max(foot_progress)), cost)
        elif longest_ready >= 10:
            rank = (2, -longest_ready, -step, cost)
        else:
            rank = (3, -step, -root_progress, cost)
        metrics = {
            "full_horizon_valid": valid,
            "first_invalid_frame": None if valid else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "longest_stopped_stagger_frames": longest_ready,
            "goalward_root_translation_m": root_progress,
            "goalward_foot_translation_m": foot_progress.tolist(),
            "lateral_root_drift_m": lateral_drift,
            "final_root_velocity_m_s": velocity[:3].tolist(),
            "final_root_height_m": float(root[2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "nonhand_object_impulse_ns": nonhand,
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    rng = np.random.default_rng(20771)
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.065)
    lower = np.full(dimensions, -0.24)
    upper = np.full(dimensions, 0.24)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, (args.population, dimensions)), lower, upper
        )
        population[0] = best_parameters
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[: args.elites]]
        mean = 0.72 * elite.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        np.savez_compressed(
            args.output.with_suffix(".checkpoint.npz"),
            parameters=best_parameters,
            mean=mean,
            std=std,
        )
        print("GEN", generation, json.dumps(best[1]), flush=True)

    _, metrics, trace, ctrl_physics = evaluate(best_parameters, record=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        balance_actuators=balance,
        nominal_frames=nominal_frames,
        nominal_poses=nominal_poses,
        residual_frames=residual_frames,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "checkpointed legal first-step staggered stance shooting",
        "population": args.population,
        "generations": args.generations,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
