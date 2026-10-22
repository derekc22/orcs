#!/usr/bin/env python3
"""Find a legal stepped, recoverable start-0 stance near the large box.

This stage deliberately contains no hand/object manipulation.  It searches a
small time-varying sagittal lower-body control family, starting from the actual
task state, for a closer stance that is upright, supported, stopped, clear of
the object, and valid through frame 299.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from probe_flat_hand_grasp import build, smoothstep


TOKENS = (
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "waist_pitch",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=14)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19017)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.stance)
    base_target = source["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    selected = np.asarray(
        [int(np.flatnonzero([token in name for name in names])[0]) for token in TOKENS]
    )
    knot_frames = np.asarray([0.0, 45.0, 90.0, 145.0, 215.0])
    dimensions = len(knot_frames) * len(selected)
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    object_initial = initial_qpos[engine.object_qpos : engine.object_qpos + 3]
    forward = np.r_[engine.reference_object_pos[-1, :2] - object_initial[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    desired_root = np.r_[object_initial[:2] - 0.36 * forward[:2], 0.63]

    def interpolate(parameters: np.ndarray, frame: float) -> np.ndarray:
        knots = parameters.reshape(len(knot_frames), len(selected))
        if frame <= knot_frames[0]:
            return knots[0]
        if frame >= knot_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(knot_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - knot_frames[segment])
            / (knot_frames[segment + 1] - knot_frames[segment])
        )
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def evaluate(parameters: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        invalid_reason = None
        nonhand_impulse = 0.0
        ready_frames = 0
        longest_ready = 0
        running_ready = 0
        cost = 0.0
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
            )
        }
        ctrl_physics = []
        held = base_target.copy()
        for step in range(300 * engine.config.substeps):
            frame = step / engine.config.substeps
            if step % engine.config.substeps == 0:
                held = base_target.copy()
                held[selected] += interpolate(parameters, frame)
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
                    nonhand_impulse += float(np.linalg.norm(force[:3])) * model.opt.timestep
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            support = int(np.sum(physical["foot_contact"]))
            position_error = root - desired_root
            ready = bool(
                frame >= 120.0
                and np.linalg.norm(position_error[:2]) < 0.09
                and root[2] > 0.56
                and physical["tilt"] < np.deg2rad(35.0)
                and np.linalg.norm(velocity[:3]) < 0.35
                and support > 0
                and nonhand_impulse <= 0.10
            )
            if step % engine.config.substeps == engine.config.substeps - 1:
                ready_frames += int(ready)
                running_ready = running_ready + 1 if ready else 0
                longest_ready = max(longest_ready, running_ready)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["root_height"].append(float(root[2]))
                    trace["torso_tilt"].append(float(physical["tilt"]))
            phase = smoothstep((frame - 55.0) / 90.0)
            cost += phase * (
                8000.0 * float(np.square(position_error[:2]).sum())
                + 5000.0 * (root[2] - desired_root[2]) ** 2
                + 1000.0 * physical["tilt"] ** 2
                + 260.0 * float(np.square(velocity[:3]).sum())
                + 80.0 * float(np.square(velocity[3:]).sum())
                + 900.0 * float(support == 0)
                + 35.0 * float(support == 1)
            )
            cost += 5.0 * float(np.square(parameters).mean())
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand_impulse > 0.10:
                invalid = "non_hand_object_impulse"
            if invalid is not None and first_invalid is None:
                first_invalid = step
                invalid_reason = invalid
                break

        valid = first_invalid is None
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        physical = generator._physical(data)
        final_xy_error = float(np.linalg.norm(root[:2] - desired_root[:2]))
        cost /= max(step + 1, 1)
        survival_frame = step / engine.config.substeps
        readiness_score = (
            final_xy_error
            + 0.5 * abs(float(root[2] - desired_root[2]))
            + 0.002 * float(np.rad2deg(physical["tilt"]))
            + 0.8 * float(np.linalg.norm(velocity[:3]))
            + 0.002 * (300.0 - survival_frame)
        )
        if valid and longest_ready >= 40:
            rank = (0, -longest_ready, final_xy_error, cost)
        elif valid:
            rank = (1, final_xy_error, -longest_ready, cost)
        elif longest_ready >= 20:
            rank = (2, readiness_score, -longest_ready, -step, cost)
        elif survival_frame >= 120.0:
            rank = (3, readiness_score, -longest_ready, -step, cost)
        else:
            rank = (4, -step, readiness_score, cost)
        metrics = {
            "full_horizon_valid": valid,
            "first_invalid_frame": None if valid else first_invalid / engine.config.substeps,
            "invalid_reason": invalid_reason,
            "ready_frames": ready_frames,
            "longest_ready_frames": longest_ready,
            "desired_root_position_m": desired_root.tolist(),
            "final_root_position_m": root.tolist(),
            "final_root_xy_error_m": final_xy_error,
            "final_root_velocity_m_s": velocity[:3].tolist(),
            "final_root_height_m": float(root[2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "nonhand_object_impulse_ns": float(nonhand_impulse),
            "readiness_score": float(readiness_score),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    lower = np.full(dimensions, -0.25)
    upper = np.full(dimensions, 0.25)
    stable_residual = base_target[selected] - initial_qpos[engine.actuator_qpos[selected]]
    scale_schedule = np.asarray([0.90, 0.90, 1.20, 1.08, 1.00])
    mean = np.concatenate([(scale - 1.0) * stable_residual for scale in scale_schedule])
    std = np.full(dimensions, 0.045)
    rng = np.random.default_rng(args.seed)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimensions)), lower, upper
        )
        population[0] = best_parameters
        if generation == 0 and len(population) > 1:
            population[1] = 0.0
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elites = population[order[: args.elites]]
        mean = 0.72 * elites.mean(axis=0) + 0.28 * mean
        std = np.maximum(0.72 * elites.std(axis=0) + 0.28 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        np.savez_compressed(
            args.output.with_suffix(".checkpoint.npz"),
            parameters=best_parameters,
            mean=mean,
            std=std,
        )
        print("GEN", generation, json.dumps(best[1]), flush=True)

    _, metrics, trace, ctrl_physics = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base_target,
        selected_actuators=selected,
        knot_frames=knot_frames,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "legal start-0 time-varying stagger-capable manipulation-ready stance",
        "population": args.population,
        "generations": args.generations,
        "runtime_s": time.perf_counter() - started,
        "selected_joints": names[selected].tolist(),
        "knot_frames": knot_frames.tolist(),
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
