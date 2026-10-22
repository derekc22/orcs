#!/usr/bin/env python3
"""Learn a small bounded physics-rate upright stabilizer before manipulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from probe_flat_hand_grasp import build


TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument(
        "--rotation-reference-scale",
        type=float,
        default=0.0,
        help=(
            "Interpolate the feedback orientation reference from the settled "
            "stance (0) toward the upright task-start root rotation (1)."
        ),
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.stance)
    base = source["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(token in name for token in tokens)],
            dtype=int,
        )
        for key, tokens in TOKENS.items()
    }
    sagittal_groups = ("hip_pitch", "knee", "ankle_pitch", "waist_pitch")
    lateral_groups = ("hip_roll", "ankle_roll", "waist_roll")
    dimensions = 4 * (len(sagittal_groups) + len(lateral_groups))
    substeps = engine.config.substeps
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    desired_root = anchor.qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_quaternion = anchor.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    settled_rotation = Rotation.from_quat(
        np.r_[root_quaternion[1:], root_quaternion[0]]
    )
    initial_root_quaternion = initial_qpos[
        engine.robot_qpos + 3 : engine.robot_qpos + 7
    ]
    initial_rotation = Rotation.from_quat(
        np.r_[initial_root_quaternion[1:], initial_root_quaternion[0]]
    )
    toward_initial = (settled_rotation.inv() * initial_rotation).as_rotvec()
    desired_rotation = settled_rotation * Rotation.from_rotvec(
        args.rotation_reference_scale * toward_initial
    )

    def correction(parameters: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        rotation_error = (desired_rotation.inv() * rotation).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal_features = np.asarray(
            [root[1] - desired_root[1], velocity[1], rotation_error[0], velocity[3]]
        )
        lateral_features = np.asarray(
            [root[0] - desired_root[0], velocity[0], rotation_error[1], velocity[4]]
        )
        gains = parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in sagittal_groups:
            delta[groups[key]] += float(np.dot(gains[row], sagittal_features))
            row += 1
        for key in lateral_groups:
            delta[groups[key]] += float(np.dot(gains[row], lateral_features))
            row += 1
        return np.clip(delta, -0.28, 0.28)

    def evaluate(parameters: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        invalid_reason = None
        nonhand = 0.0
        cost = 0.0
        ctrl_physics = []
        trace = {
            key: []
            for key in (
                "qpos", "qvel", "ctrl", "actuator_force", "foot_contact",
                "root_height", "torso_tilt", "nonhand_impulse",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            held = base.copy()
            if frame >= 90.0:
                held += correction(parameters, data)
            held = np.clip(held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = held
            ctrl_physics.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                c = data.contact[contact_index]
                first, second = int(c.geom1), int(c.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) in engine.robot_bodies and other not in hand_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    nonhand += float(np.linalg.norm(force[:3])) * model.opt.timestep
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            if frame >= 90.0:
                displacement = root - desired_root
                cost += (
                    1600.0 * float(np.square(displacement[:2]).sum())
                    + 400.0 * float(np.square(velocity[:3]).sum())
                    + 80.0 * float(np.square(velocity[3:]).sum())
                    + 1000.0 * physical["tilt"] ** 2
                    + 1800.0 * max(0.0, 0.60 - float(root[2])) ** 2
                    + 80.0 * float(np.sum(physical["foot_contact"]) == 1)
                    + 800.0 * float(np.sum(physical["foot_contact"]) == 0)
                )
            if step % substeps == substeps - 1 and record:
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(data.ctrl.copy())
                trace["actuator_force"].append(data.actuator_force.copy())
                trace["foot_contact"].append(physical["foot_contact"].copy())
                trace["root_height"].append(float(root[2]))
                trace["torso_tilt"].append(float(physical["tilt"]))
                trace["nonhand_impulse"].append(nonhand)
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
        valid = first_invalid is None
        cost /= max(step - 90 * substeps + 1, 1)
        terminal_error = float(np.linalg.norm(root[:2] - desired_root[:2]))
        terminal_speed = float(np.linalg.norm(velocity[:3]))
        if valid and nonhand <= 0.10:
            rank = (0, cost, terminal_error, terminal_speed)
        else:
            rank = (1, -step, terminal_error, terminal_speed, cost)
        metrics = {
            "full_horizon_valid": valid,
            "first_invalid_frame": None if valid else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "final_root_position_m": root.tolist(),
            "final_root_xy_error_m": terminal_error,
            "final_root_velocity_m_s": velocity[:3].tolist(),
            "final_root_speed_m_s": terminal_speed,
            "final_root_height_m": float(root[2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "nonhand_object_impulse_ns": nonhand,
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    rng = np.random.default_rng(24601)
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.18)
    lower = np.full(dimensions, -2.0)
    upper = np.full(dimensions, 2.0)
    best_parameters = (
        np.load(args.parameters)["parameters"].copy()
        if args.parameters is not None
        else mean.copy()
    )
    if args.parameters is not None:
        mean = best_parameters.copy()
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
        std = np.maximum(0.72 * elite.std(axis=0) + 0.28 * std, 0.01)
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
        desired_root=desired_root,
        desired_rotation_xyzw=desired_rotation.as_quat(),
        rotation_reference_scale=np.asarray(args.rotation_reference_scale),
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "bounded physics-rate upright root/velocity feedback search",
        "population": args.population,
        "generations": args.generations,
        "runtime_s": time.perf_counter() - started,
        "rotation_reference_scale": args.rotation_reference_scale,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
