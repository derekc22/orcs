#!/usr/bin/env python3
"""Local all-actuator MuJoCo shooting search for bilateral hand lift."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=28)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    prefix = np.load(args.prefix)
    anchor_index = args.anchor_frame
    anchor_qpos = prefix["qpos"][anchor_index].copy()
    anchor_qvel = prefix["qvel"][anchor_index].copy()
    base_target = prefix["ctrl"][anchor_index].copy()
    baseline_object_z = float(anchor_qpos[engine.object_qpos + 2])
    initial_root_height = float(anchor_qpos[engine.robot_qpos + 2])
    knot_frames = np.asarray([0.0, 40.0, 90.0])
    object_geoms = set(engine.object_geoms)

    def residual_at(knots: np.ndarray, frame: float) -> np.ndarray:
        if frame >= knot_frames[-1]:
            return knots[-1]
        index = int(np.searchsorted(knot_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - knot_frames[index]) / (knot_frames[index + 1] - knot_frames[index])
        )
        return (1.0 - amount) * knots[index] + amount * knots[index + 1]

    def group(geom: int) -> str:
        if geom == int(engine.hand_geoms[0]):
            return "left_hand"
        if geom == int(engine.hand_geoms[1]):
            return "right_hand"
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if "wrist" in name:
            return "wrist"
        if "foot" in name or "ankle" in name:
            return "feet"
        if any(token in name for token in ("knee", "shin", "hip", "thigh")):
            return "legs"
        if any(token in name for token in ("torso", "pelvis", "waist")):
            return "torso_pelvis"
        return "other_robot"

    def evaluate(parameters: np.ndarray, record: bool = False):
        knots = np.vstack([np.zeros(model.nu), parameters.reshape(2, model.nu)])
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        impulses = {
            name: 0.0
            for name in (
                "left_hand",
                "right_hand",
                "wrist",
                "feet",
                "legs",
                "torso_pelvis",
                "other_robot",
            )
        }
        hand_frames = bilateral_frames = 0
        lift_flags: list[bool] = []
        maximum_lift = 0.0
        cost = 0.0
        trace = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "actuator_force",
                "hand_contact",
                "foot_contact",
                "object_position",
                "root_height",
                "torso_tilt",
            )
        }
        for step in range(90 * engine.config.substeps):
            frame = step / engine.config.substeps
            residual = residual_at(knots, frame)
            target = np.clip(
                base_target + residual,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * float(
                    model.opt.timestep
                )
            object_position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            hand_distance = np.linalg.norm(
                data.geom_xpos[engine.hand_geoms] - object_position, axis=1
            )
            lift = float(object_position[2] - baseline_object_z)
            maximum_lift = max(maximum_lift, lift)
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            hand_count = int(np.sum(physical["hand_contact"]))
            cost += (
                8000.0 * max(0.0, initial_root_height - 0.04 - root) ** 2
                + 2200.0 * physical["tilt"] ** 2
                + 180.0
                * float(
                    np.square(data.qvel[engine.robot_dof : engine.robot_dof + 6]).sum()
                )
                + 3000.0 * float(support == 0)
                + 100.0 * float(support == 1)
                + 600.0 * float(np.square(np.maximum(hand_distance - 0.22, 0.0)).sum())
                + 5.0 * float(np.square(residual / 0.45).mean())
                - 18000.0 * max(0.0, lift)
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[name]
                for name in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            )
            if disallowed > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if invalid is not None and first_invalid is None:
                first_invalid, reason = step, invalid
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hand_count > 0)
                bilateral_frames += int(hand_count == 2)
                lift_flags.append(lift >= 0.05)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(object_position.copy())
                    trace["root_height"].append(root)
                    trace["torso_tilt"].append(float(physical["tilt"]))
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        longest = max(
            (len(list(run)) for value, run in itertools.groupby(lift_flags) if value),
            default=0,
        )
        cost /= max(step + 1, 1)
        cost -= (
            800.0 * hand_frames
            + 2200.0 * bilateral_frames
            + 80.0 * min(impulses["left_hand"], impulses["right_hand"])
            + 90000.0 * maximum_lift
            + 2500.0 * longest
        )
        rank = (0, cost) if first_invalid is None else (1, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else anchor_index + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": maximum_lift,
            "sustained_lift_frames": longest,
            "object_impulse_ns": impulses,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    dimensions = 2 * model.nu
    actuator_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    per_joint_bounds = np.full(model.nu, 0.45)
    per_joint_bounds[["wrist" in name for name in actuator_names]] = 0.30
    bounds = np.tile(per_joint_bounds, 2)
    rng = np.random.default_rng(7379)
    mean = np.zeros(dimensions)
    std = np.tile(np.minimum(per_joint_bounds * 0.35, 0.14), 2)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, (args.population, dimensions)), -bounds, bounds
        )
        population[0] = best_parameters
        population[1] = 0.0
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[:16]]
        mean = 0.70 * elite.mean(0) + 0.30 * mean
        std = np.maximum(0.70 * elite.std(0) + 0.30 * std, 0.006)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        base_target=base_target,
        knot_frames=knot_frames,
        **trace,
    )
    report = {
        "method": "local all-actuator MuJoCo bilateral lift shooting",
        "anchor_frame": anchor_index,
        "dimensions": dimensions,
        "population": args.population,
        "generations": args.generations,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
