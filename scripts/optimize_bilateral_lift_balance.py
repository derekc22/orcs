#!/usr/bin/env python3
"""Stabilize the physically bilateral hand-lift seed with balance controls."""

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
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--strict-posture", action="store_true")
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    prefix = np.load(args.prefix)
    seed = np.load(args.seed)
    stance = np.load(args.stance)
    anchor_frame = int(seed["anchor_frame"])
    anchor_qpos = prefix["qpos"][anchor_frame].copy()
    anchor_qvel = prefix["qvel"][anchor_frame].copy()
    balance = stance["selected_actuators"].astype(int)
    knot_frames = np.asarray([0.0, 12.0, 35.0, 70.0])
    object_geoms = set(engine.object_geoms)
    ground_object_z = float(anchor_qpos[engine.object_qpos + 2])

    def nominal(frame: float) -> np.ndarray:
        index = min(int(frame), len(seed["ctrl"]) - 1)
        return seed["ctrl"][index].copy()

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
        knots = np.vstack([np.zeros(len(balance)), parameters.reshape(3, -1)])
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
                "object_impulse",
            )
        }
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            residual = residual_at(knots, frame)
            target = nominal(frame)
            target[balance] += residual
            target = np.clip(
                target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
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
            lift = float(object_position[2] - ground_object_z)
            maximum_lift = max(maximum_lift, lift)
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            hand_count = int(np.sum(physical["hand_contact"]))
            root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
            cost += (
                (120000.0 if args.strict_posture else 14000.0)
                * max(0.0, (0.60 if args.strict_posture else 0.58) - root) ** 2
                + (22000.0 if args.strict_posture else 3500.0)
                * physical["tilt"] ** 2
                + (3000.0 if args.strict_posture else 600.0)
                * float(np.square(root_velocity).sum())
                + 5000.0 * float(support == 0)
                + 250.0 * float(support == 1)
                + 800.0 * float(frame > 22.0 and hand_count == 0)
                + 90.0 * float(frame > 22.0 and hand_count == 1)
                + 8.0 * float(np.square(residual / 0.55).mean())
                - 24000.0 * max(0.0, lift)
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[name]
                for name in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            if disallowed > 0.5 and invalid is None:
                invalid = "non_end_effector_object_impulse"
            if invalid is not None:
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
                    trace["object_impulse"].append(
                        np.asarray(list(impulses.values()))
                    )
            if first_invalid is not None:
                break
        longest = max(
            (len(list(run)) for value, run in itertools.groupby(lift_flags) if value),
            default=0,
        )
        cost /= max(step + 1, 1)
        cost += 30.0 * impulses["wrist"]
        if args.strict_posture:
            cost += (
                250000.0 * max(0.0, 0.58 - root) ** 2
                + 80000.0 * max(0.0, physical["tilt"] - np.deg2rad(30.0)) ** 2
            )
        cost -= (
            500.0 * hand_frames
            + 1400.0 * bilateral_frames
            + 100.0 * min(impulses["left_hand"], impulses["right_hand"])
            + 100000.0 * maximum_lift
            + 3500.0 * longest
        )
        rank = (0, cost) if first_invalid is None else (1, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else anchor_frame + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
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

    dimensions = 3 * len(balance)
    bounds = np.full(dimensions, 0.55)
    rng = np.random.default_rng(9917)
    mean = np.zeros(dimensions)
    # Seed the search toward the already validated upright stance targets.
    recovery = stance["target"][balance] - nominal(knot_frames[-1])[balance]
    mean = np.tile(np.clip(recovery, -0.55, 0.55), 3)
    std = np.full(dimensions, 0.16)
    if args.warm_start is not None:
        warm = np.load(args.warm_start)["parameters"].astype(np.float64)
        if warm.shape != (dimensions,):
            raise ValueError(
                f"warm-start shape {warm.shape} does not match {(dimensions,)}"
            )
        mean = warm.copy()
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
        elite = population[order[:12]]
        mean = 0.72 * elite.mean(0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(0) + 0.28 * std, 0.006)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_frame=anchor_frame,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        balance_actuators=balance,
        knot_frames=knot_frames,
        **trace,
    )
    report = {
        "method": "balance stabilization of actual bilateral hand-lift seed",
        "dimensions": dimensions,
        "population": args.population,
        "generations": args.generations,
        "horizon_frames": args.horizon,
        "strict_posture": args.strict_posture,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
