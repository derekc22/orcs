#!/usr/bin/env python3
"""Simulator-in-the-loop balance refinement for a bilateral flat-hand lift."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep


def interpolate(knots: np.ndarray, frames: np.ndarray, frame: float) -> np.ndarray:
    index = int(
        np.clip(np.searchsorted(frames, frame, side="right") - 1, 0, len(frames) - 2)
    )
    amount = smoothstep((frame - frames[index]) / (frames[index + 1] - frames[index]))
    return (1.0 - amount) * knots[index] + amount * knots[index + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--grasp-family", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--generations", type=int, default=28)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    family = np.load(args.grasp_family)
    balance = stance["selected_actuators"].astype(int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    anchor = mujoco.MjData(model)
    anchor.qpos[:] = stance["initial_qpos"]
    anchor.qvel[:] = stance["initial_qvel"]
    for _ in range(80 * engine.config.substeps):
        anchor.ctrl[:] = stance["target"]
        mujoco.mj_step(model, anchor)
    anchor_qpos = anchor.qpos.copy()
    anchor_qvel = anchor.qvel.copy()
    anchor_act = anchor.act.copy()
    anchor_time = float(anchor.time)
    baseline_object_z = float(anchor.qpos[engine.object_qpos + 2])
    root_height = float(anchor.qpos[engine.robot_qpos + 2])
    nominal_balance = stance["target"][balance].copy()
    poses = {
        name: family[name][6:] for name in ("pregrasp", "grasp", "squeeze", "lift")
    }
    knot_frames = np.asarray([0.0, 60.0, 120.0, 180.0, 240.0])
    free_knots = len(knot_frames) - 2
    object_geoms = set(engine.object_geoms)
    forbidden_groups = (
        "wrist",
        "feet",
        "lower_legs",
        "upper_legs",
        "torso_pelvis",
        "other_robot",
    )

    def object_group(geom: int) -> str:
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
        if "knee" in name or "shin" in name:
            return "lower_legs"
        if "hip" in name or "thigh" in name:
            return "upper_legs"
        if any(token in name for token in ("torso", "pelvis", "waist")):
            return "torso_pelvis"
        return "other_robot"

    def nominal_arm(frame: float) -> np.ndarray:
        if frame < 70.0:
            amount = smoothstep(frame / 70.0)
            return (1.0 - amount) * stance["target"] + amount * poses["pregrasp"]
        if frame < 110.0:
            amount = smoothstep((frame - 70.0) / 40.0)
            return (1.0 - amount) * poses["pregrasp"] + amount * poses["grasp"]
        if frame < 140.0:
            amount = smoothstep((frame - 110.0) / 30.0)
            return (1.0 - amount) * poses["grasp"] + amount * poses["squeeze"]
        if frame < 220.0:
            amount = smoothstep((frame - 140.0) / 80.0)
            return (1.0 - amount) * poses["squeeze"] + amount * poses["lift"]
        return poses["lift"]

    def evaluate(parameters: np.ndarray, record: bool = False):
        residual_knots = np.vstack(
            [
                np.zeros(len(balance)),
                parameters.reshape(free_knots, -1),
                np.zeros(len(balance)),
            ]
        )
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        if model.na:
            data.act[:] = anchor_act
        data.time = anchor_time
        mujoco.mj_forward(model, data)
        first_invalid = None
        invalid_reason = None
        impulses = {
            "left_hand": 0.0,
            "right_hand": 0.0,
            "wrist": 0.0,
            "feet": 0.0,
            "lower_legs": 0.0,
            "upper_legs": 0.0,
            "torso_pelvis": 0.0,
            "other_robot": 0.0,
        }
        reference_trace: list[np.ndarray] = []
        trace = {
            name: []
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "actuator_force",
                "hand_contact",
                "foot_contact",
                "root_height",
                "torso_tilt",
                "object_position",
                "forbidden",
            )
        }
        hand_frames = bilateral_frames = 0
        maximum_lift = 0.0
        lift_flags: list[bool] = []
        cost = 0.0
        steps = 240 * engine.config.substeps
        for step in range(steps):
            frame = step / engine.config.substeps
            residual = interpolate(residual_knots, knot_frames, frame)
            target = stance["target"].copy()
            target[balance] = nominal_balance + residual
            arm = nominal_arm(frame)
            target[arms] = arm[arms]
            target = np.clip(
                target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                pair = {first, second}
                if not pair & object_geoms:
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, force)
                impulses[object_group(other)] += float(
                    np.linalg.norm(force[:3])
                ) * float(model.opt.timestep)
            physical = generator._physical(data)
            current_lift = float(data.qpos[engine.object_qpos + 2] - baseline_object_z)
            maximum_lift = max(maximum_lift, current_lift)
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            hand_count = int(np.sum(physical["hand_contact"]))
            cost += (
                5000.0 * max(0.0, root_height - root) ** 2
                + 1400.0 * physical["tilt"] ** 2
                + 120.0
                * float(
                    np.square(data.qvel[engine.robot_dof : engine.robot_dof + 6]).sum()
                )
                + 2500.0 * float(support == 0)
                + 80.0 * float(support == 1)
                + 8.0 * float(np.square(physical["foot_slip"]).sum())
                + 800.0 * float(hand_count == 0 and frame >= 85.0)
                + 50.0 * float(hand_count == 1 and frame >= 85.0)
                - 12000.0 * max(0.0, current_lift)
                + 2.0 * float(np.square(residual / 0.45).mean())
            )
            invalid = generator._invalid_reason(data, physical)
            forbidden_impulse = sum(impulses[name] for name in forbidden_groups)
            if forbidden_impulse > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if invalid is not None and first_invalid is None:
                first_invalid = step
                invalid_reason = invalid
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hand_count > 0)
                bilateral_frames += int(hand_count == 2)
                lift_flags.append(current_lift >= 0.05)
                if record:
                    reference_trace.append(target.copy())
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["root_height"].append(root)
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_position"].append(
                        data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
                    )
                    trace["forbidden"].append(bool(physical["forbidden"]))
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        longest_lift = max(
            (
                len(list(group))
                for value, group in itertools.groupby(lift_flags)
                if value
            ),
            default=0,
        )
        cost /= max(step + 1, 1)
        cost -= (
            300.0 * hand_frames
            + 500.0 * bilateral_frames
            + 50000.0 * maximum_lift
            + 1000.0 * longest_lift
        )
        rank = (0, cost) if first_invalid is None else (1, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else first_invalid / engine.config.substeps,
            "invalid_reason": invalid_reason,
            "maximum_lift_m": maximum_lift,
            "sustained_lift_frames": longest_lift,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "object_impulse_ns": impulses,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "maximum_actuator_usage": float(
                np.max(np.abs(data.actuator_force) / engine.force_scale)
            ),
            "cost": float(cost),
        }
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        arrays["target"] = np.asarray(reference_trace)
        return rank, metrics, arrays

    dimensions = free_knots * len(balance)
    lower = np.full(dimensions, -0.45)
    upper = np.full(dimensions, 0.45)
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.16)
    rng = np.random.default_rng(1129)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, dimensions)), lower, upper
        )
        population[0] = best_parameters
        population[1] = 0.0
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:12]]
        mean = 0.70 * elite.mean(axis=0) + 0.30 * mean
        std = np.maximum(0.70 * elite.std(axis=0) + 0.30 * std, 0.008)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)

    _, metrics, arrays = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        anchor_act=anchor_act,
        balance_actuators=balance,
        arm_actuators=arms,
        knot_frames=knot_frames,
        **arrays,
    )
    report = {
        "method": "MuJoCo direct bilateral-hand lift balance refinement",
        "dimensions": dimensions,
        "knot_frames": knot_frames.tolist(),
        "population": args.population,
        "generations": args.generations,
        "evaluations": 1 + args.population * args.generations,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
