#!/usr/bin/env python3
"""Refine arm targets for bilateral contact on a fixed dynamic approach."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep


def interp(knots: np.ndarray, frames: np.ndarray, frame: float) -> np.ndarray:
    index = int(
        np.clip(np.searchsorted(frames, frame, side="right") - 1, 0, len(frames) - 2)
    )
    amount = smoothstep((frame - frames[index]) / (frames[index + 1] - frames[index]))
    return (1.0 - amount) * knots[index] + amount * knots[index + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--approach", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--generations", type=int, default=20)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    family = np.load(args.family)
    approach = np.load(args.approach)
    balance = stance["selected_actuators"].astype(int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    balance_frames = approach["knot_frames"]
    balance_knots = np.vstack(
        [
            np.zeros(len(balance)),
            approach["parameters"].reshape(3, -1),
            np.zeros(len(balance)),
        ]
    )
    arm_frames = np.asarray([0.0, 80.0, 115.0, 145.0, 170.0])
    free_knots = 3
    poses = {
        name: family[name][6:] for name in ("pregrasp", "grasp", "squeeze", "lift")
    }
    anchor = mujoco.MjData(model)
    anchor.qpos[:] = stance["initial_qpos"]
    anchor.qvel[:] = stance["initial_qvel"]
    for _ in range(80 * engine.config.substeps):
        anchor.ctrl[:] = stance["target"]
        mujoco.mj_step(model, anchor)
    anchor_qpos, anchor_qvel, anchor_act = (
        anchor.qpos.copy(),
        anchor.qvel.copy(),
        anchor.act.copy(),
    )
    object_geoms = set(engine.object_geoms)
    hand_geoms = set(engine.hand_geoms)

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
        amount = smoothstep((frame - 140.0) / 80.0)
        return (1.0 - amount) * poses["squeeze"] + amount * poses["lift"]

    def group(geom: int) -> str:
        if geom in hand_geoms:
            return "hand"
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if "wrist" in name:
            return "wrist"
        if "foot" in name or "ankle" in name:
            return "feet"
        if any(token in name for token in ("knee", "shin", "hip", "thigh")):
            return "legs"
        return "body"

    def evaluate(parameters: np.ndarray, record: bool = False):
        arm_knots = np.vstack(
            [
                np.zeros(len(arms)),
                parameters.reshape(free_knots, -1),
                np.zeros(len(arms)),
            ]
        )
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        if model.na:
            data.act[:] = anchor_act
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        hand_impulse = np.zeros(2)
        disallowed_impulse = 0.0
        hand_frames = bilateral_frames = 0
        cost = 0.0
        traces = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "hand_contact",
                "foot_contact",
                "object_position",
                "root_height",
                "torso_tilt",
            )
        }
        for step in range(170 * engine.config.substeps):
            frame = step / engine.config.substeps
            target = stance["target"].copy()
            target[balance] += interp(balance_knots, balance_frames, frame)
            residual = interp(arm_knots, arm_frames, frame)
            nominal = nominal_arm(frame)
            target[arms] = nominal[arms] + residual
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
                impulse = float(np.linalg.norm(force[:3])) * float(model.opt.timestep)
                if other == int(engine.hand_geoms[0]):
                    hand_impulse[0] += impulse
                elif other == int(engine.hand_geoms[1]):
                    hand_impulse[1] += impulse
                elif group(other) != "hand":
                    disallowed_impulse += impulse
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            hand_count = int(np.sum(physical["hand_contact"]))
            cost += (
                6000.0 * max(0.0, 0.61 - root) ** 2
                + 1600.0 * physical["tilt"] ** 2
                + 150.0
                * float(
                    np.square(data.qvel[engine.robot_dof : engine.robot_dof + 6]).sum()
                )
                + 3000.0 * float(support == 0)
                + 100.0 * float(support == 1)
                + 900.0 * float(frame > 105.0 and hand_count == 0)
                + 80.0 * float(frame > 105.0 and hand_count == 1)
                + 3.0 * float(np.square(residual / 0.25).mean())
            )
            invalid = generator._invalid_reason(data, physical)
            if disallowed_impulse > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if invalid is not None and first_invalid is None:
                first_invalid, reason = step, invalid
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hand_count > 0)
                bilateral_frames += int(hand_count == 2)
                if record:
                    traces["qpos"].append(data.qpos.copy())
                    traces["qvel"].append(data.qvel.copy())
                    traces["ctrl"].append(data.ctrl.copy())
                    traces["hand_contact"].append(physical["hand_contact"].copy())
                    traces["foot_contact"].append(physical["foot_contact"].copy())
                    traces["object_position"].append(
                        data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
                    )
                    traces["root_height"].append(root)
                    traces["torso_tilt"].append(float(physical["tilt"]))
            if first_invalid is not None and step > first_invalid + 4 and not record:
                break
        cost /= max(step + 1, 1)
        cost -= (
            500.0 * hand_frames
            + 1200.0 * bilateral_frames
            + 50.0 * float(np.min(hand_impulse))
        )
        rank = (0, cost) if first_invalid is None else (1, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "left_right_hand_impulse_ns": hand_impulse.tolist(),
            "disallowed_object_impulse_ns": disallowed_impulse,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in traces.items()}

    dimensions = free_knots * len(arms)
    rng = np.random.default_rng(4407)
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.08)
    bounds = np.full(dimensions, 0.25)
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
        std = np.maximum(0.72 * elite.std(0) + 0.28 * std, 0.005)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, traces = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        balance_knots=balance_knots,
        balance_frames=balance_frames,
        arm_frames=arm_frames,
        **traces,
    )
    report = {
        "method": "MuJoCo bilateral hand-contact arm refinement",
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
