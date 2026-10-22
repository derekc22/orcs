#!/usr/bin/env python3
"""Connect the legal upright start to a locally stable one-hand lean/contact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from probe_flat_hand_grasp import build, smoothstep


EQUILIBRIUM_RESIDUAL = np.asarray(
    [
        -0.001273728173330921,
        0.011538056422443807,
        0.034390132158876596,
        0.241659377256135,
        -0.09962218714829366,
        0.07298038765513559,
        0.023623742305077938,
        0.102479999279457,
        0.05910164109887407,
        -0.20370011873600175,
        -0.03354839080695084,
        -0.03751367360340259,
        0.09507631720691576,
        -0.03970718125396259,
        -0.07163380197527545,
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elites", type=int, default=8)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    base = source["balance_target"].copy()
    controlled = source["controlled_actuators"].astype(int)
    contact_pose = source["push_arm_target"].copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    settle = 90.0
    approach = 45.0
    hold_end = settle + approach + 10.0
    retract_end = hold_end + 25.0
    knot_frames = np.asarray([0.0, 24.0, 48.0, 75.0])
    dimensions = len(knot_frames) * len(balance)

    contact_control = base.copy()
    contact_control[controlled] = contact_pose
    contact_control[balance] += EQUILIBRIUM_RESIDUAL

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(int(settle) * engine.config.substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    desired_hand = anchor.geom_xpos[engine.hand_geoms[1]].copy()
    pose = mujoco.MjData(model)
    pose.qpos[:] = anchor.qpos
    pose.qpos[engine.actuator_qpos[controlled]] = contact_pose
    mujoco.mj_forward(model, pose)
    desired_hand = pose.geom_xpos[engine.hand_geoms[1]].copy()

    def residual_at(parameters: np.ndarray, local_frame: float) -> np.ndarray:
        knots = parameters.reshape(len(knot_frames), len(balance))
        if local_frame <= knot_frames[0]:
            return knots[0]
        if local_frame >= knot_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(knot_frames, local_frame, side="right") - 1)
        amount = smoothstep(
            (local_frame - knot_frames[segment])
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
        nonhand = 0.0
        hand_frames = 0
        last_hand_frame = None
        minimum_hand_error = float("inf")
        maximum_hand_impulse = 0.0
        trace = {key: [] for key in ("qpos", "qvel", "ctrl", "actuator_force", "hand_contact", "foot_contact", "object_position", "root_height", "torso_tilt", "nonhand_impulse")}
        ctrl_physics = []
        held = base.copy()
        for step in range(300 * engine.config.substeps):
            frame = step / engine.config.substeps
            if step % engine.config.substeps == 0:
                if frame < settle:
                    held = base.copy()
                elif frame < settle + approach:
                    progress = smoothstep((frame - settle) / approach)
                    held = (1.0 - progress) * base + progress * contact_control
                    held[balance] += residual_at(parameters, frame - settle)
                elif frame < hold_end:
                    held = contact_control.copy()
                    held[balance] += residual_at(parameters, frame - settle)
                elif frame < retract_end:
                    progress = smoothstep((frame - hold_end) / (retract_end - hold_end))
                    held = (1.0 - progress) * contact_control + progress * base
                    held[balance] += (1.0 - progress) * residual_at(parameters, frame - settle)
                else:
                    held = base.copy()
                held = np.clip(held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = held
            ctrl_physics.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            hand_impulse = 0.0
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
                impulse = float(np.linalg.norm(force[:3])) * model.opt.timestep
                if other in hand_geoms:
                    hand_impulse += impulse
                else:
                    nonhand += impulse
            maximum_hand_impulse += hand_impulse
            if frame >= settle:
                minimum_hand_error = min(
                    minimum_hand_error,
                    float(np.linalg.norm(data.geom_xpos[engine.hand_geoms[1]] - desired_hand)),
                )
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hands = int(np.sum(physical["hand_contact"]))
                hand_frames += int(hands > 0)
                if hands:
                    last_hand_frame = int(step // engine.config.substeps)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(data.qpos[engine.object_qpos : engine.object_qpos + 3].copy())
                    trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["nonhand_impulse"].append(nonhand)
            if invalid is not None:
                first_invalid = step
                invalid_reason = invalid
                break
        endpoint = step / engine.config.substeps
        recovery_margin = max(0.0, endpoint - hold_end)
        physical = generator._physical(data)
        valid = first_invalid is None
        useful = hand_frames > 0 and nonhand <= 0.10
        if useful and valid:
            rank = (0, -hand_frames, -recovery_margin, minimum_hand_error)
        elif useful:
            rank = (1, -recovery_margin, -hand_frames, -step, minimum_hand_error)
        elif valid:
            rank = (2, minimum_hand_error)
        else:
            rank = (3, minimum_hand_error, -step)
        metrics = {
            "full_horizon_valid": valid,
            "first_invalid_frame": None if valid else first_invalid / engine.config.substeps,
            "invalid_reason": invalid_reason,
            "hand_contact_frames": hand_frames,
            "last_hand_contact_frame": last_hand_frame,
            "hand_impulse_ns": maximum_hand_impulse,
            "nonhand_object_impulse_ns": nonhand,
            "minimum_hand_target_error_m": minimum_hand_error,
            "recovery_margin_frames": recovery_margin,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    rng = np.random.default_rng(20191)
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.075)
    lower = np.full(dimensions, -0.28)
    upper = np.full(dimensions, 0.28)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, dimensions)), lower, upper)
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
        np.savez_compressed(args.output.with_suffix(".checkpoint.npz"), parameters=best_parameters, mean=mean, std=std)
        print("GEN", generation, json.dumps(best[1]), flush=True)

    _, metrics, trace, ctrl_physics = evaluate(best_parameters, record=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        contact_control=contact_control,
        balance_actuators=balance,
        knot_frames=knot_frames,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "legal whole-body lean/contact transition shooting with recovery gate",
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
