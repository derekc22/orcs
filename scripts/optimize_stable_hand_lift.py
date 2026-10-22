#!/usr/bin/env python3
"""Lift the box from an already stable bilateral-hand physical rollout.

The lower-body/waist targets are replayed exactly from the stable seed.  CEM
only changes the 14 arm/wrist targets.  Every candidate is evaluated through
MuJoCo and ranked first by robot validity and the manipulation lift contract.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep

GROUP_NAMES = (
    "left_hand",
    "right_hand",
    "wrist",
    "feet",
    "legs",
    "torso_pelvis",
    "other_robot",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--seed-value", type=int, default=4417)
    parser.add_argument("--optimize-balance", action="store_true")
    parser.add_argument("--stance", type=Path)
    parser.add_argument("--continuation-min-lift-frames", type=int, default=10)
    parser.add_argument("--initial-std", type=float)
    parser.add_argument("--recovery-root-height", type=float, default=0.50)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--evaluation-horizon", type=int)
    parser.add_argument("--hold-index", type=int)
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=0,
        help="Start from this realized source frame (state immediately before it).",
    )
    parser.add_argument(
        "--knot-frames",
        type=float,
        nargs="+",
        help="Explicit knot locations; the first is the fixed zero boundary.",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.seed)
    source_horizon = len(source["ctrl"])
    if not 0 <= args.anchor_index < source_horizon:
        raise ValueError(f"anchor-index {args.anchor_index} outside source horizon {source_horizon}")
    horizon = args.evaluation_horizon or (source_horizon - args.anchor_index)
    anchor_frame = int(source["anchor_frame"]) if "anchor_frame" in source else 210
    anchor_frame += args.anchor_index
    if args.anchor_index == 0:
        anchor_qpos = source["anchor_qpos"].copy()
        anchor_qvel = source["anchor_qvel"].copy()
    else:
        anchor_qpos = source["qpos"][args.anchor_index - 1].copy()
        anchor_qvel = source["qvel"][args.anchor_index - 1].copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    selected = balance if args.optimize_balance else arms
    joint_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    knot_frames = (
        np.asarray(args.knot_frames, dtype=float)
        if args.knot_frames is not None
        else np.linspace(0.0, float(horizon - 1), 5)
    )
    if knot_frames[0] != 0.0 or np.any(np.diff(knot_frames) <= 0.0):
        raise ValueError("knot-frames must start at zero and increase strictly")
    if knot_frames[-1] > horizon - 1:
        raise ValueError("last knot lies outside evaluation horizon")
    free_knots = len(knot_frames) - 1
    ground_z = 0.18545216
    object_geoms = set(engine.object_geoms)

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

    def interpolate(knots: np.ndarray, frame: float) -> np.ndarray:
        if frame >= knot_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(knot_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - knot_frames[segment])
            / (knot_frames[segment + 1] - knot_frames[segment])
        )
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def evaluate(parameters: np.ndarray, record: bool = False):
        knots = np.vstack([np.zeros(len(selected)), parameters.reshape(free_knots, -1)])
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        impulses = {name: 0.0 for name in GROUP_NAMES}
        hand_frames = bilateral_frames = 0
        lift_flags: list[bool] = []
        supported_lift_flags: list[bool] = []
        peak_lift = 0.0
        peak_supported_lift = 0.0
        first_invalid = None
        reason = None
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
        for step in range(horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            nominal_index = min(
                args.anchor_index + int(frame * args.time_scale), source_horizon - 1
            )
            if args.hold_index is not None:
                nominal_index = min(
                    nominal_index, args.anchor_index + args.hold_index
                )
            residual = interpolate(knots, frame)
            target = source["ctrl"][nominal_index].copy()
            target[selected] += residual
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
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            lift = float(obj[2] - ground_z)
            peak_lift = max(peak_lift, lift)
            root = float(data.qpos[engine.robot_qpos + 2])
            hand_count = int(np.sum(physical["hand_contact"]))
            if hand_count > 0:
                peak_supported_lift = max(peak_supported_lift, lift)
            support = int(np.sum(physical["foot_contact"]))
            root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
            cost += (
                65000.0 * max(0.0, 0.48 - root) ** 2
                + 12000.0 * max(0.0, physical["tilt"] - np.deg2rad(38.0)) ** 2
                + 350.0 * float(np.square(root_velocity).sum())
                + 6000.0 * float(support == 0)
                + 200.0 * float(support == 1)
                + 6000.0 * float(hand_count == 0)
                + 180.0 * float(hand_count == 1)
                + 9.0 * float(np.square(residual / 0.25).mean())
                - 80000.0 * max(0.0, lift)
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[name]
                for name in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            if disallowed > 0.25 and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hand_count > 0)
                bilateral_frames += int(hand_count == 2)
                lift_flags.append(lift >= 0.05)
                supported_lift_flags.append(lift >= 0.05 and hand_count > 0)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(obj.copy())
                    trace["root_height"].append(root)
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(np.asarray(list(impulses.values())))
            if invalid is not None:
                first_invalid, reason = step, invalid
                break
        longest = max(
            (len(list(run)) for value, run in itertools.groupby(lift_flags) if value),
            default=0,
        )
        longest_supported = max(
            (
                len(list(run))
                for value, run in itertools.groupby(supported_lift_flags)
                if value
            ),
            default=0,
        )
        disallowed = sum(
            impulses[name]
            for name in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        cost /= max(step + 1, 1)
        cost += 80.0 * impulses["wrist"] + 2000.0 * disallowed
        cost -= (
            450.0 * hand_frames
            + 1300.0 * bilateral_frames
            + 140000.0 * peak_supported_lift
            + 12000.0 * longest_supported
        )
        valid = first_invalid is None
        contract = valid and longest_supported >= 10 and disallowed <= 0.25
        recoverable = (
            valid
            and longest_supported >= args.continuation_min_lift_frames
            and disallowed <= 0.25
            and root >= args.recovery_root_height
            and float(physical["tilt"]) < np.deg2rad(40.0)
            and int(np.sum(physical["foot_contact"])) > 0
            and float(data.qvel[engine.robot_dof + 2]) > -0.10
        )
        # Contract is lexicographically dominant.  Before it is reached, keep
        # valid candidates and maximize sustained lift rather than average loss.
        if recoverable:
            rank = (0, -longest_supported, -root, -peak_supported_lift, cost)
        elif valid and longest_supported >= args.continuation_min_lift_frames:
            rank = (1, -root, float(physical["tilt"]), -longest_supported, cost)
        elif longest_supported >= args.continuation_min_lift_frames:
            rank = (2, -step, -longest_supported, -peak_supported_lift, cost)
        elif valid:
            rank = (3, -longest_supported, -peak_supported_lift, -root, cost)
        else:
            rank = (
                4,
                -min(longest_supported, 10),
                -step,
                -peak_supported_lift,
                cost,
            )
        metrics = {
            "first_invalid_frame": None
            if valid
            else anchor_frame + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": peak_lift,
            "sustained_lift_frames": longest,
            "maximum_hand_supported_lift_m": peak_supported_lift,
            "sustained_hand_supported_lift_frames": longest_supported,
            "lift_contract": contract,
            "recoverable_terminal": bool(recoverable),
            "object_impulse_ns": impulses,
            "final_root_height_m": root,
            "final_root_vertical_velocity_m_s": float(data.qvel[engine.robot_dof + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    dimensions = free_knots * len(selected)
    names = joint_names[selected]
    per_joint_bound = np.asarray(
        [0.45 if args.optimize_balance else (0.16 if "wrist" in str(name) else 0.28) for name in names]
    )
    bounds = np.tile(per_joint_bound, free_knots)
    rng = np.random.default_rng(args.seed_value)
    mean = np.zeros(dimensions)
    if args.optimize_balance and args.stance is not None:
        stance_target = np.load(args.stance)["target"]
        recovery = np.clip(stance_target[selected] - source["ctrl"][-1, selected], -per_joint_bound, per_joint_bound)
        # Increase recovery authority smoothly across all free knots.
        mean = np.concatenate(
            [
                fraction * recovery
                for fraction in np.linspace(1.0 / free_knots, 1.0, free_knots)
            ]
        )
    std = np.tile(
        np.full(
            len(selected),
            args.initial_std if args.initial_std is not None else 0.035,
        )
        if args.optimize_balance
        else np.full(
            len(selected),
            args.initial_std if args.initial_std is not None else 0.02,
        ),
        free_knots,
    )
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
        elite = population[order[: args.elites]]
        mean = 0.75 * elite.mean(0) + 0.25 * mean
        std = np.maximum(0.75 * elite.std(0) + 0.25 * std, 0.004)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        if (
            best[1]["recoverable_terminal"]
            and best[1]["sustained_hand_supported_lift_frames"] >= 18
        ):
            break
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_frame=anchor_frame,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        optimized_actuators=selected,
        knot_frames=knot_frames,
        source_path=str(args.seed),
        **trace,
    )
    report = {
        "method": "stable bilateral hold plus MuJoCo lift refinement",
        "optimized_group": "balance" if args.optimize_balance else "arms",
        "optimized_joints": names.tolist(),
        "dimensions": dimensions,
        "population": args.population,
        "generations_completed": len(history),
        "candidate_rollouts": args.population * len(history),
        "horizon_frames": horizon,
        "time_scale": args.time_scale,
        "hold_index": args.hold_index,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
