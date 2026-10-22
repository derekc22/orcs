#!/usr/bin/env python3
"""Optimize a Cartesian-feedback, flat-hand lift/transport from a stable grip."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep

GROUPS = (
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
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--initial", type=float, nargs=7)
    parser.add_argument("--minimum-forward", type=float, default=0.0)
    parser.add_argument("--maximum-disallowed-impulse", type=float, default=2.0)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Physical evaluation horizon in 50 Hz frames (defaults to seed length).",
    )
    parser.add_argument(
        "--anchor-frame",
        type=int,
        default=210,
        help="Global reference frame corresponding to the seed anchor.",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=0,
        help="Start from this recorded seed frame (state immediately before it).",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    seed = np.load(args.seed)
    if args.seed_index == 0:
        anchor_qpos = seed["anchor_qpos"].copy()
        anchor_qvel = seed["anchor_qvel"].copy()
    else:
        anchor_qpos = seed["qpos"][args.seed_index - 1].copy()
        anchor_qvel = seed["qvel"][args.seed_index - 1].copy()
    horizon = len(seed["ctrl"]) if args.horizon is None else args.horizon
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    actuator_dofs = np.asarray(
        [model.jnt_dofadr[joint] for joint in model.actuator_trnid[:, 0]], dtype=int
    )
    arm_dofs = actuator_dofs[arms]
    object_geoms = set(engine.object_geoms)
    ground_z = 0.18545216
    goal = engine.reference_object_pos[-1]

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = anchor_qpos
    anchor.qvel[:] = anchor_qvel
    mujoco.mj_forward(model, anchor)
    initial_hands = np.asarray([anchor.geom_xpos[geom].copy() for geom in engine.hand_geoms])
    initial_object = anchor.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    forward = np.r_[goal[:2] - initial_object[:2], 0.0]
    forward /= np.linalg.norm(forward)
    lateral = np.asarray([-forward[1], forward[0], 0.0])

    lower = np.asarray(
        [args.minimum_forward, 0.02, 12.0, 0.0, 0.20, -0.12, 0.0]
    )
    upper = np.asarray([0.62, 0.22, 60.0, 0.14, 1.20, 0.12, 0.05])

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

    balance_frames = np.asarray([0.0, 30.0, 75.0])

    def balance_residual(frame: float, parameters: np.ndarray) -> np.ndarray:
        knots = np.vstack([np.zeros(len(balance)), parameters[7:].reshape(2, -1)])
        if frame >= balance_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(balance_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - balance_frames[segment])
            / (balance_frames[segment + 1] - balance_frames[segment])
        )
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def hand_targets(frame: float, parameters: np.ndarray) -> np.ndarray:
        forward_distance, lift, duration, release, _, side, squeeze = parameters[:7]
        progress = smoothstep(frame / duration)
        displacement = progress * (forward_distance * forward + np.asarray([0, 0, lift]))
        displacement += progress * side * lateral
        targets = initial_hands + displacement
        # Preserve the original asymmetric grasp, while adding a modest inward
        # squeeze in the horizontal direction perpendicular to transport.
        targets[0] -= progress * squeeze * lateral
        targets[1] += progress * squeeze * lateral
        release_progress = smoothstep((frame - duration) / 8.0)
        targets[0] += release_progress * release * lateral
        targets[1] -= release_progress * release * lateral
        return targets

    def arm_target(data: mujoco.MjData, desired: np.ndarray, nominal: np.ndarray, gain: float):
        jacobian = np.zeros((6, len(arms)))
        error = np.zeros(6)
        for hand_index, geom in enumerate(engine.hand_geoms):
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacp, jacr, int(geom))
            rows = slice(3 * hand_index, 3 * hand_index + 3)
            jacobian[rows] = jacp[:, arm_dofs]
            error[rows] = desired[hand_index] - data.geom_xpos[geom]
        damping = 2.0e-3
        current = data.qpos[engine.actuator_qpos[arms]]
        posture = nominal[arms] - current
        system = jacobian.T @ jacobian + (damping + 0.02) * np.eye(len(arms))
        rhs = jacobian.T @ error + 0.02 * posture
        delta = np.linalg.solve(system, rhs)
        delta = np.clip(gain * delta, -0.075, 0.075)
        return current + delta

    def evaluate(parameters: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        impulses = {name: 0.0 for name in GROUPS}
        lift_supported: list[bool] = []
        hand_frames = bilateral_frames = 0
        first_invalid = None
        reason = None
        peak_lift = peak_supported_lift = 0.0
        hand_transport = 0.0
        initial_goal_error = float(np.linalg.norm(initial_object - goal))
        last_hand_object = initial_object.copy()
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
                "hand_target",
            )
        }
        for step in range(horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            nominal = seed["ctrl"][
                min(args.seed_index + int(frame), len(seed["ctrl"]) - 1)
            ].copy()
            nominal[balance] += balance_residual(frame, parameters)
            if step % engine.config.substeps == 0:
                desired = hand_targets(frame, parameters)
                nominal[arms] = arm_target(data, desired, nominal, parameters[4])
                held_target = nominal
                held_desired = desired
            data.ctrl[:] = np.clip(
                held_target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
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
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            lift = float(obj[2] - ground_z)
            peak_lift = max(peak_lift, lift)
            hands = int(np.sum(physical["hand_contact"]))
            if hands:
                peak_supported_lift = max(peak_supported_lift, lift)
                hand_transport += float(np.linalg.norm(obj - last_hand_object))
                last_hand_object = obj.copy()
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            goal_error = float(np.linalg.norm(obj - goal))
            cost += (
                80000.0 * max(0.0, 0.46 - root) ** 2
                + 18000.0 * max(0.0, physical["tilt"] - np.deg2rad(42.0)) ** 2
                + 5000.0 * float(support == 0)
                + 180.0 * float(support == 1)
                + 3500.0 * float(frame < parameters[2] and hands == 0)
                + 250.0 * float(frame < parameters[2] and hands == 1)
                + 1000.0 * goal_error**2
                - 50000.0 * max(0.0, lift)
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[name]
                for name in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            # Small grazing impulses are logged but are not task-driving.  A
            # threshold remains far below task-driving strike impulses.  The
            # directional attribution is checked separately on finalists.
            if disallowed > args.maximum_disallowed_impulse and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                lift_supported.append(lift >= 0.05 and hands > 0)
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
                    trace["hand_target"].append(held_desired.copy())
            if invalid is not None:
                first_invalid, reason = step, invalid
                break
        longest = max(
            (len(list(run)) for value, run in itertools.groupby(lift_supported) if value),
            default=0,
        )
        disallowed = sum(
            impulses[name]
            for name in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        valid = first_invalid is None
        goal_progress = initial_goal_error - goal_error
        contract = (
            valid
            and longest >= 10
            and disallowed <= args.maximum_disallowed_impulse
        )
        cost /= max(step + 1, 1)
        cost += 100.0 * impulses["wrist"]
        cost -= (
            800.0 * hand_frames
            + 1800.0 * bilateral_frames
            + 180000.0 * peak_supported_lift
            + 12000.0 * longest
            + 120000.0 * goal_progress
            + 50000.0 * hand_transport
        )
        if valid and contract:
            rank = (0, goal_error, -longest, cost)
        elif valid:
            rank = (1, -longest, -peak_supported_lift, goal_error, cost)
        else:
            rank = (2, -min(longest, 10), -step, goal_error, cost)
        metrics = {
            "first_invalid_frame": None
            if valid
            else args.anchor_frame + first_invalid / 4.0,
            "invalid_reason": reason,
            "valid_frames": step / 4.0,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": peak_lift,
            "maximum_hand_supported_lift_m": peak_supported_lift,
            "sustained_hand_supported_lift_frames": longest,
            "hand_mediated_path_length_m": hand_transport,
            "goal_progress_m": goal_progress,
            "final_goal_position_error_m": goal_error,
            "final_root_height_m": root,
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "object_impulse_ns": impulses,
            "manipulation_contract_partial": contract,
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    rng = np.random.default_rng(8123)
    strategy_mean = np.asarray(
        args.initial
        if args.initial is not None
        else [0.16, 0.10, 42.0, 0.04, 0.55, 0.0, 0.018]
    )
    mean = np.r_[strategy_mean, np.zeros(2 * len(balance))]
    std = np.r_[
        [0.09, 0.04, 8.0, 0.025, 0.16, 0.035, 0.010],
        np.full(2 * len(balance), 0.075),
    ]
    lower = np.r_[lower, np.full(2 * len(balance), -0.30)]
    upper = np.r_[upper, np.full(2 * len(balance), 0.30)]
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, (args.population, len(mean))), lower, upper
        )
        population[0] = best_parameters
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[: args.elites]]
        mean = 0.72 * elite.mean(0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(0) + 0.28 * std, (upper - lower) * 0.005)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, "PARAM", best_parameters.tolist(), json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        arm_actuators=arms,
        source_path=str(args.seed),
        **trace,
    )
    report = {
        "method": "Cartesian-feedback flat-hand lift/transport",
        "parameter_names": [
            "forward_m",
            "lift_m",
            "duration_frames",
            "release_m",
            "ik_gain",
            "lateral_m",
            "squeeze_m",
        ],
        "parameters": best_parameters.tolist(),
        "balance_actuators": balance.tolist(),
        "balance_knot_frames": balance_frames.tolist(),
        "population": args.population,
        "generations": args.generations,
        "rollouts": args.population * args.generations,
        "anchor_frame": args.anchor_frame,
        "seed_index": args.seed_index,
        "horizon_frames": horizon,
        "maximum_disallowed_impulse_ns": args.maximum_disallowed_impulse,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
