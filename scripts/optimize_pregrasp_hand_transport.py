#!/usr/bin/env python3
"""Optimize a hand-only grasp/lift/transport from a stable upright pregrasp.

The floating base and object are initialized once and subsequently evolve only
through MuJoCo.  The kinematic grasp family supplies a warm motion primitive;
all success metrics and candidate ranking come from the physical rollout.
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


GROUPS = ("left_hand", "right_hand", "wrist", "feet", "legs", "torso_pelvis", "other_robot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--grasp-family", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6409)
    parser.add_argument("--maximum-disallowed-impulse", type=float, default=2.0)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    family = np.load(args.grasp_family)
    initial_qpos = stance["initial_qpos"].copy()
    initial_qvel = stance["initial_qvel"].copy()
    base_target = stance["target"].copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    actuator_dofs = np.asarray([model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0]], dtype=int)
    arm_dofs = actuator_dofs[arms]
    object_geoms = set(map(int, engine.object_geoms))
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    ground_z = 0.18545216
    family_targets = {
        key: np.asarray(family[key][6:], dtype=float)[arms]
        for key in ("pregrasp", "grasp", "squeeze", "lift")
    }

    pose = mujoco.MjData(model)
    pose.qpos[:] = initial_qpos
    pose.qvel[:] = 0.0
    pose.qpos[engine.actuator_qpos[arms]] = family_targets["lift"]
    mujoco.mj_forward(model, pose)
    lift_hands = np.asarray([pose.geom_xpos[g].copy() for g in engine.hand_geoms])
    initial_object = initial_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    direction = np.r_[goal_position[:2] - initial_object[:2], 0.0]
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    lateral = np.asarray([-direction[1], direction[0], 0.0])

    def contact_group(geom: int) -> str:
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

    def cartesian_arm_target(data: mujoco.MjData, desired: np.ndarray, nominal: np.ndarray, gain: float) -> np.ndarray:
        jacobian = np.zeros((6, len(arms)))
        error = np.zeros(6)
        for hand_index, geom in enumerate(engine.hand_geoms):
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacp, jacr, int(geom))
            rows = slice(3 * hand_index, 3 * hand_index + 3)
            jacobian[rows] = jacp[:, arm_dofs]
            error[rows] = desired[hand_index] - data.geom_xpos[geom]
        current = data.qpos[engine.actuator_qpos[arms]]
        posture = nominal - current
        system = jacobian.T @ jacobian + 0.018 * np.eye(len(arms))
        rhs = jacobian.T @ error + 0.012 * posture
        delta = np.linalg.solve(system, rhs)
        return current + np.clip(gain * delta, -0.11, 0.11)

    def evaluate(parameters: np.ndarray, record: bool = False):
        (
            wait,
            approach_duration,
            squeeze_duration,
            lift_duration,
            hold_duration,
            push_distance,
            push_rise,
            push_duration,
            yaw_skew,
            release_spread,
            release_duration,
            arm_gain,
        ) = parameters
        approach_end = wait + approach_duration
        squeeze_end = approach_end + squeeze_duration
        lift_end = squeeze_end + lift_duration
        push_start = lift_end + hold_duration
        push_end = push_start + push_duration
        release_end = push_end + release_duration

        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = base_target
        mujoco.mj_forward(model, data)
        impulses = {name: 0.0 for name in GROUPS}
        lift_flags: list[bool] = []
        hand_frames = bilateral_frames = 0
        peak_lift = 0.0
        lifted_hand_path = 0.0
        hand_path = 0.0
        last_hand_object = initial_object.copy()
        last_lifted = None
        initial_goal_error = float(np.linalg.norm(initial_object - goal_position))
        maximum_progress = 0.0
        first_invalid = reason = None
        trace = {key: [] for key in ("qpos", "qvel", "ctrl", "actuator_force", "hand_contact", "foot_contact", "object_position", "root_height", "torso_tilt", "object_impulse")}
        held_target = base_target.copy()
        for step in range(300 * engine.config.substeps):
            frame = step / engine.config.substeps
            nominal = base_target.copy()
            if frame < wait:
                nominal[arms] = family_targets["pregrasp"]
            elif frame < approach_end:
                amount = smoothstep((frame - wait) / approach_duration)
                nominal[arms] = (1.0 - amount) * family_targets["pregrasp"] + amount * family_targets["grasp"]
            elif frame < squeeze_end:
                amount = smoothstep((frame - approach_end) / squeeze_duration)
                nominal[arms] = (1.0 - amount) * family_targets["grasp"] + amount * family_targets["squeeze"]
            elif frame < lift_end:
                amount = smoothstep((frame - squeeze_end) / lift_duration)
                nominal[arms] = (1.0 - amount) * family_targets["squeeze"] + amount * family_targets["lift"]
            elif frame < push_start:
                nominal[arms] = family_targets["lift"]
            elif frame < release_end:
                push = smoothstep((frame - push_start) / push_duration)
                release = smoothstep((frame - push_end) / release_duration)
                desired = lift_hands + push * (push_distance * direction + np.asarray([0.0, 0.0, push_rise]))
                desired[0] += push * yaw_skew * direction
                desired[1] -= push * yaw_skew * direction
                desired[0] += release * release_spread * lateral
                desired[1] -= release * release_spread * lateral
                cartesian = cartesian_arm_target(data, desired, family_targets["lift"], arm_gain)
                nominal[arms] = (1.0 - release) * cartesian + release * base_target[arms]
            else:
                nominal[arms] = base_target[arms]
            if step % engine.config.substeps == 0:
                held_target = np.clip(nominal, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = held_target
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for ci in range(data.ncon):
                contact = data.contact[ci]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                impulses[contact_group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            lift = float(obj[2] - ground_z)
            peak_lift = max(peak_lift, lift)
            hands = int(np.sum(physical["hand_contact"]))
            if hands:
                increment = float(np.linalg.norm(obj - last_hand_object))
                hand_path += increment
                last_hand_object = obj.copy()
                if lift >= 0.05:
                    if last_lifted is not None:
                        lifted_hand_path += float(np.linalg.norm(obj - last_lifted))
                    last_lifted = obj.copy()
                else:
                    last_lifted = None
            maximum_progress = max(maximum_progress, initial_goal_error - float(np.linalg.norm(obj - goal_position)))
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot"))
            if disallowed > args.maximum_disallowed_impulse and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                lift_flags.append(lift >= 0.05 and hands > 0)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(obj.copy())
                    trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(np.asarray(list(impulses.values())))
            if invalid is not None:
                first_invalid, reason = step, invalid
                break

        sustained = max((len(list(run)) for value, run in itertools.groupby(lift_flags) if value), default=0)
        position_error = float(np.linalg.norm(data.qpos[engine.object_qpos : engine.object_qpos + 3] - goal_position))
        quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
        orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(quaternion, goal_quaternion)), 0.0, 1.0)))
        valid = first_invalid is None
        contract = sustained >= 10 and lifted_hand_path >= 0.03
        at_goal = position_error < 0.15 and orientation_error < 0.35
        recoverable = valid and data.qpos[engine.robot_qpos + 2] >= 0.50 and physical["tilt"] < np.deg2rad(35.0) and int(np.sum(physical["foot_contact"])) > 0
        soft = position_error + 0.25 * orientation_error - 0.8 * maximum_progress
        if valid and contract and at_goal and recoverable:
            rank = (0, soft)
        elif valid and contract:
            rank = (1, position_error, orientation_error, -lifted_hand_path)
        elif valid:
            rank = (2, -min(sustained, 10), -lifted_hand_path, -hand_frames, position_error)
        elif contract:
            rank = (3, -step, position_error, orientation_error)
        else:
            rank = (4, -min(sustained, 10), -lifted_hand_path, -hand_frames, -step, position_error)
        metrics = {
            "first_invalid_frame": None if valid else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": peak_lift,
            "sustained_hand_supported_lift_frames": sustained,
            "hand_mediated_path_m": hand_path,
            "lifted_hand_mediated_path_m": lifted_hand_path,
            "maximum_goal_progress_m": maximum_progress,
            "final_goal_position_error_m": position_error,
            "final_goal_orientation_error_rad": orientation_error,
            "at_goal": at_goal,
            "manipulation_contract": contract,
            "recoverable_terminal": recoverable,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "object_impulse_ns": impulses,
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    # The generated object begins slightly above its settled support height.
    # Reach the bilateral grasp before that short drop can brush the nearby
    # legs; subsequent phases retain broad timing freedom.
    lower = np.asarray([0.0, 1.0, 1.0, 3.0, 0.0, 0.08, -0.03, 5.0, -0.22, 0.0, 5.0, 0.15])
    upper = np.asarray([2.0, 7.0, 8.0, 18.0, 16.0, 0.70, 0.22, 24.0, 0.22, 0.20, 30.0, 1.30])
    mean = np.asarray([0.0, 3.0, 3.0, 8.0, 6.0, 0.32, 0.10, 12.0, 0.0, 0.08, 12.0, 0.65])
    std = 0.20 * (upper - lower)
    rng = np.random.default_rng(args.seed)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, len(mean))), lower, upper)
        population[0] = best_parameters
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda i: results[i][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[: args.elites]]
        mean = 0.72 * elite.mean(0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(0) + 0.28 * std, 0.006 * (upper - lower))
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, "PARAM", best_parameters.tolist(), json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), parameters=best_parameters, initial_qpos=initial_qpos, initial_qvel=initial_qvel, base_target=base_target, **trace)
    report = {"method": "stable-pregrasp hand-only grasp/lift/transport", "parameters": best_parameters.tolist(), "population": args.population, "generations": args.generations, "rollouts": args.population * args.generations, "runtime_s": time.perf_counter() - started, "best": metrics, "history": history}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
