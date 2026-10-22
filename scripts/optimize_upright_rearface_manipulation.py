#!/usr/bin/env python3
"""Jointly optimize upright balance and rear-face flat-hand manipulation.

The object and floating base are never written after initialization.  Candidate
controls are ordinary position targets for the compiled G1 actuators and every
score comes from the resulting MuJoCo trajectory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

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
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--maximum-disallowed-impulse", type=float, default=0.50)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    seed = np.load(args.seed)
    stance = np.load(args.stance)
    anchor_qpos = seed["anchor_qpos"].copy()
    anchor_qvel = seed["anchor_qvel"].copy()
    arms = seed["arm_actuators"].astype(int)
    balance = seed["balance_actuators"].astype(int)
    arm_qpos = engine.actuator_qpos[arms]
    actuator_dofs = np.asarray(
        [model.jnt_dofadr[joint] for joint in model.actuator_trnid[:, 0]], dtype=int
    )
    arm_dofs = actuator_dofs[arms]
    arm_initial = anchor_qpos[arm_qpos].copy()
    arm_contact = seed["arm_target"].copy()
    base_target = stance["target"].copy()
    base_target[arms] = arm_initial
    object_geoms = set(engine.object_geoms)
    ground_z = float(anchor_qpos[engine.object_qpos + 2])
    goal_position = engine.reference_object_pos[-1].copy()
    goal_quaternion = engine.reference_object_quat[-1].copy()

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = anchor_qpos
    anchor.qvel[:] = anchor_qvel
    mujoco.mj_forward(model, anchor)
    initial_hands = np.asarray(
        [anchor.geom_xpos[geom].copy() for geom in engine.hand_geoms]
    )
    contact_state = mujoco.MjData(model)
    contact_state.qpos[:] = anchor_qpos
    contact_state.qvel[:] = anchor_qvel
    contact_state.qpos[arm_qpos] = arm_contact
    mujoco.mj_forward(model, contact_state)
    contact_hands = np.asarray(
        [contact_state.geom_xpos[geom].copy() for geom in engine.hand_geoms]
    )
    contact_rotations = np.asarray(
        [contact_state.geom_xmat[geom].reshape(3, 3).copy() for geom in engine.hand_geoms]
    )
    initial_object = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    forward = np.r_[goal_position[:2] - initial_object[:2], 0.0]
    forward /= np.linalg.norm(forward)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    initial_goal_error = float(np.linalg.norm(initial_object - goal_position))

    balance_frames = np.asarray([0.0, 32.0, 75.0, 135.0])

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

    def interpolate_knots(frame: float, values: np.ndarray) -> np.ndarray:
        knots = values.reshape(len(balance_frames), len(balance))
        if frame <= balance_frames[0]:
            return knots[0]
        if frame >= balance_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(balance_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - balance_frames[segment])
            / (balance_frames[segment + 1] - balance_frames[segment])
        )
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def desired_hands(frame: float, strategy: np.ndarray) -> np.ndarray:
        approach, forward_m, lift_m, push_duration, release_m, release_frame, side_m, _ = strategy
        approach_progress = smoothstep(frame / approach)
        desired = (1.0 - approach_progress) * initial_hands + approach_progress * contact_hands
        transport = smoothstep((frame - approach) / push_duration)
        desired += transport * (forward_m * forward + np.asarray([0.0, 0.0, lift_m]))
        desired += transport * side_m * lateral
        # Opening occurs only after the accelerating hand motion.  It is along
        # the box lateral direction, so it cannot propel the object by itself.
        opening = smoothstep((frame - release_frame) / 10.0) * release_m
        desired[0] += opening * lateral
        desired[1] -= opening * lateral
        return desired

    def cartesian_arm_target(
        data: mujoco.MjData,
        desired: np.ndarray,
        nominal: np.ndarray,
        gain: float,
    ) -> np.ndarray:
        jacobian = np.zeros((12, len(arms)))
        error = np.zeros(12)
        for hand_index, geom in enumerate(engine.hand_geoms):
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacp, jacr, int(geom))
            rows_p = slice(6 * hand_index, 6 * hand_index + 3)
            rows_r = slice(6 * hand_index + 3, 6 * hand_index + 6)
            jacobian[rows_p] = jacp[:, arm_dofs]
            jacobian[rows_r] = 0.18 * jacr[:, arm_dofs]
            error[rows_p] = desired[hand_index] - data.geom_xpos[geom]
            current_rotation = data.geom_xmat[geom].reshape(3, 3)
            error[rows_r] = 0.18 * Rotation.from_matrix(
                contact_rotations[hand_index] @ current_rotation.T
            ).as_rotvec()
        current = data.qpos[arm_qpos]
        posture = nominal[arms] - current
        system = jacobian.T @ jacobian + 0.018 * np.eye(len(arms))
        rhs = jacobian.T @ error + 0.018 * posture
        delta = np.linalg.solve(system, rhs)
        return current + np.clip(gain * delta, -0.085, 0.085)

    def orientation_error(q1: np.ndarray, q2: np.ndarray) -> float:
        # MuJoCo uses wxyz.
        r1 = Rotation.from_quat(np.r_[q1[1:], q1[0]])
        r2 = Rotation.from_quat(np.r_[q2[1:], q2[0]])
        return float((r1.inv() * r2).magnitude())

    def evaluate(parameters: np.ndarray, record: bool = False):
        strategy = parameters[:8]
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        impulses = {name: 0.0 for name in GROUPS}
        lift_contact_flags: list[bool] = []
        hand_frames = bilateral_frames = 0
        peak_lift = peak_supported_lift = 0.0
        lifted_transport = 0.0
        last_lifted_object = initial_object.copy()
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
                "object_quaternion",
                "root_height",
                "torso_tilt",
                "object_impulse",
                "hand_target",
            )
        }
        physics_ctrl = []
        held_target = base_target.copy()
        held_desired = initial_hands.copy()
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            if step % engine.config.substeps == 0:
                held_target = base_target.copy()
                held_target[balance] += interpolate_knots(frame, parameters[8:])
                held_desired = desired_hands(frame, strategy)
                held_target[arms] = cartesian_arm_target(
                    data, held_desired, held_target, strategy[7]
                )
                held_target = np.clip(
                    held_target,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1],
                )
            data.ctrl[:] = held_target
            physics_ctrl.append(held_target.copy())
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
            obj_q = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            lift = float(obj[2] - ground_z)
            hands = int(np.sum(physical["hand_contact"]))
            support = int(np.sum(physical["foot_contact"]))
            root_height = float(data.qpos[engine.robot_qpos + 2])
            goal_error = float(np.linalg.norm(obj - goal_position))
            peak_lift = max(peak_lift, lift)
            if hands:
                peak_supported_lift = max(peak_supported_lift, lift)
            if hands and lift >= 0.05:
                lifted_transport += float(np.linalg.norm(obj - last_lifted_object))
                last_lifted_object = obj.copy()
            cost += (
                120000.0 * max(0.0, 0.46 - root_height) ** 2
                + 25000.0 * max(0.0, physical["tilt"] - np.deg2rad(40.0)) ** 2
                + 7000.0 * float(support == 0)
                + 220.0 * float(support == 1)
                + 1300.0 * goal_error**2
                - 70000.0 * max(0.0, lift)
                + 4.0 * float(np.square(parameters[8:]).mean())
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[name]
                for name in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            )
            if disallowed > args.maximum_disallowed_impulse and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                lift_contact_flags.append(lift >= 0.05 and hands > 0)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(obj.copy())
                    trace["object_quaternion"].append(obj_q.copy())
                    trace["root_height"].append(root_height)
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(np.asarray(list(impulses.values())))
                    trace["hand_target"].append(held_desired.copy())
            if invalid is not None:
                first_invalid, reason = step, invalid
                break
        longest = max(
            (len(list(run)) for value, run in itertools.groupby(lift_contact_flags) if value),
            default=0,
        )
        disallowed = sum(
            impulses[name]
            for name in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
        )
        valid = first_invalid is None
        goal_progress = initial_goal_error - goal_error
        goal_orientation_error = orientation_error(obj_q, goal_quaternion)
        contract = longest >= 10 and lifted_transport >= 0.03 and disallowed <= args.maximum_disallowed_impulse
        at_goal = goal_error < 0.15 and goal_orientation_error < 0.35
        cost /= max(step + 1, 1)
        cost += 250.0 * impulses["wrist"]
        cost -= (
            600.0 * hand_frames
            + 1000.0 * bilateral_frames
            + 220000.0 * peak_supported_lift
            + 16000.0 * longest
            + 180000.0 * goal_progress
            + 160000.0 * lifted_transport
        )
        # A manipulation-contract trajectory always outranks a no-lift hold.
        # Within the contract, survival and then goal accuracy dominate.
        if valid and contract and at_goal:
            rank = (0, goal_error, goal_orientation_error, cost)
        elif valid and contract:
            rank = (1, goal_error, goal_orientation_error, -lifted_transport, cost)
        elif contract:
            rank = (2, -step, goal_error, goal_orientation_error, cost)
        elif valid:
            rank = (3, -longest, -peak_supported_lift, goal_error, cost)
        else:
            rank = (4, -min(longest, 10), -step, goal_error, cost)
        metrics = {
            "first_invalid_frame": None if valid else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": peak_lift,
            "maximum_hand_supported_lift_m": peak_supported_lift,
            "sustained_hand_supported_lift_frames": longest,
            "lifted_hand_mediated_transport_m": lifted_transport,
            "goal_progress_m": goal_progress,
            "final_goal_position_error_m": goal_error,
            "final_goal_orientation_error_rad": goal_orientation_error,
            "final_root_height_m": root_height,
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "object_impulse_ns": impulses,
            "manipulation_contract": bool(contract),
            "at_goal": bool(at_goal),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(physics_ctrl)

    strategy_mean = np.asarray([30.0, 0.34, 0.11, 28.0, 0.07, 68.0, 0.0, 0.65])
    strategy_std = np.asarray([5.0, 0.12, 0.04, 7.0, 0.025, 10.0, 0.025, 0.14])
    strategy_lower = np.asarray([18.0, 0.08, 0.03, 14.0, 0.0, 42.0, -0.08, 0.20])
    strategy_upper = np.asarray([44.0, 0.68, 0.24, 55.0, 0.16, 115.0, 0.08, 1.20])
    mean = np.r_[strategy_mean, np.zeros(len(balance_frames) * len(balance))]
    std = np.r_[strategy_std, np.full(len(balance_frames) * len(balance), 0.14)]
    lower = np.r_[strategy_lower, np.full(len(balance_frames) * len(balance), -0.48)]
    upper = np.r_[strategy_upper, np.full(len(balance_frames) * len(balance), 0.48)]
    rng = np.random.default_rng(9321)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(
            rng.normal(mean, std, size=(args.population, len(mean))), lower, upper
        )
        population[0] = best_parameters
        if generation == 0 and len(population) > 1:
            population[1] = mean
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda index: results[index][0])
        if results[order[0]][0] < best[0]:
            best_parameters = population[order[0]].copy()
            best = results[order[0]]
        elites = population[order[: args.elites]]
        mean = 0.76 * elites.mean(axis=0) + 0.24 * mean
        std = np.maximum(0.76 * elites.std(axis=0) + 0.24 * std, (upper - lower) * 0.006)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, "PARAM", best_parameters.tolist(), json.dumps(best[1]), flush=True)

    _, metrics, trace, physics_ctrl = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        base_target=base_target,
        arm_actuators=arms,
        balance_actuators=balance,
        balance_knot_frames=balance_frames,
        ctrl_physics=physics_ctrl,
        **trace,
    )
    report = {
        "method": "rear-face flat-hand manipulation with joint balance/Cartesian hand feedback",
        "strategy_parameter_names": [
            "approach_frames",
            "forward_m",
            "lift_m",
            "push_duration_frames",
            "release_m",
            "release_frame",
            "lateral_m",
            "ik_gain",
        ],
        "dimensions": len(mean),
        "balance_knot_frames": balance_frames.tolist(),
        "population": args.population,
        "generations": args.generations,
        "rollouts": args.population * args.generations,
        "horizon_frames": args.horizon,
        "runtime_s": time.perf_counter() - started,
        "parameters": best_parameters.tolist(),
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
