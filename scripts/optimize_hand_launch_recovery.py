#!/usr/bin/env python3
"""Optimize a hand-only launch and whole-body recovery from a proven lift.

The prefix has already established the sustained-lift contract.  This local
policy uses a low-dimensional Cartesian hand push/release and blends the legal
position-actuator targets toward a physically validated upright stance.  The
object and floating base are always advanced only by MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from probe_flat_hand_grasp import build, smoothstep

GROUPS = ("left_hand", "right_hand", "wrist", "feet", "legs", "torso_pelvis", "other_robot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--anchor-index", type=int, default=53)
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=28)
    parser.add_argument("--initial-std-scale", type=float, default=0.20)
    parser.add_argument(
        "--optimize-balance-residual",
        action="store_true",
        help="Append three smooth 15-joint balance target-residual knots.",
    )
    parser.add_argument("--replay-full-prefix", action="store_true")
    parser.add_argument("--launch-start", type=float, default=0.0)
    parser.add_argument(
        "--arm-residual-bound",
        type=float,
        default=0.11,
        help="Per-update arm target-residual bound in radians.",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Prior launch artifact whose 12 search parameters seed CEM.",
    )
    parser.add_argument(
        "--warm-parameters",
        type=float,
        nargs=15,
        help="Explicit 15-parameter warm start (useful for resuming an interrupted search).",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    prefix = np.load(args.prefix)
    stance = np.load(args.stance)
    i0 = args.anchor_index
    if args.replay_full_prefix:
        anchor_qpos = prefix["anchor_qpos"].copy()
        anchor_qvel = prefix["anchor_qvel"].copy()
        base_control = prefix["ctrl"][0].copy()
    else:
        anchor_qpos = prefix["qpos"][i0 - 1].copy()
        anchor_qvel = prefix["qvel"][i0 - 1].copy()
        base_control = prefix["ctrl"][i0].copy()
    stance_target = stance["target"].copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    actuator_dofs = np.asarray([model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0]], dtype=int)
    arm_dofs = actuator_dofs[arms]
    balance_dofs = actuator_dofs[balance]
    balance_qpos = engine.actuator_qpos[balance]
    object_geoms = set(map(int, engine.object_geoms))
    goal_position = engine.reference_object_pos[-1]
    goal_quaternion = engine.reference_object_quat[-1]
    ground_z = 0.18545216

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = anchor_qpos
    anchor.qvel[:] = anchor_qvel
    mujoco.mj_forward(model, anchor)
    initial_hands = np.asarray([anchor.geom_xpos[g].copy() for g in engine.hand_geoms])
    initial_object = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    direction = np.r_[goal_position[:2] - initial_object[:2], 0.0]
    direction /= max(np.linalg.norm(direction), 1.0e-9)
    lateral = np.asarray([-direction[1], direction[0], 0.0])
    source_hands = []
    if args.replay_full_prefix:
        kinematic = mujoco.MjData(model)
        for qpos in prefix["qpos"]:
            kinematic.qpos[:] = qpos
            kinematic.qvel[:] = 0.0
            mujoco.mj_forward(model, kinematic)
            source_hands.append(
                np.asarray([kinematic.geom_xpos[g].copy() for g in engine.hand_geoms])
            )
        source_hands = np.asarray(source_hands)

    def contact_group(geom: int) -> str:
        if geom == int(engine.hand_geoms[0]):
            return "left_hand"
        if geom == int(engine.hand_geoms[1]):
            return "right_hand"
        body = int(model.geom_bodyid[geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if "wrist" in body_name:
            return "wrist"
        if "foot" in body_name or "ankle" in body_name:
            return "feet"
        if any(token in body_name for token in ("knee", "shin", "hip", "thigh")):
            return "legs"
        if any(token in body_name for token in ("torso", "pelvis", "waist")):
            return "torso_pelvis"
        return "other_robot"

    def arm_residual(data: mujoco.MjData, desired: np.ndarray, gain: float) -> np.ndarray:
        jacobian = np.zeros((6, len(arms)))
        error = np.zeros(6)
        for hand_index, geom in enumerate(engine.hand_geoms):
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacp, jacr, int(geom))
            rows = slice(3 * hand_index, 3 * hand_index + 3)
            jacobian[rows] = jacp[:, arm_dofs]
            error[rows] = desired[hand_index] - data.geom_xpos[geom]
        delta = np.linalg.solve(
            jacobian.T @ jacobian + 0.004 * np.eye(len(arms)),
            jacobian.T @ error,
        )
        return np.clip(
            gain * delta,
            -args.arm_residual_bound,
            args.arm_residual_bound,
        )

    macro_dimensions = 15

    def balance_residual_at(parameters: np.ndarray, frame: float) -> np.ndarray:
        if not args.optimize_balance_residual:
            return np.zeros(len(balance))
        free = parameters[macro_dimensions:].reshape(3, len(balance))
        frames = np.asarray(
            [args.launch_start, args.launch_start + 5.0,
             args.launch_start + 20.0, args.launch_start + 40.0,
             float(args.horizon)]
        )
        values = np.vstack([np.zeros(len(balance)), free, np.zeros(len(balance))])
        if frame <= frames[0] or frame >= frames[-1]:
            return np.zeros(len(balance))
        segment = int(np.searchsorted(frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - frames[segment]) / (frames[segment + 1] - frames[segment])
        )
        return (1.0 - amount) * values[segment] + amount * values[segment + 1]

    def evaluate(parameters: np.ndarray, record: bool = False):
        (
            push_distance,
            rise,
            push_duration,
            squeeze,
            release_spread,
            balance_blend,
            recovery_start,
            recovery_duration,
            position_gain,
            velocity_gain,
            arm_gain,
            retract_duration,
            yaw_skew,
            arm_time_scale,
            hold_duration,
        ) = parameters[:macro_dimensions]
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        first_invalid = reason = None
        peak_lift = 0.0
        hand_frames = bilateral_frames = 0
        maximum_progress = 0.0
        lift_flags: list[bool] = []
        supported_lift_area = 0.0
        hand_path = 0.0
        lifted_hand_path = 0.0
        last_hand_object = initial_object.copy()
        last_lifted_hand_object = None
        initial_error = float(np.linalg.norm(initial_object - goal_position))
        trace = {key: [] for key in ("qpos", "qvel", "ctrl", "actuator_force", "hand_contact", "foot_contact", "object_position", "root_height", "torso_tilt", "object_impulse")}
        held_target = base_control.copy()
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            phase = frame - args.launch_start
            push = smoothstep(phase / push_duration)
            recover = balance_blend * smoothstep((phase - recovery_start) / recovery_duration)
            if args.replay_full_prefix:
                source_index = min(int(frame), len(prefix["ctrl"]) - 1)
                if "ctrl_physics" in prefix:
                    source_control = prefix["ctrl_physics"][
                        min(step, len(prefix["ctrl_physics"]) - 1)
                    ].copy()
                else:
                    source_control = prefix["ctrl"][source_index].copy()
                arm_source_index = min(
                    int(frame), len(prefix["ctrl"]) - 1
                )
            else:
                source_index = 0
                arm_source_index = 0
                source_control = base_control
            nominal = (1.0 - recover) * source_control + recover * stance_target
            nominal[balance] += balance_residual_at(parameters, frame)
            if phase >= 0.0:
                # This is a residual feedback target, not a second unbounded
                # proportional controller.  Without the bound, state error
                # close to a fall can drive position targets several radians
                # away and create a positive-feedback collapse even though
                # actuator torques remain legal.  Keep the nominal trajectory
                # as the center and limit online correction authority.
                feedback = position_gain * (
                    nominal[balance] - data.qpos[balance_qpos]
                ) - velocity_gain * data.qvel[balance_dofs]
                nominal[balance] += np.clip(feedback, -0.18, 0.18)
            if phase < 0.0:
                desired = source_hands[arm_source_index] if args.replay_full_prefix else initial_hands
            elif phase <= push_duration:
                hand_origin = source_hands[arm_source_index] if args.replay_full_prefix else initial_hands
                desired = hand_origin + push * (
                    push_distance * direction + np.asarray([0.0, 0.0, rise])
                )
                desired[0] -= push * squeeze * lateral
                desired[1] += push * squeeze * lateral
                # Differential motion along the transport direction supplies
                # yaw authority while both flat hands remain the only task-
                # driving contacts.
                desired[0] += push * yaw_skew * direction
                desired[1] -= push * yaw_skew * direction
                nominal[arms] = source_control[arms] + arm_residual(data, desired, arm_gain)
            else:
                release = smoothstep(
                    (phase - push_duration - hold_duration) / retract_duration
                )
                hand_origin = source_hands[arm_source_index] if args.replay_full_prefix else initial_hands
                desired = hand_origin + push_distance * direction + np.asarray([0.0, 0.0, rise])
                desired[0] += yaw_skew * direction
                desired[1] -= yaw_skew * direction
                desired[0] += release * release_spread * lateral
                desired[1] -= release * release_spread * lateral
                push_target = source_control[arms] + arm_residual(data, desired, arm_gain)
                if arm_gain <= 1.0e-9:
                    # Exact zero-residual baseline: replay the validated arm
                    # targets without an implicit late blend to another pose.
                    nominal[arms] = source_control[arms]
                else:
                    nominal[arms] = (1.0 - release) * push_target + release * stance_target[arms]
            if (
                args.replay_full_prefix
                and "ctrl_physics" in prefix
            ) or step % engine.config.substeps == 0:
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
            hands_now = int(np.sum(physical["hand_contact"]))
            if hands_now:
                hand_path += float(np.linalg.norm(obj - last_hand_object))
                last_hand_object = obj.copy()
                supported_lift_area += max(0.0, lift - 0.04) * model.opt.timestep
                if lift >= 0.05:
                    if last_lifted_hand_object is not None:
                        lifted_hand_path += float(
                            np.linalg.norm(obj - last_lifted_hand_object)
                        )
                    last_lifted_hand_object = obj.copy()
                else:
                    last_lifted_hand_object = None
            progress = initial_error - float(np.linalg.norm(obj - goal_position))
            maximum_progress = max(maximum_progress, progress)
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot"))
            if disallowed > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hands = int(np.sum(physical["hand_contact"]))
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
        position_error = float(np.linalg.norm(data.qpos[engine.object_qpos : engine.object_qpos + 3] - goal_position))
        quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
        orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(quaternion, goal_quaternion)), 0.0, 1.0)))
        speed = float(np.linalg.norm(data.qvel[engine.object_dof : engine.object_dof + 6]))
        root_vertical_velocity = float(data.qvel[engine.robot_dof + 2])
        sustained_lift = max(
            (
                len(list(run))
                for value, run in __import__("itertools").groupby(lift_flags)
                if value
            ),
            default=0,
        )
        valid = first_invalid is None
        at_goal = position_error < 0.15 and orientation_error < 0.35
        contract = sustained_lift >= 10 and lifted_hand_path >= 0.03
        task_cost = 2500.0 * position_error**2 + 400.0 * orientation_error**2 + 40.0 * speed**2 - 1000.0 * maximum_progress
        recoverable = (
            valid
            and float(data.qpos[engine.robot_qpos + 2]) >= 0.50
            and float(physical["tilt"]) < np.deg2rad(40.0)
            and root_vertical_velocity > -0.35
            and int(np.sum(physical["foot_contact"])) > 0
        )
        if valid and at_goal and contract and recoverable:
            rank = (0, task_cost)
        elif valid and contract and recoverable:
            rank = (1, position_error, orientation_error, task_cost)
        elif valid and contract:
            # Placement is the active problem once an uninterrupted rollout
            # already satisfies the lift/hand-transport contract.  Recovery
            # remains in the score, but must not lexicographically hide metre-
            # scale object-goal improvements behind millimetres of root state.
            recovery_penalty = (
                8.0 * max(0.0, 0.50 - float(data.qpos[engine.robot_qpos + 2])) ** 2
                + 2.0 * max(0.0, float(physical["tilt"]) - np.deg2rad(35.0)) ** 2
                + 2.0 * max(0.0, -0.10 - root_vertical_velocity) ** 2
            )
            rank = (2, position_error, orientation_error, recovery_penalty, task_cost)
        elif contract:
            # Once an invalid candidate has already achieved the manipulation
            # contract, survival is the dominant learning signal.  The old
            # ordering preferred a few millimetres of extra carried path over
            # delaying a fall, which trapped CEM at the same torso-tilt event.
            rank = (3, -step, position_error, orientation_error, -lifted_hand_path, task_cost)
        elif valid:
            # A full-horizon rollout that silently drops below the required
            # five-centimetre lift is not a manipulation solution.  Keep it
            # behind every candidate that still preserves the already-proven
            # lift/transport contract, even if that candidate needs further
            # recovery work.
            rank = (4, -sustained_lift, -lifted_hand_path, -supported_lift_area, -peak_lift, position_error, task_cost)
        else:
            rank = (5, -min(sustained_lift, 10), -lifted_hand_path, -supported_lift_area, -peak_lift, -step, position_error, task_cost)
        metrics = {
            "first_invalid_local_frame": None if valid else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
            "maximum_lift_m": peak_lift,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "sustained_hand_supported_lift_frames": sustained_lift,
            "supported_lift_area_m_s": supported_lift_area,
            "hand_mediated_path_length_m": hand_path,
            "lifted_hand_mediated_path_length_m": lifted_hand_path,
            "maximum_goal_progress_m": maximum_progress,
            "final_goal_position_error_m": position_error,
            "final_goal_orientation_error_rad": orientation_error,
            "final_object_speed": speed,
            "at_goal": at_goal,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_root_vertical_velocity_m_s": root_vertical_velocity,
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "recoverable_terminal": bool(recoverable),
            "object_impulse_ns": impulses,
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    lower = np.asarray([0.0, -0.02, 6.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 5.0, -0.18, 1.0, 6.0])
    upper = np.asarray([0.65, 0.14, 20.0, 0.070, 0.18, 1.0, 20.0, 70.0, 1.5, 0.18, 1.3, 35.0, 0.18, 1.5, 18.0])
    mean = np.asarray([0.28, 0.08, 14.0, 0.025, 0.08, 0.8, 4.0, 30.0, 0.5, 0.03, 0.7, 15.0, 0.0, 1.0, 10.0])
    if args.optimize_balance_residual:
        balance_bound = np.full(3 * len(balance), 0.25)
        lower = np.r_[lower, -balance_bound]
        upper = np.r_[upper, balance_bound]
        mean = np.r_[mean, np.zeros_like(balance_bound)]
    if args.warm_start is not None:
        warm = np.load(args.warm_start)
        if "parameters" not in warm:
            raise ValueError(f"{args.warm_start} does not contain 'parameters'")
        warm_parameters = warm["parameters"].astype(float)
        if warm_parameters.shape == (12,):
            warm_parameters = np.r_[warm_parameters, 0.0, 1.0, 10.0]
        elif warm_parameters.shape == (13,):
            warm_parameters = np.r_[warm_parameters, 1.0, 10.0]
        elif warm_parameters.shape == (14,):
            warm_parameters = np.r_[warm_parameters, 10.0]
        if args.optimize_balance_residual and warm_parameters.shape == (macro_dimensions,):
            warm_parameters = np.r_[warm_parameters, np.zeros(3 * len(balance))]
        if warm_parameters.shape != mean.shape:
            raise ValueError(
                f"{args.warm_start} parameters {warm_parameters.shape} do not match {mean.shape}"
            )
        mean = np.clip(warm_parameters, lower, upper)
    if args.warm_parameters is not None:
        explicit = np.asarray(args.warm_parameters, dtype=float)
        if args.optimize_balance_residual:
            explicit = np.r_[explicit, np.zeros(3 * len(balance))]
        mean = np.clip(explicit, lower, upper)
    std = args.initial_std_scale * (upper - lower)
    if args.optimize_balance_residual:
        std[macro_dimensions:] = 0.035
    rng = np.random.default_rng(3301)
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, len(mean))), lower, upper)
        population[0] = best_parameters
        # Always retain the exact physically validated source rollout.  This
        # makes the placement search monotone in physical feasibility rather
        # than forcing every generation to perturb an already successful
        # contact trajectory.
        if len(population) > 1:
            baseline = np.asarray(
                [0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 4.0, 30.0,
                 0.0, 0.0, 0.0, 15.0, 0.0, 1.0, 10.0]
            )
            if args.optimize_balance_residual:
                baseline = np.r_[baseline, np.zeros(3 * len(balance))]
            population[1] = baseline
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda i: results[i][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[: args.elites]]
        mean = 0.72 * elite.mean(0) + 0.28 * mean
        std = np.maximum(0.72 * elite.std(0) + 0.28 * std, 0.008 * (upper - lower))
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, "PARAM", best_parameters.tolist(), json.dumps(best[1]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output.with_name(args.output.name + "_checkpoint").with_suffix(".npz"),
            parameters=best_parameters,
            generation=generation,
        )
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_anchor_frame = int(prefix["anchor_frame"]) if "anchor_frame" in prefix else 210
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        anchor_index=i0,
        anchor_frame=source_anchor_frame,
        **trace,
    )
    report = {"method": "hand-only launch plus structured stance recovery", "replay_full_prefix": args.replay_full_prefix, "launch_start": args.launch_start, "arm_residual_bound": args.arm_residual_bound, "optimize_balance_residual": args.optimize_balance_residual, "dimensions": len(mean), "population": args.population, "generations": args.generations, "rollouts": args.population * args.generations, "runtime_s": time.perf_counter() - started, "best": metrics, "history": history}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
