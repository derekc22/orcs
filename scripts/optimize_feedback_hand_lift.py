#!/usr/bin/env python3
"""Structured state-feedback refinement of a physical bilateral hand lift.

This starts from an actual stable bilateral-contact state.  MuJoCo remains the
sole dynamics/contact model.  The optimized policy combines a Cartesian palm
motion with state-dependent balance-actuator targets, avoiding a high-
dimensional open-loop knot search.
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--replay-source-arms", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--secondary-warm-start", type=Path)
    parser.add_argument(
        "--freeze-core",
        action="store_true",
        help="Hold the first 28 lift/controller parameters at the primary warm start and optimize only root-state feedback.",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    anchor_index = args.anchor_frame
    anchor_qpos = source["qpos"][anchor_index].copy()
    anchor_qvel = source["qvel"][anchor_index].copy()
    base_control = source["ctrl"][anchor_index].copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    actuator_dofs = np.asarray([model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0]], dtype=int)
    arm_dofs = actuator_dofs[arms]
    balance_dofs = actuator_dofs[balance]
    balance_qpos = engine.actuator_qpos[balance]
    object_geoms = set(map(int, engine.object_geoms))
    goal = engine.reference_object_pos[-1]
    ground_z = 0.18545216

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = anchor_qpos
    anchor.qvel[:] = anchor_qvel
    mujoco.mj_forward(model, anchor)
    initial_hands = np.asarray([anchor.geom_xpos[g].copy() for g in engine.hand_geoms])
    initial_object = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    forward = np.r_[goal[:2] - initial_object[:2], 0.0]
    forward /= max(np.linalg.norm(forward), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    source_hands = []
    source_kinematics = mujoco.MjData(model)
    for qpos in source["qpos"]:
        source_kinematics.qpos[:] = qpos
        source_kinematics.qvel[:] = 0.0
        mujoco.mj_forward(model, source_kinematics)
        source_hands.append(
            np.asarray(
                [source_kinematics.geom_xpos[g].copy() for g in engine.hand_geoms]
            )
        )
    source_hands = np.asarray(source_hands)

    # Identify a physically meaningful balance-control direction rather than
    # asking CEM to discover it in 15 dimensions.  The finite-difference
    # sensitivity measures the next control interval's vertical velocity.
    def next_vertical_velocity(control: np.ndarray) -> float:
        probe = mujoco.MjData(model)
        probe.qpos[:] = anchor_qpos
        probe.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, probe)
        for _ in range(engine.config.substeps):
            probe.ctrl[:] = control
            mujoco.mj_step(model, probe)
        return float(probe.qvel[engine.robot_dof + 2])

    baseline_vertical_velocity = next_vertical_velocity(base_control)
    vertical_sensitivity = np.empty(len(balance))
    for local, actuator in enumerate(balance):
        perturbed = base_control.copy()
        perturbed[actuator] += 0.02
        vertical_sensitivity[local] = (
            next_vertical_velocity(perturbed) - baseline_vertical_velocity
        ) / 0.02
    vertical_direction = vertical_sensitivity / max(
        float(np.max(np.abs(vertical_sensitivity))), 1.0e-9
    )
    desired_root_height = float(anchor_qpos[engine.robot_qpos + 2])

    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    group_index = np.zeros(len(balance), dtype=int)
    for i, actuator in enumerate(balance):
        joint = names[actuator]
        if "hip_pitch" in joint:
            group_index[i] = 0
        elif "knee" in joint:
            group_index[i] = 1
        elif "ankle_pitch" in joint:
            group_index[i] = 2
        else:
            group_index[i] = 3

    def group(geom: int) -> str:
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

    def arm_target(data: mujoco.MjData, desired: np.ndarray, nominal: np.ndarray, gain: float) -> np.ndarray:
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
        posture = nominal[arms] - current
        system = jacobian.T @ jacobian + 0.022 * np.eye(len(arms))
        delta = np.linalg.solve(system, jacobian.T @ error + 0.02 * posture)
        # Retain the physically tested source feedforward and apply Cartesian
        # feedback only as a conservative residual.  Replacing the target by
        # ``current + delta`` removed the squeeze-supporting feedforward and
        # caused immediate root collapse.
        return nominal[arms] + np.clip(gain * delta, -0.035, 0.035)

    def evaluate(parameters: np.ndarray, record: bool = False):
        # strategy: lift, forward, duration, squeeze, arm gain; then 15 target
        # biases, four q/qdot feedback gains, and measured vertical-state gains.
        lift, advance, duration, squeeze, arm_gain = parameters[:5]
        biases = parameters[5:20]
        position_gain = parameters[20:24]
        velocity_gain = parameters[24:28]
        root_height_gain, root_velocity_gain = parameters[28:30]
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        lift_flags: list[bool] = []
        hand_frames = bilateral_frames = 0
        first_invalid = reason = None
        peak_lift = peak_supported_lift = 0.0
        hand_path = 0.0
        last_hand_object = initial_object.copy()
        trace = {key: [] for key in ("qpos", "qvel", "ctrl", "actuator_force", "hand_contact", "foot_contact", "object_position", "root_height", "torso_tilt", "object_impulse")}
        cost = 0.0
        held_target = base_control.copy()
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            progress = smoothstep(frame / duration)
            source_index = min(anchor_index + int(frame), len(source["ctrl"]) - 1)
            desired = source_hands[source_index] + progress * (
                advance * forward + np.asarray([0.0, 0.0, lift])
            )
            desired[0] -= progress * squeeze * lateral
            desired[1] += progress * squeeze * lateral
            nominal = source["ctrl"][source_index].copy()
            balance_reference = source["qpos"][source_index, balance_qpos]
            current_q = data.qpos[balance_qpos]
            current_v = data.qvel[balance_dofs]
            nominal[balance] += progress * biases
            nominal[balance] += position_gain[group_index] * (balance_reference - current_q)
            nominal[balance] -= velocity_gain[group_index] * current_v
            root_height_error = desired_root_height - float(
                data.qpos[engine.robot_qpos + 2]
            )
            vertical_velocity = float(data.qvel[engine.robot_dof + 2])
            nominal[balance] += vertical_direction * (
                root_height_gain * root_height_error
                - root_velocity_gain * vertical_velocity
            )
            if step % engine.config.substeps == 0 and not args.replay_source_arms:
                nominal[arms] = arm_target(data, desired, nominal, arm_gain)
                held_target = np.clip(nominal, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            elif step % engine.config.substeps == 0:
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
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            height = float(obj[2] - ground_z)
            hands = int(np.sum(physical["hand_contact"]))
            support = int(np.sum(physical["foot_contact"]))
            root = float(data.qpos[engine.robot_qpos + 2])
            peak_lift = max(peak_lift, height)
            if hands:
                peak_supported_lift = max(peak_supported_lift, height)
                hand_path += float(np.linalg.norm(obj - last_hand_object))
                last_hand_object = obj.copy()
            cost += (
                150000.0 * max(0.0, 0.52 - root) ** 2
                + 50000.0 * max(0.0, physical["tilt"] - np.deg2rad(45.0)) ** 2
                + 7000.0 * float(support == 0)
                + 300.0 * float(support == 1)
                + 3000.0 * float(hands == 0)
                + 250.0 * float(hands == 1)
                - 90000.0 * max(0.0, height)
                + 30.0 * float(np.square(biases).mean())
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot"))
            if disallowed > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                lift_flags.append(height >= 0.05 and hands > 0)
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
        longest = max((len(list(run)) for value, run in itertools.groupby(lift_flags) if value), default=0)
        valid = first_invalid is None
        cost /= max(step + 1, 1)
        cost += 100.0 * impulses["wrist"]
        cost -= 700.0 * hand_frames + 1400.0 * bilateral_frames + 200000.0 * peak_supported_lift + 18000.0 * longest + 60000.0 * hand_path
        recoverable = (
            valid
            and root >= 0.50
            and float(physical["tilt"]) < np.deg2rad(45.0)
            and int(np.sum(physical["foot_contact"])) > 0
        )
        if recoverable and longest >= 10:
            rank = (0, -longest, -peak_supported_lift, cost)
        elif valid and longest >= 10:
            rank = (1, -root, float(physical["tilt"]), -longest, cost)
        elif valid:
            rank = (2, 10 - longest, -root, -peak_supported_lift, cost)
        else:
            rank = (3, -min(longest, 10), -step, -peak_supported_lift, cost)
        metrics = {
            "first_invalid_frame": None if valid else args.anchor_frame + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": peak_lift,
            "maximum_hand_supported_lift_m": peak_supported_lift,
            "sustained_hand_supported_lift_frames": longest,
            "hand_mediated_path_length_m": hand_path,
            "final_root_height_m": root,
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "recoverable_terminal": bool(recoverable),
            "object_impulse_ns": impulses,
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    lower = np.r_[[0.0, -0.03, 20.0, 0.0, 0.10], np.full(15, -0.35), np.zeros(4), np.zeros(4), 0.0, 0.0]
    upper = np.r_[[0.05, 0.12, 80.0, 0.03, 1.00], np.full(15, 0.35), np.full(4, 3.0), np.full(4, 0.20), 8.0, 2.0]
    mean = np.r_[[0.012, 0.0, 45.0, 0.004, 0.40], np.zeros(15), np.full(4, 0.2), np.full(4, 0.015), 2.0, 0.5]
    std = np.r_[[0.010, 0.025, 10.0, 0.006, 0.16], np.full(15, 0.09), np.full(4, 0.45), np.full(4, 0.025), 1.5, 0.4]
    rng = np.random.default_rng(9917)
    injected: list[np.ndarray] = []
    for path in (args.warm_start, args.secondary_warm_start):
        if path is None:
            continue
        candidate = np.load(path)["parameters"].astype(np.float64)
        if candidate.shape == (28,):
            candidate = np.r_[candidate, 0.0, 0.0]
        if candidate.shape != mean.shape:
            raise ValueError(f"warm start {path} has shape {candidate.shape}, expected {mean.shape}")
        injected.append(np.clip(candidate, lower, upper))
    if injected:
        mean = injected[0].copy()
    best_parameters = mean.copy()
    best = evaluate(best_parameters)
    history = []
    started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, len(mean))), lower, upper)
        if args.freeze_core:
            if not injected:
                raise ValueError("--freeze-core requires --warm-start")
            population[:, :28] = injected[0][:28]
        population[0] = best_parameters
        population[1, 5:] = 0.0
        for offset, candidate in enumerate(injected, start=2):
            if offset < len(population):
                population[offset] = candidate
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda i: results[i][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[: args.elites]]
        mean = 0.7 * elite.mean(0) + 0.3 * mean
        std = np.maximum(0.7 * elite.std(0) + 0.3 * std, 0.01 * (upper - lower))
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), parameters=best_parameters, anchor_frame=args.anchor_frame, anchor_qpos=anchor_qpos, anchor_qvel=anchor_qvel, base_control=base_control, balance_actuators=balance, **trace)
    report = {"method": "structured state-feedback bilateral hand lift", "dimensions": len(mean), "population": args.population, "generations": args.generations, "rollouts": args.population * args.generations, "runtime_s": time.perf_counter() - started, "best": metrics, "history": history}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
