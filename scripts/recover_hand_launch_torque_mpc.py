#!/usr/bin/env python3
"""Receding-horizon MuJoCo torque shooting recovery after a hand launch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from orcs.tasks.uolm.torque_shooting_mpc import _decode_residual
from probe_flat_hand_grasp import build, smoothstep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--anchor-index", type=int, default=8)
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-frames", type=int, default=12)
    parser.add_argument("--replan-frames", type=int, default=2)
    parser.add_argument("--evaluation-frames", type=int, default=140)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--all-actuators", action="store_true")
    parser.add_argument(
        "--preserve-launch-controls",
        action="store_true",
        help="Use the saved control trajectory as feedforward instead of blending to stance.",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    launch = np.load(args.launch)
    stance = np.load(args.stance)
    anchor = args.anchor_index
    current = mujoco.MjData(model)
    current.qpos[:] = launch["qpos"][anchor - 1]
    current.qvel[:] = launch["qvel"][anchor - 1]
    mujoco.mj_forward(model, current)
    start_control = launch["ctrl"][anchor - 1].copy()
    stance_target = stance["target"].copy()
    controlled = (
        np.arange(model.nu, dtype=int)
        if args.all_actuators
        else np.asarray(generator.balance_actuators, dtype=int)
    )
    actuator_dofs = np.asarray([model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0]], dtype=int)
    kp = model.actuator_gainprm[:, 0].copy()
    kd = -model.actuator_biasprm[:, 2].copy()
    torque_limit = np.max(np.abs(model.actuator_forcerange), axis=1)
    residual_bound = 0.55 * torque_limit[controlled]
    dimensions = 2 * len(controlled)
    rng = np.random.default_rng(7761)
    root_body = int(model.jnt_bodyid[engine.robot_free_joint])
    desired_root_z = max(float(current.qpos[engine.robot_qpos + 2]), 0.56)
    object_geoms = set(map(int, engine.object_geoms))

    def nominal_target(elapsed_frame: float) -> np.ndarray:
        if args.preserve_launch_controls:
            elapsed_physics_step = int(round(elapsed_frame * engine.config.substeps))
            if "ctrl_physics" in launch:
                first_step = anchor * engine.config.substeps
                index = first_step + elapsed_physics_step
                if index < len(launch["ctrl_physics"]):
                    return launch["ctrl_physics"][index].copy()
                source = launch["ctrl_physics"][-1].copy()
                available_frames = (
                    len(launch["ctrl_physics"]) - first_step
                ) / engine.config.substeps
            else:
                index = anchor + int(elapsed_frame)
                if index < len(launch["ctrl"]):
                    return launch["ctrl"][index].copy()
                source = launch["ctrl"][-1].copy()
                available_frames = len(launch["ctrl"]) - anchor
            blend = smoothstep((elapsed_frame - available_frames) / 20.0)
            return (1.0 - blend) * source + blend * stance_target
        blend = smoothstep(elapsed_frame / 28.0)
        return (1.0 - blend) * start_control + blend * stance_target

    def command(data: mujoco.MjData, nominal: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, int]:
        q = data.qpos[engine.actuator_qpos]
        qdot = data.qvel[actuator_dofs]
        tau_ff = kp * (nominal - q) - kd * qdot
        tau = tau_ff.copy()
        tau[controlled] += residual
        limited = np.clip(tau, -torque_limit, torque_limit)
        raw = q + (limited + kd * qdot) / kp
        target = np.clip(raw, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        clipped = int(np.sum((np.abs(tau - limited) > 1.0e-9) | (np.abs(raw - target) > 1.0e-9)))
        return target, clipped

    def disallowed_object_contact(data: mujoco.MjData) -> bool:
        for ci in range(data.ncon):
            c = data.contact[ci]
            first, second = int(c.geom1), int(c.geom2)
            if not ({first, second} & object_geoms):
                continue
            other = second if first in object_geoms else first
            body = int(model.geom_bodyid[other])
            if body not in engine.robot_bodies:
                continue
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if not any(token in body_name for token in ("hand", "wrist")):
                return True
        return False

    horizon_steps = args.horizon_frames * engine.config.substeps
    apply_steps = args.replan_frames * engine.config.substeps

    def rollout(data0: mujoco.MjData, residual_path: np.ndarray, elapsed_step: int):
        data = mujoco.MjData(model)
        data.qpos[:] = data0.qpos
        data.qvel[:] = data0.qvel
        data.act[:] = data0.act
        data.time = data0.time
        mujoco.mj_forward(model, data)
        cost = 0.0
        invalid_step = None
        reason = None
        clipped = 0
        previous = residual_path[0]
        for step in range(horizon_steps):
            elapsed_frame = (elapsed_step + step) / engine.config.substeps
            target, count = command(data, nominal_target(elapsed_frame), residual_path[step])
            clipped += count
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            root_z = float(data.qpos[engine.robot_qpos + 2])
            vertical_velocity = float(data.qvel[engine.robot_dof + 2])
            support = int(np.sum(physical["foot_contact"]))
            rate = (residual_path[step] - previous) / residual_bound
            cost += (
                9000.0 * (root_z - desired_root_z) ** 2
                + 1600.0 * physical["tilt"] ** 2
                + 900.0 * max(0.0, -vertical_velocity) ** 2
                + 3000.0 * float(support == 0)
                + 100.0 * float(support == 1)
                + 30.0 * float(np.square(residual_path[step] / residual_bound).mean())
                + 40.0 * float(np.square(rate).mean())
                + 40.0 * count
            )
            reason = generator._invalid_reason(data, physical)
            if reason is None and disallowed_object_contact(data):
                reason = "non_hand_object_contact"
            if reason is not None:
                invalid_step = step
                cost += 1.0e8 + 1.0e6 * (horizon_steps - step)
                break
            previous = residual_path[step]
        physical = generator._physical(data)
        root_z = float(data.qpos[engine.robot_qpos + 2])
        vz = float(data.qvel[engine.robot_dof + 2])
        support = int(np.sum(physical["foot_contact"]))
        cost += 12.0 * (
            12000.0 * (root_z - desired_root_z) ** 2
            + 2200.0 * physical["tilt"] ** 2
            + 1800.0 * max(0.0, -vz) ** 2
            + 6000.0 * float(support == 0)
        )
        rank = (0, cost) if invalid_step is None else (1, -invalid_step, cost)
        return rank, cost, invalid_step, reason, clipped

    current_residual = np.zeros(len(controlled))
    shifted = np.zeros(dimensions)
    trace = {key: [] for key in ("qpos", "qvel", "ctrl", "actuator_force", "hand_contact", "foot_contact", "object_position", "root_height", "torso_tilt", "residual")}
    replans = []
    first_invalid = reason = None
    started = time.perf_counter()
    total_steps = args.evaluation_frames * engine.config.substeps
    for elapsed_step in range(0, total_steps, apply_steps):
        mean = shifted.copy()
        std = np.tile(0.18 * residual_bound, 2)
        best_parameters = mean.copy()
        best_path = _decode_residual(best_parameters, current_residual, horizon_steps, 2)
        best = rollout(current, best_path, elapsed_step)
        for generation in range(args.generations):
            population = np.clip(
                rng.normal(mean, std, (args.population, dimensions)),
                -np.tile(residual_bound, 2),
                np.tile(residual_bound, 2),
            )
            population[0] = best_parameters
            population[1] = 0.0
            candidates = []
            for parameters in population:
                path = _decode_residual(parameters, current_residual, horizon_steps, 2)
                candidates.append((rollout(current, path, elapsed_step), path))
            order = sorted(range(len(population)), key=lambda i: candidates[i][0][0])
            if candidates[order[0]][0][0] < best[0]:
                best_parameters = population[order[0]].copy()
                best_path = candidates[order[0]][1]
                best = candidates[order[0]][0]
            elite = population[order[: args.elites]]
            mean = 0.7 * elite.mean(0) + 0.3 * mean
            std = np.maximum(0.7 * elite.std(0) + 0.3 * std, 0.03 * np.tile(residual_bound, 2))
        replans.append({"frame": elapsed_step / engine.config.substeps, "predicted_invalid_step": best[2], "predicted_reason": best[3], "cost": best[1]})
        for local in range(min(apply_steps, total_steps - elapsed_step)):
            elapsed_frame = (elapsed_step + local) / engine.config.substeps
            target, _ = command(current, nominal_target(elapsed_frame), best_path[local])
            current.ctrl[:] = target
            mujoco.mj_step(model, current)
            physical = generator._physical(current)
            invalid = generator._invalid_reason(current, physical)
            if invalid is None and disallowed_object_contact(current):
                invalid = "non_hand_object_contact"
            if local % engine.config.substeps == engine.config.substeps - 1:
                trace["qpos"].append(current.qpos.copy())
                trace["qvel"].append(current.qvel.copy())
                trace["ctrl"].append(current.ctrl.copy())
                trace["actuator_force"].append(current.actuator_force.copy())
                trace["hand_contact"].append(physical["hand_contact"].copy())
                trace["foot_contact"].append(physical["foot_contact"].copy())
                trace["object_position"].append(current.qpos[engine.object_qpos : engine.object_qpos + 3].copy())
                trace["root_height"].append(float(current.qpos[engine.robot_qpos + 2]))
                trace["torso_tilt"].append(float(physical["tilt"]))
                trace["residual"].append(best_path[local].copy())
            if invalid is not None:
                first_invalid = elapsed_step + local
                reason = invalid
                break
        if first_invalid is not None:
            break
        current_residual = best_path[min(apply_steps, len(best_path) - 1)].copy()
        future = best_parameters.reshape(2, -1)
        shifted = np.r_[future[1], future[1]]
        print("REPLAN", elapsed_step / engine.config.substeps, "root", current.qpos[engine.robot_qpos + 2], "tilt", np.rad2deg(physical["tilt"]), flush=True)

    arrays = {key: np.asarray(value) for key, value in trace.items()}
    position_error = float(np.linalg.norm(current.qpos[engine.object_qpos : engine.object_qpos + 3] - engine.reference_object_pos[-1]))
    quaternion = current.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(quaternion, engine.reference_object_quat[-1])), 0.0, 1.0)))
    report = {
        "method": "MuJoCo torque-shooting recovery after hand launch",
        "anchor_index": anchor,
        "controlled_actuators": controlled.tolist(),
        "horizon_frames": args.horizon_frames,
        "replan_frames": args.replan_frames,
        "population": args.population,
        "generations": args.generations,
        "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
        "invalid_reason": reason,
        "valid_frames": len(arrays["qpos"]),
        "final_position_error_m": position_error,
        "final_orientation_error_rad": orientation_error,
        "final_root_height_m": float(current.qpos[engine.robot_qpos + 2]),
        "final_torso_tilt_deg": float(np.rad2deg(generator._physical(current)["tilt"])),
        "maximum_object_lift_m": float(
            np.max(arrays["object_position"][:, 2] - 0.18545216)
        ) if len(arrays["object_position"]) else 0.0,
        "sustained_lift_frames": int(
            max(
                (
                    len(list(run))
                    for value, run in __import__("itertools").groupby(
                        arrays["object_position"][:, 2] - 0.18545216 >= 0.05
                    )
                    if value
                ),
                default=0,
            )
        ) if len(arrays["object_position"]) else 0,
        "runtime_s": time.perf_counter() - started,
        "replans": replans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), anchor_qpos=launch["qpos"][anchor - 1], anchor_qvel=launch["qvel"][anchor - 1], controlled_actuators=controlled, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
