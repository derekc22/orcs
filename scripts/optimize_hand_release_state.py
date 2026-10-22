#!/usr/bin/env python3
"""Optimize a short bilateral-hand launch toward a physically settling goal state."""

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
    parser.add_argument("--anchor-index", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--elites", type=int, default=20)
    parser.add_argument("--generations", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument(
        "--release-frame",
        type=int,
        help="End of the optimized launch phase; the rollout continues to --horizon.",
    )
    parser.add_argument("--recovery-stance", type=Path)
    parser.add_argument("--recovery-duration", type=float, default=12.0)
    parser.add_argument("--all-actuators", action="store_true")
    parser.add_argument(
        "--require-recovery",
        action="store_true",
        help="Rank only upright, two-foot, low-downward-speed terminal states first.",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    prefix = np.load(args.prefix)
    release_frame = args.horizon if args.release_frame is None else args.release_frame
    if not 0 < release_frame <= args.horizon:
        raise ValueError("release-frame must be in (0, horizon]")
    recovery_target = (
        None
        if args.recovery_stance is None
        else np.load(args.recovery_stance)["target"].copy()
    )
    i0 = args.anchor_index
    anchor_qpos = prefix["qpos"][i0 - 1].copy()
    anchor_qvel = prefix["qvel"][i0 - 1].copy()
    arms = np.setdiff1d(np.arange(model.nu), generator.balance_actuators)
    controlled = np.arange(model.nu) if args.all_actuators else arms
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    object_geoms = set(engine.object_geoms)
    goal = engine.reference_object_pos[-1]
    # A separately verified unassisted settle from this object pose reaches the
    # goal with this release velocity.  The optimizer must create it using only
    # the two flat hands; the target is not written into the simulation.
    velocity_target = np.asarray([-0.797, -3.161, -0.448, 8.002, 3.268, 11.090])
    knot_frames = np.asarray(
        [0.0, 0.20 * release_frame, 0.50 * release_frame, 0.80 * release_frame, float(release_frame)]
    )

    def baseline_control(step: int) -> np.ndarray:
        if "ctrl_physics" in prefix:
            controls = prefix["ctrl_physics"]
            offset = i0 * engine.config.substeps
            source = controls[min(offset + step, len(controls) - 1)].copy()
        else:
            frame = step / engine.config.substeps
            source_index = min(i0 + int(frame), len(prefix["ctrl"]) - 1)
            source = prefix["ctrl"][source_index].copy()
        if recovery_target is None or step < release_frame * engine.config.substeps:
            return source
        release_step = release_frame * engine.config.substeps - 1
        if "ctrl_physics" in prefix:
            controls = prefix["ctrl_physics"]
            recovery_source = controls[min(i0 * engine.config.substeps + release_step, len(controls) - 1)]
        else:
            recovery_source = prefix["ctrl"][min(i0 + release_frame - 1, len(prefix["ctrl"]) - 1)]
        frame_after = step / engine.config.substeps - release_frame
        amount = smoothstep(frame_after / args.recovery_duration)
        return (1.0 - amount) * recovery_source + amount * recovery_target

    def group(geom: int) -> str:
        if geom == int(engine.hand_geoms[0]): return "left_hand"
        if geom == int(engine.hand_geoms[1]): return "right_hand"
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if "wrist" in name: return "wrist"
        if "foot" in name or "ankle" in name: return "feet"
        if any(t in name for t in ("knee", "shin", "hip", "thigh")): return "legs"
        if any(t in name for t in ("torso", "pelvis", "waist")): return "torso_pelvis"
        return "other_robot"

    def residual_at(parameters: np.ndarray, frame: float) -> np.ndarray:
        if frame >= release_frame:
            return np.zeros(len(controlled))
        free = parameters.reshape(3, len(controlled))
        knots = np.vstack([np.zeros(len(controlled)), free, np.zeros(len(controlled))])
        if frame >= knot_frames[-1]: return knots[-1]
        segment = int(np.searchsorted(knot_frames, frame, side="right") - 1)
        amount = smoothstep((frame - knot_frames[segment]) / (knot_frames[segment + 1] - knot_frames[segment]))
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

    def evaluate(parameters: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        first_invalid = reason = None
        hand_steps = bilateral_steps = 0
        minimum_root = float("inf")
        maximum_tilt = 0.0
        trace = {key: [] for key in ("qpos", "qvel", "ctrl", "actuator_force", "hand_contact", "foot_contact", "object_position", "root_height", "torso_tilt", "object_impulse")}
        ctrl_physics = []
        release_velocity = None
        release_position = None
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            target = baseline_control(step)
            target[controlled] += residual_at(parameters, frame)
            target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            ctrl_physics.append(target.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for ci in range(data.ncon):
                contact = data.contact[ci]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms): continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies: continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            hands = int(np.sum(physical["hand_contact"]))
            hand_steps += int(hands > 0)
            bilateral_steps += int(hands == 2)
            root = float(data.qpos[engine.robot_qpos + 2])
            minimum_root = min(minimum_root, root)
            maximum_tilt = max(maximum_tilt, float(physical["tilt"]))
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(impulses[k] for k in ("feet", "legs", "torso_pelvis", "other_robot"))
            if disallowed > 0.50 and invalid is None: invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1 and record:
                trace["qpos"].append(data.qpos.copy()); trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(data.ctrl.copy()); trace["actuator_force"].append(data.actuator_force.copy())
                trace["hand_contact"].append(physical["hand_contact"].copy()); trace["foot_contact"].append(physical["foot_contact"].copy())
                trace["object_position"].append(data.qpos[engine.object_qpos:engine.object_qpos+3].copy())
                trace["root_height"].append(root); trace["torso_tilt"].append(float(physical["tilt"]))
                trace["object_impulse"].append(np.asarray(list(impulses.values())))
            if invalid is not None:
                first_invalid, reason = step, invalid
                break
            if step == release_frame * engine.config.substeps - 1:
                release_velocity = data.qvel[engine.object_dof:engine.object_dof+6].copy()
                release_position = data.qpos[engine.object_qpos:engine.object_qpos+3].copy()
        if release_velocity is None:
            release_velocity = data.qvel[engine.object_dof:engine.object_dof+6].copy()
            release_position = data.qpos[engine.object_qpos:engine.object_qpos+3].copy()
        velocity = release_velocity
        object_position = release_position
        direction = goal - object_position
        direction[2] = 0.0
        direction /= max(np.linalg.norm(direction), 1e-9)
        along = float(np.dot(velocity[:3], direction))
        lateral = float(np.linalg.norm(velocity[:2] - along * direction[:2]))
        linear_error = float(np.linalg.norm(velocity[:3] - velocity_target[:3]))
        angular_error = float(np.linalg.norm(velocity[3:] - velocity_target[3:]))
        disallowed = sum(impulses[k] for k in ("feet", "legs", "torso_pelvis", "other_robot"))
        valid = first_invalid is None
        root_height = float(data.qpos[engine.robot_qpos + 2])
        root_vertical_velocity = float(data.qvel[engine.robot_dof + 2])
        terminal_tilt = float(physical["tilt"])
        foot_contacts = int(np.sum(physical["foot_contact"]))
        recoverable = (
            valid
            and root_height >= 0.52
            and terminal_tilt < np.deg2rad(40.0)
            and root_vertical_velocity > -0.10
            and foot_contacts == 2
        )
        recovery_cost = (
            20000.0 * max(0.0, 0.52 - root_height) ** 2
            + 3000.0 * max(0.0, terminal_tilt - np.deg2rad(35.0)) ** 2
            + 2500.0 * max(0.0, -0.10 - root_vertical_velocity) ** 2
            + 1000.0 * float(foot_contacts < 2)
        )
        cost = (
            80.0 * linear_error**2 + 0.8 * angular_error**2
            - 30.0 * along + 12.0 * lateral
            + 0.15 * float(np.square(parameters).mean())
            + 0.5 * impulses["wrist"]
            - 0.1 * min(impulses["left_hand"], impulses["right_hand"])
            + (recovery_cost if args.require_recovery else 0.0)
        )
        if args.require_recovery and recoverable and hand_steps >= 20 and disallowed <= 0.5:
            rank = (0, cost)
        elif args.require_recovery and valid and hand_steps >= 20 and disallowed <= 0.5:
            rank = (1, cost)
        elif args.require_recovery and valid:
            rank = (2, -hand_steps, cost)
        elif args.require_recovery:
            rank = (3, -first_invalid, cost)
        elif valid and hand_steps >= 20 and disallowed <= 0.5:
            rank = (0, cost)
        elif valid:
            rank = (1, -hand_steps, cost)
        else:
            rank = (2, -first_invalid, cost)
        metrics = {
            "first_invalid_frame": None if valid else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "hand_contact_time_s": hand_steps * model.opt.timestep,
            "bilateral_contact_time_s": bilateral_steps * model.opt.timestep,
            "release_object_position": object_position.tolist(),
            "release_object_velocity": velocity.tolist(),
            "final_object_position": data.qpos[engine.object_qpos:engine.object_qpos+3].copy().tolist(),
            "target_velocity": velocity_target.tolist(),
            "linear_velocity_error": linear_error,
            "angular_velocity_error": angular_error,
            "goalward_speed": along,
            "lateral_speed": lateral,
            "minimum_root_height_m": minimum_root,
            "maximum_torso_tilt_deg": float(np.rad2deg(maximum_tilt)),
            "final_root_height_m": root_height,
            "final_root_vertical_velocity_m_s": root_vertical_velocity,
            "final_torso_tilt_deg": float(np.rad2deg(terminal_tilt)),
            "final_foot_contacts": foot_contacts,
            "recoverable_terminal": bool(recoverable),
            "object_impulse_ns": impulses,
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    dimensions = 3 * len(controlled)
    per = np.asarray([
        (1.05 if a in arms and "wrist" not in names[a] else 0.65 if a in arms else 0.48)
        for a in controlled
    ])
    bounds = np.tile(per, 3)
    mean = np.zeros(dimensions)
    std = np.tile(0.30 * per, 3)
    rng = np.random.default_rng(6317)
    best_parameters = mean.copy(); best = evaluate(best_parameters)
    history = []; started = time.perf_counter()
    for generation in range(args.generations):
        population = np.clip(rng.normal(mean, std, (args.population, dimensions)), -bounds, bounds)
        population[0] = best_parameters
        population[1] = 0.0
        results = [evaluate(x) for x in population]
        order = sorted(range(len(population)), key=lambda i: results[i][0])
        if results[order[0]][0] < best[0]: best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[:args.elites]]
        mean = 0.75 * elite.mean(0) + 0.25 * mean
        std = np.maximum(0.75 * elite.std(0) + 0.25 * std, 0.012)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_name(args.output.name+"_checkpoint").with_suffix(".npz"), parameters=best_parameters, generation=generation)
    _, metrics, trace, ctrl_physics = evaluate(best_parameters, record=True)
    np.savez_compressed(args.output.with_suffix(".npz"), parameters=best_parameters, anchor_qpos=anchor_qpos, anchor_qvel=anchor_qvel, anchor_index=i0, arm_actuators=arms, controlled_actuators=controlled, knot_frames=knot_frames, ctrl_physics=ctrl_physics, **trace)
    report = {"method":"bilateral flat-hand release-state shooting", "all_actuators":args.all_actuators, "require_recovery":args.require_recovery, "release_frame":release_frame, "recovery_stance":None if args.recovery_stance is None else str(args.recovery_stance), "dimensions":dimensions, "population":args.population, "generations":args.generations, "runtime_s":time.perf_counter()-started, "best":metrics, "history":history}
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
