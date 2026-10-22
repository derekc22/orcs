#!/usr/bin/env python3
"""Direct MuJoCo refinement from a stable bilateral hand-contact state."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--prefix-frame-offset", type=int, default=130)
    parser.add_argument("--anchor-frame", type=int, default=210)
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--balance-only", action="store_true")
    parser.add_argument("--use-prefix-control-sequence", action="store_true")
    parser.add_argument("--require-lift-frames", type=int, default=0)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--maximum-disallowed-impulse", type=float, default=0.5)
    parser.add_argument(
        "--residual-start",
        type=float,
        default=0.0,
        help="Delay the local residual schedule while replaying the physical prefix controls.",
    )
    parser.add_argument(
        "--preserve-contract",
        action="store_true",
        help="During a staged search, keep a proven lift/transport contract while extending survival.",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    prefix = np.load(args.prefix)
    stance = np.load(args.stance)
    index = args.anchor_frame - args.prefix_frame_offset
    # Recorded frame samples are post-step states.  The separately stored
    # anchor is the state immediately before sample zero; using qpos[0] here
    # silently advanced every local repair by one reference frame.
    if index == 0 and "anchor_qpos" in prefix and "anchor_qvel" in prefix:
        anchor_qpos = prefix["anchor_qpos"].copy()
        anchor_qvel = prefix["anchor_qvel"].copy()
    else:
        anchor_qpos = prefix["qpos"][index - 1].copy()
        anchor_qvel = prefix["qvel"][index - 1].copy()
    base_target = prefix["ctrl"][index].copy()
    balance = stance["selected_actuators"].astype(int)
    controlled = balance if args.balance_only else np.arange(model.nu)
    object_geoms = set(engine.object_geoms)
    ground_object_z = 0.18545216
    goal_position = engine.reference_object_pos[-1]
    knot_frames = np.asarray([0.0, 8.0, 35.0])

    def residual_at(knots: np.ndarray, frame: float) -> np.ndarray:
        if frame >= knot_frames[-1]:
            return knots[-1]
        segment = int(np.searchsorted(knot_frames, frame, side="right") - 1)
        amount = smoothstep(
            (frame - knot_frames[segment])
            / (knot_frames[segment + 1] - knot_frames[segment])
        )
        return (1.0 - amount) * knots[segment] + amount * knots[segment + 1]

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

    def evaluate(parameters: np.ndarray, record: bool = False):
        knots = np.vstack(
            [np.zeros(len(controlled)), parameters.reshape(2, len(controlled))]
        )
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = anchor_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        impulses = {
            name: 0.0
            for name in (
                "left_hand",
                "right_hand",
                "wrist",
                "feet",
                "legs",
                "torso_pelvis",
                "other_robot",
            )
        }
        hand_frames = bilateral_frames = 0
        lift_flags: list[bool] = []
        supported_lift_flags: list[bool] = []
        lifted_hand_transport = 0.0
        last_lifted_hand_object = None
        transport = 0.0
        maximum_lift = 0.0
        initial_goal_error = float(
            np.linalg.norm(
                anchor_qpos[engine.object_qpos : engine.object_qpos + 3]
                - goal_position
            )
        )
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
                "ctrl_physics",
            )
        }
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            residual = (
                np.zeros(len(controlled))
                if frame < args.residual_start
                else residual_at(knots, frame - args.residual_start)
            )
            if args.use_prefix_control_sequence:
                control_index = min(index + int(frame), len(prefix["ctrl"]) - 1)
                target = prefix["ctrl"][control_index].copy()
            else:
                target = base_target.copy()
            target[controlled] += residual
            target = np.clip(
                target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            data.ctrl[:] = target
            if record:
                trace["ctrl_physics"].append(data.ctrl.copy())
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
            object_position = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            goal_error = float(np.linalg.norm(object_position - goal_position))
            transport = max(transport, initial_goal_error - goal_error)
            lift = float(object_position[2] - ground_object_z)
            maximum_lift = max(maximum_lift, lift)
            root = float(data.qpos[engine.robot_qpos + 2])
            support = int(np.sum(physical["foot_contact"]))
            hand_count = int(np.sum(physical["hand_contact"]))
            if lift >= 0.05 and hand_count > 0:
                if last_lifted_hand_object is not None:
                    lifted_hand_transport += float(
                        np.linalg.norm(object_position - last_lifted_hand_object)
                    )
                last_lifted_hand_object = object_position.copy()
            else:
                last_lifted_hand_object = None
            root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
            cost += (
                18000.0 * max(0.0, 0.55 - root) ** 2
                + 4200.0 * physical["tilt"] ** 2
                + 800.0 * float(np.square(root_velocity).sum())
                + 6000.0 * float(support == 0)
                + 300.0 * float(support == 1)
                + 1800.0 * float(hand_count == 0)
                + 140.0 * float(hand_count == 1)
                + 10.0 * float(np.square(residual / 0.45).mean())
                + 1200.0 * goal_error**2
                - 50000.0 * max(0.0, lift)
            )
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[name]
                for name in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            if disallowed > args.maximum_disallowed_impulse and invalid is None:
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
                    trace["object_position"].append(object_position.copy())
                    trace["root_height"].append(root)
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(
                        np.asarray(list(impulses.values()))
                    )
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
        cost /= max(step + 1, 1)
        # Wrist-link contact is adjacent to the flat-hand end effector and is
        # retained as a strong penalty/diagnostic.  It is not an automatic
        # failure unless non-end-effector body groups drive the object.
        cost += 100.0 * impulses["wrist"]
        cost -= (
            700.0 * hand_frames
            + 1700.0 * bilateral_frames
            + 120.0 * min(impulses["left_hand"], impulses["right_hand"])
            + 180000.0 * maximum_lift
            + 9000.0 * longest
            + 100000.0 * transport
        )
        contract_missing = (
            longest_supported < args.require_lift_frames
            or lifted_hand_transport < 0.03
        )
        if first_invalid is None and not contract_missing:
            rank = (0, cost)
        elif args.preserve_contract and not contract_missing:
            rank = (1, -first_invalid, cost)
        elif first_invalid is None:
            rank = (
                2 if args.preserve_contract else 1,
                max(0, args.require_lift_frames - longest_supported),
                -maximum_lift,
                -lifted_hand_transport,
                -transport,
                cost,
            )
        elif not contract_missing:
            rank = (2, -first_invalid, cost)
        else:
            rank = (
                3,
                max(0, args.require_lift_frames - longest_supported),
                -maximum_lift,
                -lifted_hand_transport,
                -first_invalid,
                cost,
            )
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else args.anchor_frame + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "valid_frames": step / engine.config.substeps,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "maximum_lift_m": maximum_lift,
            "sustained_lift_frames": longest,
            "sustained_hand_supported_lift_frames": longest_supported,
            "lifted_hand_mediated_transport_m": lifted_hand_transport,
            "maximum_goal_progress_m": transport,
            "final_goal_position_error_m": goal_error,
            "object_impulse_ns": impulses,
            "final_root_height_m": root,
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "cost": float(cost),
        }
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}

    dimensions = 2 * len(controlled)
    actuator_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    per_joint_bounds = np.full(model.nu, 0.45)
    per_joint_bounds[["wrist" in name for name in actuator_names]] = 0.25
    controlled_bounds = per_joint_bounds[controlled]
    bounds = np.tile(controlled_bounds, 2)
    recovery = np.zeros(len(controlled))
    for local_index, actuator in enumerate(controlled):
        if actuator in balance:
            recovery[local_index] = stance["target"][actuator] - base_target[actuator]
    mean = np.tile(np.clip(recovery, -controlled_bounds, controlled_bounds), 2)
    if args.warm_start is not None:
        warm = np.load(args.warm_start)["parameters"].astype(np.float64)
        if warm.shape != (dimensions,):
            raise ValueError(
                f"warm-start shape {warm.shape} does not match {(dimensions,)}"
            )
        mean = np.clip(warm, -bounds, bounds)
    rng = np.random.default_rng(5173)
    std = np.tile(np.minimum(controlled_bounds * 0.35, 0.14), 2)
    best_parameters = mean.copy() if args.warm_start is not None else np.zeros(dimensions)
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
        order = sorted(range(len(population)), key=lambda candidate: results[candidate][0])
        if results[order[0]][0] < best[0]:
            best_parameters, best = population[order[0]].copy(), results[order[0]]
        elite = population[order[:16]]
        mean = 0.70 * elite.mean(0) + 0.30 * mean
        std = np.maximum(0.70 * elite.std(0) + 0.30 * std, 0.006)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
        # Preserve expensive progress so a long contact search can be resumed
        # without waiting for the final generation or relying on terminal logs.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output.with_name(args.output.name + "_checkpoint").with_suffix(".npz"),
            parameters=best_parameters,
            generation=generation,
        )
    _, metrics, trace = evaluate(best_parameters, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        parameters=best_parameters,
        anchor_frame=args.anchor_frame,
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        base_target=base_target,
        controlled_actuators=controlled,
        knot_frames=knot_frames,
        **trace,
    )
    report = {
        "method": (
            "balance-only hand-carry/recovery MuJoCo shooting"
            if args.balance_only
            else "all-actuator hand-carry/recovery MuJoCo shooting"
        ),
        "dimensions": dimensions,
        "population": args.population,
        "generations": args.generations,
        "horizon_frames": args.horizon,
        "uses_prefix_control_sequence": args.use_prefix_control_sequence,
        "required_lift_frames": args.require_lift_frames,
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
