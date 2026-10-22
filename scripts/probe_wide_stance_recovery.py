#!/usr/bin/env python3
"""Establish and hold a wider manipulation-ready stance before hand loading."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_flat_hand_grasp import build, smoothstep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.stance)
    base = source["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    balance_qpos = engine.actuator_qpos[balance]
    joint_ids = model.actuator_trnid[balance, 0].astype(int)
    lower = model.jnt_range[joint_ids, 0] + 0.003
    upper = model.jnt_range[joint_ids, 1] - 0.003
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    substeps = engine.config.substeps

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    anchor_qpos = anchor.qpos.copy()
    nominal = anchor_qpos[balance_qpos].copy()
    root0 = anchor_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    feet0 = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations0 = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    object_position = anchor_qpos[engine.object_qpos : engine.object_qpos + 3]
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_position[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    original_width = float(abs(np.dot(feet0[0] - feet0[1], lateral)))

    def solve_pose(outward: float):
        data = mujoco.MjData(model)
        feet_target = feet0.copy()
        feet_target[0] += outward * lateral
        feet_target[1] -= outward * lateral

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[:] = anchor_qpos
            data.qvel[:] = 0.0
            data.qpos[balance_qpos] = candidate
            mujoco.mj_forward(model, data)
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            foot_rotations = np.asarray(
                [data.xmat[body].reshape(3, 3) for body in engine.foot_bodies]
            )
            foot_orientation = np.concatenate(
                [
                    Rotation.from_matrix(
                        foot_rotations[index] @ foot_rotations0[index].T
                    ).as_rotvec()
                    for index in range(2)
                ]
            )
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            return np.r_[
                120.0 * (feet - feet_target).ravel(),
                22.0 * foot_orientation,
                20.0 * (root - root0),
                8.0 * torso_up[:2],
                0.03 * (candidate - nominal),
            ]

        result = least_squares(
            residual,
            np.clip(nominal, lower, upper),
            bounds=(lower, upper),
            max_nfev=350,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        data.qpos[:] = anchor_qpos
        data.qvel[:] = 0.0
        data.qpos[balance_qpos] = result.x
        mujoco.mj_forward(model, data)
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        physical = generator._physical(data)
        return result.x.copy(), {
            "maximum_foot_error_m": float(np.max(np.linalg.norm(feet - feet_target, axis=1))),
            "root_error_m": float(np.linalg.norm(data.qpos[engine.robot_qpos : engine.robot_qpos + 3] - root0)),
            "torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
        }

    pose_cache = {outward: solve_pose(outward) for outward in (0.04, 0.07, 0.10)}

    def evaluate(spec: dict[str, float], record: bool = False):
        pose, static = pose_cache[spec["outward_per_foot_m"]]
        wide_control = pose + (base[balance] - nominal)
        approach_end = 90.0 + spec["approach_frames"]
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        invalid_reason = None
        nonhand = 0.0
        ready_run = max_ready_run = 0
        ctrl_physics = []
        trace = {
            key: []
            for key in (
                "qpos", "qvel", "ctrl", "actuator_force", "foot_contact",
                "foot_position", "root_height", "torso_tilt", "nonhand_impulse",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            held = base.copy()
            if frame >= 90.0:
                amount = smoothstep((frame - 90.0) / spec["approach_frames"])
                held[balance] = (1.0 - amount) * base[balance] + amount * wide_control
            held = np.clip(held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = held
            ctrl_physics.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                c = data.contact[contact_index]
                first, second = int(c.geom1), int(c.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) in engine.robot_bodies and other not in hand_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    nonhand += float(np.linalg.norm(force[:3])) * model.opt.timestep
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            width = float(abs(np.dot(feet[0] - feet[1], lateral)))
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            ready = (
                frame >= approach_end
                and width > original_width + 0.05
                and root[2] > 0.58
                and np.rad2deg(physical["tilt"]) < 30.0
                and np.linalg.norm(velocity[:3]) < 0.25
                and int(np.sum(physical["foot_contact"])) == 2
                and nonhand <= 0.10
            )
            if step % substeps == substeps - 1:
                ready_run = ready_run + 1 if ready else 0
                max_ready_run = max(max_ready_run, ready_run)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["foot_position"].append(feet.copy())
                    trace["root_height"].append(float(root[2]))
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["nonhand_impulse"].append(nonhand)
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if invalid is not None:
                first_invalid = step
                invalid_reason = invalid
                break
        physical = generator._physical(data)
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        width = float(abs(np.dot(feet[0] - feet[1], lateral)))
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        valid = first_invalid is None
        metrics = {
            **spec,
            "full_horizon_valid": valid,
            "first_invalid_frame": None if valid else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "longest_wide_stopped_frames": max_ready_run,
            "original_foot_width_m": original_width,
            "final_foot_width_m": width,
            "foot_width_increase_m": width - original_width,
            "goalward_foot_translation_m": ((feet - feet0) @ forward).tolist(),
            "lateral_foot_translation_m": ((feet - feet0) @ lateral).tolist(),
            "final_root_velocity_m_s": velocity[:3].tolist(),
            "final_root_height_m": float(root[2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "nonhand_object_impulse_ns": nonhand,
            "static_ik": static,
        }
        rank = (
            not (valid and max_ready_run >= 30),
            not valid,
            -max_ready_run,
            -width,
            float(np.linalg.norm(velocity[:3])),
            -(step / substeps),
        )
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics), pose

    specs = [
        {"outward_per_foot_m": outward, "approach_frames": float(approach)}
        for outward, approach in itertools.product((0.04, 0.07, 0.10), (35, 60, 90))
    ]
    best = None
    best_spec = None
    trials = []
    started = time.perf_counter()
    for index, spec in enumerate(specs):
        result = evaluate(spec)
        trials.append(result[1])
        if best is None or result[0] < best[0]:
            best = result
            best_spec = spec
        print("TRIAL", index, json.dumps(result[1]), flush=True)
    assert best_spec is not None
    _, metrics, trace, ctrl_physics, pose = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        balance_actuators=balance,
        wide_pose=pose,
        wide_control=pose + (base[balance] - nominal),
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "legal fixed-support stance-widening and full-horizon hold grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
