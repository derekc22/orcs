#!/usr/bin/env python3
"""Establish a vertical-torso double-support squat before hand manipulation."""

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


TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}


def quat_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def rotation_to_quat(rotation: Rotation) -> np.ndarray:
    quaternion = rotation.as_quat()
    return np.r_[quaternion[3], quaternion[:3]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    feedback = np.load(args.feedback)
    base = stance["target"].copy()
    feedback_parameters = feedback["parameters"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    balance_qpos = engine.actuator_qpos[balance]
    joint_ids = model.actuator_trnid[balance, 0].astype(int)
    balance_dofs = np.asarray([model.jnt_dofadr[joint] for joint in joint_ids])
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(token in name for token in tokens)],
            dtype=int,
        )
        for key, tokens in TOKENS.items()
    }
    substeps = engine.config.substeps
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    anchor_qpos = anchor.qpos.copy()
    root0 = anchor_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_rotation0 = quat_to_rotation(
        anchor_qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    )
    desired_rotation = (
        Rotation.from_quat(feedback["desired_rotation_xyzw"])
        if "desired_rotation_xyzw" in feedback
        else root_rotation0
    )
    joints0 = anchor_qpos[balance_qpos].copy()
    feet0 = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations0 = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    lower = np.r_[
        [-0.08, -0.08, -0.16, -0.35, -0.35, -0.35],
        model.jnt_range[joint_ids, 0] + 0.004,
    ]
    upper = np.r_[
        [0.08, 0.08, 0.05, 0.35, 0.35, 0.35],
        model.jnt_range[joint_ids, 1] - 0.004,
    ]

    def feedback_correction(data: mujoco.MjData) -> np.ndarray:
        rotation = quat_to_rotation(
            data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        )
        rotation_error = (desired_rotation.inv() * rotation).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal = np.asarray(
            [root[1] - root0[1], velocity[1], rotation_error[0], velocity[3]]
        )
        lateral = np.asarray(
            [root[0] - root0[0], velocity[0], rotation_error[1], velocity[4]]
        )
        gains = feedback_parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[groups[key]] += float(np.dot(gains[row], sagittal))
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[groups[key]] += float(np.dot(gains[row], lateral))
            row += 1
        return np.clip(delta, -0.28, 0.28)

    def solve_pose(drop: float):
        data = mujoco.MjData(model)
        root_target = root0 + np.asarray([0.0, 0.0, -drop])

        def set_state(values: np.ndarray) -> None:
            data.qpos[:] = anchor_qpos
            data.qvel[:] = 0.0
            data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root0 + values[:3]
            data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = rotation_to_quat(
                Rotation.from_rotvec(values[3:6]) * root_rotation0
            )
            data.qpos[balance_qpos] = values[6:]
            mujoco.mj_forward(model, data)

        def residual(values: np.ndarray) -> np.ndarray:
            set_state(values)
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
                150.0 * (feet - feet0).ravel(),
                28.0 * foot_orientation,
                40.0 * (root - root_target),
                30.0 * torso_up[:2],
                0.03 * (values[6:] - joints0),
            ]

        seed = np.r_[np.asarray([0.0, 0.0, -drop, 0.0, 0.0, 0.0]), joints0]
        result = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=260,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        set_state(result.x)
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        physical = generator._physical(data)
        pose_qpos = data.qpos.copy()
        mapped_control = result.x[6:] + (base[balance] - joints0)

        def acceleration_residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[:] = pose_qpos
            data.qvel[:] = 0.0
            data.ctrl[:] = base
            data.ctrl[balance] = candidate
            mujoco.mj_forward(model, data)
            root_acceleration = data.qacc[
                engine.robot_dof : engine.robot_dof + 6
            ]
            joint_acceleration = data.qacc[balance_dofs]
            return np.r_[
                root_acceleration[:3],
                0.5 * root_acceleration[3:],
                0.12 * joint_acceleration,
                0.002 * (candidate - mapped_control),
            ]

        equilibrium = least_squares(
            acceleration_residual,
            np.clip(
                mapped_control,
                model.actuator_ctrlrange[balance, 0],
                model.actuator_ctrlrange[balance, 1],
            ),
            bounds=(
                model.actuator_ctrlrange[balance, 0],
                model.actuator_ctrlrange[balance, 1],
            ),
            max_nfev=140,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        mapped_acceleration_norm = float(
            np.linalg.norm(acceleration_residual(mapped_control)[:-len(balance)])
        )
        equilibrium_acceleration_norm = float(
            np.linalg.norm(acceleration_residual(equilibrium.x)[:-len(balance)])
        )
        return result.x[6:].copy(), equilibrium.x.copy(), {
            "maximum_foot_error_m": float(
                np.max(np.linalg.norm(feet - feet0, axis=1))
            ),
            "root_error_m": float(
                np.linalg.norm(
                    data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
                    - root_target
                )
            ),
            "torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "mapped_acceleration_residual": mapped_acceleration_norm,
            "equilibrium_acceleration_residual": equilibrium_acceleration_norm,
        }

    pose_cache = {drop: solve_pose(drop) for drop in (0.05, 0.08, 0.11)}

    def evaluate(spec: dict[str, float], record: bool = False):
        pose, squat_control, static = pose_cache[spec["root_drop_m"]]
        approach_end = 90.0 + spec["approach_frames"]
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        first_invalid = None
        reason = None
        nonhand = 0.0
        ready_run = maximum_ready_run = 0
        ctrl_physics = []
        trace = {key: [] for key in (
            "qpos", "qvel", "ctrl", "actuator_force", "foot_contact",
            "root_height", "torso_tilt", "nonhand_impulse",
        )}
        for step in range(300 * substeps):
            frame = step / substeps
            held = base.copy()
            if frame >= 90.0:
                amount = smoothstep((frame - 90.0) / spec["approach_frames"])
                held[balance] = (
                    (1.0 - amount) * base[balance] + amount * squat_control
                )
                held += feedback_correction(data)
            held = np.clip(
                held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            data.ctrl[:] = held
            ctrl_physics.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for index in range(data.ncon):
                contact = data.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if (
                    int(model.geom_bodyid[other]) in engine.robot_bodies
                    and other not in hand_geoms
                ):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    nonhand += float(np.linalg.norm(force[:3])) * model.opt.timestep
            root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
            velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
            ready = (
                frame >= approach_end
                and root[2] < root0[2] - 0.025
                and root[2] > 0.48
                and np.rad2deg(physical["tilt"]) < 32.0
                and np.linalg.norm(velocity[:3]) < 0.18
                and int(np.sum(physical["foot_contact"])) == 2
                and nonhand <= 0.10
            )
            if step % substeps == substeps - 1:
                ready_run = ready_run + 1 if ready else 0
                maximum_ready_run = max(maximum_ready_run, ready_run)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["root_height"].append(float(root[2]))
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["nonhand_impulse"].append(nonhand)
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if invalid is not None:
                first_invalid = step
                reason = invalid
                break
        physical = generator._physical(data)
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        valid = first_invalid is None
        metrics = {
            **spec,
            "full_horizon_valid": valid,
            "first_invalid_frame": None if valid else first_invalid / substeps,
            "invalid_reason": reason,
            "longest_ready_squat_frames": maximum_ready_run,
            "achieved_root_drop_m": float(root0[2] - root[2]),
            "final_root_velocity_m_s": velocity[:3].tolist(),
            "final_root_speed_m_s": float(np.linalg.norm(velocity[:3])),
            "final_root_height_m": float(root[2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "nonhand_object_impulse_ns": nonhand,
            "static_ik": static,
        }
        rank = (
            not (valid and maximum_ready_run >= 30),
            not valid,
            -maximum_ready_run,
            abs(metrics["achieved_root_drop_m"] - spec["root_drop_m"]),
            metrics["final_root_speed_m_s"],
            -(step / substeps),
        )
        return (
            rank,
            metrics,
            {key: np.asarray(value) for key, value in trace.items()},
            np.asarray(ctrl_physics),
            pose,
            squat_control,
        )

    specs = [
        {"root_drop_m": drop, "approach_frames": float(approach)}
        for drop, approach in itertools.product((0.05, 0.08, 0.11), (40, 70))
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
    _, metrics, trace, ctrl_physics, pose, squat_control = evaluate(
        best_spec, record=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        balance_actuators=balance,
        squat_pose_qpos=pose,
        squat_control=squat_control,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "feedback-stabilized vertical-torso double-support squat grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
