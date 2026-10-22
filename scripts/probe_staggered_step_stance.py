#!/usr/bin/env python3
"""Legally step the upright G1 toward a bilateral manipulation-ready stance."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-distance", type=float, default=0.12)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.stance)
    base = source["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    balance_qpos = engine.actuator_qpos[balance]
    feedback_artifact = None if args.feedback is None else np.load(args.feedback)
    feedback_parameters = (
        None if feedback_artifact is None else feedback_artifact["parameters"].copy()
    )
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    feedback_groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(token in name for token in tokens)],
            dtype=int,
        )
        for key, tokens in TOKENS.items()
    }
    joint_ids = model.actuator_trnid[balance, 0].astype(int)
    lower = np.r_[
        [-0.30, -0.30, -0.16, -0.55, -0.55, -0.55],
        model.jnt_range[joint_ids, 0] + 0.004,
    ]
    upper = np.r_[
        [0.30, 0.30, 0.16, 0.55, 0.55, 0.55],
        model.jnt_range[joint_ids, 1] - 0.004,
    ]
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
    joints0 = anchor_qpos[balance_qpos].copy()
    feet0 = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations0 = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    object_position = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    root_quaternion = anchor_qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    desired_rotation = (
        Rotation.from_quat(feedback_artifact["desired_rotation_xyzw"])
        if feedback_artifact is not None
        and "desired_rotation_xyzw" in feedback_artifact
        else Rotation.from_quat(np.r_[root_quaternion[1:], root_quaternion[0]])
    )
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_position[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])

    def feedback_correction(
        data: mujoco.MjData, desired_root_position: np.ndarray
    ) -> np.ndarray:
        if feedback_parameters is None:
            return np.zeros(model.nu)
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        rotation_error = (desired_rotation.inv() * rotation).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal_features = np.asarray(
            [
                root[1] - desired_root_position[1],
                velocity[1],
                rotation_error[0],
                velocity[3],
            ]
        )
        lateral_features = np.asarray(
            [
                root[0] - desired_root_position[0],
                velocity[0],
                rotation_error[1],
                velocity[4],
            ]
        )
        gains = feedback_parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[feedback_groups[key]] += float(np.dot(gains[row], sagittal_features))
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[feedback_groups[key]] += float(np.dot(gains[row], lateral_features))
            row += 1
        return np.clip(delta, -0.28, 0.28)

    def solve_pose(
        seed: np.ndarray,
        feet_target: np.ndarray,
        root_target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        data = mujoco.MjData(model)

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
                140.0 * (feet - feet_target).ravel(),
                28.0 * foot_orientation,
                30.0 * (root - root_target),
                12.0 * torso_up[:2],
                0.04 * (values[6:] - joints0),
            ]

        result = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=350,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        set_state(result.x)
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        physical = generator._physical(data)
        diagnostics = {
            "cost": float(result.cost),
            "maximum_foot_error_m": float(
                np.max(np.linalg.norm(feet - feet_target, axis=1))
            ),
            "root_error_m": float(
                np.linalg.norm(
                    data.qpos[engine.robot_qpos : engine.robot_qpos + 3] - root_target
                )
            ),
            "torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
        }
        return result.x.copy(), diagnostics

    def make_poses(step_length: float, lift: float, first_foot: int):
        second_foot = 1 - first_foot
        shift_sign = 1.0 if first_foot == 0 else -1.0
        root_load_first = root0 + 0.055 * shift_sign * lateral
        root_load_second = root0 + 0.60 * step_length * forward - 0.055 * shift_sign * lateral
        root_final = root0 + step_length * forward
        targets: list[tuple[np.ndarray, np.ndarray]] = []
        targets.append((feet0.copy(), root_load_first))
        feet = feet0.copy()
        feet[first_foot] += step_length * forward + np.asarray([0.0, 0.0, lift])
        targets.append((feet.copy(), root_load_first + 0.20 * step_length * forward))
        feet[first_foot, 2] = feet0[first_foot, 2]
        targets.append((feet.copy(), root0 + 0.42 * step_length * forward))
        targets.append((feet.copy(), root_load_second))
        feet[second_foot] += step_length * forward + np.asarray([0.0, 0.0, lift])
        targets.append((feet.copy(), root_load_second + 0.18 * step_length * forward))
        feet[second_foot, 2] = feet0[second_foot, 2]
        targets.append((feet.copy(), root_final))
        seed = np.r_[np.zeros(6), joints0]
        poses = []
        root_targets = []
        diagnostics = []
        for feet_target, root_target in targets:
            seed, diagnostic = solve_pose(seed, feet_target, root_target)
            poses.append(seed[6:].copy())
            root_targets.append(root_target.copy())
            diagnostics.append(diagnostic)
        return np.asarray(poses), np.asarray(root_targets), diagnostics

    def make_shuffle_poses(step_length: float):
        seed = np.r_[np.zeros(6), joints0]
        poses = []
        root_targets = []
        diagnostics = []
        for fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            offset = fraction * step_length * forward
            seed, diagnostic = solve_pose(seed, feet0 + offset, root0 + offset)
            poses.append(seed[6:].copy())
            root_targets.append((root0 + offset).copy())
            diagnostics.append(diagnostic)
        return np.asarray(poses), np.asarray(root_targets), diagnostics

    pose_cache = {}
    if args.shuffle:
        pose_cache[(args.shuffle_distance, 0.0, -1)] = make_shuffle_poses(
            args.shuffle_distance
        )
    else:
        for step_length, lift, first_foot in itertools.product(
            (0.14,), (0.060,), (0, 1)
        ):
            pose_cache[(step_length, lift, first_foot)] = make_poses(
                step_length, lift, first_foot
            )

    def evaluate(spec: dict[str, float | int], record: bool = False):
        poses, root_targets, diagnostics = pose_cache[
            (spec["step_length_m"], spec["swing_height_m"], spec["first_foot"])
        ]
        duration_scale = spec["duration_scale"]
        durations = duration_scale * (
            np.asarray([30.0, 30.0, 30.0])
            if args.shuffle
            else np.asarray([14.0, 22.0, 10.0, 14.0, 22.0, 10.0])
        )
        frames = np.r_[90.0, 90.0 + np.cumsum(durations)]
        pose_targets = poses + (base[balance] - joints0)
        pose_controls = np.vstack([base[balance], pose_targets])
        root_controls = np.vstack([root0, root_targets])
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        nonhand = 0.0
        first_invalid = None
        invalid_reason = None
        ctrl_physics = []
        trace = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "actuator_force",
                "foot_contact",
                "root_height",
                "torso_tilt",
                "nonhand_impulse",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            held = base.copy()
            if frame >= frames[0]:
                segment = min(
                    int(np.searchsorted(frames, frame, side="right") - 1),
                    len(frames) - 2,
                )
                amount = smoothstep(
                    (frame - frames[segment])
                    / max(frames[segment + 1] - frames[segment], 1.0e-9)
                )
                held[balance] = (
                    (1.0 - amount) * pose_controls[segment]
                    + amount * pose_controls[segment + 1]
                )
                desired_root_position = (
                    (1.0 - amount) * root_controls[segment]
                    + amount * root_controls[segment + 1]
                )
                held += feedback_correction(data, desired_root_position)
            held = np.clip(
                held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            data.ctrl[:] = held
            ctrl_physics.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if (
                    int(model.geom_bodyid[other]) in engine.robot_bodies
                    and other not in hand_geoms
                ):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    nonhand += float(np.linalg.norm(force[:3])) * model.opt.timestep
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if step % substeps == substeps - 1 and record:
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(data.ctrl.copy())
                trace["actuator_force"].append(data.actuator_force.copy())
                trace["foot_contact"].append(physical["foot_contact"].copy())
                trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
                trace["torso_tilt"].append(float(physical["tilt"]))
                trace["nonhand_impulse"].append(nonhand)
            if invalid is not None:
                first_invalid = step
                invalid_reason = invalid
                break
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        physical = generator._physical(data)
        achieved = float(np.dot(root - root0, forward))
        lateral_drift = float(abs(np.dot(root - root0, lateral)))
        metrics = {
            **spec,
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "command_end_frame": float(frames[-1]),
            "goalward_root_translation_m": achieved,
            "lateral_root_drift_m": lateral_drift,
            "final_root_position_m": root.tolist(),
            "final_root_velocity_m_s": velocity[:3].tolist(),
            "final_root_height_m": float(root[2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "nonhand_object_impulse_ns": nonhand,
            "maximum_static_ik_foot_error_m": max(
                item["maximum_foot_error_m"] for item in diagnostics
            ),
            "maximum_static_ik_root_error_m": max(
                item["root_error_m"] for item in diagnostics
            ),
        }
        stopped = float(np.linalg.norm(velocity[:3])) < 0.25
        rank = (
            not metrics["full_horizon_valid"],
            not (achieved > 0.07 and stopped and nonhand <= 0.10),
            abs(achieved - spec["step_length_m"]),
            float(np.linalg.norm(velocity[:3])),
            lateral_drift,
            -(step / substeps),
        )
        return (
            rank,
            metrics,
            {key: np.asarray(value) for key, value in trace.items()},
            np.asarray(ctrl_physics),
            poses,
            root_targets,
            frames,
        )

    if args.shuffle:
        specs = [
            {
                "step_length_m": args.shuffle_distance,
                "swing_height_m": 0.0,
                "first_foot": -1,
                "duration_scale": duration_scale,
            }
            for duration_scale in (0.8, 1.2)
        ]
    else:
        specs = [
            {
                "step_length_m": step_length,
                "swing_height_m": lift,
                "first_foot": first_foot,
                "duration_scale": duration_scale,
            }
            for step_length, lift, first_foot, duration_scale in itertools.product(
                (0.14,), (0.060,), (0, 1), (0.85, 1.15)
            )
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
    _, metrics, trace, ctrl_physics, poses, root_targets, frames = evaluate(
        best_spec, record=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        balance_actuators=balance,
        pose_qpos=poses,
        pose_controls=poses + (base[balance] - joints0),
        root_targets=root_targets,
        phase_frames=frames,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": (
            "legal double-support three-stage forward shuffle probe"
            if args.shuffle
            else "legal two-step translated-support manipulation-stance probe"
        ),
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
