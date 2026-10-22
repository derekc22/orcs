#!/usr/bin/env python3
"""Probe stacked bilateral rear/top-edge tipping from upright double support."""

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


GROUPS = ("left_hand", "right_hand", "wrist", "feet", "legs", "torso_pelvis", "other_robot")
TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.stance)
    base = source["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    controlled = np.arange(model.nu, dtype=int)
    controlled_qpos = engine.actuator_qpos[controlled]
    joint_ids = model.actuator_trnid[controlled, 0].astype(int)
    lower = model.jnt_range[joint_ids, 0] + 0.003
    upper = model.jnt_range[joint_ids, 1] - 0.003
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    substeps = engine.config.substeps
    feedback_artifact = np.load(args.feedback)
    feedback_parameters = feedback_artifact["parameters"].copy()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    feedback_groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(token in name for token in tokens)],
            dtype=int,
        )
        for key, tokens in TOKENS.items()
    }

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    anchor_qpos = anchor.qpos.copy()
    nominal = anchor_qpos[controlled_qpos].copy()
    feet0 = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations0 = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    object_anchor = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    root0 = anchor_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_quaternion = anchor_qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    desired_rotation = (
        Rotation.from_quat(feedback_artifact["desired_rotation_xyzw"])
        if "desired_rotation_xyzw" in feedback_artifact
        else Rotation.from_quat(np.r_[root_quaternion[1:], root_quaternion[0]])
    )
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])

    def feedback_correction(data: mujoco.MjData) -> np.ndarray:
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        rotation_error = (desired_rotation.inv() * rotation).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal_features = np.asarray(
            [root[1] - root0[1], velocity[1], rotation_error[0], velocity[3]]
        )
        lateral_features = np.asarray(
            [root[0] - root0[0], velocity[0], rotation_error[1], velocity[4]]
        )
        gains = feedback_parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[feedback_groups[key]] += float(
                np.dot(gains[row], sagittal_features)
            )
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[feedback_groups[key]] += float(
                np.dot(gains[row], lateral_features)
            )
            row += 1
        return np.clip(delta, -0.28, 0.28)

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

    def solve_pose(
        left_lateral: float,
        right_lateral: float,
        depth: float,
        goalward_shift: float,
        vertical_shift: float,
        seed: np.ndarray | None = None,
    ):
        data = mujoco.MjData(model)
        targets = np.asarray(
            [
                object_anchor
                + left_lateral * lateral
                + (-0.205 + depth + goalward_shift) * forward,
                object_anchor
                + right_lateral * lateral
                + (-0.185 + depth + goalward_shift) * forward,
            ]
        )
        targets[:, 2] = object_anchor[2] + np.asarray([0.255, 0.195]) + vertical_shift

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[:] = anchor_qpos
            data.qvel[:] = 0.0
            data.qpos[controlled_qpos] = candidate
            mujoco.mj_forward(model, data)
            hands = np.asarray([data.geom_xpos[geom] for geom in engine.hand_geoms])
            hand_rotations = np.asarray(
                [data.geom_xmat[geom].reshape(3, 3) for geom in engine.hand_geoms]
            )
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
            return np.r_[
                80.0 * (hands - targets).ravel(),
                0.20 * (hand_rotations[0, :, 1] - forward),
                0.20 * (hand_rotations[1, :, 1] - forward),
                100.0 * (feet - feet0).ravel(),
                12.0 * foot_orientation,
                30.0 * torso_up[:2],
                0.02 * (candidate - nominal),
            ]

        result = least_squares(
            residual,
            np.clip(nominal if seed is None else seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=180,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        data.qpos[:] = anchor_qpos
        data.qvel[:] = 0.0
        data.qpos[controlled_qpos] = result.x
        mujoco.mj_forward(model, data)
        hands = np.asarray([data.geom_xpos[geom] for geom in engine.hand_geoms])
        physical = generator._physical(data)
        return (
            result.x.copy(),
            targets,
            np.linalg.norm(hands - targets, axis=1),
            float(np.rad2deg(physical["tilt"])),
        )

    pose_cache = {}
    right_lateral = 0.02
    depth = 0.045
    load_down = -0.015
    for left_lateral in (0.08, 0.12):
        contact = solve_pose(
            left_lateral, right_lateral, depth, 0.0, 0.0
        )
        for load_forward in (0.025,):
            load = solve_pose(
                left_lateral,
                right_lateral,
                depth,
                load_forward,
                load_down,
                contact[0],
            )
            pose_cache[
                (left_lateral, right_lateral, depth, load_forward, load_down)
            ] = (contact, load)

    def evaluate(spec: dict[str, float], record: bool = False):
        contact, load = pose_cache[
            (
                spec["left_lateral_m"],
                spec["right_lateral_m"],
                spec["depth_m"],
                spec["load_forward_m"],
                spec["load_down_m"],
            )
        ]
        contact_pose, targets, errors, static_tilt = contact
        load_pose, load_targets, load_errors, load_static_tilt = load
        contact_control = contact_pose + (base - nominal)
        load_control = load_pose + (base - nominal)
        approach_end = 90.0 + spec["approach_frames"]
        contact_end = approach_end + 6.0
        load_end = contact_end + spec["load_frames"]
        retract_end = load_end + spec["retract_frames"]
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        hand_frames = 0
        bilateral_frames = 0
        last_hand_frame = None
        first_invalid = None
        invalid_reason = None
        max_lift = 0.0
        max_goalward = 0.0
        max_speed = 0.0
        first_contact_speed = None
        ctrl_physics = []
        trace = {
            key: []
            for key in (
                "qpos", "qvel", "ctrl", "actuator_force", "hand_contact",
                "foot_contact", "object_position", "root_height", "torso_tilt",
                "object_impulse",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            if frame < 90.0:
                held = base.copy()
            elif frame < approach_end:
                amount = smoothstep((frame - 90.0) / spec["approach_frames"])
                held = (1.0 - amount) * base + amount * contact_control
            elif frame < contact_end:
                held = contact_control.copy()
            elif frame < load_end:
                amount = smoothstep(
                    (frame - contact_end) / spec["load_frames"]
                )
                held = (1.0 - amount) * contact_control + amount * load_control
            elif frame < retract_end:
                amount = smoothstep(
                    (frame - load_end) / spec["retract_frames"]
                )
                held = (1.0 - amount) * load_control + amount * base
            else:
                held = base.copy()
            if frame >= 90.0:
                held += feedback_correction(data)
            held = np.clip(held, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
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
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            velocity = data.qvel[engine.object_dof : engine.object_dof + 3]
            if frame >= 90.0:
                max_lift = max(max_lift, float(obj[2] - object_anchor[2]))
                max_goalward = max(max_goalward, float(np.dot(obj - object_anchor, forward)))
                max_speed = max(max_speed, float(np.dot(velocity, forward)))
            nonhand = sum(impulses[key] for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot"))
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if step % substeps == substeps - 1:
                count = int(np.sum(physical["hand_contact"]))
                hand_frames += int(count > 0)
                bilateral_frames += int(count == 2)
                if count:
                    last_hand_frame = int(step // substeps)
                    if first_contact_speed is None:
                        root_velocity = data.qvel[
                            engine.robot_dof : engine.robot_dof + 3
                        ]
                        first_contact_speed = float(np.linalg.norm(root_velocity))
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
                first_invalid = step
                invalid_reason = invalid
                break
        physical = generator._physical(data)
        endpoint = step / substeps
        metrics = {
            **spec,
            "static_hand_target_error_m": errors.tolist(),
            "static_torso_tilt_deg": static_tilt,
            "load_hand_target_error_m": load_errors.tolist(),
            "load_static_torso_tilt_deg": load_static_tilt,
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "last_hand_contact_frame": last_hand_frame,
            "hand_impulse_ns": impulses["left_hand"] + impulses["right_hand"],
            "nonhand_object_impulse_ns": nonhand,
            "maximum_postsettle_lift_m": max_lift,
            "maximum_goalward_displacement_m": max_goalward,
            "maximum_goalward_speed_m_s": max_speed,
            "first_contact_root_speed_m_s": first_contact_speed,
            "recovery_margin_frames": max(0.0, endpoint - load_end),
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        clean = hand_frames > 0 and nonhand <= 0.10
        rank = (
            not clean,
            not metrics["full_horizon_valid"],
            -metrics["recovery_margin_frames"],
            -bilateral_frames,
            -hand_frames,
            -max_goalward,
            float(np.max(errors)),
        )
        return (
            rank,
            metrics,
            {key: np.asarray(value) for key, value in trace.items()},
            np.asarray(ctrl_physics),
            contact_control,
            load_control,
            targets,
            load_targets,
        )

    specs = [
        {
            "left_lateral_m": left_lateral,
            "right_lateral_m": right_lateral,
            "depth_m": depth,
            "approach_frames": float(approach),
            "load_forward_m": load_forward,
            "load_down_m": load_down,
            "load_frames": float(load_frames),
            "retract_frames": 24.0,
        }
        for left_lateral, right_lateral, depth, approach, load_forward, load_down, load_frames in itertools.product(
            (0.08, 0.12), (0.02,), (0.045,), (55,),
            (0.025,), (-0.015,), (8,)
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
    (
        _, metrics, trace, ctrl_physics, contact_control, load_control,
        targets, load_targets,
    ) = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        contact_control=contact_control,
        load_control=load_control,
        hand_targets=targets,
        load_hand_targets=load_targets,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "yaw-aligned upright stacked bilateral rear-edge recovery probe",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
