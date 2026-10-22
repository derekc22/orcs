#!/usr/bin/env python3
"""Test side-face hand loading with upright feedback active before contact."""

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


TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}
GROUPS = ("left_hand", "right_hand", "wrist", "feet", "legs", "torso_pelvis", "other_robot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--side-pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    feedback = np.load(args.feedback)
    side = np.load(args.side_pose)
    initial_qpos = feedback["initial_qpos"].copy()
    initial_qvel = feedback["initial_qvel"].copy()
    base = feedback["base_target"].copy()
    parameters = feedback["parameters"].copy()
    controlled = side["controlled_actuators"].astype(int)
    contact_control = side["contact_pose"] + side["equilibrium_offset"]
    load_control = side["load_pose"] + side["equilibrium_offset"]
    substeps = engine.config.substeps
    object_geoms = set(map(int, engine.object_geoms))
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(token in name for token in tokens)],
            dtype=int,
        )
        for key, tokens in TOKENS.items()
    }
    sagittal_groups = ("hip_pitch", "knee", "ankle_pitch", "waist_pitch")
    lateral_groups = ("hip_roll", "ankle_roll", "waist_roll")

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    desired_root = anchor.qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_quaternion = anchor.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    desired_rotation = Rotation.from_quat(np.r_[root_quaternion[1:], root_quaternion[0]])
    object_anchor = anchor.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    object_quaternion = anchor.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    object_rotation = Rotation.from_quat(np.r_[object_quaternion[1:], object_quaternion[0]])
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])

    def balance_correction(data: mujoco.MjData) -> np.ndarray:
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        rotation_error = (desired_rotation.inv() * rotation).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal_features = np.asarray(
            [root[1] - desired_root[1], velocity[1], rotation_error[0], velocity[3]]
        )
        lateral_features = np.asarray(
            [root[0] - desired_root[0], velocity[0], rotation_error[1], velocity[4]]
        )
        gains = parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in sagittal_groups:
            delta[groups[key]] += float(np.dot(gains[row], sagittal_features))
            row += 1
        for key in lateral_groups:
            delta[groups[key]] += float(np.dot(gains[row], lateral_features))
            row += 1
        return np.clip(delta, -0.28, 0.28)

    def contact_group(geom: int) -> str:
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

    def evaluate(spec: dict[str, float], record: bool = False):
        approach_end = 90.0 + spec["approach_frames"]
        load_end = approach_end + spec["load_frames"]
        hold_end = load_end + 3.0
        retract_end = hold_end + spec["retract_frames"]
        scaled_load = contact_control + spec["load_scale"] * (load_control - contact_control)
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        hand_frames = 0
        last_hand_frame = None
        first_invalid = None
        invalid_reason = None
        first_contact_root_speed = None
        first_contact_vertical_speed = None
        max_lift = max_goalward = max_lateral = max_goalward_speed = max_tip = 0.0
        recovery_run = max_recovery_run = 0
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
            held = base.copy()
            if frame >= 90.0:
                if frame < approach_end:
                    amount = smoothstep((frame - 90.0) / spec["approach_frames"])
                    held[controlled] = (1.0 - amount) * base[controlled] + amount * contact_control
                elif frame < load_end:
                    amount = smoothstep((frame - approach_end) / spec["load_frames"])
                    held[controlled] = (1.0 - amount) * contact_control + amount * scaled_load
                elif frame < hold_end:
                    held[controlled] = scaled_load
                elif frame < retract_end:
                    amount = smoothstep((frame - hold_end) / spec["retract_frames"])
                    held[controlled] = (1.0 - amount) * scaled_load + amount * base[controlled]
                held += balance_correction(data)
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
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                impulses[contact_group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            velocity = data.qvel[engine.object_dof : engine.object_dof + 3]
            quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
            if frame >= 90.0:
                max_lift = max(max_lift, float(obj[2] - object_anchor[2]))
                max_goalward = max(max_goalward, float(np.dot(obj - object_anchor, forward)))
                max_lateral = max(max_lateral, abs(float(np.dot(obj - object_anchor, lateral))))
                max_goalward_speed = max(max_goalward_speed, float(np.dot(velocity, forward)))
                max_tip = max(max_tip, float((object_rotation.inv() * rotation).magnitude()))
            nonhand = sum(
                impulses[key]
                for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            )
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if step % substeps == substeps - 1:
                count = int(np.sum(physical["hand_contact"]))
                hand_frames += int(count > 0)
                if count:
                    last_hand_frame = int(step // substeps)
                    if first_contact_root_speed is None:
                        root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
                        first_contact_root_speed = float(np.linalg.norm(root_velocity))
                        first_contact_vertical_speed = float(root_velocity[2])
                recovered = (
                    frame >= retract_end
                    and data.qpos[engine.robot_qpos + 2] > 0.60
                    and np.rad2deg(physical["tilt"]) < 35.0
                    and np.linalg.norm(data.qvel[engine.robot_dof : engine.robot_dof + 3]) < 0.25
                    and int(np.sum(physical["foot_contact"])) == 2
                )
                recovery_run = recovery_run + 1 if recovered else 0
                max_recovery_run = max(max_recovery_run, recovery_run)
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
        lower_body = sum(
            impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        metrics = {
            **spec,
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "hand_contact_frames": hand_frames,
            "last_hand_contact_frame": last_hand_frame,
            "left_right_hand_impulse_ns": [impulses["left_hand"], impulses["right_hand"]],
            "wrist_impulse_ns": impulses["wrist"],
            "lower_body_torso_impulse_ns": lower_body,
            "first_contact_root_speed_m_s": first_contact_root_speed,
            "first_contact_vertical_speed_m_s": first_contact_vertical_speed,
            "maximum_postsettle_lift_m": max_lift,
            "maximum_goalward_displacement_m": max_goalward,
            "maximum_lateral_displacement_m": max_lateral,
            "maximum_goalward_speed_m_s": max_goalward_speed,
            "maximum_object_tip_rad": max_tip,
            "recovery_margin_after_load_frames": max(0.0, endpoint - load_end),
            "maximum_recovered_run_frames": max_recovery_run,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        clean = hand_frames > 0 and impulses["wrist"] <= 0.10 and lower_body <= 0.10
        deliberate = first_contact_root_speed is not None and first_contact_root_speed < 0.50
        motion = max_lift > 0.001 or max_goalward > 0.002 or max_tip > 0.01
        recoverable = max_recovery_run >= 20
        rank = (
            not (clean and deliberate and motion and recoverable),
            not clean,
            not deliberate,
            not recoverable,
            -max_recovery_run,
            -max_lift,
            -max_goalward,
            -max_tip,
            -hand_frames,
            -(step / substeps),
        )
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    specs = [
        {
            "approach_frames": float(approach),
            "load_frames": float(load_frames),
            "load_scale": float(load_scale),
            "retract_frames": float(retract),
        }
        for approach, load_frames, load_scale, retract in itertools.product(
            (40, 55), (8, 14), (0.5, 1.0), (12, 20)
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
    _, metrics, trace, ctrl_physics = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        feedback_parameters=parameters,
        controlled_actuators=controlled,
        contact_control=contact_control,
        load_control=load_control,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "upright-feedback side-face load/retract recovery-margin grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
