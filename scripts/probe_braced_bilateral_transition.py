#!/usr/bin/env python3
"""Transition from the clean right-hand brace to bilateral box support."""

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


GROUPS = (
    "left_hand", "right_hand", "wrist", "feet", "legs", "torso_pelvis",
    "other_robot",
)
TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}


def quat_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    feedback = np.load(args.feedback)
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    source_ctrl = source["ctrl_physics"].copy()
    base = source["base_target"].copy()
    feedback_parameters = feedback["parameters"].copy()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    feedback_groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(token in name for token in tokens)],
            dtype=int,
        )
        for key, tokens in TOKENS.items()
    }
    all_actuators = np.arange(model.nu, dtype=int)
    actuator_qpos = engine.actuator_qpos[all_actuators]
    joints = model.actuator_trnid[:, 0].astype(int)
    lower = model.jnt_range[joints, 0] + 0.003
    upper = model.jnt_range[joints, 1] - 0.003
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    substeps = engine.config.substeps
    source_index = 132
    transition_start = 133.0

    qpos_anchor = source["qpos"][source_index].copy()
    anchor = mujoco.MjData(model)
    anchor.qpos[:] = qpos_anchor
    anchor.qvel[:] = 0.0
    mujoco.mj_forward(model, anchor)
    nominal = qpos_anchor[actuator_qpos].copy()
    feet_anchor = np.asarray(
        [anchor.xpos[body].copy() for body in engine.foot_bodies]
    )
    right_anchor = anchor.geom_xpos[engine.hand_geoms[1]].copy()
    object_anchor = qpos_anchor[
        engine.object_qpos : engine.object_qpos + 3
    ].copy()
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    source_control = source_ctrl[int(transition_start * substeps) - 1].copy()
    control_offset = source_control - nominal
    desired_root = feedback["desired_root"].copy()
    desired_rotation = Rotation.from_quat(feedback["desired_rotation_xyzw"])

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

    def feedback_correction(data: mujoco.MjData) -> np.ndarray:
        rotation = quat_rotation(
            data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        )
        rotation_error = (desired_rotation.inv() * rotation).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal = np.asarray(
            [root[1] - desired_root[1], velocity[1], rotation_error[0], velocity[3]]
        )
        lateral_features = np.asarray(
            [root[0] - desired_root[0], velocity[0], rotation_error[1], velocity[4]]
        )
        gains = feedback_parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[feedback_groups[key]] += float(np.dot(gains[row], sagittal))
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[feedback_groups[key]] += float(
                np.dot(gains[row], lateral_features)
            )
            row += 1
        return np.clip(delta, -0.28, 0.28)

    def solve_pose(left_lateral: float, height: float):
        data = mujoco.MjData(model)
        target = object_anchor + left_lateral * lateral - 0.17 * forward
        target[2] = object_anchor[2] + height

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[:] = qpos_anchor
            data.qvel[:] = 0.0
            data.qpos[actuator_qpos] = candidate
            mujoco.mj_forward(model, data)
            hands = np.asarray(
                [data.geom_xpos[geom] for geom in engine.hand_geoms]
            )
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            return np.r_[
                90.0 * (hands[0] - target),
                110.0 * (hands[1] - right_anchor),
                110.0 * (feet - feet_anchor).ravel(),
                5.0 * torso_up[:2],
                0.02 * (candidate - nominal),
            ]

        result = least_squares(
            residual,
            np.clip(nominal, lower, upper),
            bounds=(lower, upper),
            max_nfev=240,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        data.qpos[:] = qpos_anchor
        data.qvel[:] = 0.0
        data.qpos[actuator_qpos] = result.x
        mujoco.mj_forward(model, data)
        hands = np.asarray([data.geom_xpos[geom] for geom in engine.hand_geoms])
        return result.x.copy(), target, {
            "left_error_m": float(np.linalg.norm(hands[0] - target)),
            "right_error_m": float(np.linalg.norm(hands[1] - right_anchor)),
            "maximum_foot_error_m": float(
                np.max(
                    np.linalg.norm(
                        np.asarray([data.xpos[body] for body in engine.foot_bodies])
                        - feet_anchor,
                        axis=1,
                    )
                )
            ),
            "torso_tilt_deg": float(np.rad2deg(generator._physical(data)["tilt"])),
            "static_hand_contact": generator._physical(data)["hand_contact"].tolist(),
        }

    pose_cache = {
        (left_lateral, height): solve_pose(left_lateral, height)
        for left_lateral, height in itertools.product((0.06, 0.12), (0.22, 0.28))
    }

    def evaluate(spec: dict[str, float], record: bool = False):
        pose, left_target, static = pose_cache[
            (spec["left_lateral_m"], spec["left_height_m"])
        ]
        bilateral_control = pose + control_offset
        approach_end = transition_start + spec["approach_frames"]
        hold_end = approach_end + 6.0
        retract_end = hold_end + 28.0
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        first_invalid = None
        reason = None
        hand_frames = bilateral_frames = 0
        first_bilateral_speed = None
        last_hand_frame = None
        max_lift = max_goalward = 0.0
        recovered_run = maximum_recovered_run = 0
        ctrl_physics = []
        trace = {key: [] for key in (
            "qpos", "qvel", "ctrl", "actuator_force", "hand_contact",
            "foot_contact", "object_position", "root_height", "torso_tilt",
            "object_impulse",
        )}
        for step in range(300 * substeps):
            frame = step / substeps
            if frame < transition_start:
                held = source_ctrl[min(step, len(source_ctrl) - 1)].copy()
            elif frame < approach_end:
                amount = smoothstep(
                    (frame - transition_start) / spec["approach_frames"]
                )
                held = (1.0 - amount) * source_control + amount * bilateral_control
                held += feedback_correction(data)
            elif frame < hold_end:
                held = bilateral_control + feedback_correction(data)
            elif frame < retract_end:
                amount = smoothstep((frame - hold_end) / 28.0)
                held = (1.0 - amount) * bilateral_control + amount * base
                held += feedback_correction(data)
            else:
                held = base + feedback_correction(data)
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
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                impulses[group(other)] += (
                    float(np.linalg.norm(force[:3])) * model.opt.timestep
                )
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            if frame >= 90.0:
                max_lift = max(max_lift, float(obj[2] - object_anchor[2]))
                max_goalward = max(
                    max_goalward, float(np.dot(obj - object_anchor, forward))
                )
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
                bilateral_frames += int(count == 2)
                if count:
                    last_hand_frame = int(step // substeps)
                if count == 2 and first_bilateral_speed is None:
                    velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
                    first_bilateral_speed = float(np.linalg.norm(velocity))
                recovered = (
                    frame >= retract_end
                    and data.qpos[engine.robot_qpos + 2] > 0.57
                    and np.rad2deg(physical["tilt"]) < 35.0
                    and int(np.sum(physical["foot_contact"])) == 2
                )
                recovered_run = recovered_run + 1 if recovered else 0
                maximum_recovered_run = max(maximum_recovered_run, recovered_run)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["actuator_force"].append(data.actuator_force.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(obj.copy())
                    trace["root_height"].append(
                        float(data.qpos[engine.robot_qpos + 2])
                    )
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(
                        np.asarray(list(impulses.values()))
                    )
            if invalid is not None:
                first_invalid = step
                reason = invalid
                break
        endpoint = step / substeps
        physical = generator._physical(data)
        metrics = {
            **spec,
            "static_ik": static,
            "left_target_m": left_target.tolist(),
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": (
                None if first_invalid is None else first_invalid / substeps
            ),
            "invalid_reason": reason,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "last_hand_contact_frame": last_hand_frame,
            "first_bilateral_root_speed_m_s": first_bilateral_speed,
            "hand_impulse_ns": impulses["left_hand"] + impulses["right_hand"],
            "nonhand_object_impulse_ns": nonhand,
            "maximum_lift_m": max_lift,
            "maximum_goalward_displacement_m": max_goalward,
            "recovery_margin_frames": max(0.0, endpoint - hold_end),
            "maximum_recovered_run_frames": maximum_recovered_run,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        clean = bilateral_frames > 0 and nonhand <= 0.10
        rank = (
            not clean,
            nonhand > 0.10,
            not metrics["full_horizon_valid"],
            -bilateral_frames,
            -hand_frames,
            -metrics["maximum_recovered_run_frames"],
            -metrics["recovery_margin_frames"],
            first_bilateral_speed if first_bilateral_speed is not None else 10.0,
        )
        return (
            rank,
            metrics,
            {key: np.asarray(value) for key, value in trace.items()},
            np.asarray(ctrl_physics),
            bilateral_control,
        )

    specs = [
        {
            "left_lateral_m": left_lateral,
            "left_height_m": height,
            "approach_frames": float(duration),
        }
        for left_lateral, height, duration in itertools.product(
            (0.06, 0.12), (0.22, 0.28), (18, 30)
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
    _, metrics, trace, ctrl_physics, bilateral_control = evaluate(
        best_spec, record=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        source_ctrl_physics=source_ctrl,
        ctrl_physics=ctrl_physics,
        bilateral_control=bilateral_control,
        ctrl_transition_start=np.asarray(transition_start),
        **trace,
    )
    report = {
        "method": "clean right-hand brace to bilateral support transition grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
