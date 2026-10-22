#!/usr/bin/env python3
"""Probe a fixed-foot whole-body one-hand tilt with explicit recovery margin."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    balance_target = stance["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    right = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name.startswith("right_")
            and any(token in name for token in ("shoulder", "elbow", "wrist"))
        ],
        dtype=int,
    )
    controlled = np.unique(np.r_[generator.balance_actuators, right]).astype(int)
    controlled_qpos = engine.actuator_qpos[controlled]
    controlled_joints = model.actuator_trnid[controlled, 0].astype(int)
    lower = model.jnt_range[controlled_joints, 0] + 0.003
    upper = model.jnt_range[controlled_joints, 1] - 0.003
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    goal = engine.reference_object_pos[-1]
    settle = 90

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(settle * engine.config.substeps):
        anchor.ctrl[:] = balance_target
        mujoco.mj_step(model, anchor)
    anchor_qpos = anchor.qpos.copy()
    object_anchor = anchor_qpos[engine.object_qpos : engine.object_qpos + 3]
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    foot_positions = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    quaternion_anchor = anchor_qpos[engine.object_qpos + 3 : engine.object_qpos + 7].copy()

    def contact_group(geom: int) -> str:
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

    def solve(height: float, lateral_offset: float, along: float, seed: np.ndarray | None = None):
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = 0.0
        target = object_anchor + lateral_offset * lateral + along * forward
        target[2] = object_anchor[2] + height
        nominal = anchor_qpos[controlled_qpos]

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[controlled_qpos] = candidate
            mujoco.mj_forward(model, data)
            rotation = data.geom_xmat[engine.hand_geoms[1]].reshape(3, 3)
            current_feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            current_foot_rotations = np.asarray(
                [data.xmat[body].reshape(3, 3) for body in engine.foot_bodies]
            )
            foot_orientation = np.concatenate(
                [
                    Rotation.from_matrix(
                        current_foot_rotations[index] @ foot_rotations[index].T
                    ).as_rotvec()
                    for index in range(2)
                ]
            )
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            return np.r_[
                80.0 * (data.geom_xpos[engine.hand_geoms[1]] - target),
                0.25 * (rotation[:, 1] - forward),
                100.0 * (current_feet - foot_positions).ravel(),
                12.0 * foot_orientation,
                3.0 * torso_up[:2],
                0.02 * (candidate - nominal),
            ]

        result = least_squares(
            residual,
            np.clip(nominal if seed is None else np.asarray(seed), lower, upper),
            bounds=(lower, upper),
            max_nfev=250,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        data.qpos[controlled_qpos] = result.x
        mujoco.mj_forward(model, data)
        error = float(np.linalg.norm(data.geom_xpos[engine.hand_geoms[1]] - target))
        return result.x.copy(), target, error

    def evaluate(spec: dict[str, float], record: bool = False):
        approach = int(spec["approach_frames"])
        contact_end = settle + approach
        push_start = contact_end + 8
        push_end = push_start + 18
        retract_end = push_end + 30
        contact_arm, contact_target, contact_error = solve(
            spec["height_m"], spec["lateral_m"], spec["along_m"]
        )
        push_arm, push_target, push_error = solve(
            spec["height_m"],
            spec["lateral_m"],
            spec["along_m"] + spec["push_m"],
            contact_arm,
        )
        initial_control = balance_target[controlled].copy()
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        object0 = data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
        impulses = {key: 0.0 for key in GROUPS}
        hand_frames = bilateral_frames = 0
        first_invalid = None
        invalid_reason = None
        last_hand_frame = None
        peak_lift = 0.0
        peak_goalward_speed = 0.0
        peak_tip = 0.0
        ctrl_physics = []
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
            )
        }
        held = balance_target.copy()
        for step in range(300 * engine.config.substeps):
            frame = step / engine.config.substeps
            if step % engine.config.substeps == 0:
                held = balance_target.copy()
                if frame < settle:
                    target_arm = initial_control
                elif frame < contact_end:
                    amount = smoothstep((frame - settle) / approach)
                    target_arm = (1.0 - amount) * initial_control + amount * contact_arm
                elif frame < push_start:
                    target_arm = contact_arm
                elif frame < push_end:
                    amount = smoothstep((frame - push_start) / (push_end - push_start))
                    target_arm = (1.0 - amount) * contact_arm + amount * push_arm
                elif frame < retract_end:
                    amount = smoothstep((frame - push_end) / (retract_end - push_end))
                    target_arm = (1.0 - amount) * push_arm + amount * initial_control
                else:
                    target_arm = initial_control
                held[controlled] = target_arm
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
                impulses[contact_group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            if frame >= settle:
                peak_lift = max(peak_lift, float(obj[2] - 0.18545216))
            velocity = data.qvel[engine.object_dof : engine.object_dof + 3]
            peak_goalward_speed = max(peak_goalward_speed, float(np.dot(velocity, forward)))
            q = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            r0 = Rotation.from_quat(
                np.r_[quaternion_anchor[1:], quaternion_anchor[0]]
            )
            r = Rotation.from_quat(np.r_[q[1:], q[0]])
            peak_tip = max(peak_tip, float((r0.inv() * r).magnitude()))
            disallowed = sum(impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot"))
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and disallowed > 0.10:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hands = int(np.sum(physical["hand_contact"]))
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                if hands:
                    last_hand_frame = int(step // engine.config.substeps)
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
        endpoint = step / engine.config.substeps
        recovery_margin = max(0.0, endpoint - push_end)
        physical = generator._physical(data)
        metrics = {
            **spec,
            "contact_ik_error_m": contact_error,
            "push_ik_error_m": push_error,
            "contact_target_m": contact_target.tolist(),
            "push_target_m": push_target.tolist(),
            "push_end_frame": float(push_end),
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": invalid_reason,
            "recovery_margin_frames": recovery_margin,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "last_hand_contact_frame": last_hand_frame,
            "maximum_lift_m": peak_lift,
            "maximum_goalward_speed_m_s": peak_goalward_speed,
            "maximum_object_tip_rad": peak_tip,
            "object_impulse_ns": impulses,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        return metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics), contact_arm, push_arm

    specs = [
        {
            "height_m": height,
            "lateral_m": lateral_offset,
            "along_m": along,
            "approach_frames": float(approach),
            "push_m": push,
        }
        for height, lateral_offset, along, approach, push in itertools.product(
            (0.16, 0.18, 0.20, 0.22),
            (-0.18, -0.14),
            (-0.220, -0.200),
            (45,),
            (0.025,),
        )
    ]
    trials = []
    best_rank = None
    best_spec = None
    started = time.perf_counter()
    for index, spec in enumerate(specs):
        metrics, _, _, _, _ = evaluate(spec)
        disallowed = sum(metrics["object_impulse_ns"][key] for key in ("feet", "legs", "torso_pelvis", "other_robot"))
        useful = metrics["hand_contact_frames"] > 0 and metrics["maximum_object_tip_rad"] > 0.01 and disallowed <= 0.10
        rank = (
            not useful,
            not metrics["full_horizon_valid"],
            -metrics["recovery_margin_frames"],
            -metrics["hand_contact_frames"],
            -metrics["maximum_object_tip_rad"],
            metrics["contact_ik_error_m"],
        )
        trials.append(metrics)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_spec = spec
        print("TRIAL", index, json.dumps(metrics), flush=True)

    assert best_spec is not None
    metrics, trace, ctrl_physics, contact_arm, push_arm = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        balance_target=balance_target,
        controlled_actuators=controlled,
        contact_arm_target=contact_arm,
        push_arm_target=push_arm,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "legal fixed-foot whole-body right-hand high-face tilt/retract recovery grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
