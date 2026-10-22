#!/usr/bin/env python3
"""Structured hand-load and recovery probes from the legal v3 lean transition."""

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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pose-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    pose_source = np.load(args.pose_source)
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    source_ctrl = source["ctrl_physics"].copy()
    base = source["base_target"].copy()
    controlled = pose_source["controlled_actuators"].astype(int)
    controlled_qpos = engine.actuator_qpos[controlled]
    controlled_joints = model.actuator_trnid[controlled, 0].astype(int)
    lower = model.jnt_range[controlled_joints, 0] + 0.003
    upper = model.jnt_range[controlled_joints, 1] - 0.003
    source_pose = pose_source["push_arm_target"].copy()
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    goal = engine.reference_object_pos[-1]
    substeps = engine.config.substeps

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    anchor_qpos = anchor.qpos.copy()
    object_anchor = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    foot_positions = np.asarray([anchor.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )

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

    def solve_pose(along: float, height: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        data = mujoco.MjData(model)
        data.qpos[:] = anchor_qpos
        data.qvel[:] = 0.0
        target = object_anchor - 0.14 * lateral + along * forward
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
            np.clip(source_pose, lower, upper),
            bounds=(lower, upper),
            max_nfev=300,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        data.qpos[controlled_qpos] = result.x
        mujoco.mj_forward(model, data)
        error = float(np.linalg.norm(data.geom_xpos[engine.hand_geoms[1]] - target))
        torso_tilt = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        data.xmat[engine.torso_body].reshape(3, 3)[2, 2], -1.0, 1.0
                    )
                )
            )
        )
        return result.x.copy(), target, error, torso_tilt

    pose_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, float, float]] = {}
    for extra_along, height_offset in itertools.product(
        (0.050, 0.080, 0.110), (-0.040, -0.060, -0.080)
    ):
        key = (extra_along, height_offset)
        pose_cache[key] = solve_pose(-0.195 + extra_along, 0.220 + height_offset)

    def envelope(frame: float, start: float, ramp: float, hold: float, retract: float) -> float:
        if frame < start:
            return 0.0
        if frame < start + ramp:
            return smoothstep((frame - start) / ramp)
        if frame < start + ramp + hold:
            return 1.0
        if frame < start + ramp + hold + retract:
            return 1.0 - smoothstep(
                (frame - start - ramp - hold) / retract
            )
        return 0.0

    def evaluate(spec: dict[str, float], record: bool = False):
        pose, target, ik_error, static_tilt = pose_cache[
            (spec["extra_along_m"], spec["height_offset_m"])
        ]
        pose_delta = spec["load_scale"] * (pose - source_pose)
        load_end = (
            spec["load_start_frame"]
            + spec["ramp_frames"]
            + spec["hold_frames"]
        )
        retract_end = load_end + spec["retract_frames"]
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        hand_frames = 0
        last_hand_frame = None
        first_invalid = None
        invalid_reason = None
        max_lift = 0.0
        max_goalward_displacement = 0.0
        max_goalward_speed = 0.0
        recovery_run = 0
        maximum_recovery_run = 0
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
        for step in range(300 * substeps):
            frame = step / substeps
            if step < len(source_ctrl):
                held = source_ctrl[step].copy()
            else:
                held = base.copy()
            amount = envelope(
                frame,
                spec["load_start_frame"],
                spec["ramp_frames"],
                spec["hold_frames"],
                spec["retract_frames"],
            )
            held[controlled] += amount * pose_delta
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
                impulses[contact_group(other)] += (
                    float(np.linalg.norm(force[:3])) * model.opt.timestep
                )
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            velocity = data.qvel[engine.object_dof : engine.object_dof + 3]
            if frame >= 90.0:
                max_lift = max(max_lift, float(obj[2] - object_anchor[2]))
                max_goalward_displacement = max(
                    max_goalward_displacement,
                    float(np.dot(obj - object_anchor, forward)),
                )
                max_goalward_speed = max(
                    max_goalward_speed, float(np.dot(velocity, forward))
                )
            nonhand = sum(
                impulses[key]
                for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            )
            invalid = generator._invalid_reason(data, physical)
            if invalid is None and nonhand > 0.10:
                invalid = "non_hand_object_impulse"
            if step % substeps == substeps - 1:
                reference_frame = int(step // substeps)
                hands = int(np.sum(physical["hand_contact"]))
                hand_frames += int(hands > 0)
                if hands:
                    last_hand_frame = reference_frame
                recovered = (
                    frame >= retract_end
                    and data.qpos[engine.robot_qpos + 2] > 0.57
                    and np.rad2deg(physical["tilt"]) < 35.0
                    and int(np.sum(physical["foot_contact"])) == 2
                )
                recovery_run = recovery_run + 1 if recovered else 0
                maximum_recovery_run = max(maximum_recovery_run, recovery_run)
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
        endpoint = step / substeps
        physical = generator._physical(data)
        metrics = {
            **spec,
            "ik_target_m": target.tolist(),
            "static_ik_error_m": ik_error,
            "static_torso_tilt_deg": static_tilt,
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "hand_contact_frames": hand_frames,
            "last_hand_contact_frame": last_hand_frame,
            "hand_impulse_ns": impulses["left_hand"] + impulses["right_hand"],
            "nonhand_object_impulse_ns": sum(
                impulses[key]
                for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            ),
            "maximum_postsettle_lift_m": max_lift,
            "maximum_goalward_displacement_m": max_goalward_displacement,
            "maximum_goalward_speed_m_s": max_goalward_speed,
            "recovery_margin_after_load_frames": max(0.0, endpoint - load_end),
            "maximum_recovered_run_frames": maximum_recovery_run,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        useful_motion = max_lift > 0.001 or max_goalward_displacement > 0.001
        clean_contact = hand_frames > 0 and metrics["nonhand_object_impulse_ns"] <= 0.10
        recoverable = metrics["recovery_margin_after_load_frames"] >= 30.0
        rank = (
            not (clean_contact and useful_motion and recoverable),
            not clean_contact,
            not recoverable,
            -max_lift,
            -max_goalward_displacement,
            -hand_frames,
            -metrics["recovery_margin_after_load_frames"],
        )
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics), pose

    specs = [
        {
            "extra_along_m": extra_along,
            "height_offset_m": height_offset,
            "load_scale": float(load_scale),
            "load_start_frame": 127.0,
            "ramp_frames": float(ramp),
            "hold_frames": float(hold),
            "retract_frames": 12.0,
        }
        for extra_along, height_offset, load_scale, ramp, hold in itertools.product(
            (0.050, 0.080, 0.110),
            (-0.040, -0.060, -0.080),
            (1.0, 2.0),
            (2, 4),
            (4,),
        )
    ]
    trials = []
    best_rank = None
    best_spec = None
    started = time.perf_counter()
    for index, spec in enumerate(specs):
        rank, metrics, _, _, _ = evaluate(spec)
        trials.append(metrics)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_spec = spec
        print("TRIAL", index, json.dumps(metrics), flush=True)

    assert best_spec is not None
    _, metrics, trace, ctrl_physics, pose = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        source_ctrl_physics=source_ctrl,
        ctrl_physics=ctrl_physics,
        controlled_actuators=controlled,
        load_pose=pose,
        object_anchor=object_anchor,
        goal_direction=forward,
        **trace,
    )
    report = {
        "method": "structured deeper fixed-foot whole-body hand load/retract recovery grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
