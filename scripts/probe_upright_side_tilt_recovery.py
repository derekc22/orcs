#!/usr/bin/env python3
"""Probe a supported left-palm side tilt that keeps the box out of the legs."""

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
    source = np.load(args.stance)
    base = source["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    left_arm = np.asarray(
        [
            i
            for i, name in enumerate(names)
            if name.startswith("left_")
            and any(token in name for token in ("shoulder", "elbow", "wrist"))
        ],
        dtype=int,
    )
    controlled = np.unique(np.r_[generator.balance_actuators, left_arm]).astype(int)
    controlled_qpos = engine.actuator_qpos[controlled]
    joint_ids = model.actuator_trnid[controlled, 0].astype(int)
    lower = model.jnt_range[joint_ids, 0] + 0.003
    upper = model.jnt_range[joint_ids, 1] - 0.003
    object_geoms = set(map(int, engine.object_geoms))
    substeps = engine.config.substeps

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
    object_quaternion = anchor_qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    initial_rotation = Rotation.from_quat(
        np.r_[object_quaternion[1:], object_quaternion[0]]
    )
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    lateral = np.asarray([-forward[1], forward[0], 0.0])

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

    def solve_pose(along: float, lateral_offset: float, seed: np.ndarray):
        data = mujoco.MjData(model)
        target = object_anchor + lateral_offset * lateral + along * forward
        target[2] = object_anchor[2] + 0.220

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[:] = anchor_qpos
            data.qvel[:] = 0.0
            data.qpos[controlled_qpos] = candidate
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
            hand_rotation = data.geom_xmat[engine.hand_geoms[0]].reshape(3, 3)
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            return np.r_[
                80.0 * (data.geom_xpos[engine.hand_geoms[0]] - target),
                0.50 * (hand_rotation[:, 1] + lateral),
                100.0 * (feet - feet0).ravel(),
                12.0 * foot_orientation,
                3.0 * torso_up[:2],
                0.02 * (candidate - nominal),
            ]

        result = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=400,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        data.qpos[:] = anchor_qpos
        data.qvel[:] = 0.0
        data.qpos[controlled_qpos] = result.x
        mujoco.mj_forward(model, data)
        physical = generator._physical(data)
        return (
            result.x.copy(),
            target,
            float(np.linalg.norm(data.geom_xpos[engine.hand_geoms[0]] - target)),
            float(np.rad2deg(physical["tilt"])),
        )

    pose_cache = {}
    for along in (-0.20, -0.16):
        contact = solve_pose(along, 0.22, nominal)
        for load_lateral in (0.18, 0.14):
            load = solve_pose(along, load_lateral, contact[0])
            pose_cache[(along, load_lateral)] = (contact, load)

    def evaluate(spec: dict[str, float], record: bool = False):
        contact, load = pose_cache[(spec["along_m"], spec["load_lateral_m"])]
        equilibrium_offset = base[controlled] - nominal
        contact_control = contact[0] + equilibrium_offset
        load_control = load[0] + equilibrium_offset
        approach_end = 90.0 + spec["approach_frames"]
        load_end = approach_end + spec["load_frames"]
        hold_end = load_end + 4.0
        retract_end = hold_end + spec["retract_frames"]
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        impulses = {key: 0.0 for key in GROUPS}
        hand_frames = 0
        last_hand_frame = None
        first_invalid = None
        invalid_reason = None
        max_lift = max_goalward = max_lateral = max_speed = max_tip = 0.0
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
            if frame < 90.0:
                target = base[controlled]
            elif frame < approach_end:
                amount = smoothstep((frame - 90.0) / spec["approach_frames"])
                target = (1.0 - amount) * base[controlled] + amount * contact_control
            elif frame < load_end:
                amount = smoothstep((frame - approach_end) / spec["load_frames"])
                target = (1.0 - amount) * contact_control + amount * load_control
            elif frame < hold_end:
                target = load_control
            elif frame < retract_end:
                amount = smoothstep((frame - hold_end) / spec["retract_frames"])
                target = (1.0 - amount) * load_control + amount * base[controlled]
            else:
                target = base[controlled]
            held[controlled] = target
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
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            velocity = data.qvel[engine.object_dof : engine.object_dof + 3]
            quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
            if frame >= 90.0:
                max_lift = max(max_lift, float(obj[2] - object_anchor[2]))
                max_goalward = max(max_goalward, float(np.dot(obj - object_anchor, forward)))
                max_lateral = max(max_lateral, abs(float(np.dot(obj - object_anchor, lateral))))
                max_speed = max(max_speed, float(np.dot(velocity, forward)))
                max_tip = max(max_tip, float((initial_rotation.inv() * rotation).magnitude()))
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
                recovered = (
                    frame >= retract_end
                    and data.qpos[engine.robot_qpos + 2] > 0.58
                    and np.rad2deg(physical["tilt"]) < 32.0
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
        metrics = {
            **spec,
            "contact_ik_error_m": contact[2],
            "load_ik_error_m": load[2],
            "static_contact_torso_tilt_deg": contact[3],
            "static_load_torso_tilt_deg": load[3],
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else first_invalid / substeps,
            "invalid_reason": invalid_reason,
            "hand_contact_frames": hand_frames,
            "last_hand_contact_frame": last_hand_frame,
            "left_right_hand_impulse_ns": [impulses["left_hand"], impulses["right_hand"]],
            "wrist_impulse_ns": impulses["wrist"],
            "lower_body_torso_impulse_ns": sum(
                impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot")
            ),
            "maximum_postsettle_lift_m": max_lift,
            "maximum_goalward_displacement_m": max_goalward,
            "maximum_lateral_displacement_m": max_lateral,
            "maximum_goalward_speed_m_s": max_speed,
            "maximum_object_tip_rad": max_tip,
            "recovery_margin_after_load_frames": max(0.0, endpoint - load_end),
            "maximum_recovered_run_frames": max_recovery_run,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        clean = hand_frames > 0 and metrics["lower_body_torso_impulse_ns"] <= 0.10
        motion = max_tip > 0.01 or max_lateral > 0.001 or max_lift > 0.001
        recoverable = max_recovery_run >= 20
        rank = (
            not (clean and motion and recoverable),
            not clean,
            not recoverable,
            -max_recovery_run,
            -max_tip,
            -max_lift,
            -max_lateral,
            -hand_frames,
            -(step / substeps),
        )
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics), contact, load

    specs = [
        {
            "along_m": along,
            "load_lateral_m": load_lateral,
            "approach_frames": float(approach),
            "load_frames": float(load_frames),
            "retract_frames": 20.0,
        }
        for along, load_lateral, approach, load_frames in itertools.product(
            (-0.20, -0.16), (0.18, 0.14), (40, 55), (8, 14)
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
    _, metrics, trace, ctrl_physics, contact, load = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        base_target=base,
        controlled_actuators=controlled,
        contact_pose=contact[0],
        load_pose=load[0],
        equilibrium_offset=base[controlled] - nominal,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "legal upright left-palm side-tilt/load/retract recovery grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
