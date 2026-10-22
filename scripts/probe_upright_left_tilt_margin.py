#!/usr/bin/env python3
"""Stop/retract recovery-margin probes for the upright left-hand tilt path."""

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


GROUPS = ("left_hand", "right_hand", "wrist", "feet", "legs", "torso_pelvis", "other_robot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    base = source["base_target"].copy()
    contact_pose = source["contact_pose"].copy()
    object_geoms = set(map(int, engine.object_geoms))
    substeps = engine.config.substeps

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    object_anchor = anchor.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    object_quaternion = anchor.qpos[engine.object_qpos + 3 : engine.object_qpos + 7].copy()
    initial_rotation = Rotation.from_quat(
        np.r_[object_quaternion[1:], object_quaternion[0]]
    )
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_anchor[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)

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

    def evaluate(spec: dict[str, float], record: bool = False):
        stop = spec["stop_frame"]
        retract_end = stop + spec["retract_frames"]
        stop_amount = smoothstep((stop - 90.0) / 45.0)
        stop_control = (1.0 - stop_amount) * base + stop_amount * contact_pose
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
        max_goalward = 0.0
        max_speed = 0.0
        max_tip = 0.0
        recovery_run = 0
        max_recovery_run = 0
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
            elif frame < stop:
                amount = smoothstep((frame - 90.0) / 45.0)
                held = (1.0 - amount) * base + amount * contact_pose
            elif frame < retract_end:
                amount = smoothstep((frame - stop) / spec["retract_frames"])
                held = (1.0 - amount) * stop_control + amount * base
            else:
                held = base.copy()
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
            quaternion = data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
            if frame >= 90.0:
                max_lift = max(max_lift, float(obj[2] - object_anchor[2]))
                max_goalward = max(max_goalward, float(np.dot(obj - object_anchor, forward)))
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
            "maximum_goalward_speed_m_s": max_speed,
            "maximum_object_tip_rad": max_tip,
            "recovery_margin_after_stop_frames": max(0.0, endpoint - stop),
            "maximum_recovered_run_frames": max_recovery_run,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_foot_contact": physical["foot_contact"].tolist(),
        }
        clean = hand_frames > 0 and metrics["lower_body_torso_impulse_ns"] <= 0.10
        motion = max_goalward > 0.001 or max_tip > 0.01 or max_lift > 0.001
        recoverable = max_recovery_run >= 20
        rank = (
            not (clean and motion and recoverable),
            not clean,
            not recoverable,
            -max_recovery_run,
            -max_tip,
            -max_goalward,
            -hand_frames,
            -(step / substeps),
        )
        return rank, metrics, {key: np.asarray(value) for key, value in trace.items()}, np.asarray(ctrl_physics)

    specs = [
        {"stop_frame": float(stop), "retract_frames": float(retract)}
        for stop, retract in itertools.product((130, 134, 138, 142), (12, 20, 28))
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
        contact_pose=contact_pose,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "structured upright left-hand tilt stop/retract recovery-margin grid",
        "candidate_count": len(specs),
        "runtime_s": time.perf_counter() - started,
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
