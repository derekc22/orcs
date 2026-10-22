#!/usr/bin/env python3
"""Probe fast bilateral hand transport from the first sustained-lift state."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build, smoothstep
from scipy.optimize import least_squares


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--balance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    anchor = np.load(args.anchor)
    balance_solution = np.load(args.balance)
    balance = balance_solution["controlled_actuators"].astype(int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_joint_ids = model.actuator_trnid[arms, 0].astype(int)
    lower = model.jnt_range[arm_joint_ids, 0] + 0.001
    upper = model.jnt_range[arm_joint_ids, 1] - 0.001
    qpos0 = anchor["anchor_qpos"].copy()
    qvel0 = anchor["anchor_qvel"].copy()
    base_target = anchor["base_target"].copy()
    ground_z = 0.18545216
    goal = engine.reference_object_pos[-1]
    object_geoms = set(engine.object_geoms)

    kframes = balance_solution["knot_frames"]
    bknots = np.vstack(
        [np.zeros(len(balance)), balance_solution["parameters"].reshape(2, -1)]
    )

    def interp(knots: np.ndarray, frames: np.ndarray, frame: float) -> np.ndarray:
        if frame >= frames[-1]:
            return knots[-1]
        index = int(np.searchsorted(frames, frame, side="right") - 1)
        amount = smoothstep((frame - frames[index]) / (frames[index + 1] - frames[index]))
        return (1.0 - amount) * knots[index] + amount * knots[index + 1]

    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = qpos0
    ik_data.qvel[:] = 0.0
    mujoco.mj_forward(model, ik_data)
    arm0 = np.clip(ik_data.qpos[engine.actuator_qpos[arms]].copy(), lower, upper)
    hand0 = np.asarray([ik_data.geom_xpos[geom].copy() for geom in engine.hand_geoms])
    box0 = qpos0[engine.object_qpos : engine.object_qpos + 3]
    direction = goal - box0
    direction /= np.linalg.norm(direction)

    def solve_endpoint(distance: float, rise: float) -> np.ndarray:
        target = hand0 + distance * direction
        target[:, 2] += rise

        def residual(candidate: np.ndarray) -> np.ndarray:
            ik_data.qpos[engine.actuator_qpos[arms]] = candidate
            mujoco.mj_forward(model, ik_data)
            positions = np.asarray(
                [ik_data.geom_xpos[geom] for geom in engine.hand_geoms]
            )
            return np.r_[45.0 * (positions - target).ravel(), 0.03 * (candidate - arm0)]

        result = least_squares(
            residual,
            arm0,
            bounds=(lower, upper),
            max_nfev=1200,
        )
        return result.x

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

    def rollout(endpoint: np.ndarray, duration: int, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(model, data)
        impulses = {name: 0.0 for name in (
            "left_hand", "right_hand", "wrist", "feet", "legs",
            "torso_pelvis", "other_robot")}
        lift_flags: list[bool] = []
        first_invalid = reason = None
        trace = {name: [] for name in (
            "qpos", "qvel", "ctrl", "object_position", "hand_contact",
            "foot_contact", "root_height", "torso_tilt")}
        for step in range(78 * engine.config.substeps):
            frame = step / engine.config.substeps
            target = base_target.copy()
            target[balance] += interp(bknots, kframes, frame)
            amount = smoothstep(frame / duration)
            target[arms] = (1.0 - amount) * base_target[arms] + amount * endpoint
            target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            for ci in range(data.ncon):
                contact = data.contact[ci]
                first, second = int(contact.geom1), int(contact.geom2)
                if not ({first, second} & object_geoms):
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(impulses[name] for name in ("feet", "legs", "torso_pelvis", "other_robot"))
            if disallowed > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if invalid is not None:
                first_invalid, reason = step, invalid
            if step % engine.config.substeps == engine.config.substeps - 1:
                lift_flags.append(data.qpos[engine.object_qpos + 2] - ground_z >= 0.05)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["object_position"].append(data.qpos[engine.object_qpos:engine.object_qpos+3].copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["root_height"].append(data.qpos[engine.robot_qpos+2])
                    trace["torso_tilt"].append(physical["tilt"])
            if invalid is not None:
                break
        longest = max((len(list(run)) for value, run in itertools.groupby(lift_flags) if value), default=0)
        object_position = data.qpos[engine.object_qpos:engine.object_qpos+3]
        metrics = {
            "first_invalid_frame": None if first_invalid is None else 222 + first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "sustained_lift_frames": longest,
            "maximum_lift_m": float(max((q[engine.object_qpos+2] - ground_z for q in trace["qpos"]), default=data.qpos[engine.object_qpos+2]-ground_z)) if record else None,
            "final_goal_position_error_m": float(np.linalg.norm(object_position-goal)),
            "object_goal_progress_m": float(np.linalg.norm(box0-goal)-np.linalg.norm(object_position-goal)),
            "object_impulse_ns": impulses,
            "final_root_height_m": float(data.qpos[engine.robot_qpos+2]),
            "final_torso_tilt_deg": float(np.degrees(physical["tilt"])),
        }
        return metrics, {key: np.asarray(value) for key, value in trace.items()}

    trials = []
    best_key = None
    best_rank = None
    endpoints = {}
    for distance, rise, duration in itertools.product(
        (0.08, 0.12, 0.16, 0.22, 0.30), (0.0, 0.04, 0.08), (4, 7, 10, 14)
    ):
        endpoint = solve_endpoint(distance, rise)
        metrics, _ = rollout(endpoint, duration)
        key = f"d{distance:.2f}_z{rise:.2f}_t{duration}"
        rank = (
            metrics["first_invalid_frame"] is not None,
            -(metrics["first_invalid_frame"] or 300.0),
            -metrics["sustained_lift_frames"],
            metrics["final_goal_position_error_m"],
        )
        trials.append({"key": key, **metrics})
        endpoints[key] = endpoint
        if best_rank is None or rank < best_rank:
            best_key, best_rank = key, rank
        print(key, json.dumps(metrics), flush=True)
    assert best_key is not None
    best_metrics, trace = rollout(endpoints[best_key], int(best_key.rsplit("t", 1)[1]), record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        anchor_qpos=qpos0,
        anchor_qvel=qvel0,
        endpoint=endpoints[best_key],
        best_key=best_key,
        **trace,
    )
    args.output.with_suffix(".json").write_text(json.dumps({"best_key": best_key, "best": best_metrics, "trials": trials}, indent=2)+"\n")
    print("BEST", best_key, json.dumps(best_metrics), flush=True)


if __name__ == "__main__":
    main()
