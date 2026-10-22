#!/usr/bin/env python3
"""Probe hand contact from a rigidly translated, dynamically stable stance.

Unlike the earlier whole-body IK probe, this keeps the proven lower-body
geometry unchanged.  The complete robot is translated horizontally, then only
the 14 arm/wrist joints are solved to approach the unchanged, settled box.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from probe_flat_hand_grasp import build, smoothstep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--settled-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--rear-face",
        action="store_true",
        help="Place both palms on the robot-facing box face instead of squeezing opposite sides.",
    )
    parser.add_argument(
        "--exact",
        type=float,
        nargs=5,
        metavar=("ROOT_Y", "SPACING", "HEIGHT", "ALONG", "DURATION"),
        help="Evaluate and save one contact construction instead of the grid.",
    )
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    settled = np.load(args.settled_object)
    balance = stance["selected_actuators"].astype(int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_qpos = engine.actuator_qpos[arms]
    arm_joint_ids = model.actuator_trnid[arms, 0].astype(int)
    lower = model.jnt_range[arm_joint_ids, 0] + 0.003
    upper = model.jnt_range[arm_joint_ids, 1] - 0.003
    qpos_base = stance["initial_qpos"].copy()
    qvel_base = stance["initial_qvel"].copy()
    qpos_base[engine.object_qpos : engine.object_qpos + 7] = settled["qpos"][
        0, engine.object_qpos : engine.object_qpos + 7
    ]
    qvel_base[engine.object_dof : engine.object_dof + 6] = 0.0
    object_position = qpos_base[engine.object_qpos : engine.object_qpos + 3].copy()
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_position[:2], 0.0]
    forward /= np.linalg.norm(forward)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    object_geoms = set(engine.object_geoms)

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

    def solve(qpos: np.ndarray, distance: float, height: float, along: float):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        arm0 = np.clip(qpos[arm_qpos], lower, upper)
        if args.rear_face:
            targets = np.asarray(
                [
                    object_position + distance * lateral + along * forward,
                    object_position - distance * lateral + along * forward,
                ]
            )
            normals = np.asarray([forward, forward])
            longitudinal = np.asarray([lateral, lateral])
        else:
            targets = np.asarray(
                [
                    object_position + distance * lateral + along * forward,
                    object_position - distance * lateral + along * forward,
                ]
            )
            normals = np.asarray([-lateral, lateral])
            longitudinal = np.asarray([forward, forward])
        targets[:, 2] = object_position[2] + height

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[arm_qpos] = candidate
            mujoco.mj_forward(model, data)
            positions = np.asarray([data.geom_xpos[g] for g in engine.hand_geoms])
            rotations = np.asarray(
                [data.geom_xmat[g].reshape(3, 3) for g in engine.hand_geoms]
            )
            return np.r_[
                45.0 * (positions - targets).ravel(),
                0.8 * (rotations[:, :, 1] - normals).ravel(),
                0.3 * (rotations[:, :, 0] - longitudinal).ravel(),
                0.025 * (candidate - arm0),
            ]

        result = least_squares(
            residual,
            arm0,
            bounds=(lower, upper),
            max_nfev=160,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        data.qpos[arm_qpos] = result.x
        mujoco.mj_forward(model, data)
        errors = np.linalg.norm(
            np.asarray([data.geom_xpos[g] for g in engine.hand_geoms]) - targets,
            axis=1,
        )
        return result.x, errors

    def rollout(
        qpos: np.ndarray,
        arm_target: np.ndarray,
        approach_frames: int,
        record: bool = False,
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel_base
        mujoco.mj_forward(model, data)
        initial_arm = qpos[arm_qpos].copy()
        target = stance["target"].copy()
        impulses = {
            name: 0.0
            for name in (
                "left_hand",
                "right_hand",
                "wrist",
                "feet",
                "legs",
                "torso_pelvis",
                "other_robot",
            )
        }
        hand_frames = bilateral_frames = 0
        first_invalid = reason = None
        trace = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "hand_contact",
                "foot_contact",
                "object_position",
                "root_height",
                "torso_tilt",
                "object_impulse",
            )
        }
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            amount = smoothstep(frame / approach_frames)
            target[arms] = (1.0 - amount) * initial_arm + amount * arm_target
            data.ctrl[:] = np.clip(
                target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
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
            disallowed = sum(
                impulses[name]
                for name in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            if disallowed > 0.5 and invalid is None:
                invalid = "non_hand_object_impulse"
            if step % engine.config.substeps == engine.config.substeps - 1:
                hands = int(np.sum(physical["hand_contact"]))
                hand_frames += int(hands > 0)
                bilateral_frames += int(hands == 2)
                if record:
                    trace["qpos"].append(data.qpos.copy())
                    trace["qvel"].append(data.qvel.copy())
                    trace["ctrl"].append(data.ctrl.copy())
                    trace["hand_contact"].append(physical["hand_contact"].copy())
                    trace["foot_contact"].append(physical["foot_contact"].copy())
                    trace["object_position"].append(
                        data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
                    )
                    trace["root_height"].append(data.qpos[engine.robot_qpos + 2])
                    trace["torso_tilt"].append(float(physical["tilt"]))
                    trace["object_impulse"].append(np.asarray(list(impulses.values())))
            if invalid is not None:
                first_invalid, reason = step, invalid
                break
        metrics = {
            "first_invalid_frame": None
            if first_invalid is None
            else first_invalid / engine.config.substeps,
            "invalid_reason": reason,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "object_impulse_ns": impulses,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
        }
        return metrics, {key: np.asarray(value) for key, value in trace.items()}

    trials = []
    best = None
    best_payload = None
    grid = (
        [(args.exact[0], args.exact[1], args.exact[2], args.exact[3])]
        if args.exact is not None
        else
        itertools.product(
            (0.05, 0.10, 0.15, 0.20),
            (0.10, 0.14, 0.18),
            (0.14, 0.20, 0.26),
            (-0.24, -0.20, -0.16),
        )
        if args.rear_face
        else itertools.product(
            (0.0, -0.08, -0.14, -0.18),
            (0.24, 0.21, 0.18),
            (0.14, 0.18, 0.22),
            (0.0, 0.04),
        )
    )
    for root_y, distance, height, along in grid:
        qpos = qpos_base.copy()
        qpos[engine.robot_qpos + 1] += root_y
        arm_target, errors = solve(qpos, distance, height, along)
        durations = (int(args.exact[4]),) if args.exact is not None else (30, 45)
        for duration in durations:
            metrics, _ = rollout(qpos, arm_target, duration)
            rank = (
                metrics["first_invalid_frame"] is not None,
                -(metrics["first_invalid_frame"] or args.horizon),
                -metrics["bilateral_hand_contact_frames"],
                -metrics["hand_contact_frames"],
                float(errors.sum()),
            )
            item = {
                "root_y_delta_m": root_y,
                "lateral_distance_m": distance,
                "height_offset_m": height,
                "longitudinal_offset_m": along,
                "approach_frames": duration,
                "hand_position_error_m": errors.tolist(),
                **metrics,
            }
            trials.append(item)
            if best is None or rank < best:
                best = rank
                best_payload = (qpos.copy(), arm_target.copy(), duration, item)
            print(json.dumps(item), flush=True)

    assert best_payload is not None
    qpos, arm_target, duration, item = best_payload
    metrics, trace = rollout(qpos, arm_target, duration, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        anchor_qpos=qpos,
        anchor_qvel=qvel_base,
        arm_target=arm_target,
        arm_actuators=arms,
        balance_actuators=balance,
        **trace,
    )
    report = {"method": "rigid-translated upright stance plus arm-only IK", "best": item, "trials": trials}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
