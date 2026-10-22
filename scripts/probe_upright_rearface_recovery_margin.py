#!/usr/bin/env python3
"""Cheap recovery-margin probe for a legal upright rear-face manipulation.

Every rollout begins at the actual start-0 MuJoCo state.  It first replays the
known upright balance target, approaches a bilateral rear-face contact, applies
a modest hand-only forward/upward load, retracts, and then leaves a long balance
recovery window.  IK is used only to construct actuator targets; floating-base
and object states are never written after initialization.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from probe_flat_hand_grasp import build, smoothstep


GROUPS = (
    "left_hand",
    "right_hand",
    "wrist",
    "feet",
    "legs",
    "torso_pelvis",
    "other_robot",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--maximum-disallowed-impulse", type=float, default=0.10)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    stance = np.load(args.stance)
    balance_target = stance["target"].copy()
    initial_qpos = generator.initial_data.qpos.copy()
    initial_qvel = generator.initial_data.qvel.copy()
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_qpos = engine.actuator_qpos[arms]
    arm_joints = model.actuator_trnid[arms, 0].astype(int)
    lower = model.jnt_range[arm_joints, 0] + 0.003
    upper = model.jnt_range[arm_joints, 1] - 0.003
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    goal = engine.reference_object_pos[-1].copy()

    def group(geom: int) -> str:
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

    def stance_anchor(frame: int) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        for _ in range(frame * engine.config.substeps):
            data.ctrl[:] = balance_target
            mujoco.mj_step(model, data)
        return data.qpos.copy(), data.qvel.copy()

    def solve_arms(
        qpos: np.ndarray,
        spacing: float,
        height: float,
        along: float,
        forward_shift: float,
        lift_shift: float,
        seed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        obj = qpos[engine.object_qpos : engine.object_qpos + 3]
        forward = np.r_[goal[:2] - obj[:2], 0.0]
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        lateral = np.asarray([-forward[1], forward[0], 0.0])
        targets = np.asarray(
            [
                obj + spacing * lateral + (along + forward_shift) * forward,
                obj - spacing * lateral + (along + forward_shift) * forward,
            ]
        )
        targets[:, 2] = obj[2] + height + lift_shift
        nominal = qpos[arm_qpos].copy()
        start = nominal if seed is None else np.asarray(seed).copy()

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[arm_qpos] = candidate
            mujoco.mj_forward(model, data)
            positions = np.asarray([data.geom_xpos[g] for g in engine.hand_geoms])
            rotations = np.asarray(
                [data.geom_xmat[g].reshape(3, 3) for g in engine.hand_geoms]
            )
            # The palm's thin local-y axis faces taskward; local-x lies across
            # the rear face.  Position dominates because this is a flat palm.
            return np.r_[
                55.0 * (positions - targets).ravel(),
                1.6 * (rotations[:, :, 1] - forward).ravel(),
                0.7 * (rotations[:, :, 0] - lateral).ravel(),
                0.025 * (candidate - nominal),
            ]

        solved = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=220,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        data.qpos[arm_qpos] = solved.x
        mujoco.mj_forward(model, data)
        errors = np.linalg.norm(
            np.asarray([data.geom_xpos[g] for g in engine.hand_geoms]) - targets,
            axis=1,
        )
        return solved.x.copy(), errors, targets

    def evaluate(spec: dict[str, float], record: bool = False):
        settle = int(spec["settle_frames"])
        approach = int(spec["approach_frames"])
        contact_hold = 8
        load_duration = 14
        retract_duration = 28
        contact_end = settle + approach
        load_start = contact_end + contact_hold
        load_end = load_start + load_duration
        retract_end = load_end + retract_duration
        qpos_anchor, _ = anchors[settle]
        contact_arm, contact_error, hand_targets = solve_arms(
            qpos_anchor,
            spec["spacing_m"],
            spec["height_m"],
            spec["along_m"],
            0.0,
            0.0,
        )
        load_arm, load_error, load_targets = solve_arms(
            qpos_anchor,
            spec["spacing_m"],
            spec["height_m"],
            spec["along_m"],
            spec["load_forward_m"],
            spec["load_up_m"],
            contact_arm,
        )
        initial_arm = initial_qpos[arm_qpos].copy()
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        initial_object = data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
        forward = np.r_[goal[:2] - initial_object[:2], 0.0]
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        impulses = {key: 0.0 for key in GROUPS}
        first_invalid = None
        invalid_reason = None
        hand_frames = 0
        bilateral_frames = 0
        maximum_lift = 0.0
        maximum_goalward_displacement = 0.0
        maximum_goalward_speed = 0.0
        maximum_upward_speed = 0.0
        last_hand_frame = None
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
        ctrl_physics = []
        held = balance_target.copy()
        ground_z = None
        for step in range(args.horizon * engine.config.substeps):
            frame = step / engine.config.substeps
            if step % engine.config.substeps == 0:
                held = balance_target.copy()
                if frame < settle:
                    arm = initial_arm
                elif frame < contact_end:
                    amount = smoothstep((frame - settle) / approach)
                    arm = (1.0 - amount) * initial_arm + amount * contact_arm
                elif frame < load_start:
                    arm = contact_arm
                elif frame < load_end:
                    amount = smoothstep((frame - load_start) / load_duration)
                    arm = (1.0 - amount) * contact_arm + amount * load_arm
                elif frame < retract_end:
                    amount = smoothstep((frame - load_end) / retract_duration)
                    arm = (1.0 - amount) * load_arm + amount * initial_arm
                else:
                    arm = initial_arm
                held[arms] = arm
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
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                impulses[group(other)] += float(np.linalg.norm(force[:3])) * model.opt.timestep
            obj = data.qpos[engine.object_qpos : engine.object_qpos + 3]
            if ground_z is None and frame >= max(20, settle - 10):
                ground_z = float(obj[2])
            base_z = float(obj[2]) if ground_z is None else ground_z
            maximum_lift = max(maximum_lift, float(obj[2] - base_z))
            maximum_goalward_displacement = max(
                maximum_goalward_displacement, float(np.dot(obj - initial_object, forward))
            )
            velocity = data.qvel[engine.object_dof : engine.object_dof + 3]
            maximum_goalward_speed = max(maximum_goalward_speed, float(np.dot(velocity, forward)))
            maximum_upward_speed = max(maximum_upward_speed, float(velocity[2]))
            invalid = generator._invalid_reason(data, physical)
            disallowed = sum(
                impulses[key] for key in ("feet", "legs", "torso_pelvis", "other_robot")
            )
            if invalid is None and disallowed > args.maximum_disallowed_impulse:
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
            if invalid is not None and first_invalid is None:
                first_invalid = step
                invalid_reason = invalid
                if not record:
                    break
        completed_frame = step / engine.config.substeps
        recovery_endpoint = (
            completed_frame if first_invalid is None else first_invalid / engine.config.substeps
        )
        recovery_margin = max(0.0, recovery_endpoint - load_end)
        p = generator._physical(data)
        metrics = {
            **spec,
            "contact_ik_error_m": contact_error.tolist(),
            "load_ik_error_m": load_error.tolist(),
            "contact_targets_m": hand_targets.tolist(),
            "load_targets_m": load_targets.tolist(),
            "load_end_frame": float(load_end),
            "first_invalid_frame": None if first_invalid is None else first_invalid / engine.config.substeps,
            "invalid_reason": invalid_reason,
            "full_horizon_valid": first_invalid is None,
            "recovery_margin_frames": float(recovery_margin),
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "last_hand_contact_frame": last_hand_frame,
            "maximum_lift_m": float(maximum_lift),
            "maximum_goalward_displacement_m": float(maximum_goalward_displacement),
            "maximum_goalward_speed_m_s": float(maximum_goalward_speed),
            "maximum_upward_speed_m_s": float(maximum_upward_speed),
            "object_impulse_ns": impulses,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(p["tilt"])),
            "final_foot_contact": p["foot_contact"].tolist(),
        }
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        return metrics, arrays, np.asarray(ctrl_physics), contact_arm, load_arm

    settle_values = (90, 120, 150)
    anchors = {frame: stance_anchor(frame) for frame in settle_values}
    specs = []
    for settle, spacing, height, duration, forward_load, upward_load in itertools.product(
        settle_values,
        (0.10, 0.14),
        (0.14, 0.20, 0.26),
        (40, 60),
        (0.018, 0.035),
        (0.010,),
    ):
        specs.append(
            {
                "settle_frames": float(settle),
                "spacing_m": spacing,
                "height_m": height,
                "along_m": -0.240,
                "approach_frames": float(duration),
                "load_forward_m": forward_load,
                "load_up_m": upward_load,
            }
        )

    started = time.perf_counter()
    trials = []
    best_rank = None
    best_spec = None
    for index, spec in enumerate(specs):
        metrics, _, _, _, _ = evaluate(spec)
        disallowed = sum(
            metrics["object_impulse_ns"][key]
            for key in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        useful = (
            metrics["hand_contact_frames"] > 0
            and metrics["maximum_goalward_speed_m_s"] > 0.02
            and metrics["maximum_upward_speed_m_s"] > 0.01
            and disallowed <= args.maximum_disallowed_impulse
        )
        rank = (
            not useful,
            not metrics["full_horizon_valid"],
            -metrics["recovery_margin_frames"],
            -metrics["bilateral_hand_contact_frames"],
            -metrics["maximum_lift_m"],
            -metrics["maximum_goalward_speed_m_s"],
        )
        trials.append(metrics)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_spec = spec
        print("TRIAL", index, json.dumps(metrics), flush=True)

    assert best_spec is not None
    metrics, trace, ctrl_physics, contact_arm, load_arm = evaluate(best_spec, record=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        balance_target=balance_target,
        arm_actuators=arms,
        contact_arm_target=contact_arm,
        load_arm_target=load_arm,
        ctrl_physics=ctrl_physics,
        **trace,
    )
    report = {
        "method": "legal upright rear-face contact/load/retract recovery-margin grid",
        "runtime_s": time.perf_counter() - started,
        "candidate_count": len(specs),
        "best": metrics,
        "trials": trials,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
