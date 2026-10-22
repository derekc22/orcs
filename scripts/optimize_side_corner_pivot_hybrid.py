#!/usr/bin/env python3
"""Bounded reduced hybrid shooting for the final side/corner pivot gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from probe_flat_hand_grasp import build, smoothstep
from run_side_corner_pivot_gate import IMPULSE_GROUPS, contact_impulses
from search_reach_support_feasibility import quat_rotation, support_margin


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/hand_manipulation_research"


def envelope(frame: float, begin: float, loaded: float, release: float, end: float) -> float:
    if frame < begin or frame >= end:
        return 0.0
    if frame < loaded:
        return smoothstep((frame - begin) / max(loaded - begin, 1.0e-9))
    if frame <= release:
        return 1.0
    return 1.0 - smoothstep((frame - release) / max(end - release, 1.0e-9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=LOG / "side_corner_pivot_hybrid_v1"
    )
    parser.add_argument("--maxfev", type=int, default=100)
    args = parser.parse_args()
    started = time.perf_counter()

    engine, generator = build(0.55)
    model = engine.model
    substeps = engine.config.substeps
    source = np.load(LOG / "upright_feedback_side_load_v1.npz")
    stable = np.load(LOG / "upright_balance_vertical_v4.npz")
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    base = source["base_target"].copy()
    feedback_parameters = source["feedback_parameters"].copy()
    controlled = source["controlled_actuators"].astype(int)
    contact_values = source["contact_control"].copy()
    load_values = source["load_control"].copy()
    controlled_lookup = {
        int(actuator): index for index, actuator in enumerate(controlled)
    }
    names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    name_to_actuator = {name: index for index, name in enumerate(names)}
    left_arm = np.asarray(
        [
            name_to_actuator[name]
            for name in (
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            )
        ],
        dtype=int,
    )
    balance = np.asarray(generator.balance_actuators, dtype=int)
    left_leg = np.asarray(
        [
            name_to_actuator[name]
            for name in (
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            )
        ],
        dtype=int,
    )
    support_pulse = np.asarray([0.0, 0.006, 0.040, -0.022, -0.004])
    sagittal_groups = [
        np.asarray([i for i, name in enumerate(names) if "hip_pitch" in name]),
        np.asarray([i for i, name in enumerate(names) if "knee" in name]),
        np.asarray([i for i, name in enumerate(names) if "ankle_pitch" in name]),
        np.asarray([i for i, name in enumerate(names) if "waist_pitch" in name]),
    ]
    feedback_tokens = {
        "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
        "knee": ("left_knee", "right_knee"),
        "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
        "waist_pitch": ("waist_pitch",),
        "hip_roll": ("left_hip_roll", "right_hip_roll"),
        "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
        "waist_roll": ("waist_roll",),
    }
    feedback_groups = {
        key: np.asarray(
            [i for i, name in enumerate(names) if any(t in name for t in tokens)],
            dtype=int,
        )
        for key, tokens in feedback_tokens.items()
    }

    anchor = mujoco.MjData(model)
    anchor.qpos[:] = initial_qpos
    anchor.qvel[:] = initial_qvel
    mujoco.mj_forward(model, anchor)
    for _ in range(90 * substeps):
        anchor.ctrl[:] = base
        mujoco.mj_step(model, anchor)
    desired_root = anchor.qpos[
        engine.robot_qpos : engine.robot_qpos + 3
    ].copy()
    desired_rotation = quat_rotation(
        anchor.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    )
    object_anchor = anchor.qpos[
        engine.object_qpos : engine.object_qpos + 3
    ].copy()
    object_rotation0 = quat_rotation(
        anchor.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    )

    # Preserve the demonstrated dynamic side-contact path exactly at the unit
    # seed.  The supported static solution remains the geometric target and is
    # used to justify the contact family, but its mapped controls were not a
    # dynamically realizable feedforward path.
    source_balance_contact = np.asarray(
        [contact_values[controlled_lookup[int(index)]] for index in balance]
    )
    source_balance_delta = source_balance_contact - base[balance]
    arm_contact = np.asarray(
        [contact_values[controlled_lookup[int(index)]] for index in left_arm]
    )
    arm_load_source = np.asarray(
        [load_values[controlled_lookup[int(index)]] for index in left_arm]
    )
    arm_load_delta = arm_load_source - arm_contact

    def feedback_correction(data: mujoco.MjData) -> np.ndarray:
        rotation_error = (
            desired_rotation.inv()
            * quat_rotation(
                data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
            )
        ).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal = np.asarray(
            [root[1] - desired_root[1], velocity[1], rotation_error[0], velocity[3]]
        )
        lateral = np.asarray(
            [root[0] - desired_root[0], velocity[0], rotation_error[1], velocity[4]]
        )
        gains = feedback_parameters.reshape(-1, 4)
        correction = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            correction[feedback_groups[key]] += float(np.dot(gains[row], sagittal))
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            correction[feedback_groups[key]] += float(np.dot(gains[row], lateral))
            row += 1
        return np.clip(correction, -0.28, 0.28)

    evaluation_count = 0
    best_rank: tuple[Any, ...] | None = None
    best_parameters = None
    best_metrics = None

    def evaluate(parameters: np.ndarray, record: bool = False):
        nonlocal evaluation_count, best_rank, best_parameters, best_metrics
        evaluation_count += 1
        data = mujoco.MjData(model)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        cumulative = {key: 0.0 for key in IMPULSE_GROUPS}
        first_invalid = None
        invalid_reason = None
        first_contact_step = None
        first_contact_speed = None
        first_contact_vz = None
        hand_frames = 0
        contact_run = 0
        longest_contact = 0
        recovered_run = 0
        maximum_recovered_run = 0
        maximum_rotation = 0.0
        maximum_tip = 0.0
        maximum_yaw = 0.0
        maximum_displacement = 0.0
        minimum_support = math.inf
        best_contact_detail = None
        best_contact_force = -math.inf
        postpivot_index = None
        postpivot_score = -math.inf
        controls = []
        trace: dict[str, list[Any]] = {
            key: []
            for key in (
                "qpos", "qvel", "ctrl", "hand_contact", "foot_contact",
                "root_height", "torso_tilt", "support_margin",
                "object_position", "object_orientation", "object_impulse",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            held = base.copy()
            if frame >= 90.0:
                held += feedback_correction(data)
                approach = envelope(frame, 90.0, 145.0, 162.0, 180.0)
                arm_amount = smoothstep((frame - 90.0) / 55.0) if frame < 145.0 else 1.0
                if frame >= 162.0:
                    arm_amount = 1.0 - smoothstep((frame - 162.0) / 18.0)
                arm_amount = float(np.clip(arm_amount, 0.0, 1.0))
                load_amount = envelope(frame, 145.0, 159.0, 162.0, 174.0)
                held[balance] += approach * parameters[0] * source_balance_delta
                held[left_arm] = (
                    (1.0 - arm_amount) * base[left_arm]
                    + arm_amount
                    * (
                        base[left_arm]
                        + parameters[1] * (arm_contact - base[left_arm])
                        + load_amount * parameters[2] * arm_load_delta
                    )
                )
                support_amount = envelope(frame, 112.0, 130.0, 154.0, 170.0)
                held[left_leg] += support_amount * parameters[3] * support_pulse
                approach_residual = envelope(frame, 96.0, 120.0, 158.0, 178.0)
                recovery_residual = envelope(frame, 150.0, 178.0, 242.0, 275.0)
                for group, value in zip(
                    sagittal_groups, parameters[4:8], strict=True
                ):
                    held[group] += approach_residual * value
                for group, value in zip(
                    sagittal_groups, parameters[8:12], strict=True
                ):
                    held[group] += recovery_residual * value
            held = np.clip(
                held,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            data.ctrl[:] = held
            controls.append(held.copy())
            mujoco.mj_step(model, data)
            physical = generator._physical(data)
            impulses, details = contact_impulses(model, data, engine)
            for key, value in impulses.items():
                cumulative[key] += value
            for detail in details:
                if detail["normal_force_n"] > best_contact_force:
                    best_contact_force = detail["normal_force_n"]
                    best_contact_detail = detail
            strict_forbidden = sum(
                cumulative[key]
                for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            )
            support, _, _ = support_margin(model, data, engine)
            if support is not None:
                minimum_support = min(minimum_support, support)
            root_velocity = data.qvel[engine.robot_dof : engine.robot_dof + 3]
            if physical["hand_contact"][0] and first_contact_step is None:
                first_contact_step = step
                first_contact_speed = float(np.linalg.norm(root_velocity))
                first_contact_vz = float(root_velocity[2])
            object_position = data.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ].copy()
            object_quaternion = data.qpos[
                engine.object_qpos + 3 : engine.object_qpos + 7
            ].copy()
            relative_rotation = (
                object_rotation0.inv() * quat_rotation(object_quaternion)
            ).as_rotvec()
            rotation_magnitude = float(np.linalg.norm(relative_rotation))
            tip = float(np.linalg.norm(relative_rotation[:2]))
            yaw = float(abs(relative_rotation[2]))
            displacement = float(
                np.linalg.norm(object_position[:2] - object_anchor[:2])
            )
            if frame >= 90.0:
                maximum_rotation = max(maximum_rotation, rotation_magnitude)
                maximum_tip = max(maximum_tip, tip)
                maximum_yaw = max(maximum_yaw, yaw)
                maximum_displacement = max(maximum_displacement, displacement)
            reason = generator._invalid_reason(data, physical)
            if reason is None and strict_forbidden > 0.10:
                reason = "strict_non_hand_object_impulse"
            if reason is not None and first_invalid is None:
                first_invalid = step
                invalid_reason = reason
            if step % substeps == substeps - 1:
                active = bool(physical["hand_contact"][0])
                hand_frames += int(active)
                contact_run = contact_run + 1 if active else 0
                longest_contact = max(longest_contact, contact_run)
                recovered = bool(
                    frame >= 180.0
                    and data.qpos[engine.robot_qpos + 2] >= 0.50
                    and np.rad2deg(physical["tilt"]) <= 35.0
                    and np.linalg.norm(root_velocity) <= 0.30
                    and np.all(physical["foot_contact"])
                    and strict_forbidden <= 0.10
                )
                recovered_run = recovered_run + 1 if recovered else 0
                maximum_recovered_run = max(maximum_recovered_run, recovered_run)
                if (
                    frame >= 145.0
                    and strict_forbidden <= 0.10
                    and data.qpos[engine.robot_qpos + 2] >= 0.50
                    and np.rad2deg(physical["tilt"]) <= 35.0
                    and np.all(physical["foot_contact"])
                ):
                    score = rotation_magnitude + 0.25 * displacement
                    if score > postpivot_score:
                        postpivot_score = score
                        postpivot_index = len(trace["qpos"])
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(held.copy())
                trace["hand_contact"].append(physical["hand_contact"].copy())
                trace["foot_contact"].append(physical["foot_contact"].copy())
                trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
                trace["torso_tilt"].append(float(physical["tilt"]))
                trace["support_margin"].append(np.nan if support is None else support)
                trace["object_position"].append(object_position)
                trace["object_orientation"].append(object_quaternion)
                trace["object_impulse"].append(
                    np.asarray([cumulative[key] for key in IMPULSE_GROUPS])
                )
            if reason is not None:
                break

        arrays = {key: np.asarray(value) for key, value in trace.items()}
        final_rotation = (
            object_rotation0.inv()
            * quat_rotation(
                data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
            )
        ).magnitude()
        strict_forbidden = sum(
            cumulative[key]
            for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
        )
        valid_frames = 300.0 if first_invalid is None else first_invalid / substeps
        deliberate = bool(
            first_contact_speed is not None
            and first_contact_speed <= 0.25
            and abs(first_contact_vz) <= 0.10
        )
        useful_motion = bool(maximum_rotation >= 0.10 and final_rotation >= 0.08)
        recoverable = maximum_recovered_run >= 20
        gate_pass = bool(
            first_invalid is None
            and strict_forbidden <= 0.10
            and cumulative["wrist"] <= 0.10
            and deliberate
            and longest_contact >= 8
            and useful_motion
            and recoverable
            and postpivot_index is not None
        )
        metrics = {
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else valid_frames,
            "invalid_reason": invalid_reason,
            "first_hand_contact_frame": None if first_contact_step is None else first_contact_step / substeps,
            "first_contact_root_speed_m_s": first_contact_speed,
            "first_contact_vertical_speed_m_s": first_contact_vz,
            "hand_contact_frames": hand_frames,
            "longest_unilateral_contact_frames": longest_contact,
            "maximum_object_reorientation_rad": maximum_rotation,
            "maximum_object_tip_rad": maximum_tip,
            "maximum_object_yaw_rad": maximum_yaw,
            "maximum_horizontal_displacement_m": maximum_displacement,
            "final_object_reorientation_rad": float(final_rotation),
            "minimum_support_margin_m": None if not np.isfinite(minimum_support) else minimum_support,
            "maximum_recovered_run_frames": maximum_recovered_run,
            "final_root_height_m": float(data.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_root_speed_m_s": float(np.linalg.norm(root_velocity)),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "object_impulse_ns": cumulative,
            "strict_forbidden_impulse_ns": strict_forbidden,
            "strongest_hand_contact": best_contact_detail,
            "postpivot_reference_frame": postpivot_index,
            "pivot_prefix_gate_pass": gate_pass,
        }
        rank = (
            hand_frames == 0,
            strict_forbidden > 0.10,
            not deliberate,
            longest_contact < 8,
            first_invalid is not None,
            -valid_frames,
            not useful_motion,
            not recoverable,
            -maximum_rotation,
            -maximum_displacement,
        )
        cost = (
            1.0e10 * float(hand_frames == 0)
            + 1.0e9 * float(strict_forbidden > 0.10)
            + 2.0e8 * float(not deliberate)
            + 2.0e7 * max(0, 8 - longest_contact)
            + 1.0e8 * float(first_invalid is not None)
            + 2.0e5 * (300.0 - valid_frames)
            + 5.0e7 * float(not useful_motion)
            + 2.0e7 * float(not recoverable)
            - 2.0e6 * maximum_rotation
            - 1.0e6 * maximum_displacement
            + 2.0e3 * float(np.dot(parameters[4:], parameters[4:]))
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_parameters = parameters.copy()
            best_metrics = metrics
            print(
                "PROMOTE",
                evaluation_count,
                json.dumps({"parameters": parameters.tolist(), **metrics}),
                flush=True,
            )
        if not record:
            return float(cost), metrics, None
        arrays["ctrl_physics"] = np.asarray(controls)
        arrays["postpivot_qpos"] = arrays["qpos"][
            postpivot_index if postpivot_index is not None else -1
        ]
        arrays["postpivot_qvel"] = arrays["qvel"][
            postpivot_index if postpivot_index is not None else -1
        ]
        return float(cost), metrics, arrays

    lower = np.asarray([0.00, 0.85, 0.30, -1.0] + [-0.16] * 8)
    upper = np.asarray([1.15, 1.50, 1.60, 1.0] + [0.16] * 8)
    midpoint = 0.5 * (lower + upper)
    halfspan = 0.5 * (upper - lower)
    seeds = [
        np.asarray([1.0, 1.0, 1.0, 0.0] + [0.0] * 8),
        np.asarray([0.70, 1.20, 0.8, 0.7] + [0.0] * 8),
    ]
    solver_runs = []
    seed_metrics = []
    for seed_index, seed in enumerate(seeds):
        seed = np.clip(seed, lower, upper)
        _, metrics, _ = evaluate(seed)
        seed_metrics.append({"seed_index": seed_index, "parameters": seed.tolist(), "metrics": metrics})
        normalized_seed = (seed - midpoint) / halfspan
        result = minimize(
            lambda normalized: evaluate(midpoint + halfspan * normalized)[0],
            normalized_seed,
            method="COBYQA",
            bounds=[(-1.0, 1.0)] * len(seed),
            options={
                "maxfev": args.maxfev,
                "initial_tr_radius": 0.35,
                "final_tr_radius": 0.035,
                "disp": True,
            },
        )
        solver_runs.append(
            {
                "seed_index": seed_index,
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "fun": float(result.fun),
                "parameters": (midpoint + halfspan * result.x).tolist(),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output.with_name(args.output.stem + "_checkpoint.npz"),
            best_parameters=best_parameters,
            best_metrics_json=np.asarray(json.dumps(best_metrics)),
            evaluation_count=np.asarray(evaluation_count),
            solver_runs_json=np.asarray(json.dumps(solver_runs)),
        )

    assert best_parameters is not None and best_metrics is not None
    _, final_metrics, arrays = evaluate(best_parameters, record=True)
    assert arrays is not None
    replay = mujoco.MjData(model)
    replay.qpos[:] = initial_qpos
    replay.qvel[:] = initial_qvel
    mujoco.mj_forward(model, replay)
    replay_qpos = []
    replay_qvel = []
    for step, held in enumerate(arrays["ctrl_physics"]):
        replay.ctrl[:] = held
        mujoco.mj_step(model, replay)
        if step % substeps == substeps - 1:
            replay_qpos.append(replay.qpos.copy())
            replay_qvel.append(replay.qvel.copy())
    replay_qpos = np.asarray(replay_qpos)
    replay_qvel = np.asarray(replay_qvel)
    replay_result = {
        "physics_control_samples": len(arrays["ctrl_physics"]),
        "max_qpos_error": float(np.max(np.abs(replay_qpos - arrays["qpos"]))),
        "max_qvel_error": float(np.max(np.abs(replay_qvel - arrays["qvel"]))),
    }
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        best_parameters=best_parameters,
        impulse_groups=np.asarray(IMPULSE_GROUPS),
        **arrays,
    )
    report = {
        "method": "reduced hybrid phase shooting around demonstrated horizontal-normal side contact",
        "solver": "SciPy COBYQA over 12 bounded phase variables",
        "dimensions": 12,
        "evaluation_count": evaluation_count,
        "phase_structure": [
            "stable prefix",
            "support shaping before side approach",
            "left-palm side/corner acquisition",
            "unilateral load and floor-supported pivot",
            "arm retract overlapping support recovery",
        ],
        "seed_metrics": seed_metrics,
        "solver_runs": solver_runs,
        "best_parameters": best_parameters.tolist(),
        "best": final_metrics,
        "replay": replay_result,
        "pivot_gate_pass": bool(final_metrics["pivot_prefix_gate_pass"]),
        "runtime_s": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(final_metrics), flush=True)
    print("PIVOT_GATE_PASS" if report["pivot_gate_pass"] else "PIVOT_GATE_FAIL", flush=True)


if __name__ == "__main__":
    main()
