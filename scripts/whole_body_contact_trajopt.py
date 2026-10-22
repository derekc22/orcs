#!/usr/bin/env python3
"""Hybrid whole-body/contact trajectory optimization for largebox start 0.

The optimizer uses exact MuJoCo interval maps at 200 Hz.  This file also owns
the replay/contact-attribution gates used before expensive optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from probe_flat_hand_grasp import build
from scipy.optimize import least_squares, minimize
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/hand_manipulation_research"
IMPULSE_GROUPS = (
    "left_hand",
    "right_hand",
    "wrist",
    "feet",
    "legs",
    "torso_pelvis",
    "other_robot",
)


def _geom_name(model: mujoco.MjModel, geom: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
        or f"geom_{geom}"
    )


def _contact_group(model: mujoco.MjModel, engine: Any, geom: int) -> str:
    geom = int(geom)
    if geom == int(engine.hand_geoms[0]):
        return "left_hand"
    if geom == int(engine.hand_geoms[1]):
        return "right_hand"
    body = int(model.geom_bodyid[geom])
    name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
    ).lower()
    geom_name = _geom_name(model, geom).lower()
    if "wrist" in name or "wrist" in geom_name or "hand" in geom_name:
        return "wrist"
    if "foot" in name or "ankle" in name:
        return "feet"
    if any(token in name for token in ("knee", "shin", "hip", "thigh")):
        return "legs"
    if any(token in name for token in ("torso", "pelvis", "waist")):
        return "torso_pelvis"
    return "other_robot"


def _contact_impulses(
    model: mujoco.MjModel, data: mujoco.MjData, engine: Any
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    impulses = {key: 0.0 for key in IMPULSE_GROUPS}
    contacts: list[dict[str, Any]] = []
    object_geoms = set(map(int, engine.object_geoms))
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in object_geoms and second not in object_geoms:
            continue
        other = second if first in object_geoms else first
        if int(model.geom_bodyid[other]) not in engine.robot_bodies:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        impulse = float(np.linalg.norm(wrench[:3])) * model.opt.timestep
        group = _contact_group(model, engine, other)
        impulses[group] += impulse
        contacts.append(
            {
                "group": group,
                "robot_geom": _geom_name(model, other),
                "object_geom": _geom_name(model, first if first in object_geoms else second),
                "impulse_ns": impulse,
            }
        )
    return impulses, contacts


def _orientation_error(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    first = Rotation.from_quat(np.r_[q1_wxyz[1:], q1_wxyz[0]])
    second = Rotation.from_quat(np.r_[q2_wxyz[1:], q2_wxyz[0]])
    return float((first.inv() * second).magnitude())


def _replay_artifact(
    model: mujoco.MjModel,
    engine: Any,
    generator: Any,
    path: Path,
) -> dict[str, Any]:
    artifact = np.load(path)
    if "anchor_qpos" in artifact:
        initial_qpos = artifact["anchor_qpos"].copy()
        initial_qvel = artifact["anchor_qvel"].copy()
        anchor_kind = "explicit_pre_step_anchor"
    else:
        initial_qpos = artifact["initial_qpos"].copy()
        initial_qvel = artifact["initial_qvel"].copy()
        anchor_kind = "initial_state"
    controls = artifact["ctrl_physics"]
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    cumulative = {key: 0.0 for key in IMPULSE_GROUPS}
    contact_geoms: dict[str, set[str]] = {key: set() for key in IMPULSE_GROUPS}
    replay_qpos = []
    replay_qvel = []
    first_invalid_step = None
    invalid_reason = None
    for step, control in enumerate(controls):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        impulses, contacts = _contact_impulses(model, data, engine)
        for key, value in impulses.items():
            cumulative[key] += value
        for contact in contacts:
            contact_geoms[contact["group"]].add(contact["robot_geom"])
        physical = generator._physical(data)
        reason = generator._invalid_reason(data, physical)
        strict_nonhand = sum(
            cumulative[key]
            for key in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        if reason is None and strict_nonhand > 0.10:
            reason = "non_hand_object_impulse"
        if reason is not None and first_invalid_step is None:
            first_invalid_step = step
            invalid_reason = reason
        if step % engine.config.substeps == engine.config.substeps - 1:
            replay_qpos.append(data.qpos.copy())
            replay_qvel.append(data.qvel.copy())
    replay_qpos = np.asarray(replay_qpos)
    replay_qvel = np.asarray(replay_qvel)
    stored_qpos = artifact["qpos"][: len(replay_qpos)]
    stored_qvel = artifact["qvel"][: len(replay_qvel)]
    qpos_error = float(np.max(np.abs(replay_qpos - stored_qpos)))
    qvel_error = float(np.max(np.abs(replay_qvel - stored_qvel)))
    return {
        "artifact": str(path.relative_to(ROOT)),
        "anchor_convention": anchor_kind,
        "physics_hz": 1.0 / model.opt.timestep,
        "reference_hz": 1.0 / (model.opt.timestep * engine.config.substeps),
        "physics_control_samples": int(len(controls)),
        "complete_reference_samples": int(len(replay_qpos)),
        "max_qpos_replay_error": qpos_error,
        "max_qvel_replay_error": qvel_error,
        "exact_replay": bool(qpos_error == 0.0 and qvel_error == 0.0),
        "first_invalid_frame": (
            None
            if first_invalid_step is None
            else first_invalid_step / engine.config.substeps
        ),
        "invalid_reason": invalid_reason,
        "object_impulse_ns": cumulative,
        "contact_geoms": {
            key: sorted(value) for key, value in contact_geoms.items() if value
        },
    }


def run_audit(output: Path) -> dict[str, Any]:
    engine, generator = build(0.55)
    model = engine.model
    artifacts = [
        LOG / "upright_balance_vertical_v4.npz",
        LOG / "early_recovery_during_lift_v11.npz",
        LOG / "braced_bilateral_transition_v2.npz",
    ]
    results = [
        _replay_artifact(model, engine, generator, path) for path in artifacts
    ]
    near = np.load(LOG / "reach_support_feasibility_v1_near_state.npz")
    candidate = mujoco.MjData(model)
    candidate.qpos[:] = near["candidate_qpos"]
    candidate.qvel[:] = near["candidate_qvel"]
    mujoco.mj_forward(model, candidate)
    near_physical = generator._physical(candidate)
    near_summary = {
        "artifact": "logs/hand_manipulation_research/reach_support_feasibility_v1_near_state.npz",
        "candidate_index": int(near["candidate_index"]),
        "root_height_m": float(candidate.qpos[engine.robot_qpos + 2]),
        "torso_tilt_deg": float(np.rad2deg(near_physical["tilt"])),
        "foot_contact": near_physical["foot_contact"].tolist(),
        "hand_contact": near_physical["hand_contact"].tolist(),
        "object_position_m": candidate.qpos[
            engine.object_qpos : engine.object_qpos + 3
        ].tolist(),
    }
    report = {
        "date": "2026-10-20",
        "method": "independent exact MuJoCo physics-rate replay and contact attribution gate",
        "model": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "physics_timestep_s": model.opt.timestep,
            "substeps_per_reference_frame": engine.config.substeps,
        },
        "replays": results,
        "reach_support_near_state": near_summary,
        "gate_pass": bool(all(item["exact_replay"] for item in results)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    quaternion = rotation.as_quat()
    return np.r_[quaternion[3], quaternion[:3]]


def _wxyz_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def _solve_step_posture(
    model: mujoco.MjModel,
    engine: Any,
    generator: Any,
    anchor_qpos: np.ndarray,
    balance: np.ndarray,
    feet_reference: np.ndarray,
    foot_rotations_reference: np.ndarray,
    feet_target: np.ndarray,
    root_target: np.ndarray,
    seed: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    data = mujoco.MjData(model)
    balance_qpos = engine.actuator_qpos[balance]
    joint_ids = model.actuator_trnid[balance, 0].astype(int)
    root0 = anchor_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_rotation0 = _wxyz_to_rotation(
        anchor_qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    )
    lower = np.r_[
        [-0.18, -0.18, -0.10, -0.35, -0.35, -0.25],
        model.jnt_range[joint_ids, 0] + 0.004,
    ]
    upper = np.r_[
        [0.18, 0.18, 0.10, 0.35, 0.35, 0.25],
        model.jnt_range[joint_ids, 1] - 0.004,
    ]
    joints0 = anchor_qpos[balance_qpos].copy()

    def set_state(values: np.ndarray) -> None:
        data.qpos[:] = anchor_qpos
        data.qvel[:] = 0.0
        data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root0 + values[:3]
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = (
            _rotation_to_wxyz(
                Rotation.from_rotvec(values[3:6]) * root_rotation0
            )
        )
        data.qpos[balance_qpos] = values[6:]
        mujoco.mj_forward(model, data)

    def residual(values: np.ndarray) -> np.ndarray:
        set_state(values)
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        rotations = np.asarray(
            [data.xmat[body].reshape(3, 3) for body in engine.foot_bodies]
        )
        rotation_error = np.concatenate(
            [
                Rotation.from_matrix(
                    rotations[index] @ foot_rotations_reference[index].T
                ).as_rotvec()
                for index in range(2)
            ]
        )
        torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        return np.r_[
            150.0 * (feet - feet_target).ravel(),
            26.0 * rotation_error,
            38.0 * (root - root_target),
            18.0 * torso_up[:2],
            0.025 * (values[6:] - joints0),
        ]

    result = least_squares(
        residual,
        np.clip(seed, lower, upper),
        bounds=(lower, upper),
        max_nfev=300,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    set_state(result.x)
    feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
    return result.x.copy(), {
        "cost": float(result.cost),
        "maximum_foot_error_m": float(
            np.max(np.linalg.norm(feet - feet_target, axis=1))
        ),
        "root_error_m": float(
            np.linalg.norm(
                data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
                - root_target
            )
        ),
        "torso_tilt_deg": float(
            np.rad2deg(generator._physical(data)["tilt"])
        ),
    }


def run_support_gate(
    output: Path,
    *,
    max_nfev: int,
    interval_frames: int,
) -> dict[str, Any]:
    """Solve a small planned unload/swing/replant trajectory by multiple shooting."""
    engine, generator = build(0.55)
    model = engine.model
    stable = np.load(LOG / "upright_balance_vertical_v4.npz")
    anchor_frame = 100
    anchor_qpos = stable["qpos"][anchor_frame - 1].copy()
    anchor_qvel = stable["qvel"][anchor_frame - 1].copy()
    base_control = stable["ctrl"][anchor_frame - 1].copy()
    feedback_parameters = stable["parameters"].copy()
    desired_rotation = Rotation.from_quat(stable["desired_rotation_xyzw"])
    balance = np.asarray(generator.balance_actuators, dtype=int)
    balance_qpos = engine.actuator_qpos[balance]
    anchor = mujoco.MjData(model)
    anchor.qpos[:] = anchor_qpos
    anchor.qvel[:] = anchor_qvel
    mujoco.mj_forward(model, anchor)
    feet0 = np.asarray(
        [anchor.xpos[body].copy() for body in engine.foot_bodies]
    )
    foot_rotations0 = np.asarray(
        [anchor.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    root0 = anchor_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    object0 = anchor_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object0[:2], 0.0]
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    # Swing the foot farther from the box first.  The small forward component is
    # enough to prove a real support exchange without revisiting the failed
    # 14 cm gait family.
    swing_foot = 0
    support_foot = 1
    toward_support = feet0[support_foot] - feet0[swing_foot]
    toward_support[2] = 0.0
    root_transfer = root0 + 0.08 * toward_support
    step_delta = 0.012 * forward
    root_final = root0 + 0.5 * step_delta

    feet_transfer = feet0.copy()
    feet_swing = feet0.copy()
    feet_swing[swing_foot] += step_delta + np.asarray([0.0, 0.0, 0.015])
    feet_plant = feet0.copy()
    feet_plant[swing_foot] += step_delta
    joints0 = anchor_qpos[balance_qpos].copy()
    seed = np.r_[np.zeros(6), joints0]
    transfer, transfer_diag = _solve_step_posture(
        model,
        engine,
        generator,
        anchor_qpos,
        balance,
        feet0,
        foot_rotations0,
        feet_transfer,
        root_transfer,
        seed,
    )
    swing, swing_diag = _solve_step_posture(
        model,
        engine,
        generator,
        anchor_qpos,
        balance,
        feet0,
        foot_rotations0,
        feet_swing,
        root_transfer + 0.35 * step_delta,
        transfer,
    )
    plant, plant_diag = _solve_step_posture(
        model,
        engine,
        generator,
        anchor_qpos,
        balance,
        feet0,
        foot_rotations0,
        feet_plant,
        root_final,
        swing,
    )

    intervals = 10
    steps_per_interval = interval_frames * engine.config.substeps
    key_nodes = np.asarray([0, 2, 5, 7, 10], dtype=int)
    offset = base_control[balance] - joints0
    key_controls = np.asarray(
        [
            base_control[balance],
            transfer[6:] + offset,
            swing[6:] + offset,
            plant[6:] + offset,
            plant[6:] + offset,
        ]
    )
    control_seed = np.empty((intervals + 1, len(balance)))
    for node in range(intervals + 1):
        segment = min(np.searchsorted(key_nodes, node, side="right") - 1, 3)
        segment = max(segment, 0)
        amount = (node - key_nodes[segment]) / (
            key_nodes[segment + 1] - key_nodes[segment]
        )
        control_seed[node] = (
            (1.0 - amount) * key_controls[segment]
            + amount * key_controls[segment + 1]
        )

    node_targets = []
    for node in range(intervals + 1):
        if node <= 2:
            amount = node / 2.0
            feet_target = feet0.copy()
            root_target = (1.0 - amount) * root0 + amount * root_transfer
        elif node <= 5:
            amount = (node - 2) / 3.0
            feet_target = (1.0 - amount) * feet0 + amount * feet_swing
            root_target = root_transfer + 0.35 * amount * step_delta
        elif node <= 7:
            amount = (node - 5) / 2.0
            feet_target = (1.0 - amount) * feet_swing + amount * feet_plant
            root_target = (
                (1.0 - amount) * (root_transfer + 0.35 * step_delta)
                + amount * root_final
            )
        else:
            feet_target = feet_plant.copy()
            root_target = root_final.copy()
        node_targets.append((feet_target, root_target))

    joint_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    group_tokens = {
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
            [
                index
                for index, name in enumerate(joint_names)
                if any(token in name for token in tokens)
            ],
            dtype=int,
        )
        for key, tokens in group_tokens.items()
    }

    def feedback_correction(
        data: mujoco.MjData, desired_root_position: np.ndarray
    ) -> np.ndarray:
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation_error = (
            desired_rotation.inv() * _wxyz_to_rotation(quaternion)
        ).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal = np.asarray(
            [
                root[1] - desired_root_position[1],
                velocity[1],
                rotation_error[0],
                velocity[3],
            ]
        )
        lateral_features = np.asarray(
            [
                root[0] - desired_root_position[0],
                velocity[0],
                rotation_error[1],
                velocity[4],
            ]
        )
        gains = feedback_parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row_index = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[feedback_groups[key]] += float(
                np.dot(gains[row_index], sagittal)
            )
            row_index += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[feedback_groups[key]] += float(
                np.dot(gains[row_index], lateral_features)
            )
            row_index += 1
        return np.clip(delta, -0.28, 0.28)

    def rollout_interval(
        interval: int,
        qpos: np.ndarray,
        qvel: np.ndarray,
        control0: np.ndarray,
        control1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        for substep in range(steps_per_interval):
            amount = (substep + 1) / steps_per_interval
            held = base_control.copy()
            held[balance] = (1.0 - amount) * control0 + amount * control1
            desired_root_position = (
                (1.0 - amount) * node_targets[interval][1]
                + amount * node_targets[interval + 1][1]
            )
            held += feedback_correction(data, desired_root_position)
            data.ctrl[:] = np.clip(
                held,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(model, data)
        return data.qpos.copy(), data.qvel.copy()

    nominal_qpos = [anchor_qpos]
    nominal_qvel = [anchor_qvel]
    for interval in range(intervals):
        qpos, qvel = rollout_interval(
            interval,
            nominal_qpos[-1],
            nominal_qvel[-1],
            control_seed[interval],
            control_seed[interval + 1],
        )
        nominal_qpos.append(qpos)
        nominal_qvel.append(qvel)
    nominal_qpos = np.asarray(nominal_qpos)
    nominal_qvel = np.asarray(nominal_qvel)

    nq_tangent = model.nv
    state_width = 2 * model.nv
    state_slices: list[slice | None] = [None]
    cursor = 0
    for _node in range(1, intervals + 1):
        state_slices.append(slice(cursor, cursor + state_width))
        cursor += state_width
    control_slices = []
    for _node in range(intervals + 1):
        control_slices.append(slice(cursor, cursor + len(balance)))
        cursor += len(balance)
    dimensions = cursor

    q_scale = np.full(model.nv, 0.30)
    q_scale[:3] = 0.08
    q_scale[3:6] = 0.25
    q_scale[engine.object_dof : engine.object_dof + 3] = 0.006
    q_scale[engine.object_dof + 3 : engine.object_dof + 6] = 0.02
    v_scale = np.full(model.nv, 1.8)
    v_scale[:3] = 0.45
    v_scale[3:6] = 1.2
    v_scale[engine.object_dof : engine.object_dof + 6] = 0.08
    control_scale = np.full(len(balance), 0.28)

    lower = np.full(dimensions, -1.0)
    upper = np.full(dimensions, 1.0)
    for node, control_slice in enumerate(control_slices):
        lower[control_slice] = (
            model.actuator_ctrlrange[balance, 0] - control_seed[node]
        ) / control_scale
        upper[control_slice] = (
            model.actuator_ctrlrange[balance, 1] - control_seed[node]
        ) / control_scale

    def decode(values: np.ndarray):
        qposes = [anchor_qpos]
        qvels = [anchor_qvel]
        for node in range(1, intervals + 1):
            block = values[state_slices[node]]
            qpos = nominal_qpos[node].copy()
            mujoco.mj_integratePos(
                model, qpos, block[:nq_tangent] * q_scale, 1.0
            )
            qposes.append(qpos)
            qvels.append(
                nominal_qvel[node] + block[nq_tangent:] * v_scale
            )
        controls = [
            control_seed[node] + values[control_slices[node]] * control_scale
            for node in range(intervals + 1)
        ]
        return qposes, qvels, controls

    position_tolerance = np.full(model.nv, 0.012)
    position_tolerance[:3] = 0.004
    position_tolerance[3:6] = 0.012
    position_tolerance[engine.object_dof : engine.object_dof + 3] = 0.001
    position_tolerance[engine.object_dof + 3 : engine.object_dof + 6] = 0.004
    velocity_tolerance = np.full(model.nv, 0.18)
    velocity_tolerance[:3] = 0.06
    velocity_tolerance[3:6] = 0.16
    velocity_tolerance[engine.object_dof : engine.object_dof + 6] = 0.03

    dynamics_weight = 1.0

    def residual(values: np.ndarray) -> np.ndarray:
        qposes, qvels, controls = decode(values)
        parts: list[np.ndarray] = []
        for interval in range(intervals):
            predicted_qpos, predicted_qvel = rollout_interval(
                interval,
                qposes[interval],
                qvels[interval],
                controls[interval],
                controls[interval + 1],
            )
            difference = np.zeros(model.nv)
            mujoco.mj_differentiatePos(
                model, difference, 1.0, predicted_qpos, qposes[interval + 1]
            )
            parts.append(dynamics_weight * difference / position_tolerance)
            parts.append(
                dynamics_weight
                * (qvels[interval + 1] - predicted_qvel)
                / velocity_tolerance
            )
        for node in range(1, intervals + 1):
            data = mujoco.MjData(model)
            data.qpos[:] = qposes[node]
            data.qvel[:] = qvels[node]
            mujoco.mj_forward(model, data)
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            feet_target, root_target = node_targets[node]
            root = qposes[node][engine.robot_qpos : engine.robot_qpos + 3]
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            task = np.r_[
                (feet - feet_target).ravel() / 0.012,
                (root - root_target) / 0.025,
                torso_up[:2] / 0.20,
                max(0.0, 0.54 - root[2]) / 0.02,
            ]
            if 3 <= node <= 6:
                masses = model.body_mass[list(engine.robot_bodies)]
                com = np.average(
                    data.xipos[list(engine.robot_bodies)], axis=0, weights=masses
                )
                task = np.r_[
                    task,
                    (com[:2] - feet[support_foot, :2]) / 0.08,
                ]
            else:
                task = np.r_[task, np.zeros(2)]
            parts.append(task)
        for node in range(intervals):
            parts.append(
                0.10 * (controls[node + 1] - controls[node]) / 0.20
            )
        parts.append(0.35 * qvels[-1] / velocity_tolerance)
        parts.append(0.04 * values)
        return np.concatenate(parts)

    # Construct the exact block sparsity of the residual.  The final small
    # regularizer is diagonal; local task terms only touch one state node.
    probe = residual(np.zeros(dimensions))
    sparsity = lil_matrix((len(probe), dimensions), dtype=np.int8)
    row = 0
    for interval in range(intervals):
        count = 2 * model.nv
        if interval > 0:
            sparsity[row : row + count, state_slices[interval]] = 1
        sparsity[row : row + count, state_slices[interval + 1]] = 1
        sparsity[row : row + count, control_slices[interval]] = 1
        sparsity[row : row + count, control_slices[interval + 1]] = 1
        row += count
    node_task_width = 6 + 3 + 2 + 1 + 2
    for node in range(1, intervals + 1):
        sparsity[row : row + node_task_width, state_slices[node]] = 1
        row += node_task_width
    for node in range(intervals):
        sparsity[row : row + len(balance), control_slices[node]] = 1
        sparsity[row : row + len(balance), control_slices[node + 1]] = 1
        row += len(balance)
    sparsity[row : row + model.nv, state_slices[-1]] = 1
    row += model.nv
    for column in range(dimensions):
        sparsity[row + column, column] = 1
    assert row + dimensions == len(probe)

    values = np.zeros(dimensions)
    history = []
    started = time.perf_counter()
    stage_weights = (0.2,) if max_nfev <= 1 else (0.2, 1.0, 8.0, 50.0)
    for stage, weight in enumerate(stage_weights):
        dynamics_weight = weight
        solved = least_squares(
            residual,
            np.clip(values, lower, upper),
            bounds=(lower, upper),
            jac_sparsity=sparsity.tocsr(),
            method="trf",
            max_nfev=max_nfev,
            diff_step=2.0e-3,
            ftol=2.0e-5,
            xtol=2.0e-5,
            gtol=2.0e-5,
            verbose=2,
        )
        values = solved.x.copy()
        history.append(
            {
                "stage": stage,
                "dynamics_weight": weight,
                "cost": float(solved.cost),
                "optimality": float(solved.optimality),
                "nfev": int(solved.nfev),
                "status": int(solved.status),
            }
        )
        np.savez_compressed(
            output.with_name(output.stem + "_checkpoint.npz"),
            variables=values,
            history_json=np.asarray(json.dumps(history)),
        )

    _qposes, _qvels, optimized_controls = decode(values)
    data = mujoco.MjData(model)
    data.qpos[:] = anchor_qpos
    data.qvel[:] = anchor_qvel
    mujoco.mj_forward(model, data)
    ctrl_physics = []
    trace = {
        key: []
        for key in (
            "qpos",
            "qvel",
            "ctrl",
            "foot_contact",
            "hand_contact",
            "foot_position",
            "root_height",
            "torso_tilt",
            "object_impulse",
        )
    }
    cumulative = {key: 0.0 for key in IMPULSE_GROUPS}
    first_invalid = None
    invalid_reason = None
    total_transition_steps = intervals * steps_per_interval
    hold_steps = 40 * engine.config.substeps
    for step in range(total_transition_steps + hold_steps):
        if step < total_transition_steps:
            interval = step // steps_per_interval
            local = step % steps_per_interval
            amount = (local + 1) / steps_per_interval
            control = (
                (1.0 - amount) * optimized_controls[interval]
                + amount * optimized_controls[interval + 1]
            )
        else:
            control = optimized_controls[-1]
        held = base_control.copy()
        held[balance] = control
        if step < total_transition_steps:
            desired_root_position = (
                (1.0 - amount) * node_targets[interval][1]
                + amount * node_targets[interval + 1][1]
            )
        else:
            desired_root_position = node_targets[-1][1]
        held += feedback_correction(data, desired_root_position)
        held = np.clip(
            held,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        data.ctrl[:] = held
        ctrl_physics.append(held.copy())
        mujoco.mj_step(model, data)
        impulses, _ = _contact_impulses(model, data, engine)
        for key, value in impulses.items():
            cumulative[key] += value
        physical = generator._physical(data)
        strict_nonhand = sum(
            cumulative[key]
            for key in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        reason = generator._invalid_reason(data, physical)
        if reason is None and strict_nonhand > 0.10:
            reason = "non_hand_object_impulse"
        if reason is not None and first_invalid is None:
            first_invalid = step
            invalid_reason = reason
        if step % engine.config.substeps == engine.config.substeps - 1:
            feet = np.asarray(
                [data.xpos[body].copy() for body in engine.foot_bodies]
            )
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["ctrl"].append(data.ctrl.copy())
            trace["foot_contact"].append(physical["foot_contact"].copy())
            trace["hand_contact"].append(physical["hand_contact"].copy())
            trace["foot_position"].append(feet)
            trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
            trace["torso_tilt"].append(float(physical["tilt"]))
            trace["object_impulse"].append(
                np.asarray([cumulative[key] for key in IMPULSE_GROUPS])
            )
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    foot_positions = arrays["foot_position"]
    maximum_clearance = float(
        np.max(foot_positions[:, swing_foot, 2] - feet0[swing_foot, 2])
    )
    final_step = float(
        np.dot(foot_positions[-1, swing_foot] - feet0[swing_foot], forward)
    )
    planted_tail = int(
        np.sum(np.all(arrays["foot_contact"][-10:], axis=1))
    )
    dynamics_weight = 50.0
    qposes, qvels, controls = decode(values)
    max_position_defect = 0.0
    max_velocity_defect = 0.0
    for interval in range(intervals):
        predicted_qpos, predicted_qvel = rollout_interval(
            interval,
            qposes[interval], qvels[interval], controls[interval], controls[interval + 1]
        )
        difference = np.zeros(model.nv)
        mujoco.mj_differentiatePos(
            model, difference, 1.0, predicted_qpos, qposes[interval + 1]
        )
        max_position_defect = max(max_position_defect, float(np.max(np.abs(difference))))
        max_velocity_defect = max(
            max_velocity_defect,
            float(np.max(np.abs(qvels[interval + 1] - predicted_qvel))),
        )
    gate_pass = bool(
        first_invalid is None
        and maximum_clearance >= 0.008
        and final_step >= 0.004
        and planted_tail >= 5
        and cumulative["feet"]
        + cumulative["legs"]
        + cumulative["torso_pelvis"]
        + cumulative["other_robot"]
        <= 0.10
    )
    metrics = {
        "method": "sparse hybrid direct multiple shooting manipulation-free support transition gate",
        "solver": "scipy least_squares TRF with block-sparse finite differences and penalty continuation",
        "dimensions": dimensions,
        "shooting_intervals": intervals,
        "interval_reference_frames": interval_frames,
        "physics_steps_per_interval": steps_per_interval,
        "runtime_s": time.perf_counter() - started,
        "history": history,
        "static_pose_diagnostics": {
            "transfer": transfer_diag,
            "swing": swing_diag,
            "plant": plant_diag,
        },
        "maximum_swing_foot_clearance_m": maximum_clearance,
        "final_swing_foot_goalward_reposition_m": final_step,
        "two_foot_contact_frames_in_final_10": planted_tail,
        "first_invalid_local_frame": (
            None if first_invalid is None else first_invalid / engine.config.substeps
        ),
        "invalid_reason": invalid_reason,
        "final_root_height_m": float(arrays["root_height"][-1]),
        "final_torso_tilt_deg": float(np.rad2deg(arrays["torso_tilt"][-1])),
        "object_impulse_ns": cumulative,
        "maximum_position_shooting_defect": max_position_defect,
        "maximum_velocity_shooting_defect": max_velocity_defect,
        "exact_replay_gate_pass": gate_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"),
        anchor_qpos=anchor_qpos,
        anchor_qvel=anchor_qvel,
        anchor_frame=anchor_frame,
        balance_actuators=balance,
        shooting_variables=values,
        shooting_qpos=np.asarray(qposes),
        shooting_qvel=np.asarray(qvels),
        shooting_controls=np.asarray(controls),
        ctrl_physics=np.asarray(ctrl_physics),
        impulse_groups=np.asarray(IMPULSE_GROUPS),
        **arrays,
    )
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def run_unload_gate(output: Path) -> dict[str, Any]:
    """Create a full-horizon, exact-replay foot unload/replant seed."""
    engine, generator = build(0.55)
    model = engine.model
    stable = np.load(LOG / "upright_balance_vertical_v4.npz")
    base = stable["base_target"].copy()
    parameters = stable["parameters"].copy()
    desired_root = stable["desired_root"].copy()
    desired_rotation = Rotation.from_quat(stable["desired_rotation_xyzw"])
    joint_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    tokens = {
        "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
        "knee": ("left_knee", "right_knee"),
        "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
        "waist_pitch": ("waist_pitch",),
        "hip_roll": ("left_hip_roll", "right_hip_roll"),
        "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
        "waist_roll": ("waist_roll",),
    }
    groups = {
        key: np.asarray(
            [
                index
                for index, name in enumerate(joint_names)
                if any(token in name for token in values)
            ],
            dtype=int,
        )
        for key, values in tokens.items()
    }
    name_to_actuator = {name: index for index, name in enumerate(joint_names)}
    pulse_actuators = np.asarray(
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
    pulse = np.asarray([0.0, 0.006, 0.040, -0.022, -0.004])
    anchor_frame = 100
    pulse_start_frame = 108
    pulse_duration_frames = 14

    def correction(data: mujoco.MjData) -> np.ndarray:
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation_error = (
            desired_rotation.inv() * _wxyz_to_rotation(quaternion)
        ).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal = np.asarray(
            [
                root[1] - desired_root[1],
                velocity[1],
                rotation_error[0],
                velocity[3],
            ]
        )
        lateral = np.asarray(
            [
                root[0] - desired_root[0],
                velocity[0],
                rotation_error[1],
                velocity[4],
            ]
        )
        gains = parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[groups[key]] += float(np.dot(gains[row], sagittal))
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[groups[key]] += float(np.dot(gains[row], lateral))
            row += 1
        return np.clip(delta, -0.28, 0.28)

    data = mujoco.MjData(model)
    data.qpos[:] = stable["initial_qpos"]
    data.qvel[:] = stable["initial_qvel"]
    mujoco.mj_forward(model, data)
    initial_foot_z = None
    cumulative = {key: 0.0 for key in IMPULSE_GROUPS}
    first_invalid = None
    invalid_reason = None
    controls = []
    pre_step_qpos = []
    pre_step_qvel = []
    trace = {
        key: []
        for key in (
            "qpos",
            "qvel",
            "ctrl",
            "actuator_force",
            "foot_contact",
            "hand_contact",
            "foot_position",
            "root_height",
            "torso_tilt",
            "object_impulse",
        )
    }
    for step in range(300 * engine.config.substeps):
        pre_step_qpos.append(data.qpos.copy())
        pre_step_qvel.append(data.qvel.copy())
        frame = step / engine.config.substeps
        if frame < anchor_frame:
            held = stable["ctrl_physics"][step].copy()
        else:
            held = base + correction(data)
            if pulse_start_frame <= frame < pulse_start_frame + pulse_duration_frames:
                phase = (frame - pulse_start_frame) / pulse_duration_frames
                held[pulse_actuators] += np.sin(np.pi * phase) ** 2 * pulse
        held = np.clip(
            held,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        data.ctrl[:] = held
        controls.append(held.copy())
        mujoco.mj_step(model, data)
        if initial_foot_z is None and frame >= anchor_frame:
            initial_foot_z = float(data.xpos[engine.foot_bodies[0], 2])
        impulses, _ = _contact_impulses(model, data, engine)
        for key, value in impulses.items():
            cumulative[key] += value
        physical = generator._physical(data)
        reason = generator._invalid_reason(data, physical)
        strict_nonhand = sum(
            cumulative[key]
            for key in ("feet", "legs", "torso_pelvis", "other_robot")
        )
        if reason is None and strict_nonhand > 0.10:
            reason = "non_hand_object_impulse"
        if reason is not None and first_invalid is None:
            first_invalid = step
            invalid_reason = reason
        if step % engine.config.substeps == engine.config.substeps - 1:
            feet = np.asarray(
                [data.xpos[body].copy() for body in engine.foot_bodies]
            )
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["ctrl"].append(data.ctrl.copy())
            trace["actuator_force"].append(data.actuator_force.copy())
            trace["foot_contact"].append(physical["foot_contact"].copy())
            trace["hand_contact"].append(physical["hand_contact"].copy())
            trace["foot_position"].append(feet)
            trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
            trace["torso_tilt"].append(float(physical["tilt"]))
            trace["object_impulse"].append(
                np.asarray([cumulative[key] for key in IMPULSE_GROUPS])
            )
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    controls = np.asarray(controls)
    pre_step_qpos = np.asarray(pre_step_qpos)
    pre_step_qvel = np.asarray(pre_step_qvel)
    # Count at physics rate to avoid overstating a sub-reference-frame event.
    replay = mujoco.MjData(model)
    replay.qpos[:] = stable["initial_qpos"]
    replay.qvel[:] = stable["initial_qvel"]
    mujoco.mj_forward(model, replay)
    replay_qpos = []
    replay_qvel = []
    physics_one_foot_steps = 0
    maximum_clearance = 0.0
    anchor_foot_z = None
    for step, held in enumerate(controls):
        replay.ctrl[:] = held
        mujoco.mj_step(model, replay)
        physical = generator._physical(replay)
        if step >= anchor_frame * engine.config.substeps:
            if anchor_foot_z is None:
                anchor_foot_z = float(replay.xpos[engine.foot_bodies[0], 2])
            maximum_clearance = max(
                maximum_clearance,
                float(replay.xpos[engine.foot_bodies[0], 2] - anchor_foot_z),
            )
        physics_one_foot_steps += int(np.sum(physical["foot_contact"]) == 1)
        if step % engine.config.substeps == engine.config.substeps - 1:
            replay_qpos.append(replay.qpos.copy())
            replay_qvel.append(replay.qvel.copy())
    qpos_error = float(
        np.max(np.abs(np.asarray(replay_qpos) - arrays["qpos"]))
    )
    qvel_error = float(
        np.max(np.abs(np.asarray(replay_qvel) - arrays["qvel"]))
    )
    # Explicit shooting-node reproduction: every interval starts from its
    # stored pre-step state and consumes the original physics-rate controls.
    node_steps = np.asarray([400, 432, 460, 488, 520, 800, 1200], dtype=int)
    max_node_qpos_error = 0.0
    max_node_qvel_error = 0.0
    for begin, end in zip(node_steps[:-1], node_steps[1:], strict=True):
        interval_data = mujoco.MjData(model)
        interval_data.qpos[:] = pre_step_qpos[begin]
        interval_data.qvel[:] = pre_step_qvel[begin]
        mujoco.mj_forward(model, interval_data)
        for held in controls[begin:end]:
            interval_data.ctrl[:] = held
            mujoco.mj_step(model, interval_data)
        if end < len(pre_step_qpos):
            max_node_qpos_error = max(
                max_node_qpos_error,
                float(np.max(np.abs(interval_data.qpos - pre_step_qpos[end]))),
            )
            max_node_qvel_error = max(
                max_node_qvel_error,
                float(np.max(np.abs(interval_data.qvel - pre_step_qvel[end]))),
            )
    full_valid = first_invalid is None
    gate_pass = bool(
        full_valid
        and physics_one_foot_steps >= 4
        and maximum_clearance >= 0.002
        and int(np.sum(np.all(arrays["foot_contact"][-10:], axis=1))) >= 8
        and qpos_error == 0.0
        and qvel_error == 0.0
        and max_node_qpos_error <= 1.0e-12
        and max_node_qvel_error <= 1.0e-12
    )
    metrics = {
        "method": "phase-structured exact-closed-loop foot unload/replant seed represented as zero-defect multiple shooting",
        "full_horizon_valid": full_valid,
        "first_invalid_frame": (
            None if first_invalid is None else first_invalid / engine.config.substeps
        ),
        "invalid_reason": invalid_reason,
        "pulse_start_frame": pulse_start_frame,
        "pulse_duration_frames": pulse_duration_frames,
        "pulse_actuators": joint_names[pulse_actuators].tolist(),
        "pulse_target_offsets_rad": pulse.tolist(),
        "physics_single_support_steps": physics_one_foot_steps,
        "approximate_single_support_duration_s": physics_one_foot_steps
        * model.opt.timestep,
        "maximum_swing_foot_clearance_m": maximum_clearance,
        "final_root_height_m": float(arrays["root_height"][-1]),
        "final_torso_tilt_deg": float(np.rad2deg(arrays["torso_tilt"][-1])),
        "final_foot_contact": arrays["foot_contact"][-1].tolist(),
        "object_impulse_ns": cumulative,
        "replay": {
            "physics_control_samples": len(controls),
            "max_qpos_error": qpos_error,
            "max_qvel_error": qvel_error,
        },
        "multiple_shooting_reproduction": {
            "node_physics_steps": node_steps.tolist(),
            "max_qpos_defect": max_node_qpos_error,
            "max_qvel_defect": max_node_qvel_error,
        },
        "gate_pass": gate_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"),
        initial_qpos=stable["initial_qpos"],
        initial_qvel=stable["initial_qvel"],
        base_target=base,
        feedback_parameters=parameters,
        desired_root=desired_root,
        desired_rotation_xyzw=stable["desired_rotation_xyzw"],
        pulse_actuators=pulse_actuators,
        pulse_target_offsets=pulse,
        ctrl_physics=controls,
        shooting_node_physics_steps=node_steps,
        shooting_node_qpos=pre_step_qpos[node_steps[:-1]],
        shooting_node_qvel=pre_step_qvel[node_steps[:-1]],
        impulse_groups=np.asarray(IMPULSE_GROUPS),
        **arrays,
    )
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def run_coupled_search(output: Path, *, maxfev: int) -> dict[str, Any]:
    """Bounded hybrid phase shooting around the clean unilateral brace.

    This is the serious sequential-manipulation hypothesis: support exchange,
    unilateral loading, delayed second-hand acquisition, and recovery are
    parameterized and evaluated in one uninterrupted start-0 rollout.
    """
    engine, generator = build(0.55)
    model = engine.model
    brace = np.load(LOG / "braced_bilateral_transition_v2.npz")
    stable = np.load(LOG / "upright_balance_vertical_v4.npz")
    baseline_controls = brace["ctrl_physics"]
    base = stable["base_target"].copy()
    feedback_parameters = stable["parameters"].copy()
    desired_root = stable["desired_root"].copy()
    desired_rotation = Rotation.from_quat(stable["desired_rotation_xyzw"])
    joint_names = np.asarray(engine.joint_names)[engine.actuator_joint_indexes]
    name_to_actuator = {name: index for index, name in enumerate(joint_names)}
    tokens = {
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
            [
                index
                for index, name in enumerate(joint_names)
                if any(token in name for token in values)
            ],
            dtype=int,
        )
        for key, values in tokens.items()
    }
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
        ]
    )
    right_arm = np.asarray(
        [
            name_to_actuator[name]
            for name in (
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            )
        ]
    )
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
        ]
    )
    support_pulse = np.asarray([0.0, 0.006, 0.040, -0.022, -0.004])
    recovery_groups = (
        feedback_groups["hip_pitch"],
        feedback_groups["knee"],
        feedback_groups["ankle_pitch"],
        feedback_groups["waist_pitch"],
        np.r_[feedback_groups["hip_roll"], feedback_groups["ankle_roll"]],
    )

    def feedback_correction(data: mujoco.MjData) -> np.ndarray:
        quaternion = data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
        rotation_error = (
            desired_rotation.inv() * _wxyz_to_rotation(quaternion)
        ).as_rotvec()
        root = data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
        velocity = data.qvel[engine.robot_dof : engine.robot_dof + 6]
        sagittal = np.asarray(
            [
                root[1] - desired_root[1],
                velocity[1],
                rotation_error[0],
                velocity[3],
            ]
        )
        lateral = np.asarray(
            [
                root[0] - desired_root[0],
                velocity[0],
                rotation_error[1],
                velocity[4],
            ]
        )
        gains = feedback_parameters.reshape(-1, 4)
        delta = np.zeros(model.nu)
        row = 0
        for key in ("hip_pitch", "knee", "ankle_pitch", "waist_pitch"):
            delta[feedback_groups[key]] += float(np.dot(gains[row], sagittal))
            row += 1
        for key in ("hip_roll", "ankle_roll", "waist_roll"):
            delta[feedback_groups[key]] += float(np.dot(gains[row], lateral))
            row += 1
        return np.clip(delta, -0.28, 0.28)

    object_settled = brace["qpos"][100, engine.object_qpos : engine.object_qpos + 7]
    object_position0 = object_settled[:3].copy()
    object_rotation0 = object_settled[3:7].copy()
    goal_position = engine.reference_object_pos[-1].copy()
    goal_rotation = engine.reference_object_quat[-1].copy()
    goalward = np.r_[goal_position[:2] - object_position0[:2], 0.0]
    goalward /= max(float(np.linalg.norm(goalward)), 1.0e-12)
    lateral_axis = np.asarray([-goalward[1], goalward[0], 0.0])

    def arm_directions(frame: int, geom: int, actuators: np.ndarray, axes: np.ndarray):
        data = mujoco.MjData(model)
        data.qpos[:] = brace["qpos"][frame]
        data.qvel[:] = brace["qvel"][frame]
        mujoco.mj_forward(model, data)
        jacobian = np.zeros((3, model.nv))
        rotational = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jacobian, rotational, int(geom))
        dofs = engine.actuator_qpos[actuators] - 1
        reduced = jacobian[:, dofs]
        inverse = reduced.T @ np.linalg.inv(
            reduced @ reduced.T + 2.5e-3 * np.eye(3)
        )
        return np.asarray([inverse @ axis for axis in axes]).T

    left_axes = np.asarray([goalward, [0.0, 0.0, 1.0], lateral_axis])
    left_directions = arm_directions(
        164, int(engine.hand_geoms[0]), left_arm, left_axes
    )
    right_data = mujoco.MjData(model)
    right_data.qpos[:] = brace["qpos"][132]
    right_data.qvel[:] = brace["qvel"][132]
    mujoco.mj_forward(model, right_data)
    right_to_object = (
        object_position0 - right_data.geom_xpos[int(engine.hand_geoms[1])]
    )
    right_to_object /= max(float(np.linalg.norm(right_to_object)), 1.0e-12)
    right_axes = np.asarray(
        [right_to_object, [0.0, 0.0, 1.0], lateral_axis]
    )
    right_directions = arm_directions(
        132, int(engine.hand_geoms[1]), right_arm, right_axes
    )
    right_hold_control = brace["ctrl"][132, right_arm].copy()

    def envelope(frame: float, begin: float, loaded: float, release: float, end: float) -> float:
        if frame < begin or frame >= end:
            return 0.0
        if frame < loaded:
            x = (frame - begin) / max(loaded - begin, 1.0e-9)
            return x * x * (3.0 - 2.0 * x)
        if frame <= release:
            return 1.0
        x = (frame - release) / max(end - release, 1.0e-9)
        x = x * x * (3.0 - 2.0 * x)
        return 1.0 - x

    evaluation_count = 0
    best_rank: tuple[Any, ...] | None = None
    best_parameters = None
    best_metrics = None

    def evaluate(parameters: np.ndarray, record: bool = False):
        nonlocal evaluation_count, best_rank, best_parameters, best_metrics
        evaluation_count += 1
        data = mujoco.MjData(model)
        data.qpos[:] = brace["initial_qpos"]
        data.qvel[:] = brace["initial_qvel"]
        mujoco.mj_forward(model, data)
        cumulative = {key: 0.0 for key in IMPULSE_GROUPS}
        first_invalid = None
        invalid_reason = None
        reference_hand_contact = []
        reference_foot_contact = []
        object_positions = []
        object_orientations = []
        root_heights = []
        torso_tilts = []
        controls = []
        qposes = []
        qvels = []
        hand_frames = 0
        bilateral_frames = 0
        single_support_steps = 0
        for step in range(300 * engine.config.substeps):
            frame = step / engine.config.substeps
            if step < 197 * engine.config.substeps:
                held = baseline_controls[min(step, len(baseline_controls) - 1)].copy()
            else:
                held = base + feedback_correction(data)
            support_amount = envelope(frame, 146.0, 151.0, 166.0, 174.0)
            held[left_leg] += support_amount * parameters[6] * support_pulse
            left_amount = envelope(frame, 158.0, 168.0, 185.0, 202.0)
            right_amount = envelope(frame, 127.0, 134.0, 186.0, 204.0)
            held[left_arm] += left_amount * (left_directions @ parameters[:3])
            held[right_arm] += right_amount * parameters[3] * (
                right_hold_control - held[right_arm]
            )
            held[right_arm] += right_amount * (
                right_directions[:, 1:] @ parameters[4:6]
            )
            recovery_amount = envelope(frame, 184.0, 199.0, 260.0, 286.0)
            for group, value in zip(
                recovery_groups, parameters[7:12], strict=True
            ):
                held[group] += recovery_amount * value
            held = np.clip(
                held,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            data.ctrl[:] = held
            controls.append(held.copy())
            mujoco.mj_step(model, data)
            impulses, _ = _contact_impulses(model, data, engine)
            for key, value in impulses.items():
                cumulative[key] += value
            physical = generator._physical(data)
            single_support_steps += int(np.sum(physical["foot_contact"]) == 1)
            strict_forbidden = sum(
                cumulative[key]
                for key in (
                    "wrist",
                    "feet",
                    "legs",
                    "torso_pelvis",
                    "other_robot",
                )
            )
            reason = generator._invalid_reason(data, physical)
            if reason is None and strict_forbidden > 0.10:
                reason = "strict_non_hand_object_impulse"
            if reason is not None and first_invalid is None:
                first_invalid = step
                invalid_reason = reason
            if step % engine.config.substeps == engine.config.substeps - 1:
                hand = physical["hand_contact"].copy()
                reference_hand_contact.append(hand)
                reference_foot_contact.append(physical["foot_contact"].copy())
                hand_frames += int(np.any(hand))
                bilateral_frames += int(np.all(hand))
                object_positions.append(
                    data.qpos[
                        engine.object_qpos : engine.object_qpos + 3
                    ].copy()
                )
                object_orientations.append(
                    data.qpos[
                        engine.object_qpos + 3 : engine.object_qpos + 7
                    ].copy()
                )
                root_heights.append(float(data.qpos[engine.robot_qpos + 2]))
                torso_tilts.append(float(physical["tilt"]))
                if record:
                    qposes.append(data.qpos.copy())
                    qvels.append(data.qvel.copy())
            if reason is not None:
                break
        object_positions_array = np.asarray(object_positions)
        object_orientations_array = np.asarray(object_orientations)
        hand_contact_array = np.asarray(reference_hand_contact)
        foot_contact_array = np.asarray(reference_foot_contact)
        task_slice = slice(100, None)
        lifts = object_positions_array[task_slice, 2] - object_position0[2]
        goalward_displacement = (
            object_positions_array[task_slice] - object_position0
        ) @ goalward
        orientation_change = np.asarray(
            [
                _orientation_error(object_rotation0, quaternion)
                for quaternion in object_orientations_array[task_slice]
            ]
        )
        task_hand_contact = hand_contact_array[task_slice]
        supported = (lifts > 0.05) & np.any(task_hand_contact, axis=1)
        longest_supported = 0
        run = 0
        lifted_path = 0.0
        for index, active in enumerate(supported):
            run = run + 1 if active else 0
            longest_supported = max(longest_supported, run)
            if active and index > 0 and np.any(task_hand_contact[index - 1]):
                lifted_path += float(
                    np.linalg.norm(
                        object_positions_array[index + 100]
                        - object_positions_array[index + 99]
                    )
                )
        final_position_error = float(
            np.linalg.norm(object_positions_array[-1] - goal_position)
        )
        final_orientation_error = _orientation_error(
            object_orientations_array[-1], goal_rotation
        )
        valid_frames = (
            300.0
            if first_invalid is None
            else first_invalid / engine.config.substeps
        )
        strict_forbidden = sum(
            cumulative[key]
            for key in (
                "wrist",
                "feet",
                "legs",
                "torso_pelvis",
                "other_robot",
            )
        )
        final_root_velocity = data.qvel[
            engine.robot_dof : engine.robot_dof + 6
        ].copy()
        recovered = bool(
            first_invalid is None
            and root_heights[-1] > 0.55
            and np.rad2deg(torso_tilts[-1]) < 40.0
            and np.linalg.norm(final_root_velocity[:3]) < 0.30
            and np.all(foot_contact_array[-1])
        )
        metrics = {
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": None if first_invalid is None else valid_frames,
            "invalid_reason": invalid_reason,
            "hand_contact_frames": hand_frames,
            "bilateral_hand_contact_frames": bilateral_frames,
            "physics_single_support_steps": single_support_steps,
            "maximum_object_lift_m": float(np.max(lifts)),
            "maximum_goalward_displacement_m": float(
                np.max(goalward_displacement)
            ),
            "maximum_object_reorientation_rad": float(np.max(orientation_change)),
            "sustained_hand_supported_lift_frames": longest_supported,
            "lifted_hand_path_m": lifted_path,
            "final_position_error_m": final_position_error,
            "final_orientation_error_rad": final_orientation_error,
            "final_root_height_m": root_heights[-1],
            "final_torso_tilt_deg": float(np.rad2deg(torso_tilts[-1])),
            "final_root_speed_m_s": float(np.linalg.norm(final_root_velocity[:3])),
            "final_foot_contact": foot_contact_array[-1].tolist(),
            "recovered": recovered,
            "object_impulse_ns": cumulative,
            "strict_forbidden_impulse_ns": strict_forbidden,
        }
        # Lexicographic tuple is the source of truth.  The scalar only guides
        # the bounded derivative-free trust region within that hierarchy.
        rank = (
            strict_forbidden > 0.10,
            first_invalid is not None,
            -valid_frames,
            hand_frames < 10,
            -hand_frames,
            -float(np.max(orientation_change)),
            -float(np.max(goalward_displacement)),
            -bilateral_frames,
            final_position_error,
            final_orientation_error,
        )
        cost = (
            1.0e9 * float(strict_forbidden > 0.10)
            + 1.0e8 * float(first_invalid is not None)
            + 1.0e5 * (300.0 - valid_frames)
            + 2.0e7 * max(0, 10 - hand_frames)
            - 2.0e5 * hand_frames
            - 5.0e6 * float(np.max(orientation_change))
            - 3.0e6 * float(np.max(goalward_displacement))
            - 1.0e5 * bilateral_frames
            + 2.0e4 * final_position_error
            + 1.0e4 * final_orientation_error
            + 2.0e3 * float(np.dot(parameters, parameters))
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_parameters = np.asarray(parameters).copy()
            best_metrics = metrics
            print(
                "PROMOTE",
                evaluation_count,
                json.dumps({"parameters": parameters.tolist(), **metrics}),
                flush=True,
            )
        record_arrays = None
        if record:
            record_arrays = {
                "qpos": np.asarray(qposes),
                "qvel": np.asarray(qvels),
                "ctrl_physics": np.asarray(controls),
                "hand_contact": hand_contact_array,
                "foot_contact": foot_contact_array,
                "object_position": object_positions_array,
                "object_orientation": object_orientations_array,
                "root_height": np.asarray(root_heights),
                "torso_tilt": np.asarray(torso_tilts),
            }
        return float(cost), metrics, record_arrays

    lower = np.asarray(
        [-0.020, -0.040, -0.040, 0.000, -0.040, -0.050, -1.0]
        + [-0.16] * 5
    )
    upper = np.asarray(
        [0.100, 0.040, 0.040, 1.500, 0.040, 0.050, 1.0]
        + [0.16] * 5
    )
    seeds = [
        np.zeros(12),
        np.asarray([0.040, -0.020, 0.0, 1.000, 0.0, 0.0, 0.5] + [0.0] * 5),
        np.asarray([0.080, -0.030, 0.0, 1.250, 0.0, 0.0, -0.5] + [0.0] * 5),
    ]
    midpoint = 0.5 * (lower + upper)
    halfspan = 0.5 * (upper - lower)
    started = time.perf_counter()
    solver_runs = []
    seed_probe_metrics = []
    for seed_index, seed in enumerate(seeds):
        seed_normalized = np.clip((seed - midpoint) / halfspan, -1.0, 1.0)
        _, seed_metrics, _ = evaluate(seed)
        seed_probe_metrics.append(
            {
                "seed_index": seed_index,
                "parameters": seed.tolist(),
                "metrics": seed_metrics,
            }
        )
        result = minimize(
            lambda normalized: evaluate(midpoint + halfspan * normalized)[0],
            seed_normalized,
            method="COBYQA",
            bounds=[(-1.0, 1.0)] * len(seed),
            options={
                "maxfev": maxfev,
                "initial_tr_radius": 0.35,
                "final_tr_radius": 0.03,
                "disp": True,
            },
        )
        physical_result = midpoint + halfspan * result.x
        solver_runs.append(
            {
                "seed_index": seed_index,
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "fun": float(result.fun),
                "x": physical_result.tolist(),
            }
        )
        np.savez_compressed(
            output.with_name(output.stem + "_checkpoint.npz"),
            best_parameters=best_parameters,
            best_metrics_json=np.asarray(json.dumps(best_metrics)),
            solver_runs_json=np.asarray(json.dumps(solver_runs)),
            evaluation_count=evaluation_count,
        )
    assert best_parameters is not None and best_metrics is not None
    _, final_metrics, arrays = evaluate(best_parameters, record=True)
    assert arrays is not None
    # Independent exact replay of every promoted physics-rate target.
    replay = mujoco.MjData(model)
    replay.qpos[:] = brace["initial_qpos"]
    replay.qvel[:] = brace["initial_qvel"]
    mujoco.mj_forward(model, replay)
    replay_qpos = []
    replay_qvel = []
    for step, held in enumerate(arrays["ctrl_physics"]):
        replay.ctrl[:] = held
        mujoco.mj_step(model, replay)
        if step % engine.config.substeps == engine.config.substeps - 1:
            replay_qpos.append(replay.qpos.copy())
            replay_qvel.append(replay.qvel.copy())
    replay_qpos = np.asarray(replay_qpos)
    replay_qvel = np.asarray(replay_qvel)
    replay_error = {
        "physics_control_samples": len(arrays["ctrl_physics"]),
        "max_qpos_error": float(np.max(np.abs(replay_qpos - arrays["qpos"]))),
        "max_qvel_error": float(np.max(np.abs(replay_qvel - arrays["qvel"]))),
    }
    report = {
        "method": "bounded hybrid condensed phase shooting: preserve right-hand brace through support transfer plus left-hand join/load plus recovery",
        "solver": "SciPy COBYQA bounded trust region over 12 phase variables",
        "dimensions": 12,
        "phase_structure": [
            "stable/braced approach",
            "preserve first right-hand brace",
            "left-foot unload overlapping left-hand acquisition",
            "left-hand goalward/upward/lateral load",
            "whole-body recovery bias",
        ],
        "evaluation_count": evaluation_count,
        "maxfev_per_seed": maxfev,
        "runtime_s": time.perf_counter() - started,
        "seed_probe_metrics": seed_probe_metrics,
        "solver_runs": solver_runs,
        "best_parameters": best_parameters.tolist(),
        "best": final_metrics,
        "replay": replay_error,
        "success": bool(
            final_metrics["full_horizon_valid"]
            and final_metrics["sustained_hand_supported_lift_frames"] >= 10
            and final_metrics["maximum_object_lift_m"] > 0.05
            and final_metrics["lifted_hand_path_m"] > 0.03
            and final_metrics["strict_forbidden_impulse_ns"] <= 0.10
            and final_metrics["final_position_error_m"] < 0.15
            and final_metrics["final_orientation_error_rad"] < 0.35
            and final_metrics["recovered"]
            and replay_error["max_qpos_error"] == 0.0
            and replay_error["max_qvel_error"] == 0.0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"),
        initial_qpos=brace["initial_qpos"],
        initial_qvel=brace["initial_qvel"],
        parameters=best_parameters,
        left_arm_actuators=left_arm,
        right_arm_actuators=right_arm,
        left_arm_directions=left_directions,
        right_arm_directions=right_directions,
        impulse_groups=np.asarray(IMPULSE_GROUPS),
        **arrays,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL", json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument(
        "--output",
        type=Path,
        default=LOG / "whole_body_contact_trajopt_sanity_v1.json",
    )
    support = subparsers.add_parser("support-gate")
    support.add_argument(
        "--output",
        type=Path,
        default=LOG / "whole_body_support_transition_v1",
    )
    support.add_argument("--max-nfev", type=int, default=10)
    support.add_argument("--interval-frames", type=int, default=6)
    unload = subparsers.add_parser("unload-gate")
    unload.add_argument(
        "--output",
        type=Path,
        default=LOG / "whole_body_support_unload_v1",
    )
    coupled = subparsers.add_parser("coupled-search")
    coupled.add_argument(
        "--output",
        type=Path,
        default=LOG / "whole_body_sequential_coupled_v1",
    )
    coupled.add_argument("--maxfev", type=int, default=80)
    args = parser.parse_args()
    if args.command == "audit":
        report = run_audit(args.output)
        if not report["gate_pass"]:
            sys.exit(2)
    elif args.command == "support-gate":
        report = run_support_gate(
            args.output,
            max_nfev=args.max_nfev,
            interval_frames=args.interval_frames,
        )
        if not report["exact_replay_gate_pass"]:
            sys.exit(3)
    elif args.command == "unload-gate":
        report = run_unload_gate(args.output)
        if not report["gate_pass"]:
            sys.exit(4)
    elif args.command == "coupled-search":
        report = run_coupled_search(args.output, maxfev=args.maxfev)
        if report["success"]:
            print("START-0 PHYSICAL SUCCESS", flush=True)


if __name__ == "__main__":
    main()
