#!/usr/bin/env python3
"""Final bounded gate for a recoverable unilateral side/corner box pivot.

The contact family is selected from the useful-normal contact recorded in
upright_side_tilt_recovery_v3.  Only three upper-side/corner targets and two
approach timings are evaluated.  Every candidate is an uninterrupted start-0
MuJoCo rollout with physics-rate controls and strict contact attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_flat_hand_grasp import build, smoothstep
from search_reach_support_feasibility import (
    minimum_pair_distance,
    quat_rotation,
    rotation_quat,
    state_metrics,
    support_margin,
)


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
    "ground",
)


def contact_group(model: mujoco.MjModel, engine: Any, geom: int) -> str:
    if geom == int(engine.hand_geoms[0]):
        return "left_hand"
    if geom == int(engine.hand_geoms[1]):
        return "right_hand"
    if geom in engine.terrain_geoms:
        return "ground"
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


def contact_impulses(
    model: mujoco.MjModel, data: mujoco.MjData, engine: Any
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    impulses = {key: 0.0 for key in IMPULSE_GROUPS}
    details: list[dict[str, Any]] = []
    objects = set(map(int, engine.object_geoms))
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in objects and second not in objects:
            continue
        other = second if first in objects else first
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        impulse = float(np.linalg.norm(force[:3])) * model.opt.timestep
        if int(model.geom_bodyid[other]) in engine.robot_bodies:
            group = contact_group(model, engine, other)
        elif other in engine.terrain_geoms:
            group = "ground"
        else:
            continue
        impulses[group] += impulse
        if group in ("left_hand", "right_hand"):
            normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
            object_position = data.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ]
            arm = np.asarray(contact.pos, dtype=np.float64) - object_position
            details.append(
                {
                    "group": group,
                    "point_m": np.asarray(contact.pos).tolist(),
                    "normal": normal.tolist(),
                    "tangent_1": np.asarray(contact.frame[3:6]).tolist(),
                    "tangent_2": np.asarray(contact.frame[6:9]).tolist(),
                    "moment_arm_m": arm.tolist(),
                    "normal_moment_arm": np.cross(arm, normal).tolist(),
                    "horizontal_normal_magnitude": float(
                        np.linalg.norm(normal[:2])
                    ),
                    "distance_m": float(contact.dist),
                    "normal_force_n": float(abs(force[0])),
                }
            )
    return impulses, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=LOG / "side_corner_pivot_gate_v1"
    )
    args = parser.parse_args()
    started = time.perf_counter()

    engine, generator = build(0.55)
    model = engine.model
    substeps = engine.config.substeps
    stable = np.load(LOG / "upright_balance_vertical_v4.npz")
    side_source = np.load(LOG / "upright_side_tilt_recovery_v3.npz")
    initial_qpos = stable["initial_qpos"].copy()
    initial_qvel = stable["initial_qvel"].copy()
    # The exact pre-step convention makes frame 89 the state reached by the
    # fixed 90-frame prefix used below.  Solve and track from that state rather
    # than from the unrelated frame-299 endpoint of the stable rollout.
    stable_qpos = stable["qpos"][89].copy()
    stable_control = stable["base_target"].copy()
    feedback_parameters = stable["parameters"].copy()
    stable_root = stable_qpos[
        engine.robot_qpos : engine.robot_qpos + 3
    ].copy()
    stable_rotation = quat_rotation(
        stable_qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    )

    data = mujoco.MjData(model)
    data.qpos[:] = stable_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    stable_feet = np.asarray(
        [data.xpos[body].copy() for body in engine.foot_bodies]
    )
    stable_foot_rotations = np.asarray(
        [data.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    robot_masses = model.body_mass[list(engine.robot_bodies)]
    stable_com = np.average(
        data.xipos[list(engine.robot_bodies)], axis=0, weights=robot_masses
    )

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
    controlled = np.unique(np.r_[generator.balance_actuators, left_arm]).astype(int)
    controlled_qpos = engine.actuator_qpos[controlled]
    joint_ids = model.actuator_trnid[controlled, 0].astype(int)
    joint_lower = model.jnt_range[joint_ids, 0] + 0.004
    joint_upper = model.jnt_range[joint_ids, 1] - 0.004
    stable_joints = stable_qpos[controlled_qpos].copy()

    source_controlled = side_source["controlled_actuators"].astype(int)
    source_map = {
        int(actuator): float(value)
        for actuator, value in zip(
            source_controlled, side_source["load_pose"], strict=True
        )
    }
    source_joint_seed = np.asarray(
        [
            source_map.get(int(actuator), stable_joints[index])
            for index, actuator in enumerate(controlled)
        ]
    )
    source_static = side_source["qpos"][89].copy()
    source_static[engine.actuator_qpos[source_controlled]] = side_source["load_pose"]
    data.qpos[:] = source_static
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    source_hand_rotation = data.geom_xmat[int(engine.hand_geoms[0])].reshape(3, 3).copy()

    root_lower = np.asarray([-0.12, -0.02, -0.10, -0.35, -0.35, -0.25])
    root_upper = np.asarray([0.12, 0.16, 0.08, 0.35, 0.35, 0.25])
    lower = np.r_[root_lower, joint_lower]
    upper = np.r_[root_upper, joint_upper]

    def set_configuration(values: np.ndarray) -> None:
        data.qpos[:] = stable_qpos
        data.qvel[:] = 0.0
        data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = (
            stable_root + values[:3]
        )
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = rotation_quat(
            Rotation.from_rotvec(values[3:6]) * stable_rotation
        )
        data.qpos[controlled_qpos] = values[6:]
        mujoco.mj_forward(model, data)

    def solve_contact(target: np.ndarray) -> dict[str, Any]:
        def residual(values: np.ndarray) -> np.ndarray:
            set_configuration(values)
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            foot_rotations = np.asarray(
                [data.xmat[body].reshape(3, 3) for body in engine.foot_bodies]
            )
            foot_orientation = np.concatenate(
                [
                    Rotation.from_matrix(
                        foot_rotations[index] @ stable_foot_rotations[index].T
                    ).as_rotvec()
                    for index in range(2)
                ]
            )
            hand_rotation = data.geom_xmat[int(engine.hand_geoms[0])].reshape(3, 3)
            com = np.average(
                data.xipos[list(engine.robot_bodies)],
                axis=0,
                weights=robot_masses,
            )
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            return np.r_[
                100.0 * (data.geom_xpos[int(engine.hand_geoms[0])] - target),
                4.0 * (hand_rotation[:, 1] - source_hand_rotation[:, 1]),
                2.0 * (hand_rotation[:, 2] - source_hand_rotation[:, 2]),
                140.0 * (feet - stable_feet).ravel(),
                25.0 * foot_orientation,
                80.0 * (com[:2] - stable_com[:2]),
                5.0 * torso_up[:2],
                3.0 * values[:3],
                2.0 * values[3:6],
                0.01 * (values[6:] - stable_joints),
            ]

        seeds = [
            np.r_[np.zeros(6), source_joint_seed],
            np.r_[np.zeros(6), stable_joints],
            np.r_[[0.0, 0.08, 0.0, 0.0, 0.0, 0.0], source_joint_seed],
        ]
        best = None
        for seed in seeds:
            result = least_squares(
                residual,
                np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=300,
                ftol=2.0e-7,
                xtol=2.0e-7,
                gtol=2.0e-7,
            )
            set_configuration(result.x)
            distance, _ = minimum_pair_distance(
                model,
                data,
                [int(engine.hand_geoms[0])],
                list(engine.object_geoms),
            )
            hand_error = float(
                np.linalg.norm(data.geom_xpos[int(engine.hand_geoms[0])] - target)
            )
            score = hand_error + max(0.0, distance)
            if best is None or score < best[0]:
                best = (score, result.x.copy(), hand_error, distance, int(result.nfev))
        assert best is not None
        set_configuration(best[1])
        qpos = data.qpos.copy()
        metrics = state_metrics(
            model, data, engine, generator, stable_qpos, stable_feet
        )
        impulses, details = contact_impulses(model, data, engine)
        mapped_control = stable_control.copy()
        mapped_control[controlled] += qpos[controlled_qpos] - stable_qpos[controlled_qpos]
        mapped_control = np.clip(
            mapped_control,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        return {
            "target": target.copy(),
            "values": best[1],
            "qpos": qpos,
            "mapped_control": mapped_control,
            "hand_target_error_m": best[2],
            "hand_collision_distance_m": best[3],
            "solver_evaluations": best[4],
            "state_metrics": metrics,
            "static_contact_details": details,
            "static_impulses": impulses,
        }

    contact_specs = [
        ("upper_side_center", np.asarray([0.866, 0.500, 0.380]), np.asarray([-0.028, 0.000, 0.006])),
        ("upper_positive_corner", np.asarray([0.866, 0.580, 0.380]), np.asarray([-0.024, 0.018, 0.006])),
        ("upper_negative_corner", np.asarray([0.866, 0.440, 0.380]), np.asarray([-0.024, -0.018, 0.006])),
    ]
    contact_solutions = []
    for name, target, load_translation in contact_specs:
        solution = solve_contact(target)
        solution["name"] = name
        solution["load_translation"] = load_translation
        contact_solutions.append(solution)
        print(
            "CONTACT",
            name,
            json.dumps(
                {
                    "target": target.tolist(),
                    "hand_target_error_m": solution["hand_target_error_m"],
                    "hand_collision_distance_m": solution["hand_collision_distance_m"],
                    "support_margin_m": solution["state_metrics"]["support_polygon_margin_m"],
                    "root_height_m": solution["state_metrics"]["root_height_m"],
                    "torso_tilt_deg": solution["state_metrics"]["torso_tilt_deg"],
                    "forbidden_clearance_m": solution["state_metrics"]["closest_forbidden_body_object_distance_m"],
                    "active_self_contacts": len(solution["state_metrics"]["actual_self_collisions"]),
                    "contact_details": solution["static_contact_details"],
                }
            ),
            flush=True,
        )

    token_groups = {
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
                for index, name in enumerate(names)
                if any(token in name for token in tokens)
            ],
            dtype=int,
        )
        for key, tokens in token_groups.items()
    }

    def feedback_correction(
        rollout_data: mujoco.MjData,
        desired_root: np.ndarray,
        desired_rotation: Rotation,
    ) -> np.ndarray:
        root_rotation = quat_rotation(
            rollout_data.qpos[
                engine.robot_qpos + 3 : engine.robot_qpos + 7
            ]
        )
        rotation_error = (desired_rotation.inv() * root_rotation).as_rotvec()
        root = rollout_data.qpos[
            engine.robot_qpos : engine.robot_qpos + 3
        ]
        velocity = rollout_data.qvel[
            engine.robot_dof : engine.robot_dof + 6
        ]
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

    def arm_load_delta(solution: dict[str, Any]) -> np.ndarray:
        load_data = mujoco.MjData(model)
        load_data.qpos[:] = solution["qpos"]
        load_data.qvel[:] = 0.0
        mujoco.mj_forward(model, load_data)
        jacobian = np.zeros((3, model.nv))
        rotational = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(
            model,
            load_data,
            jacobian,
            rotational,
            int(engine.hand_geoms[0]),
        )
        dofs = engine.actuator_qpos[left_arm] - 1
        reduced = jacobian[:, dofs]
        inverse = reduced.T @ np.linalg.inv(
            reduced @ reduced.T + 2.5e-3 * np.eye(3)
        )
        return inverse @ solution["load_translation"]

    actuator_dofs = np.asarray(
        [model.jnt_dofadr[joint] for joint in model.actuator_trnid[:, 0]],
        dtype=int,
    )
    kp = model.actuator_gainprm[:, 0].copy()
    kd = -model.actuator_biasprm[:, 2].copy()
    torque_limit = np.max(np.abs(model.actuator_forcerange), axis=1)

    def inverse_dynamics_path(
        endpoint: np.ndarray, duration_frames: float
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Create nominal position targets for a supported configuration path."""
        steps = int(round(duration_frames * substeps))
        tangent = np.zeros(model.nv)
        mujoco.mj_differentiatePos(
            model, tangent, 1.0, stable_qpos, endpoint
        )
        qrefs = []
        for index in range(steps + 1):
            amount = smoothstep(index / max(steps, 1))
            qpos = stable_qpos.copy()
            mujoco.mj_integratePos(model, qpos, amount * tangent, 1.0)
            qrefs.append(qpos)
        qrefs = np.asarray(qrefs)
        dt = model.opt.timestep
        qvels = np.zeros((steps + 1, model.nv))
        for index in range(1, steps):
            mujoco.mj_differentiatePos(
                model,
                qvels[index],
                2.0 * dt,
                qrefs[index - 1],
                qrefs[index + 1],
            )
        qaccs = np.zeros_like(qvels)
        qaccs[1:-1] = (qvels[2:] - qvels[:-2]) / (2.0 * dt)
        inverse_data = mujoco.MjData(model)
        controls = []
        clipped = 0
        for index in range(1, steps + 1):
            inverse_data.qpos[:] = qrefs[index]
            # Use the contact-consistent quasi-static generalized force.  The
            # fully prescribed free-root acceleration makes inverse dynamics
            # demand unactuated constraint forces and saturates the position
            # actuators; velocity feedforward below supplies the slow path.
            inverse_data.qvel[:] = 0.0
            inverse_data.qacc[:] = 0.0
            mujoco.mj_inverse(model, inverse_data)
            requested_torque = inverse_data.qfrc_inverse[actuator_dofs]
            torque = np.clip(requested_torque, -torque_limit, torque_limit)
            raw = (
                qrefs[index, engine.actuator_qpos]
                + (torque + kd * qvels[index, actuator_dofs]) / kp
            )
            target = np.clip(
                raw,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            clipped += int(
                np.sum(
                    (np.abs(requested_torque - torque) > 1.0e-8)
                    | (np.abs(raw - target) > 1.0e-8)
                )
            )
            controls.append(target)
        return np.asarray(controls), qrefs[1:], clipped

    object_anchor = stable_qpos[
        engine.object_qpos : engine.object_qpos + 3
    ].copy()
    object_rotation0 = quat_rotation(
        stable_qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    )

    def evaluate(
        solution: dict[str, Any], approach_frames: float, load_scale: float, record: bool
    ) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, np.ndarray] | None]:
        rollout = mujoco.MjData(model)
        rollout.qpos[:] = initial_qpos
        rollout.qvel[:] = initial_qvel
        mujoco.mj_forward(model, rollout)
        contact_root = solution["qpos"][
            engine.robot_qpos : engine.robot_qpos + 3
        ].copy()
        contact_rotation = quat_rotation(
            solution["qpos"][engine.robot_qpos + 3 : engine.robot_qpos + 7]
        )
        approach_start = 90.0
        lower_end = approach_start + approach_frames
        settle_end = lower_end + 6.0
        approach_end = settle_end + 30.0
        load_end = approach_end + 10.0
        hold_end = load_end + 4.0
        withdraw_end = hold_end + 8.0
        arm_retract_end = withdraw_end + 18.0
        recover_end = arm_retract_end + approach_frames
        inverse_clip_count = 0
        contact_control = solution["mapped_control"].copy()
        precontact_control = stable_control.copy()
        precontact_control[generator.balance_actuators] = contact_control[
            generator.balance_actuators
        ]
        load_control = contact_control.copy()
        load_control[left_arm] += load_scale * arm_load_delta(solution)
        load_control = np.clip(
            load_control,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        cumulative = {key: 0.0 for key in IMPULSE_GROUPS}
        first_invalid_step = None
        invalid_reason = None
        first_hand_step = None
        first_hand_root_speed = None
        first_hand_vertical_speed = None
        best_contact_detail = None
        best_contact_force = -math.inf
        longest_contact = 0
        contact_run = 0
        recovered_run = 0
        maximum_recovered_run = 0
        minimum_support = math.inf
        maximum_rotation = 0.0
        maximum_tip = 0.0
        maximum_yaw = 0.0
        maximum_displacement = 0.0
        postpivot_index = None
        postpivot_score = -math.inf
        controls = []
        traces: dict[str, list[Any]] = {
            key: []
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "hand_contact",
                "foot_contact",
                "root_height",
                "torso_tilt",
                "support_margin",
                "object_position",
                "object_orientation",
                "object_impulse",
            )
        }
        for step in range(300 * substeps):
            frame = step / substeps
            if frame < approach_start:
                held = stable["ctrl_physics"][step].copy()
                root_target = stable_root
                rotation_target = stable_rotation
            else:
                if frame < lower_end:
                    amount = smoothstep(
                        (frame - approach_start) / max(approach_frames, 1.0)
                    )
                    held = (
                        (1.0 - amount) * stable_control
                        + amount * precontact_control
                    )
                    root_amount = amount
                elif frame < settle_end:
                    held = precontact_control.copy()
                    root_amount = 1.0
                elif frame < approach_end:
                    amount = smoothstep((frame - settle_end) / 30.0)
                    held = (
                        (1.0 - amount) * precontact_control
                        + amount * contact_control
                    )
                    root_amount = 1.0
                elif frame < load_end:
                    amount = smoothstep((frame - approach_end) / 10.0)
                    held = (1.0 - amount) * contact_control + amount * load_control
                    root_amount = 1.0
                elif frame < hold_end:
                    held = load_control.copy()
                    root_amount = 1.0
                elif frame < withdraw_end:
                    amount = smoothstep((frame - hold_end) / 8.0)
                    held = (1.0 - amount) * load_control + amount * contact_control
                    root_amount = 1.0
                elif frame < arm_retract_end:
                    amount = smoothstep((frame - withdraw_end) / 18.0)
                    held = (
                        (1.0 - amount) * contact_control
                        + amount * precontact_control
                    )
                    root_amount = 1.0
                elif frame < recover_end:
                    amount = smoothstep(
                        (frame - arm_retract_end) / max(approach_frames, 1.0)
                    )
                    held = (
                        (1.0 - amount) * precontact_control
                        + amount * stable_control
                    )
                    root_amount = 1.0 - amount
                else:
                    held = stable_control.copy()
                    root_amount = 0.0
                root_target = (
                    (1.0 - root_amount) * stable_root
                    + root_amount * contact_root
                )
                relative_rotation = (
                    stable_rotation.inv() * contact_rotation
                ).as_rotvec()
                rotation_target = stable_rotation * Rotation.from_rotvec(
                    root_amount * relative_rotation
                )
                held += feedback_correction(rollout, root_target, rotation_target)
            held = np.clip(
                held,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            rollout.ctrl[:] = held
            controls.append(held.copy())
            mujoco.mj_step(model, rollout)
            physical = generator._physical(rollout)
            impulses, contact_details = contact_impulses(model, rollout, engine)
            for key, value in impulses.items():
                cumulative[key] += value
            for detail in contact_details:
                if detail["normal_force_n"] > best_contact_force:
                    best_contact_force = detail["normal_force_n"]
                    best_contact_detail = detail
            strict_forbidden = sum(
                cumulative[key]
                for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
            )
            support, _, _ = support_margin(model, rollout, engine)
            if support is not None:
                minimum_support = min(minimum_support, support)
            root_velocity = rollout.qvel[
                engine.robot_dof : engine.robot_dof + 3
            ]
            if np.any(physical["hand_contact"]):
                if first_hand_step is None:
                    first_hand_step = step
                    first_hand_root_speed = float(np.linalg.norm(root_velocity))
                    first_hand_vertical_speed = float(root_velocity[2])
            object_position = rollout.qpos[
                engine.object_qpos : engine.object_qpos + 3
            ].copy()
            object_quaternion = rollout.qpos[
                engine.object_qpos + 3 : engine.object_qpos + 7
            ].copy()
            relative_object_rotation = (
                object_rotation0.inv() * quat_rotation(object_quaternion)
            ).as_rotvec()
            rotation_magnitude = float(np.linalg.norm(relative_object_rotation))
            tip_magnitude = float(np.linalg.norm(relative_object_rotation[:2]))
            yaw_magnitude = float(abs(relative_object_rotation[2]))
            displacement = float(np.linalg.norm(object_position[:2] - object_anchor[:2]))
            if frame >= approach_start:
                maximum_rotation = max(maximum_rotation, rotation_magnitude)
                maximum_tip = max(maximum_tip, tip_magnitude)
                maximum_yaw = max(maximum_yaw, yaw_magnitude)
                maximum_displacement = max(maximum_displacement, displacement)
            reason = generator._invalid_reason(rollout, physical)
            if reason is None and strict_forbidden > 0.10:
                reason = "strict_non_hand_object_impulse"
            if reason is not None and first_invalid_step is None:
                first_invalid_step = step
                invalid_reason = reason
            if step % substeps == substeps - 1:
                active = bool(physical["hand_contact"][0])
                contact_run = contact_run + 1 if active else 0
                longest_contact = max(longest_contact, contact_run)
                recovered = bool(
                    frame >= recover_end
                    and rollout.qpos[engine.robot_qpos + 2] >= 0.50
                    and np.rad2deg(physical["tilt"]) <= 35.0
                    and np.linalg.norm(root_velocity) <= 0.30
                    and np.all(physical["foot_contact"])
                    and strict_forbidden <= 0.10
                )
                recovered_run = recovered_run + 1 if recovered else 0
                maximum_recovered_run = max(maximum_recovered_run, recovered_run)
                if (
                    frame >= approach_end
                    and strict_forbidden <= 0.10
                    and rollout.qpos[engine.robot_qpos + 2] >= 0.50
                    and np.rad2deg(physical["tilt"]) <= 35.0
                    and np.all(physical["foot_contact"])
                ):
                    score = rotation_magnitude + 0.25 * displacement
                    if score > postpivot_score:
                        postpivot_score = score
                        postpivot_index = len(traces["qpos"])
                traces["qpos"].append(rollout.qpos.copy())
                traces["qvel"].append(rollout.qvel.copy())
                traces["ctrl"].append(held.copy())
                traces["hand_contact"].append(physical["hand_contact"].copy())
                traces["foot_contact"].append(physical["foot_contact"].copy())
                traces["root_height"].append(float(rollout.qpos[engine.robot_qpos + 2]))
                traces["torso_tilt"].append(float(physical["tilt"]))
                traces["support_margin"].append(np.nan if support is None else support)
                traces["object_position"].append(object_position)
                traces["object_orientation"].append(object_quaternion)
                traces["object_impulse"].append(
                    np.asarray([cumulative[key] for key in IMPULSE_GROUPS])
                )
            if reason is not None:
                break
        arrays = {key: np.asarray(value) for key, value in traces.items()}
        final_relative_rotation = (
            object_rotation0.inv()
            * quat_rotation(
                rollout.qpos[
                    engine.object_qpos + 3 : engine.object_qpos + 7
                ]
            )
        ).as_rotvec()
        strict_forbidden = sum(
            cumulative[key]
            for key in ("wrist", "feet", "legs", "torso_pelvis", "other_robot")
        )
        full_valid = first_invalid_step is None
        deliberate = bool(
            first_hand_root_speed is not None
            and first_hand_root_speed <= 0.25
            and abs(first_hand_vertical_speed) <= 0.10
        )
        useful_motion = bool(
            maximum_rotation >= 0.10
            and np.linalg.norm(final_relative_rotation) >= 0.08
        )
        recoverable = maximum_recovered_run >= 20
        gate_prefix = bool(
            full_valid
            and strict_forbidden <= 0.10
            and cumulative["wrist"] <= 0.10
            and deliberate
            and longest_contact >= 8
            and useful_motion
            and recoverable
            and postpivot_index is not None
        )
        metrics = {
            "contact_name": solution["name"],
            "approach_frames": approach_frames,
            "load_scale": load_scale,
            "inverse_dynamics_clip_count": inverse_clip_count,
            "full_horizon_valid": full_valid,
            "first_invalid_frame": None if full_valid else first_invalid_step / substeps,
            "invalid_reason": invalid_reason,
            "first_hand_contact_frame": None if first_hand_step is None else first_hand_step / substeps,
            "first_contact_root_speed_m_s": first_hand_root_speed,
            "first_contact_vertical_speed_m_s": first_hand_vertical_speed,
            "longest_unilateral_contact_frames": longest_contact,
            "maximum_object_reorientation_rad": maximum_rotation,
            "maximum_object_tip_rad": maximum_tip,
            "maximum_object_yaw_rad": maximum_yaw,
            "maximum_horizontal_displacement_m": maximum_displacement,
            "final_object_reorientation_rad": float(np.linalg.norm(final_relative_rotation)),
            "minimum_support_margin_m": None if not np.isfinite(minimum_support) else minimum_support,
            "maximum_recovered_run_frames": maximum_recovered_run,
            "final_root_height_m": float(rollout.qpos[engine.robot_qpos + 2]),
            "final_torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "final_root_speed_m_s": float(np.linalg.norm(root_velocity)),
            "final_foot_contact": physical["foot_contact"].tolist(),
            "object_impulse_ns": cumulative,
            "strict_forbidden_impulse_ns": strict_forbidden,
            "strongest_hand_contact": best_contact_detail,
            "postpivot_reference_frame": postpivot_index,
            "pivot_prefix_gate_pass": gate_prefix,
        }
        rank = (
            strict_forbidden > 0.10,
            not full_valid,
            not deliberate,
            longest_contact < 8,
            not recoverable,
            not useful_motion,
            -maximum_rotation,
            -float(np.linalg.norm(final_relative_rotation)),
            -maximum_displacement,
            -longest_contact,
        )
        if not record:
            return rank, metrics, None
        arrays["ctrl_physics"] = np.asarray(controls)
        arrays["postpivot_qpos"] = (
            arrays["qpos"][postpivot_index]
            if postpivot_index is not None
            else arrays["qpos"][-1]
        )
        arrays["postpivot_qvel"] = (
            arrays["qvel"][postpivot_index]
            if postpivot_index is not None
            else arrays["qvel"][-1]
        )
        return rank, metrics, arrays

    trials = []
    best = None
    best_solution = None
    for solution in contact_solutions:
        static_metrics = solution["state_metrics"]
        static_ok = bool(
            solution["hand_collision_distance_m"] <= 0.005
            and static_metrics["support_polygon_margin_m"] is not None
            and static_metrics["support_polygon_margin_m"] >= 0.015
            and static_metrics["root_height_m"] >= 0.50
            and static_metrics["torso_tilt_deg"] <= 35.0
            and static_metrics["closest_forbidden_body_object_distance_m"] >= -0.001
            and not static_metrics["nonhand_object_contacts"]
            and not static_metrics["forbidden_terrain_contact"]
            and not static_metrics["actual_self_collisions"]
        )
        if not static_ok:
            trials.append(
                {
                    "contact_name": solution["name"],
                    "static_gate_pass": False,
                    "reason": "static_contact_or_support_gate",
                }
            )
            continue
        for approach_frames in (40.0, 50.0):
            result = evaluate(solution, approach_frames, 1.0, record=False)
            trials.append({"static_gate_pass": True, **result[1]})
            print("TRIAL", json.dumps(result[1]), flush=True)
            if best is None or result[0] < best[0]:
                best = result
                best_solution = solution

    # One mechanically justified escalation: if a clean deliberate contact was
    # reached but the pivot magnitude was insufficient, deepen that same load.
    if best is not None and best_solution is not None:
        best_metrics = best[1]
        if (
            best_metrics["strict_forbidden_impulse_ns"] <= 0.10
            and best_metrics["first_contact_root_speed_m_s"] is not None
            and best_metrics["first_contact_root_speed_m_s"] <= 0.25
            and best_metrics["longest_unilateral_contact_frames"] >= 3
            and best_metrics["maximum_object_reorientation_rad"] < 0.10
        ):
            for load_scale in (1.35, 1.70):
                result = evaluate(
                    best_solution,
                    float(best_metrics["approach_frames"]),
                    load_scale,
                    record=False,
                )
                trials.append({"static_gate_pass": True, **result[1]})
                print("ESCALATION", json.dumps(result[1]), flush=True)
                if result[0] < best[0]:
                    best = result
                    best_metrics = result[1]

    if best is None or best_solution is None:
        report = {
            "method": "bounded mechanically selected unilateral side/corner pivot gate",
            "contact_candidate_count": len(contact_solutions),
            "dynamic_rollout_count": 0,
            "promotion_thresholds": {
                "maximum_first_contact_root_speed_m_s": 0.25,
                "maximum_first_contact_vertical_speed_m_s": 0.10,
                "minimum_consecutive_contact_frames": 8,
                "minimum_persistent_reorientation_rad": 0.08,
                "minimum_peak_reorientation_rad": 0.10,
                "minimum_recovery_frames": 20,
                "maximum_wrist_or_forbidden_impulse_ns": 0.10,
            },
            "static_contacts": [
                {
                    "name": solution["name"],
                    "target_m": solution["target"].tolist(),
                    "load_translation_m": solution["load_translation"].tolist(),
                    "hand_target_error_m": solution["hand_target_error_m"],
                    "hand_collision_distance_m": solution["hand_collision_distance_m"],
                    "state_metrics": solution["state_metrics"],
                    "contact_details": solution["static_contact_details"],
                }
                for solution in contact_solutions
            ],
            "trials": trials,
            "best": None,
            "pivot_gate_pass": False,
            "runtime_s": time.perf_counter() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print("PIVOT_GATE_FAIL no static candidate", flush=True)
        return

    # Re-run the promoted parameters with full traces and independently replay
    # the exact physics-rate controls.
    _, final_metrics, arrays = evaluate(
        best_solution,
        float(best[1]["approach_frames"]),
        float(best[1]["load_scale"]),
        record=True,
    )
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        stable_qpos=stable_qpos,
        contact_qpos=best_solution["qpos"],
        contact_control=best_solution["mapped_control"],
        contact_target=best_solution["target"],
        load_translation=best_solution["load_translation"],
        impulse_groups=np.asarray(IMPULSE_GROUPS),
        **arrays,
    )
    report = {
        "method": "bounded mechanically selected unilateral side/corner pivot gate",
        "contact_candidate_count": len(contact_solutions),
        "dynamic_rollout_count": sum("full_horizon_valid" in trial for trial in trials),
        "promotion_thresholds": {
            "maximum_first_contact_root_speed_m_s": 0.25,
            "maximum_first_contact_vertical_speed_m_s": 0.10,
            "minimum_consecutive_contact_frames": 8,
            "minimum_persistent_reorientation_rad": 0.08,
            "minimum_peak_reorientation_rad": 0.10,
            "minimum_recovery_frames": 20,
            "maximum_wrist_or_forbidden_impulse_ns": 0.10,
        },
        "static_contacts": [
            {
                "name": solution["name"],
                "target_m": solution["target"].tolist(),
                "load_translation_m": solution["load_translation"].tolist(),
                "hand_target_error_m": solution["hand_target_error_m"],
                "hand_collision_distance_m": solution["hand_collision_distance_m"],
                "state_metrics": solution["state_metrics"],
                "contact_details": solution["static_contact_details"],
            }
            for solution in contact_solutions
        ],
        "trials": trials,
        "best": final_metrics,
        "replay": replay_result,
        "pivot_gate_pass": bool(final_metrics["pivot_prefix_gate_pass"]),
        "runtime_s": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(final_metrics), flush=True)
    print(
        "PIVOT_GATE_PASS" if final_metrics["pivot_prefix_gate_pass"] else "PIVOT_GATE_FAIL",
        flush=True,
    )


if __name__ == "__main__":
    main()
