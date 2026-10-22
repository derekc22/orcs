#!/usr/bin/env python3
"""Bounded configuration-space reach/support feasibility study for start 0.

This deliberately does not optimize a task trajectory.  It constructs broad
families of double-support postures around the settled largebox, solves arm IK
against the *actual* MuJoCo palm/object collision distance, and only runs a
short physical recovery probe for configurations which pass every geometric
gate.
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
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from probe_flat_hand_grasp import build


ROOT = Path(__file__).resolve().parents[1]


def quat_rotation(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[q[1:], q[0]])


def rotation_quat(rotation: Rotation) -> np.ndarray:
    q = rotation.as_quat()
    return np.r_[q[3], q[:3]]


def geom_name(model: mujoco.MjModel, geom: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"geom_{geom}"


def pair_distance(
    model: mujoco.MjModel, data: mujoco.MjData, first: int, second: int
) -> float:
    return float(
        mujoco.mj_geomDistance(
            model, data, int(first), int(second), 2.0, np.zeros(6, dtype=np.float64)
        )
    )


def minimum_pair_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: list[int] | set[int] | np.ndarray,
    second: list[int] | set[int] | np.ndarray,
    *,
    skip_same: bool = True,
) -> tuple[float, tuple[int, int] | None]:
    best = 2.0
    best_pair = None
    for a in first:
        for b in second:
            if skip_same and int(a) == int(b):
                continue
            distance = pair_distance(model, data, int(a), int(b))
            if distance < best:
                best = distance
                best_pair = (int(a), int(b))
    return best, best_pair


def support_margin(
    model: mujoco.MjModel, data: mujoco.MjData, engine: Any
) -> tuple[float | None, list[list[float]], list[float]]:
    points: list[np.ndarray] = []
    for contact in data.contact:
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in engine.terrain_geoms and second not in engine.terrain_geoms:
            continue
        other = second if first in engine.terrain_geoms else first
        if int(model.geom_bodyid[other]) in engine.foot_bodies:
            points.append(np.asarray(contact.pos[:2], dtype=np.float64).copy())
    masses = model.body_mass[list(engine.robot_bodies)]
    com = np.average(data.xipos[list(engine.robot_bodies)], axis=0, weights=masses)
    if len(points) < 3:
        return None, [p.tolist() for p in points], com[:2].tolist()
    points_array = np.unique(np.round(np.asarray(points), decimals=9), axis=0)
    if len(points_array) < 3:
        return None, points_array.tolist(), com[:2].tolist()
    try:
        hull = ConvexHull(points_array)
    except Exception:
        return None, points_array.tolist(), com[:2].tolist()
    equations = hull.equations
    signed = -(
        equations[:, :2] @ com[:2] + equations[:, 2]
    ) / np.linalg.norm(equations[:, :2], axis=1)
    return float(np.min(signed)), points_array[hull.vertices].tolist(), com[:2].tolist()


def enabled_robot_geoms(model: mujoco.MjModel, engine: Any) -> list[int]:
    return [
        geom
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in engine.robot_bodies
        and (int(model.geom_contype[geom]) != 0 or int(model.geom_conaffinity[geom]) != 0)
    ]


def collision_compatible(model: mujoco.MjModel, first: int, second: int) -> bool:
    if not (
        (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
        or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
    ):
        return False
    body_a, body_b = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
    if body_a == body_b:
        return False
    if int(model.body_parentid[body_a]) == body_b or int(model.body_parentid[body_b]) == body_a:
        return False
    return True


def state_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    engine: Any,
    generator: Any,
    stable_qpos: np.ndarray,
    stable_feet: np.ndarray,
) -> dict[str, Any]:
    physical = generator._physical(data)
    hands = list(map(int, engine.hand_geoms))
    objects = list(map(int, engine.object_geoms))
    robot_geoms = enabled_robot_geoms(model, engine)
    nonhand = [geom for geom in robot_geoms if geom not in hands]
    hand_distances = []
    hand_pairs = []
    for hand in hands:
        distance, pair = minimum_pair_distance(model, data, [hand], objects)
        hand_distances.append(distance)
        hand_pairs.append(pair)
    forbidden_object_distance, forbidden_object_pair = minimum_pair_distance(
        model, data, nonhand, objects
    )
    foot_bodies = set(map(int, engine.foot_bodies))
    forbidden_terrain = [
        geom for geom in robot_geoms if int(model.geom_bodyid[geom]) not in foot_bodies
    ]
    forbidden_terrain_distance, forbidden_terrain_pair = minimum_pair_distance(
        model, data, forbidden_terrain, list(engine.terrain_geoms)
    )
    self_distance = 2.0
    self_pair = None
    for index, first in enumerate(robot_geoms):
        for second in robot_geoms[index + 1 :]:
            if not collision_compatible(model, first, second):
                continue
            distance = pair_distance(model, data, first, second)
            if distance < self_distance:
                self_distance, self_pair = distance, (first, second)
    actual_self = []
    nonhand_contacts = []
    for contact in data.contact:
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        if pair & engine.object_geoms:
            other = second if first in engine.object_geoms else first
            if int(model.geom_bodyid[other]) in engine.robot_bodies and other not in hands:
                nonhand_contacts.append(
                    {
                        "robot_geom": geom_name(model, other),
                        "distance_m": float(contact.dist),
                    }
                )
        elif (
            int(model.geom_bodyid[first]) in engine.robot_bodies
            and int(model.geom_bodyid[second]) in engine.robot_bodies
        ):
            actual_self.append(
                {
                    "geoms": [geom_name(model, first), geom_name(model, second)],
                    "distance_m": float(contact.dist),
                }
            )
    support, polygon, com_xy = support_margin(model, data, engine)
    joint_values = data.qpos[engine.joint_qpos]
    limits = engine.joint_limits
    joint_margin = float(np.min(np.minimum(joint_values - limits[:, 0], limits[:, 1] - joint_values)))
    feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])

    def named_pair(pair: tuple[int, int] | None) -> list[str] | None:
        return None if pair is None else [geom_name(model, pair[0]), geom_name(model, pair[1])]

    return {
        "hand_collision_distance_m": hand_distances,
        "hand_closest_geom_pairs": [named_pair(pair) for pair in hand_pairs],
        "actual_hand_contact": physical["hand_contact"].tolist(),
        "foot_contact": physical["foot_contact"].tolist(),
        "support_polygon_margin_m": support,
        "support_polygon_xy_m": polygon,
        "robot_com_projection_xy_m": com_xy,
        "root_height_m": float(data.qpos[engine.robot_qpos + 2]),
        "torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
        "joint_limit_margin_rad": joint_margin,
        "closest_forbidden_body_object_distance_m": forbidden_object_distance,
        "closest_forbidden_body_object_pair": named_pair(forbidden_object_pair),
        "nonhand_object_contacts": nonhand_contacts,
        "closest_forbidden_terrain_distance_m": forbidden_terrain_distance,
        "closest_forbidden_terrain_pair": named_pair(forbidden_terrain_pair),
        "forbidden_terrain_contact": bool(physical["forbidden"]),
        "self_collision_margin_m_equivalent": self_distance,
        "closest_self_collision_pair": named_pair(self_pair),
        "actual_self_collisions": actual_self,
        "root_displacement_from_stable_m": float(
            np.linalg.norm(data.qpos[engine.robot_qpos : engine.robot_qpos + 3] - stable_qpos[engine.robot_qpos : engine.robot_qpos + 3])
        ),
        "stance_rms_displacement_from_stable_m": float(
            np.sqrt(np.mean(np.square(feet - stable_feet)))
        ),
        "root_position_m": data.qpos[engine.robot_qpos : engine.robot_qpos + 3].tolist(),
        "foot_body_positions_m": feet.tolist(),
        "palm_positions_m": data.geom_xpos[engine.hand_geoms].tolist(),
    }


def geometric_pass(metrics: dict[str, Any]) -> bool:
    support = metrics["support_polygon_margin_m"]
    return bool(
        max(metrics["hand_collision_distance_m"]) <= 0.005
        and min(metrics["hand_collision_distance_m"]) >= -0.004
        and all(metrics["foot_contact"])
        and support is not None
        and support >= 0.015
        and metrics["root_height_m"] >= 0.46
        and metrics["torso_tilt_deg"] <= 40.0
        and metrics["joint_limit_margin_rad"] >= 0.004
        and metrics["closest_forbidden_body_object_distance_m"] >= -0.001
        and not metrics["nonhand_object_contacts"]
        and metrics["closest_forbidden_terrain_distance_m"] >= -0.001
        and not metrics["forbidden_terrain_contact"]
        and not metrics["actual_self_collisions"]
    )


def recovery_probe(
    model: mujoco.MjModel,
    engine: Any,
    generator: Any,
    qpos: np.ndarray,
    base_control: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    robot_dofs = np.r_[
        np.arange(engine.robot_dof, engine.robot_dof + 6), engine.joint_dof
    ]
    seed = np.clip(base_control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    def acceleration(control: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = control
        mujoco.mj_forward(model, data)
        return np.r_[
            data.qacc[robot_dofs[:6]],
            0.12 * data.qacc[robot_dofs[6:]],
            0.001 * (control - seed),
        ]

    equilibrium = least_squares(
        acceleration,
        seed,
        bounds=(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]),
        max_nfev=100,
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
    )
    equilibrium_control = equilibrium.x.copy()
    equilibrium_acceleration_residual = float(
        np.linalg.norm(acceleration(equilibrium_control)[:-model.nu])
    )
    arms = np.setdiff1d(np.arange(model.nu), generator.balance_actuators)
    arm_qpos = engine.actuator_qpos[arms]
    arm_joint_ids = model.actuator_trnid[arms, 0].astype(int)
    lower = model.jnt_range[arm_joint_ids, 0] + 0.004
    upper = model.jnt_range[arm_joint_ids, 1] - 0.004
    start_arm = qpos[arm_qpos].copy()
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    start_hands = data.geom_xpos[engine.hand_geoms].copy()
    object_position = qpos[engine.object_qpos : engine.object_qpos + 3]
    goal = engine.reference_object_pos[-1]
    goalward = np.r_[goal[:2] - object_position[:2], 0.0]
    goalward /= max(np.linalg.norm(goalward), 1.0e-9)
    hand_target = start_hands + 0.012 * np.asarray([0.0, 0.0, 1.0]) + 0.012 * goalward

    def load_ik(values: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.qpos[arm_qpos] = values
        mujoco.mj_forward(model, data)
        return np.r_[
            80.0 * (data.geom_xpos[engine.hand_geoms] - hand_target).ravel(),
            0.01 * (values - start_arm),
        ]

    loaded = least_squares(
        load_ik,
        start_arm,
        bounds=(lower, upper),
        max_nfev=160,
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
    )
    load_ik(loaded.x)
    hand_ik_error = np.linalg.norm(
        data.geom_xpos[engine.hand_geoms] - hand_target, axis=1
    ).tolist()
    loaded_control = equilibrium_control.copy()
    loaded_control[arms] += loaded.x - start_arm
    loaded_control = np.clip(
        loaded_control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    )
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_object = data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    first_invalid = None
    invalid_reason = None
    nonhand_impulse = 0.0
    hand_impulse = 0.0
    minimum_support = math.inf
    two_foot_steps = 0
    maximum_lift = 0.0
    maximum_goalward = 0.0
    first_negative_support_s = None
    stage_samples: dict[str, list[dict[str, Any]]] = {
        "quiet_hold": [],
        "load_ramp": [],
        "loaded_hold": [],
        "retract_and_recover": [],
    }
    trace = {key: [] for key in ("qpos", "qvel", "ctrl", "foot_contact", "hand_contact", "root_height", "torso_tilt")}
    total_steps = 200
    for step in range(total_steps):
        # 0.25 s quiet, 0.20 s ramp, 0.20 s hold, then retract/quiet.
        if step < 50:
            amount = 0.0
        elif step < 90:
            amount = (step - 50) / 40.0
        elif step < 130:
            amount = 1.0
        elif step < 170:
            amount = 1.0 - (step - 130) / 40.0
        else:
            amount = 0.0
        data.ctrl[:] = (1.0 - amount) * equilibrium_control + amount * loaded_control
        mujoco.mj_step(model, data)
        physical = generator._physical(data)
        support, _, _ = support_margin(model, data, engine)
        if support is not None:
            minimum_support = min(minimum_support, support)
            if support < 0.0 and first_negative_support_s is None:
                first_negative_support_s = (step + 1) * model.opt.timestep
        two_foot_steps += int(bool(np.all(physical["foot_contact"])))
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair & object_geoms:
                continue
            other = int(contact.geom2) if int(contact.geom1) in object_geoms else int(contact.geom1)
            if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            impulse = float(np.linalg.norm(force[:3])) * model.opt.timestep
            if other in hand_geoms:
                hand_impulse += impulse
            else:
                nonhand_impulse += impulse
        delta = data.qpos[engine.object_qpos : engine.object_qpos + 3] - initial_object
        maximum_lift = max(maximum_lift, float(delta[2]))
        maximum_goalward = max(maximum_goalward, float(np.dot(delta, goalward)))
        reason = generator._invalid_reason(data, physical)
        if reason is None and nonhand_impulse > 0.10:
            reason = "non_hand_object_impulse"
        if reason is not None and first_invalid is None:
            first_invalid, invalid_reason = step, reason
        if step in (24, 49, 69, 89, 109, 129, 149, 169, 199):
            if step < 50:
                stage = "quiet_hold"
            elif step < 90:
                stage = "load_ramp"
            elif step < 130:
                stage = "loaded_hold"
            else:
                stage = "retract_and_recover"
            stage_samples[stage].append(
                {
                    "time_s": (step + 1) * model.opt.timestep,
                    "load_fraction": amount,
                    "valid_so_far": first_invalid is None,
                    "foot_contact": physical["foot_contact"].tolist(),
                    "support_margin_m": support,
                    "root_height_m": float(data.qpos[engine.robot_qpos + 2]),
                    "torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
                    "cumulative_hand_impulse_ns": hand_impulse,
                    "cumulative_nonhand_impulse_ns": nonhand_impulse,
                    "object_lift_m": float(delta[2]),
                    "goalward_displacement_m": float(np.dot(delta, goalward)),
                }
            )
        if step % engine.config.substeps == engine.config.substeps - 1:
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["ctrl"].append(data.ctrl.copy())
            trace["foot_contact"].append(physical["foot_contact"].copy())
            trace["hand_contact"].append(physical["hand_contact"].copy())
            trace["root_height"].append(float(data.qpos[engine.robot_qpos + 2]))
            trace["torso_tilt"].append(float(physical["tilt"]))
    physical = generator._physical(data)
    final_root_height = float(data.qpos[engine.robot_qpos + 2])
    final_torso_tilt = float(np.rad2deg(physical["tilt"]))
    final_foot_contact = physical["foot_contact"].tolist()
    result = {
        "horizon_s": total_steps * model.opt.timestep,
        "full_probe_valid": first_invalid is None,
        "first_invalid_s": None if first_invalid is None else first_invalid * model.opt.timestep,
        "invalid_reason": invalid_reason,
        "equilibrium_acceleration_residual": equilibrium_acceleration_residual,
        "requested_hand_motion_m": [0.012, 0.012],
        "hand_ik_error_m": hand_ik_error,
        "hand_object_impulse_ns": hand_impulse,
        "nonhand_object_impulse_ns": nonhand_impulse,
        "maximum_object_lift_m": maximum_lift,
        "maximum_goalward_displacement_m": maximum_goalward,
        "minimum_support_margin_m": None if not np.isfinite(minimum_support) else minimum_support,
        "first_negative_support_margin_s": first_negative_support_s,
        "two_foot_support_fraction": two_foot_steps / total_steps,
        "final_root_height_m": final_root_height,
        "final_torso_tilt_deg": final_torso_tilt,
        "final_foot_contact": final_foot_contact,
        "increasing_load_stage_samples": stage_samples,
    }
    arrays = {
        "qpos_seed": qpos,
        "qvel_seed": np.zeros(model.nv),
        "equilibrium_control": equilibrium_control,
        "loaded_control": loaded_control,
        **{key: np.asarray(value) for key, value in trace.items()},
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--postures-per-side", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8128)
    args = parser.parse_args()
    started = time.perf_counter()
    engine, generator = build(0.55)
    model = engine.model
    stable_artifact = ROOT / "logs/hand_manipulation_research/upright_balance_vertical_v4.npz"
    stable = np.load(stable_artifact)
    stable_qpos = stable["qpos"][-1].copy()
    stable_control = stable["ctrl"][-1].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = stable_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    stable_root_rotation = quat_rotation(stable_qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7])
    stable_feet = np.asarray([data.xpos[body].copy() for body in engine.foot_bodies])
    stable_foot_rotations = np.asarray([data.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies])
    object_position = stable_qpos[engine.object_qpos : engine.object_qpos + 3].copy()
    stable_outward = stable_qpos[engine.robot_qpos : engine.robot_qpos + 2] - object_position[:2]
    stable_outward /= np.linalg.norm(stable_outward)
    stable_angle = math.atan2(stable_outward[1], stable_outward[0])
    balance = np.asarray(generator.balance_actuators, dtype=int)
    arms = np.setdiff1d(np.arange(model.nu), balance)
    balance_qpos = engine.actuator_qpos[balance]
    arm_qpos = engine.actuator_qpos[arms]
    balance_joint_ids = model.actuator_trnid[balance, 0].astype(int)
    arm_joint_ids = model.actuator_trnid[arms, 0].astype(int)
    balance_lower = model.jnt_range[balance_joint_ids, 0] + 0.004
    balance_upper = model.jnt_range[balance_joint_ids, 1] - 0.004
    arm_lower = model.jnt_range[arm_joint_ids, 0] + 0.004
    arm_upper = model.jnt_range[arm_joint_ids, 1] - 0.004
    root0 = stable_qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
    balance0 = stable_qpos[balance_qpos].copy()
    arm0 = stable_qpos[arm_qpos].copy()
    rng = np.random.default_rng(args.seed)
    ranges = {
        "approach_side_deg": [0.0, 90.0, 180.0, 270.0],
        "root_horizontal_distance_m": [0.34, 0.54],
        "root_height_m": [0.48, 0.68],
        "stance_width_m": [0.24, 0.48],
        "fore_aft_stagger_m": [-0.12, 0.12],
        "root_pitch_delta_deg": [-20.0, 20.0],
        "hand_height_m": [0.20, 0.46],
        "palm_contact_target_distance_m": 0.0005,
    }
    records: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    base_controls: list[np.ndarray] = []
    posture_failures = 0
    candidate_index = 0
    for side_index, side_deg in enumerate(ranges["approach_side_deg"]):
        phi = math.radians(side_deg)
        outward = np.asarray([math.cos(phi), math.sin(phi)])
        lateral = np.asarray([-outward[1], outward[0]])
        yaw_delta = phi - stable_angle
        yaw_rotation = Rotation.from_rotvec([0.0, 0.0, yaw_delta])
        rotated_foot = np.asarray([yaw_rotation.apply(rotation) for rotation in stable_foot_rotations])
        for local_index in range(args.postures_per_side):
            # A deterministic Latin-style spread puts every variable across its
            # full range without a large Cartesian product.
            fractions = np.mod(
                (local_index + 0.5) / args.postures_per_side
                + np.asarray([0.0, 0.37, 0.61, 0.19, 0.79]),
                1.0,
            )
            if local_index == 0:
                fractions[:] = 0.0
            elif local_index == 1:
                fractions[:] = 1.0
            distance = 0.34 + 0.20 * fractions[0]
            height = 0.48 + 0.20 * fractions[1]
            width = 0.24 + 0.24 * fractions[2]
            stagger = -0.12 + 0.24 * fractions[3]
            pitch_delta = math.radians(-20.0 + 40.0 * fractions[4])
            root_xy = object_position[:2] + distance * outward
            root_target = np.r_[root_xy, height]
            foot_center = root_xy + 0.09 * outward
            foot_targets = np.asarray(
                [
                    np.r_[foot_center - 0.5 * width * lateral - 0.5 * stagger * outward, stable_feet[0, 2]],
                    np.r_[foot_center + 0.5 * width * lateral + 0.5 * stagger * outward, stable_feet[1, 2]],
                ]
            )
            pitch_axis = np.r_[lateral, 0.0]
            desired_root_rotation = Rotation.from_rotvec(pitch_delta * pitch_axis) * yaw_rotation * stable_root_rotation
            pose_data = mujoco.MjData(model)

            def set_posture(values: np.ndarray) -> None:
                pose_data.qpos[:] = stable_qpos
                pose_data.qvel[:] = 0.0
                pose_data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root0 + values[:3]
                root_rotation = Rotation.from_rotvec(values[3:6]) * stable_root_rotation
                pose_data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = rotation_quat(root_rotation)
                pose_data.qpos[balance_qpos] = values[6:]
                pose_data.qpos[engine.object_qpos : engine.object_qpos + 7] = stable_qpos[engine.object_qpos : engine.object_qpos + 7]
                mujoco.mj_forward(model, pose_data)

            def posture_residual(values: np.ndarray) -> np.ndarray:
                set_posture(values)
                feet = np.asarray([pose_data.xpos[body] for body in engine.foot_bodies])
                foot_rotations = np.asarray([pose_data.xmat[body].reshape(3, 3) for body in engine.foot_bodies])
                foot_orientation = np.concatenate(
                    [
                        Rotation.from_matrix(foot_rotations[index] @ rotated_foot[index].T).as_rotvec()
                        for index in range(2)
                    ]
                )
                root_rotation = quat_rotation(pose_data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7])
                root_error = (desired_root_rotation.inv() * root_rotation).as_rotvec()
                torso_up = pose_data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
                return np.r_[
                    90.0 * (pose_data.qpos[engine.robot_qpos : engine.robot_qpos + 3] - root_target),
                    22.0 * root_error,
                    120.0 * (feet - foot_targets).ravel(),
                    25.0 * foot_orientation,
                    10.0 * torso_up[:2],
                    0.015 * (values[6:] - balance0),
                ]

            root_seed_rotation = desired_root_rotation * stable_root_rotation.inv()
            pose_seed = np.r_[root_target - root0, root_seed_rotation.as_rotvec(), balance0]
            pose_lower = np.r_[[-0.8, -0.8, -0.24, -1.2, -1.2, -3.4], balance_lower]
            pose_upper = np.r_[[0.8, 0.8, 0.20, 1.2, 1.2, 3.4], balance_upper]
            solved_pose = least_squares(
                posture_residual,
                np.clip(pose_seed, pose_lower, pose_upper),
                bounds=(pose_lower, pose_upper),
                max_nfev=140,
                ftol=2.0e-7,
                xtol=2.0e-7,
                gtol=2.0e-7,
            )
            set_posture(solved_pose.x)
            posture_qpos = pose_data.qpos.copy()
            foot_error = float(
                np.max(
                    np.linalg.norm(
                        np.asarray([pose_data.xpos[body] for body in engine.foot_bodies]) - foot_targets,
                        axis=1,
                    )
                )
            )
            root_error = float(
                np.linalg.norm(pose_data.qpos[engine.robot_qpos : engine.robot_qpos + 3] - root_target)
            )
            if foot_error > 0.018 or root_error > 0.035:
                posture_failures += 1
            best = None
            height_values = [0.20, 0.33, 0.46]
            for hand_height in height_values:
                arm_data = mujoco.MjData(model)

                def hand_distances() -> np.ndarray:
                    result = []
                    for hand in engine.hand_geoms:
                        distance_value, _ = minimum_pair_distance(
                            model, arm_data, [int(hand)], list(engine.object_geoms)
                        )
                        result.append(distance_value)
                    return np.asarray(result)

                def arm_residual(values: np.ndarray) -> np.ndarray:
                    arm_data.qpos[:] = posture_qpos
                    arm_data.qvel[:] = 0.0
                    arm_data.qpos[arm_qpos] = values
                    mujoco.mj_forward(model, arm_data)
                    distances = hand_distances()
                    hands = arm_data.geom_xpos[engine.hand_geoms]
                    side_coordinates = (hands[:, :2] - object_position[:2]) @ lateral
                    expected_sides = np.asarray([-0.08, 0.08])
                    return np.r_[
                        120.0 * (distances - 0.0005),
                        7.0 * (hands[:, 2] - hand_height),
                        2.5 * (side_coordinates - expected_sides),
                        0.004 * (values - arm0),
                    ]

                seeds = [arm0]
                random_seed = arm_lower + rng.random(len(arms)) * (arm_upper - arm_lower)
                seeds.append(random_seed)
                for arm_seed in seeds:
                    solved_arm = least_squares(
                        arm_residual,
                        np.clip(arm_seed, arm_lower, arm_upper),
                        bounds=(arm_lower, arm_upper),
                        max_nfev=110,
                        ftol=5.0e-7,
                        xtol=5.0e-7,
                        gtol=5.0e-7,
                    )
                    arm_residual(solved_arm.x)
                    distances = hand_distances()
                    score = float(np.max(np.abs(distances - 0.0005)))
                    if best is None or score < best[0]:
                        best = (
                            score,
                            arm_data.qpos.copy(),
                            int(solved_arm.nfev),
                            hand_height,
                        )
            assert best is not None
            data.qpos[:] = best[1]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            metrics = state_metrics(model, data, engine, generator, stable_qpos, stable_feet)
            passed = geometric_pass(metrics)
            mapped_control = stable_control.copy()
            mapped_control[balance] += data.qpos[balance_qpos] - stable_qpos[balance_qpos]
            mapped_control[arms] += data.qpos[arm_qpos] - stable_qpos[arm_qpos]
            mapped_control = np.clip(mapped_control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            record = {
                "candidate_index": candidate_index,
                "family": "direct_double_support_configuration_search",
                "approach_side_deg": side_deg,
                "requested_root_distance_m": distance,
                "requested_root_height_m": height,
                "requested_stance_width_m": width,
                "requested_stagger_m": stagger,
                "requested_root_pitch_delta_deg": math.degrees(pitch_delta),
                "selected_hand_height_m": best[3],
                "posture_solver_evaluations": int(solved_pose.nfev),
                "arm_solver_evaluations": best[2],
                "maximum_foot_target_error_m": foot_error,
                "root_target_error_m": root_error,
                "geometric_gate_pass": passed,
                **metrics,
            }
            records.append(record)
            states.append(data.qpos.copy())
            base_controls.append(mapped_control)
            candidate_index += 1
            if candidate_index % 12 == 0:
                print(
                    "PROGRESS",
                    candidate_index,
                    "best_gap",
                    min(max(r["hand_collision_distance_m"]) for r in records),
                    "passes",
                    sum(r["geometric_gate_pass"] for r in records),
                    flush=True,
                )

    def ranking(record: dict[str, Any]) -> tuple[Any, ...]:
        support = record["support_polygon_margin_m"]
        collision_violation = max(
            0.0, -record["closest_forbidden_body_object_distance_m"] - 0.001
        ) + max(0.0, -record["closest_forbidden_terrain_distance_m"] - 0.001)
        return (
            not record["geometric_gate_pass"],
            bool(record["nonhand_object_contacts"]),
            bool(record["forbidden_terrain_contact"]),
            collision_violation,
            max(record["hand_collision_distance_m"]),
            -(support if support is not None else -1.0),
            record["torso_tilt_deg"],
        )

    order = sorted(range(len(records)), key=lambda index: ranking(records[index]))
    recovery_results: list[dict[str, Any]] = []
    recovery_arrays = None
    feasible_state_index = None
    for index in order:
        if not records[index]["geometric_gate_pass"]:
            continue
        probe, arrays = recovery_probe(
            model, engine, generator, states[index], base_controls[index]
        )
        records[index]["recovery_probe"] = probe
        recovery_results.append({"candidate_index": index, **probe})
        if (
            probe["full_probe_valid"]
            and probe["nonhand_object_impulse_ns"] <= 0.10
            and probe["hand_object_impulse_ns"] > 0.01
            and probe["maximum_object_lift_m"] > 0.0002
            and probe["maximum_goalward_displacement_m"] > 0.0002
        ):
            feasible_state_index = index
            recovery_arrays = arrays
            break
        if len(recovery_results) >= 6:
            break
    feasible = feasible_state_index is not None
    baseline_probe, _ = recovery_probe(
        model, engine, generator, stable_qpos, stable_control
    )
    if feasible:
        conclusion = (
            "FEASIBLE: a collision-valid bilateral manipulation-ready state passed "
            "the short nominal-physics recovery and modest hand-load probe."
        )
    else:
        conclusion = (
            "BLOCKED (bounded evidence): under the nominal flat-hand G1 morphology, "
            "current largebox geometry, and tested start-0 task pose, broad direct "
            "configuration search found no state simultaneously satisfying bilateral "
            "collision reach, collision-free support margin, and the recovery gate. "
            "This does not establish that flat-hand G1 can never manipulate the object."
        )
    near = [records[index] for index in order[:12]]
    constraint_counts = {
        "bilateral_within_5mm": int(
            sum(max(r["hand_collision_distance_m"]) <= 0.005 for r in records)
        ),
        "actual_bilateral_contact": int(
            sum(all(r["actual_hand_contact"]) for r in records)
        ),
        "double_foot_contact": int(sum(all(r["foot_contact"]) for r in records)),
        "support_margin_at_least_15mm": int(
            sum(
                r["support_polygon_margin_m"] is not None
                and r["support_polygon_margin_m"] >= 0.015
                for r in records
            )
        ),
        "bilateral_reach_and_15mm_support": int(
            sum(
                max(r["hand_collision_distance_m"]) <= 0.005
                and r["support_polygon_margin_m"] is not None
                and r["support_polygon_margin_m"] >= 0.015
                and all(r["foot_contact"])
                for r in records
            )
        ),
        "bilateral_reach_support_and_no_forbidden_contact": int(
            sum(
                max(r["hand_collision_distance_m"]) <= 0.005
                and r["support_polygon_margin_m"] is not None
                and r["support_polygon_margin_m"] >= 0.015
                and all(r["foot_contact"])
                and not r["nonhand_object_contacts"]
                and not r["forbidden_terrain_contact"]
                and not r["actual_self_collisions"]
                for r in records
            )
        ),
        "complete_static_geometric_gate": int(
            sum(r["geometric_gate_pass"] for r in records)
        ),
        "complete_recovery_gate": int(feasible),
    }
    per_side = {}
    for side in ranges["approach_side_deg"]:
        group = [r for r in records if r["approach_side_deg"] == side]
        per_side[str(int(side))] = {
            "evaluations": len(group),
            "bilateral_within_5mm": int(
                sum(max(r["hand_collision_distance_m"]) <= 0.005 for r in group)
            ),
            "best_maximum_hand_distance_m": float(
                min(max(r["hand_collision_distance_m"]) for r in group)
            ),
            "best_support_margin_among_bilateral_reach_m": (
                None
                if not any(max(r["hand_collision_distance_m"]) <= 0.005 for r in group)
                else float(
                    max(
                        r["support_polygon_margin_m"]
                        if r["support_polygon_margin_m"] is not None
                        else -1.0
                        for r in group
                        if max(r["hand_collision_distance_m"]) <= 0.005
                    )
                )
            ),
        }
    report = {
        "date": "2026-10-20",
        "scope": "start 0; nominal flat-hand G1 and nominal MuJoCo largebox/physics",
        "method": "bounded direct configuration-space support-posture plus actual collision-distance arm IK",
        "source_stable_state": str(stable_artifact.relative_to(ROOT)),
        "search_ranges": ranges,
        "search_coverage": {
            "approach_sides": ["world +x", "world +y", "world -x", "world -y"],
            "front_side_rear_interpretation": "all four horizontal approach quadrants; palms may close on any legal object collision surface",
            "postures_per_side": args.postures_per_side,
            "hand_heights_per_posture": 3,
            "arm_initializations_per_height": 2,
            "stance_and_posture_sampling": "deterministic full-range Latin-style spread with both range corners included",
        },
        "number_of_posture_evaluations": len(records),
        "number_of_arm_ik_solves": len(records) * 3 * 2,
        "posture_target_miss_count": posture_failures,
        "geometric_gate_pass_count": int(sum(r["geometric_gate_pass"] for r in records)),
        "constraint_intersection_counts": constraint_counts,
        "per_approach_side_summary": per_side,
        "recovery_probe_count": len(recovery_results),
        "recovery_probe_results": recovery_results,
        "recovery_controller_validation_on_source_stable_state": baseline_probe,
        "best_feasible_candidate_index": feasible_state_index,
        "best_near_feasible_states": near,
        "gates": {
            "maximum_each_palm_collision_distance_m": 0.005,
            "maximum_palm_penetration_m": 0.004,
            "required_feet_in_contact": 2,
            "minimum_support_polygon_margin_m": 0.015,
            "minimum_root_height_m": 0.46,
            "maximum_torso_tilt_deg": 40.0,
            "minimum_joint_limit_margin_rad": 0.004,
            "maximum_forbidden_penetration_m": 0.001,
            "self_collision": "none",
        },
        "conclusion": conclusion,
        "claim_boundary": (
            "A negative result supports a reach-versus-support incompatibility only for "
            "the documented ranges/task pose; it is not a universal morphology impossibility proof."
        ),
        "runtime_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if feasible and recovery_arrays is not None and feasible_state_index is not None:
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            source_stable_qpos=stable_qpos,
            candidate_qpos=states[feasible_state_index],
            candidate_qvel=np.zeros(model.nv),
            mapped_control=base_controls[feasible_state_index],
            candidate_index=np.asarray(feasible_state_index),
            **recovery_arrays,
        )
    elif order:
        # Preserve the exact best near-feasible static state even for a negative
        # study so the frontier and recovery probe can be reproduced directly.
        best_index = order[0]
        np.savez_compressed(
            args.output.with_name(args.output.stem + "_near_state.npz"),
            source_stable_qpos=stable_qpos,
            candidate_qpos=states[best_index],
            candidate_qvel=np.zeros(model.nv),
            mapped_control=base_controls[best_index],
            candidate_index=np.asarray(best_index),
        )
    print("CONCLUSION", conclusion, flush=True)
    print("OUTPUT", args.output, flush=True)


if __name__ == "__main__":
    main()
