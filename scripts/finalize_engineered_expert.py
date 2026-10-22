#!/usr/bin/env python3
"""Freeze, independently replay, and validate the engineered physical expert."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_locomomanipulation"
MODEL_PATH = OUT / "engineered_locomomanipulation_model.mjb"
EXPERT_MODEL_PATH = OUT / "engineered_locomomanipulation_expert_start0.mjb"
SOURCE_PATH = OUT / "engineered_release_retraction_v1.npz"
EXPERT_PATH = OUT / "engineered_locomomanipulation_expert_start0.npz"
REPORT_PATH = OUT / "engineered_locomomanipulation_expert_start0.json"
DT = 0.005
SUBSTEPS = 4
EXPERT_FRAMES = 161


def cube_orientation_error(rotation: Rotation, goal: Rotation) -> float:
    """Minimum orientation error over the cube's 24 exact rotations."""
    errors = []
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                equivalent = rotation * Rotation.from_matrix(matrix)
                errors.append((equivalent.inv() * goal).magnitude())
    return float(min(errors))


def replay(model: mujoco.MjModel, initial_qpos: np.ndarray, initial_qvel: np.ndarray, controls: np.ndarray, goal: np.ndarray, goal_quat: np.ndarray, record: bool):
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    robot_dof = int(model.jnt_dofadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    foot_bodies = {
        side: {
            i for i in range(model.nbody)
            if (name := (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "")).startswith("robot/")
            and side in name and any(token in name for token in ("foot", "ankle"))
        }
        for side in ("left", "right")
    }
    foot_markers = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{side}_ankle_roll_link") for side in ("left", "right")]
    hand_geoms = [
        {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_front_guard_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_trailing_guard_collision"),
        }
        for side in ("left", "right")
    ]
    object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == object_body}
    impulse_names = ("left_end_effector", "right_end_effector", "feet", "legs", "pelvis_torso", "other_robot", "environment")
    impulses = {name: 0.0 for name in impulse_names}
    trace = {key: [] for key in ("qpos", "qvel", "ctrl", "object_pos", "object_quat", "hand_contact", "foot_contact", "root_height", "torso_tilt", "impulses")}
    first_invalid = None
    initial_object = data.qpos[object_qpos:object_qpos + 3].copy()
    initial_feet = None
    last_feet = None
    max_foot_height = np.zeros(2)
    for step, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if not pair & object_geoms:
                continue
            other = next((geom for geom in pair if geom not in object_geoms), None)
            if other is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, force)
            impulse = float(np.linalg.norm(force[:3]) * DT)
            if other in hand_geoms[0]:
                group = "left_end_effector"
            elif other in hand_geoms[1]:
                group = "right_end_effector"
            else:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
                if body_name.startswith("robot/"):
                    if any(token in body_name for token in ("foot", "ankle")):
                        group = "feet"
                    elif any(token in body_name for token in ("knee", "shin", "hip", "thigh")):
                        group = "legs"
                    elif any(token in body_name for token in ("pelvis", "torso", "waist")):
                        group = "pelvis_torso"
                    else:
                        group = "other_robot"
                else:
                    group = "environment"
            impulses[group] += impulse
        if step % SUBSTEPS != SUBSTEPS - 1:
            continue
        frame = step // SUBSTEPS
        root_height = float(data.qpos[robot_qpos + 2])
        up = data.xmat[torso].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
        if first_invalid is None and (root_height < 0.45 or tilt > 50.0):
            first_invalid = [frame, "root_height" if root_height < 0.45 else "torso_tilt"]
        hand_contact = np.zeros(2, dtype=bool)
        foot_contact = np.zeros(2, dtype=bool)
        for ci in range(data.ncon):
            pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
            for side, geoms in enumerate(hand_geoms):
                hand_contact[side] |= bool(pair & geoms) and bool(pair & object_geoms)
            for side, name in enumerate(("left", "right")):
                foot_contact[side] |= any(int(model.geom_bodyid[g]) in foot_bodies[name] for g in pair)
        feet = np.asarray([data.xpos[body].copy() for body in foot_markers])
        if initial_feet is None:
            initial_feet = feet.copy()
        last_feet = feet
        max_foot_height = np.maximum(max_foot_height, feet[:, 2])
        if record:
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["ctrl"].append(ctrl.copy())
            trace["object_pos"].append(data.qpos[object_qpos:object_qpos + 3].copy())
            trace["object_quat"].append(data.qpos[object_qpos + 3:object_qpos + 7].copy())
            trace["hand_contact"].append(hand_contact)
            trace["foot_contact"].append(foot_contact)
            trace["root_height"].append(root_height)
            trace["torso_tilt"].append(tilt)
            trace["impulses"].append([impulses[name] for name in impulse_names])
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    quat = data.qpos[object_qpos + 3:object_qpos + 7]
    final_rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
    goal_rotation = Rotation.from_quat(np.r_[goal_quat[1:], goal_quat[0]])
    raw_orientation_error = float((final_rotation.inv() * goal_rotation).magnitude())
    metrics = {
        "full_horizon_valid": first_invalid is None,
        "first_invalid": first_invalid,
        "final_position_error_m": float(np.linalg.norm(data.qpos[object_qpos:object_qpos + 3] - goal)),
        "final_orientation_error_rad": cube_orientation_error(final_rotation, goal_rotation),
        "final_raw_quaternion_error_rad": raw_orientation_error,
        "orientation_metric": "minimum over the 24 exact rotational symmetries of the cubic payload",
        "final_object_speed": float(np.linalg.norm(data.qvel[object_dof:object_dof + 6])),
        "final_root_height_m": float(data.qpos[robot_qpos + 2]),
        "final_root_velocity_6d": data.qvel[robot_dof:robot_dof + 6].tolist(),
        "final_root_speed_6d": float(np.linalg.norm(data.qvel[robot_dof:robot_dof + 6])),
        "object_impulse_ns": impulses,
        "non_end_effector_robot_object_impulse_ns": float(sum(impulses[name] for name in ("feet", "legs", "pelvis_torso", "other_robot"))),
        "foot_translation_m": np.linalg.norm(last_feet[:, :2] - initial_feet[:, :2], axis=1).tolist(),
        "maximum_foot_center_height_m": max_foot_height.tolist(),
    }
    if record:
        lift = arrays["object_pos"][:, 2] - initial_object[2]
        supported = arrays["hand_contact"].any(axis=1) & (lift > 0.05)
        run = longest = 0
        for active in supported:
            run = run + 1 if active else 0
            longest = max(longest, run)
        indexes = np.flatnonzero(supported)
        carry = float(np.linalg.norm(arrays["object_pos"][indexes[-1], :2] - arrays["object_pos"][indexes[0], :2])) if len(indexes) else 0.0
        metrics.update({
            "maximum_lift_m": float(lift.max()),
            "longest_hand_supported_lift_frames": int(longest),
            "carried_displacement_m": carry,
            "minimum_root_height_m": float(arrays["root_height"].min()),
            "maximum_torso_tilt_deg": float(arrays["torso_tilt"].max()),
            "final_hand_contact": arrays["hand_contact"][-1].tolist(),
            "final_foot_contact": arrays["foot_contact"][-1].tolist(),
        })
    return metrics, arrays


def main() -> None:
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    source = np.load(SOURCE_PATH)
    initial_qpos = source["initial_qpos"].copy()
    initial_qvel = source["initial_qvel"].copy()
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    object_qpos = int(model.jnt_qposadr[object_joint])
    controls = source["ctrl_physics"][: EXPERT_FRAMES * SUBSTEPS].copy()
    continuation_controls = source["ctrl_physics"].copy()
    goal = source["goal_position"].copy()
    goal_quat = source["goal_quaternion"].copy()

    metrics, trace = replay(model, initial_qpos, initial_qvel, controls, goal, goal_quat, True)
    continuation, _ = replay(model, initial_qpos, initial_qvel, continuation_controls, goal, goal_quat, False)
    repeat, repeat_trace = replay(model, initial_qpos, initial_qvel, controls, goal, goal_quat, True)
    max_qpos_error = float(np.max(np.abs(trace["qpos"] - repeat_trace["qpos"])))
    max_qvel_error = float(np.max(np.abs(trace["qvel"] - repeat_trace["qvel"])))
    success = bool(
        metrics["full_horizon_valid"]
        and continuation["full_horizon_valid"]
        and metrics["maximum_lift_m"] > 0.05
        and metrics["longest_hand_supported_lift_frames"] >= 25
        and metrics["carried_displacement_m"] > 0.5
        and max(metrics["foot_translation_m"]) > 0.5
        and metrics["non_end_effector_robot_object_impulse_ns"] < 0.1
        and metrics["final_position_error_m"] < 0.15
        and metrics["final_orientation_error_rad"] < 0.35
        and metrics["final_object_speed"] < 0.25
        and not any(metrics["final_hand_contact"])
        and max_qpos_error == 0.0
        and max_qvel_error == 0.0
    )
    report = {
        "success": success,
        "method": "frozen SONIC physical carry controls with smooth release retraction",
        "expert_horizon_reference_frames": len(controls) // SUBSTEPS,
        "expert_horizon_seconds": len(controls) * DT,
        "post_expert_exact_continuation_frames": len(continuation_controls) // SUBSTEPS - len(controls) // SUBSTEPS,
        "initial_object_quaternion_wxyz": initial_qpos[object_qpos + 3:object_qpos + 7].tolist(),
        **metrics,
        "continuation_full_horizon_valid": continuation["full_horizon_valid"],
        "continuation_first_invalid": continuation["first_invalid"],
        "continuation_final_root_height_m": continuation["final_root_height_m"],
        "replay_max_qpos_error": max_qpos_error,
        "replay_max_qvel_error": max_qvel_error,
    }
    np.savez_compressed(
        EXPERT_PATH,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        ctrl_physics=controls,
        continuation_ctrl_physics=continuation_controls,
        impulse_groups=np.asarray(("left_end_effector", "right_end_effector", "feet", "legs", "pelvis_torso", "other_robot", "environment")),
        goal_position=goal,
        goal_quaternion=goal_quat,
        **trace,
    )
    mujoco.mj_saveModel(model, str(EXPERT_MODEL_PATH), None)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps(report))


if __name__ == "__main__":
    main()
