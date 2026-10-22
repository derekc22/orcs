#!/usr/bin/env python3
"""Bounded post-grasp feedback search for the engineered carry expert.

The successful acquisition/carry prefix and all arm controls are immutable.
Only lower-body/waist position targets receive bounded state feedback after
frame 95.  Every candidate is evaluated in exact 200 Hz CPU MuJoCo.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_locomomanipulation"
MODEL_PATH = OUT / "engineered_locomomanipulation_model.mjb"
SOURCE_PATH = OUT / "engineered_locomomanipulation_v1.npz"
MOTION_PATH = OUT / "released_policy_motion_v1.npz"
DT = 0.005
SUBSTEPS = 4
START_FRAME = 130

GROUP_TOKENS = {
    "hip_pitch": ("left_hip_pitch", "right_hip_pitch"),
    "knee": ("left_knee", "right_knee"),
    "ankle_pitch": ("left_ankle_pitch", "right_ankle_pitch"),
    "waist_pitch": ("waist_pitch",),
    "hip_roll": ("left_hip_roll", "right_hip_roll"),
    "ankle_roll": ("left_ankle_roll", "right_ankle_roll"),
    "waist_roll": ("waist_roll",),
}
SAGITTAL = ("hip_pitch", "knee", "ankle_pitch", "waist_pitch")
LATERAL = ("hip_roll", "ankle_roll", "waist_roll")


def main() -> None:
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    source = np.load(SOURCE_PATH)
    motion = np.load(MOTION_PATH)
    base_controls = source["ctrl_physics"].copy()
    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    robot_dof = int(model.jnt_dofadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    actuator_names = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        actuator_names.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "").split("/", 1)[-1])
    groups = {
        key: np.asarray([i for i, name in enumerate(actuator_names) if any(token in name for token in tokens)], dtype=int)
        for key, tokens in GROUP_TOKENS.items()
    }
    ref_pos = motion["body_pos_w"][:, 0].astype(float)
    ref_quat = motion["body_quat_w"][:, 0].astype(float)
    ref_vel = motion["body_lin_vel_w"][:, 0].astype(float)
    ref_ang = motion["body_ang_vel_w"][:, 0].astype(float)
    ref_rot = Rotation.from_quat(np.c_[ref_quat[:, 1:], ref_quat[:, :1]])
    goal = source["goal_position"]
    goal_quat = source["goal_quaternion"]

    def evaluate(parameters: np.ndarray, record: bool = False):
        gains = parameters.reshape(7, 4)
        data = mujoco.MjData(model)
        data.qpos[:] = source["initial_qpos"]
        data.qvel[:] = source["initial_qvel"]
        mujoco.mj_forward(model, data)
        first_invalid = None
        max_tilt = 0.0
        min_height = 10.0
        cost = 0.0
        controls = []
        trace = {key: [] for key in ("qpos", "qvel", "ctrl", "root_height", "torso_tilt")}
        for step, base in enumerate(base_controls):
            frame = min(step // SUBSTEPS, len(ref_pos) - 1)
            ctrl = base.copy()
            if frame >= START_FRAME:
                quat = data.qpos[robot_qpos + 3 : robot_qpos + 7]
                cur_rot = Rotation.from_quat(np.r_[quat[1:], quat[0]])
                local_pos = ref_rot[frame].inv().apply(data.qpos[robot_qpos : robot_qpos + 3] - ref_pos[frame])
                local_vel = ref_rot[frame].inv().apply(data.qvel[robot_dof : robot_dof + 3] - ref_vel[frame])
                local_rot = (ref_rot[frame].inv() * cur_rot).as_rotvec()
                local_ang = ref_rot[frame].inv().apply(data.qvel[robot_dof + 3 : robot_dof + 6] - ref_ang[frame])
                sagittal = np.asarray([local_pos[0], local_vel[0], local_rot[1], local_ang[1]])
                lateral = np.asarray([local_pos[1], local_vel[1], local_rot[0], local_ang[0]])
                delta = np.zeros(model.nu)
                row = 0
                for key in SAGITTAL:
                    delta[groups[key]] += float(gains[row] @ sagittal)
                    row += 1
                for key in LATERAL:
                    delta[groups[key]] += float(gains[row] @ lateral)
                    row += 1
                ctrl += np.clip(delta, -0.35, 0.35)
                ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            controls.append(ctrl)
            if step % SUBSTEPS != SUBSTEPS - 1:
                continue
            root_height = float(data.qpos[robot_qpos + 2])
            torso_up = data.xmat[torso].reshape(3, 3)[:, 2]
            tilt = float(np.arccos(np.clip(torso_up[2], -1.0, 1.0)))
            min_height = min(min_height, root_height)
            max_tilt = max(max_tilt, tilt)
            if first_invalid is None and (root_height < 0.45 or tilt > np.deg2rad(50.0)):
                first_invalid = frame
            if frame >= START_FRAME:
                cost += 1200.0 * max(0.0, 0.60 - root_height) ** 2 + 500.0 * tilt**2
            if record:
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(ctrl.copy())
                trace["root_height"].append(root_height)
                trace["torso_tilt"].append(np.rad2deg(tilt))
        obj = data.qpos[object_qpos : object_qpos + 3]
        oq = data.qpos[object_qpos + 3 : object_qpos + 7]
        pos_error = float(np.linalg.norm(obj - goal))
        ori_error = float((Rotation.from_quat(np.r_[oq[1:], oq[0]]).inv() * Rotation.from_quat(np.r_[goal_quat[1:], goal_quat[0]])).magnitude())
        root_speed = float(np.linalg.norm(data.qvel[robot_dof : robot_dof + 3]))
        if first_invalid is None:
            rank = (0, cost + 200.0 * root_speed**2, pos_error)
        else:
            rank = (1, -first_invalid, cost, pos_error)
        metrics = {
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": first_invalid,
            "minimum_root_height_m": min_height,
            "maximum_torso_tilt_deg": float(np.rad2deg(max_tilt)),
            "final_root_speed_m_s": root_speed,
            "final_position_error_m": pos_error,
            "final_orientation_error_rad": ori_error,
        }
        return rank, metrics, np.asarray(controls), {k: np.asarray(v) for k, v in trace.items()}

    rng = np.random.default_rng(8128)
    dimensions = 28
    mean = np.zeros(dimensions)
    std = np.full(dimensions, 0.80)
    best_params = mean.copy()
    best = evaluate(best_params)
    history = []
    for generation in range(10):
        population = np.clip(rng.normal(mean, std, (48, dimensions)), -5.0, 5.0)
        population[0] = best_params
        results = [evaluate(candidate) for candidate in population]
        order = sorted(range(len(population)), key=lambda i: results[i][0])
        if results[order[0]][0] < best[0]:
            best_params = population[order[0]].copy()
            best = results[order[0]]
        elite = population[order[:10]]
        mean = 0.7 * elite.mean(axis=0) + 0.3 * mean
        std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, 0.02)
        history.append({"generation": generation, **best[1]})
        print("GEN", generation, json.dumps(best[1]), flush=True)
    _, metrics, controls, trace = evaluate(best_params, record=True)
    out = OUT / "engineered_locomomanipulation_recovered_v1.npz"
    np.savez_compressed(
        out,
        initial_qpos=source["initial_qpos"],
        initial_qvel=source["initial_qvel"],
        ctrl_physics=controls,
        parameters=best_params,
        goal_position=goal,
        goal_quaternion=goal_quat,
        **trace,
    )
    report = {"method": "locked manipulation prefix plus bounded lower-body feedback", "start_frame": START_FRAME, "best": metrics, "history": history}
    (OUT / "engineered_locomomanipulation_recovered_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
