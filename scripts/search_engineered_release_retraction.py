#!/usr/bin/env python3
"""Bounded exact-physics search for collision-free release-arm retraction.

The demonstrated grasp/carry prefix and every lower-body control are frozen.
Only arm position targets after release are smoothly redirected toward later
donor targets whose wrist geometry is clear of the placed object.
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
DT = 0.005
SUBSTEPS = 4


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def main() -> None:
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    source = np.load(SOURCE_PATH)
    base = source["ctrl_physics_policy"].copy()
    names = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        names.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "").split("/")[-1])
    arms = np.asarray([i for i, name in enumerate(names) if any(k in name for k in ("shoulder", "elbow", "wrist"))])
    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_dof = int(model.jnt_dofadr[object_joint])
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    hand_geoms = [
        {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_front_guard_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_hand_trailing_guard_collision"),
        }
        for side in ("left", "right")
    ]
    object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == object_body}
    goal = source["goal_position"]
    goal_quat = source["goal_quaternion"]

    def evaluate(ctrl: np.ndarray, record: bool = False):
        data = mujoco.MjData(model)
        data.qpos[:] = source["initial_qpos"]
        data.qvel[:] = source["initial_qvel"]
        mujoco.mj_forward(model, data)
        first_invalid = None
        late_contacts = 0
        trace = {k: [] for k in ("qpos", "qvel", "ctrl", "object_pos", "object_quat", "hand_contact", "root_height", "torso_tilt")}
        for step, command in enumerate(ctrl):
            data.ctrl[:] = command
            mujoco.mj_step(model, data)
            if step % SUBSTEPS != SUBSTEPS - 1:
                continue
            frame = step // SUBSTEPS
            contacts = np.zeros(2, dtype=bool)
            for ci in range(data.ncon):
                pair = {int(data.contact[ci].geom1), int(data.contact[ci].geom2)}
                for side, geoms in enumerate(hand_geoms):
                    contacts[side] |= bool(pair & geoms) and bool(pair & object_geoms)
            if frame >= 112:
                late_contacts += int(contacts.sum())
            height = float(data.qpos[robot_qpos + 2])
            up = data.xmat[torso].reshape(3, 3)[:, 2]
            tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
            if first_invalid is None and (height < 0.45 or tilt > 50.0):
                first_invalid = frame
            if record:
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(command.copy())
                trace["object_pos"].append(data.qpos[object_qpos:object_qpos + 3].copy())
                trace["object_quat"].append(data.qpos[object_qpos + 3:object_qpos + 7].copy())
                trace["hand_contact"].append(contacts)
                trace["root_height"].append(height)
                trace["torso_tilt"].append(tilt)
        pos = data.qpos[object_qpos:object_qpos + 3]
        quat = data.qpos[object_qpos + 3:object_qpos + 7]
        pos_error = float(np.linalg.norm(pos - goal))
        ori_error = float((Rotation.from_quat(np.r_[quat[1:], quat[0]]).inv() * Rotation.from_quat(np.r_[goal_quat[1:], goal_quat[0]])).magnitude())
        speed = float(np.linalg.norm(data.qvel[object_dof:object_dof + 6]))
        metrics = {
            "full_horizon_valid": first_invalid is None,
            "first_invalid_frame": first_invalid,
            "final_position_error_m": pos_error,
            "final_orientation_error_rad": ori_error,
            "final_object_speed": speed,
            "post_release_contact_frames": late_contacts,
        }
        rank = (first_invalid is not None, pos_error >= 0.15, speed >= 0.25, late_contacts, pos_error + 0.03 * speed)
        return rank, metrics, {k: np.asarray(v) for k, v in trace.items()}

    candidates = []
    side_indexes = {
        "both": arms,
        "left": np.asarray([i for i in arms if names[i].startswith("left_")]),
        "right": np.asarray([i for i in arms if names[i].startswith("right_")]),
    }
    # Retract just before the observed first post-placement contact toward
    # native future donor arm targets.  One-sided variants avoid injecting an
    # unnecessary whole-upper-body transient.
    for side, selected in side_indexes.items():
        for start_frame in (104, 106, 108, 110, 112):
            for duration in (3, 5, 8, 12, 20):
                for target_frame in (136, 140, 150, 160, 175):
                    ctrl = base.copy()
                    target = base[target_frame * SUBSTEPS + (SUBSTEPS - 1), selected].copy()
                    start_step = start_frame * SUBSTEPS
                    duration_steps = duration * SUBSTEPS
                    for step in range(start_step, len(ctrl)):
                        alpha = smoothstep((step - start_step) / max(duration_steps, 1))
                        ctrl[step, selected] = (1.0 - alpha) * base[step, selected] + alpha * target
                    rank, metrics, _ = evaluate(ctrl)
                    candidates.append((rank, side, start_frame, duration, target_frame, metrics, ctrl))
    candidates.sort(key=lambda x: x[0])
    summary = []
    for rank, side, start, duration, target, metrics, _ in candidates[:20]:
        item = {"side": side, "start_frame": start, "duration_frames": duration, "target_frame": target, **metrics}
        summary.append(item)
        print("CANDIDATE", json.dumps(item), flush=True)
    _, side, start, duration, target, metrics, controls = candidates[0]
    _, _, trace = evaluate(controls, record=True)
    np.savez_compressed(
        OUT / "engineered_release_retraction_v1.npz",
        initial_qpos=source["initial_qpos"],
        initial_qvel=source["initial_qvel"],
        ctrl_physics=controls,
        goal_position=goal,
        goal_quaternion=goal_quat,
        arm_actuator_indexes=arms,
        **trace,
    )
    report = {
        "method": "frozen successful carry prefix and lower body; smooth bilateral arm retraction",
        "best_parameters": {"side": side, "start_frame": start, "duration_frames": duration, "target_frame": target},
        "best": metrics,
        "top_candidates": summary,
    }
    (OUT / "engineered_release_retraction_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BEST", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
