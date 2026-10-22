#!/usr/bin/env python3
"""Independently replay and render the exact engineered expert controls."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_locomomanipulation"
MODEL_PATH = OUT / "engineered_locomomanipulation_model.mjb"
EXPERT_PATH = OUT / "engineered_locomomanipulation_expert_start0.npz"
VIDEO_PATH = OUT / "engineered_locomomanipulation_expert_start0.mp4"
MONTAGE_PATH = OUT / "engineered_locomomanipulation_expert_start0_montage.png"
DENSE_PATH = None
SUBSTEPS = 4
DT = 0.005


def cube_orientation_error(rotation: Rotation, goal: Rotation) -> float:
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


def main() -> None:
    model = mujoco.MjModel.from_binary_path(str(MODEL_PATH))
    model.opt.timestep = DT
    expert = np.load(EXPERT_PATH)
    data = mujoco.MjData(model)
    data.qpos[:] = expert["initial_qpos"]
    data.qvel[:] = expert["initial_qvel"]
    mujoco.mj_forward(model, data)

    robot_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint")
    robot_qpos = int(model.jnt_qposadr[robot_joint])
    object_qpos = int(model.jnt_qposadr[object_joint])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{s}_ankle_roll_link") for s in ("left", "right")]
    hand_geoms = [
        {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_hand_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_hand_front_guard_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_hand_trailing_guard_collision"),
        }
        for s in ("left", "right")
    ]
    object_geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == object_body}
    initial_z = float(data.qpos[object_qpos + 2])
    goal = expert["goal_position"]
    goal_quat = expert["goal_quaternion"]

    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (-1.62, 3.08, 0.85)
    camera.distance = 3.15
    camera.azimuth = 118.0
    camera.elevation = -14.0
    font = ImageFont.load_default(size=18)
    frames: list[np.ndarray] = []
    non_ee_impulse = 0.0
    controls = expert["ctrl_physics"]
    for step, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if not (pair & object_geoms):
                continue
            other = next((g for g in pair if g not in object_geoms), None)
            if other is None or any(other in geoms for geoms in hand_geoms):
                continue
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
            if body_name.startswith("robot/"):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                non_ee_impulse += float(np.linalg.norm(force[:3]) * DT)
        if step % SUBSTEPS != SUBSTEPS - 1:
            continue

        frame = step // SUBSTEPS
        hand_contact = np.zeros(2, dtype=bool)
        foot_contact = np.zeros(2, dtype=bool)
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            for side, geoms in enumerate(hand_geoms):
                hand_contact[side] |= bool(pair & geoms) and bool(pair & object_geoms)
            for side, body in enumerate(foot_bodies):
                if int(model.geom_bodyid[int(con.geom1)]) == body or int(model.geom_bodyid[int(con.geom2)]) == body:
                    other = int(con.geom2) if int(model.geom_bodyid[int(con.geom1)]) == body else int(con.geom1)
                    foot_contact[side] |= other not in object_geoms
        root_height = float(data.qpos[robot_qpos + 2])
        torso_up = data.xmat[torso].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(torso_up[2], -1.0, 1.0))))
        valid = root_height >= 0.45 and tilt <= 50.0
        lift = float(data.qpos[object_qpos + 2] - initial_z)
        pos_error = float(np.linalg.norm(data.qpos[object_qpos:object_qpos + 3] - goal))
        quat = data.qpos[object_qpos + 3:object_qpos + 7]
        ori_error = cube_orientation_error(
            Rotation.from_quat(np.r_[quat[1:], quat[0]]),
            Rotation.from_quat(np.r_[goal_quat[1:], goal_quat[0]]),
        )
        if foot_contact.all():
            support = "two feet"
        elif foot_contact[0]:
            support = "left foot"
        elif foot_contact[1]:
            support = "right foot"
        else:
            support = "flight/transition"
        phase = "CARRY" if hand_contact.any() and lift > 0.05 else ("GRASP" if hand_contact.any() else ("PLACED" if frame > 110 else "APPROACH"))

        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        lines = [
            f"frame {frame:03d} / {len(controls) // SUBSTEPS - 1:03d}    phase: {phase}",
            f"valid: {valid}    support: {support}    root: {root_height:.3f} m    tilt: {tilt:.1f} deg",
            f"left/right end-effector contact: {bool(hand_contact[0])} / {bool(hand_contact[1])}",
            f"object lift: {lift:+.3f} m    non-end-effector impulse: {non_ee_impulse:.4f} Ns",
            f"goal position error: {pos_error:.3f} m    cube-sym orientation error: {ori_error:.3f} rad",
        ]
        box = (12, 12, 800, 133)
        draw.rounded_rectangle(box, radius=7, fill=(0, 0, 0, 190), outline=(230, 230, 230))
        for i, line in enumerate(lines):
            draw.text((24, 21 + i * 22), line, font=font, fill=(255, 255, 255))
        frames.append(np.asarray(image))
    renderer.close()

    with imageio.get_writer(VIDEO_PATH, fps=50, codec="libx264", quality=8, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    selected = [0, 20, 40, 60, 80, 100, 112, 126, len(frames) - 1]
    thumb_w, thumb_h = 480, 360
    montage = Image.new("RGB", (thumb_w * 3, thumb_h * 3))
    for slot, index in enumerate(selected):
        thumb = Image.fromarray(frames[index]).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        montage.paste(thumb, ((slot % 3) * thumb_w, (slot // 3) * thumb_h))
    montage.save(MONTAGE_PATH)
    if DENSE_PATH is not None:
        dense_indexes = np.linspace(0, len(frames) - 1, 20).round().astype(int)
        dense_w, dense_h = 384, 288
        dense = Image.new("RGB", (dense_w * 5, dense_h * 4))
        for slot, index in enumerate(dense_indexes):
            thumb = Image.fromarray(frames[index]).resize((dense_w, dense_h), Image.Resampling.LANCZOS)
            dense.paste(thumb, ((slot % 5) * dense_w, (slot // 5) * dense_h))
        dense.save(DENSE_PATH)
    print(VIDEO_PATH)
    print(MONTAGE_PATH)
    if DENSE_PATH is not None:
        print(DENSE_PATH)


if __name__ == "__main__":
    main()
