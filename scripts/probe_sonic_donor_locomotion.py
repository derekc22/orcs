#!/usr/bin/env python3
"""Cheap gate: track the known valid stepping rollout with frozen SONIC.

This converts the exact 200 Hz kick-expert robot states into a 50 Hz,
MuJoCo-ordered tracking clip.  The object is intentionally absent: this gate
asks whether the repository's released locomotion controller can reproduce the
support transition without relying on the old forbidden kick contact.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import SonicBaseModel

from mocke import PRETRAINED_DIR, default_motion_file
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from probe_flat_hand_grasp import build


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/engineering_locomomanipulation"
DONOR = ROOT / "logs/unified_physical_reference_200hz/final_start_000/physical_reference.npz"


def make_motion(output: Path) -> None:
    engine, _ = build()
    model = engine.model
    donor = np.load(DONOR)
    qpos = np.concatenate([donor["initial_qpos"][None], donor["qpos"][3::4]], axis=0)
    qvel = np.concatenate([donor["initial_qvel"][None], donor["qvel"][3::4]], axis=0)
    # The tracking entity contains the robot bodies only, in compiled order
    # without the world body.  The donor model has robot bodies 1..31.
    robot_bodies = [
        i
        for i in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("robot/")
    ]
    body_pos = np.empty((len(qpos), len(robot_bodies), 3))
    body_quat = np.empty((len(qpos), len(robot_bodies), 4))
    body_lin = np.empty_like(body_pos)
    body_ang = np.empty_like(body_pos)
    data = mujoco.MjData(model)
    for frame, (q, v) in enumerate(zip(qpos, qvel, strict=True)):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[robot_bodies]
        body_quat[frame] = data.xquat[robot_bodies]
        body_ang[frame] = data.cvel[robot_bodies, :3]
        body_lin[frame] = data.cvel[robot_bodies, 3:]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        joint_pos=qpos[:, engine.actuator_qpos],
        joint_vel=qvel[:, engine.joint_dof],
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-start", type=int)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if args.default_start is None:
        motion = LOG / "donor_step_motion_mj.npz"
        make_motion(motion)
        label = "known exact support-transition robot motion"
    else:
        source = np.load(default_motion_file())
        start, end = args.default_start, args.default_start + args.frames + 50
        motion = LOG / f"sonic_default_segment_{start:04d}.npz"
        np.savez_compressed(
            motion,
            **{
                key: source[key][start:end]
                for key in (
                    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w",
                )
            },
        )
        label = f"released SONIC demo segment {start}:{end}"
    cfg = sonic_tracking_env_cfg(str(motion), play=True, il_ordered=False)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.terminations = {}
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    obs = env.get_observations()
    model = SonicBaseModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=29,
        base_checkpoint=str(PRETRAINED_DIR / "sonic/last_ported.pt"),
    ).to("cuda:0")
    model.eval()
    robot = raw.scene["robot"]
    feet = [robot.body_names.index(x) for x in ("left_ankle_roll_link", "right_ankle_roll_link")]
    root0 = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    foot0 = robot.data.body_link_pos_w[0, feet].detach().cpu().numpy().copy()
    first_invalid = None
    maximum_tilt = 0.0
    minimum_height = math.inf
    foot_clearance = np.zeros(2)
    trace = []
    qpos_trace = []
    with torch.inference_mode():
        for frame in range(args.frames):
            action = model(obs)
            obs, _, _, _ = env.step(action)
            root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
            quat = robot.data.root_link_quat_w[0].detach().cpu().numpy().copy()
            # wxyz root quaternion: up-axis z component after rotation.
            w, x, y, z = quat
            up_z = 1.0 - 2.0 * (x * x + y * y)
            tilt = math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0))))
            foot = robot.data.body_link_pos_w[0, feet].detach().cpu().numpy().copy()
            maximum_tilt = max(maximum_tilt, tilt)
            minimum_height = min(minimum_height, float(root[2]))
            foot_clearance = np.maximum(foot_clearance, foot[:, 2])
            if first_invalid is None:
                if root[2] < 0.45:
                    first_invalid = [frame, "root_height"]
                elif tilt > 50.0:
                    first_invalid = [frame, "torso_tilt"]
            trace.append(np.r_[root, quat, foot.ravel()])
            qpos_trace.append(raw.sim.data.qpos[0].detach().cpu().numpy().copy())
    rootf = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    footf = robot.data.body_link_pos_w[0, feet].detach().cpu().numpy().copy()
    report = {
        "method": f"frozen SONIC tracking of {label}",
        "full_horizon_physically_valid": first_invalid is None,
        "first_invalid": first_invalid,
        "root_translation_m": float(np.linalg.norm(rootf[:2] - root0[:2])),
        "foot_translation_m": np.linalg.norm(footf[:, :2] - foot0[:, :2], axis=1).tolist(),
        "maximum_foot_center_height_m": foot_clearance.tolist(),
        "minimum_root_height_m": minimum_height,
        "maximum_root_tilt_deg": maximum_tilt,
        "motion_file": str(motion.relative_to(ROOT)),
    }
    np.savez_compressed(
        LOG / "sonic_donor_locomotion_probe.npz",
        trace=np.asarray(trace),
        qpos=np.asarray(qpos_trace),
    )
    (LOG / "sonic_donor_locomotion_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
