#!/usr/bin/env python3
"""Solve bilateral flat-hand IK in an actual dynamic rollout state.

The target hand transforms are expressed relative to the physical box, so the
solution remains meaningful after the box has settled or rotated.  Only the
fourteen arm/wrist joints are changed; the floating base and lower body remain
at the recorded MuJoCo state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from probe_flat_hand_grasp import build
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def quaternion_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[130, 140, 150, 160])
    parser.add_argument("--max-nfev", type=int, default=1600)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    rollout = np.load(args.rollout)
    family = np.load(args.family)
    balance = np.asarray(
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28]
    )
    arms = np.setdiff1d(np.arange(model.nu), balance)
    arm_joint_ids = model.actuator_trnid[arms, 0].astype(int)
    lower = model.jnt_range[arm_joint_ids, 0] + 0.005
    upper = model.jnt_range[arm_joint_ids, 1] - 0.005

    # Reconstruct the static bilateral-grasp state and store hand transforms in
    # the box frame.  Applying those relative transforms to the actual box pose
    # removes the stale pre-settling world-space assumption.
    template = mujoco.MjData(model)
    template.qpos[:] = generator.initial_data.qpos
    template.qvel[:] = 0.0
    values = family["grasp"]
    template.qpos[engine.robot_qpos : engine.robot_qpos + 3] += values[:3]
    root_quat = template.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    root_rotation = Rotation.from_rotvec(values[3:6]) * quaternion_rotation(root_quat)
    quat = root_rotation.as_quat()
    template.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = np.r_[
        quat[3], quat[:3]
    ]
    template.qpos[engine.actuator_qpos] = values[6:]
    mujoco.mj_forward(model, template)
    template_object_pos = template.qpos[engine.object_qpos : engine.object_qpos + 3]
    template_object_rotation = quaternion_rotation(
        template.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
    )
    hand_relative_positions = np.asarray(
        [
            template_object_rotation.inv().apply(
                template.geom_xpos[geom] - template_object_pos
            )
            for geom in engine.hand_geoms
        ]
    )
    hand_relative_rotations = [
        template_object_rotation.inv()
        * Rotation.from_matrix(template.geom_xmat[geom].reshape(3, 3))
        for geom in engine.hand_geoms
    ]

    solutions: dict[str, np.ndarray] = {}
    reports: dict[str, object] = {}
    for frame in args.frames:
        data = mujoco.MjData(model)
        data.qpos[:] = rollout["qpos"][frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        nominal = data.qpos[engine.actuator_qpos[arms]].copy()
        object_pos = data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
        object_rotation = quaternion_rotation(
            data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7]
        )
        target_positions = np.asarray(
            [
                object_pos + object_rotation.apply(relative)
                for relative in hand_relative_positions
            ]
        )
        target_rotations = [
            object_rotation * relative for relative in hand_relative_rotations
        ]

        def set_arms(candidate: np.ndarray, state: mujoco.MjData = data) -> None:
            state.qpos[engine.actuator_qpos[arms]] = candidate
            mujoco.mj_forward(model, state)

        def residual(
            candidate: np.ndarray,
            state: mujoco.MjData = data,
            targets: np.ndarray = target_positions,
            rotations: list[Rotation] = target_rotations,
            prior: np.ndarray = nominal,
        ) -> np.ndarray:
            set_arms(candidate)
            hand_positions = np.asarray(
                [state.geom_xpos[geom] for geom in engine.hand_geoms]
            )
            orientation_errors = np.concatenate(
                [
                    (
                        Rotation.from_matrix(state.geom_xmat[geom].reshape(3, 3)).inv()
                        * rotations[index]
                    ).as_rotvec()
                    for index, geom in enumerate(engine.hand_geoms)
                ]
            )
            return np.r_[
                45.0 * (hand_positions - targets).ravel(),
                7.0 * orientation_errors,
                0.08 * (candidate - prior),
            ]

        result = least_squares(
            residual,
            np.clip(nominal, lower, upper),
            bounds=(lower, upper),
            max_nfev=args.max_nfev,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        set_arms(result.x)
        contacts = []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            contacts.append(
                {
                    "geom1": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ),
                    "geom2": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ),
                    "distance_m": float(contact.dist),
                }
            )
        solved_positions = np.asarray(
            [data.geom_xpos[geom] for geom in engine.hand_geoms]
        )
        key = f"frame_{frame:03d}"
        solutions[key] = result.x
        reports[key] = {
            "cost": float(result.cost),
            "evaluations": int(result.nfev),
            "hand_position_error_m": np.linalg.norm(
                solved_positions - target_positions, axis=1
            ).tolist(),
            "arm_rms_change_rad": float(np.sqrt(np.mean((result.x - nominal) ** 2))),
            "arm_max_change_rad": float(np.max(np.abs(result.x - nominal))),
            "contacts": contacts,
        }
        print(key, json.dumps(reports[key]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        arm_actuators=arms,
        hand_relative_positions=hand_relative_positions,
        **solutions,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "actual-state box-relative bilateral flat-hand IK",
                "frames": args.frames,
                "arm_joint_names": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint))
                    for joint in arm_joint_ids
                ],
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
