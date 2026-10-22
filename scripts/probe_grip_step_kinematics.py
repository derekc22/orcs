#!/usr/bin/env python3
"""Probe whether the supported side grip admits a stepped-under stance.

The object is held fixed only for this kinematic feasibility calculation.  The
probe starts from an actual MuJoCo bilateral-hand state and asks whether the
feet/support base can move toward the task direction while preserving both
flat-hand transforms and reducing torso lean.  It is not a task rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_flat_hand_grasp import build


def quat_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def rotation_to_quat(rotation: Rotation) -> np.ndarray:
    quaternion = rotation.as_quat()
    return np.r_[quaternion[3], quaternion[:3]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shifts", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25])
    parser.add_argument("--max-nfev", type=int, default=2500)
    args = parser.parse_args()

    engine, generator = build(0.55)
    model = engine.model
    source = np.load(args.source)
    qpos0 = source["qpos"][args.source_frame].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    root0 = qpos0[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_rotation0 = quat_to_rotation(qpos0[engine.robot_qpos + 3 : engine.robot_qpos + 7])
    joints0 = qpos0[engine.actuator_qpos].copy()
    hands0 = np.asarray([data.geom_xpos[geom].copy() for geom in engine.hand_geoms])
    hand_rotations0 = np.asarray(
        [data.geom_xmat[geom].reshape(3, 3).copy() for geom in engine.hand_geoms]
    )
    feet0 = np.asarray([data.xpos[body].copy() for body in engine.foot_bodies])
    foot_rotations0 = np.asarray(
        [data.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )
    object_position = qpos0[engine.object_qpos : engine.object_qpos + 3].copy()
    goal = engine.reference_object_pos[-1]
    forward = np.r_[goal[:2] - object_position[:2], 0.0]
    forward /= max(np.linalg.norm(forward), 1.0e-9)

    joint_ids = model.actuator_trnid[:, 0].astype(int)
    lower = np.r_[[-0.25, -0.35, -0.18, -0.60, -0.60, -0.50], model.jnt_range[joint_ids, 0] + 0.005]
    upper = np.r_[[0.25, 0.15, 0.20, 0.60, 0.60, 0.50], model.jnt_range[joint_ids, 1] - 0.005]
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))

    def set_state(values: np.ndarray) -> None:
        data.qpos[:] = qpos0
        data.qvel[:] = 0.0
        data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root0 + values[:3]
        rotation = Rotation.from_rotvec(values[3:6]) * root_rotation0
        data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = rotation_to_quat(rotation)
        data.qpos[engine.actuator_qpos] = values[6:]
        mujoco.mj_forward(model, data)

    results: dict[str, object] = {}
    solutions: dict[str, np.ndarray] = {}
    seed = np.r_[np.zeros(6), joints0]
    for shift in args.shifts:
        foot_targets = feet0 + shift * forward

        def residual(values: np.ndarray) -> np.ndarray:
            set_state(values)
            hands = np.asarray([data.geom_xpos[geom] for geom in engine.hand_geoms])
            hand_rotations = np.asarray(
                [data.geom_xmat[geom].reshape(3, 3) for geom in engine.hand_geoms]
            )
            feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
            foot_rotations = np.asarray(
                [data.xmat[body].reshape(3, 3) for body in engine.foot_bodies]
            )
            hand_orientation = np.concatenate(
                [
                    Rotation.from_matrix(hand_rotations[i] @ hand_rotations0[i].T).as_rotvec()
                    for i in range(2)
                ]
            )
            foot_orientation = np.concatenate(
                [
                    Rotation.from_matrix(foot_rotations[i] @ foot_rotations0[i].T).as_rotvec()
                    for i in range(2)
                ]
            )
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            root_height = data.qpos[engine.robot_qpos + 2]
            return np.r_[
                90.0 * (hands - hands0).ravel(),
                12.0 * hand_orientation,
                120.0 * (feet - foot_targets).ravel(),
                30.0 * foot_orientation,
                18.0 * torso_up[:2],
                8.0 * (root_height - max(root0[2], 0.58)),
                0.04 * (values[6:] - joints0),
            ]

        solved = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=args.max_nfev,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        set_state(solved.x)
        seed = solved.x.copy()
        hands = np.asarray([data.geom_xpos[geom] for geom in engine.hand_geoms])
        feet = np.asarray([data.xpos[body] for body in engine.foot_bodies])
        physical = generator._physical(data)
        nonhand_object: list[dict[str, object]] = []
        self_collisions: list[dict[str, object]] = []
        maximum_foot_penetration = 0.0
        hand_object = [False, False]
        contacts: list[dict[str, object]] = []
        for ci in range(data.ncon):
            contact = data.contact[ci]
            first, second = int(contact.geom1), int(contact.geom2)
            record = {
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                "distance_m": float(contact.dist),
            }
            contacts.append(record)
            pair = {first, second}
            if pair & engine.terrain_geoms:
                other = second if first in engine.terrain_geoms else first
                if int(model.geom_bodyid[other]) in engine.foot_bodies:
                    maximum_foot_penetration = max(maximum_foot_penetration, -float(contact.dist))
            if pair & object_geoms:
                other = second if first in object_geoms else first
                if other in hand_geoms:
                    hand_object[list(engine.hand_geoms).index(other)] = True
                elif int(model.geom_bodyid[other]) in engine.robot_bodies:
                    nonhand_object.append(record)
            elif int(model.geom_bodyid[first]) in engine.robot_bodies and int(model.geom_bodyid[second]) in engine.robot_bodies:
                self_collisions.append(record)
        key = f"shift_{shift:.3f}"
        solutions[key] = solved.x.copy()
        results[key] = {
            "shift_m": shift,
            "cost": float(solved.cost),
            "evaluations": int(solved.nfev),
            "root_position_m": data.qpos[engine.robot_qpos : engine.robot_qpos + 3].tolist(),
            "torso_tilt_deg": float(np.rad2deg(physical["tilt"])),
            "hand_position_error_m": np.linalg.norm(hands - hands0, axis=1).tolist(),
            "foot_position_error_m": np.linalg.norm(feet - foot_targets, axis=1).tolist(),
            "hand_object_contact": hand_object,
            "maximum_foot_penetration_m": maximum_foot_penetration,
            "nonhand_object_contacts": nonhand_object,
            "self_collisions": self_collisions,
            "forbidden_terrain": bool(physical["forbidden"]),
            "contacts": contacts,
        }
        print(key, json.dumps(results[key]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **solutions)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "stepped-support-base bilateral-grip kinematic probe",
                "source": str(args.source),
                "source_frame": args.source_frame,
                "forward": forward.tolist(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
