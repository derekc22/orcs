#!/usr/bin/env python3
"""Probe hand-only under-edge support configurations for the nominal largebox.

This is a geometric/mechanical feasibility probe.  The box is placed in a
family of tipped poses only while solving IK; no result from this script is a
dynamic task success.  The probe answers whether the flat hands can reach the
opening below the near edge while both feet remain at the known stable stance.
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

ROOT = Path(__file__).resolve().parents[1]


def quat_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])


def rotation_to_quat(rotation: Rotation) -> np.ndarray:
    quaternion = rotation.as_quat()
    return np.r_[quaternion[3], quaternion[:3]]


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or f"id_{index}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", nargs="+", type=float, default=[10, 20, 30, 40, 50])
    parser.add_argument("--max-nfev", type=int, default=1800)
    parser.add_argument(
        "--mode",
        choices=("bilateral_under", "left_under", "right_under"),
        default="bilateral_under",
    )
    parser.add_argument("--root-y-min-delta", type=float, default=-0.12)
    parser.add_argument("--upright-weight", type=float, default=20.0)
    parser.add_argument("--foot-y-shift", type=float, default=0.0)
    parser.add_argument("--minimum-root-z-delta", type=float, default=-0.18)
    parser.add_argument("--foot-weight", type=float, default=75.0)
    parser.add_argument("--foot-orientation-weight", type=float, default=10.0)
    parser.add_argument("--stance-step", type=int, default=-1)
    args = parser.parse_args()

    engine, generator = build(0.60)
    model = engine.model
    stance = np.load(
        ROOT / "logs/hand_manipulation_research/stable_upright_stance_v1.npz"
    )
    # The trace is recorded at 200 Hz and begins with qpos.  Its terminal state
    # is the best available dynamically supported, collision-free stance.
    qpos0 = stance["trace"][args.stance_step, : model.nq].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    root0 = qpos0[engine.robot_qpos : engine.robot_qpos + 3].copy()
    root_rotation0 = quat_to_rotation(
        qpos0[engine.robot_qpos + 3 : engine.robot_qpos + 7]
    )
    joints0 = qpos0[engine.actuator_qpos].copy()
    object_position0 = qpos0[engine.object_qpos : engine.object_qpos + 3].copy()
    object_rotation0 = quat_to_rotation(
        qpos0[engine.object_qpos + 3 : engine.object_qpos + 7]
    )
    foot_positions0 = np.asarray(
        [data.xpos[body].copy() for body in engine.foot_bodies]
    )
    foot_targets = foot_positions0.copy()
    foot_targets[:, 1] += args.foot_y_shift
    foot_rotations0 = np.asarray(
        [data.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
    )

    vertices = []
    for geom in engine.object_geoms:
        mesh = int(model.geom_dataid[geom])
        begin = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = model.mesh_vert[begin : begin + count]
        vertices.append(
            local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
    vertices0 = np.concatenate(vertices)
    bounds_min = vertices0.min(axis=0)
    bounds_max = vertices0.max(axis=0)
    centre_x = 0.5 * (bounds_min[0] + bounds_max[0])
    # Tip away from the robot about the far (negative-y) terrain edge.
    pivot = np.asarray([centre_x, bounds_min[1], 0.0])
    settled_support_points = np.asarray(
        [
            [centre_x + 0.090, bounds_max[1] - 0.055, 0.0],
            [centre_x - 0.090, bounds_max[1] - 0.055, 0.0],
        ]
    )

    joint_ids = model.actuator_trnid[:, 0].astype(int)
    lower = np.r_[
        [
            -0.18,
            args.root_y_min_delta,
            args.minimum_root_z_delta,
            -0.65,
            -0.65,
            -0.45,
        ],
        model.jnt_range[joint_ids, 0] + 0.005,
    ]
    upper = np.r_[
        [0.18, 0.12, 0.20, 0.65, 0.65, 0.45],
        model.jnt_range[joint_ids, 1] - 0.005,
    ]
    nominal = np.r_[[0.0, args.foot_y_shift, 0.0, 0.0, 0.0, 0.0], joints0]
    object_geoms = set(map(int, engine.object_geoms))
    hand_geoms = set(map(int, engine.hand_geoms))
    results: dict[str, object] = {}
    solutions: dict[str, np.ndarray] = {}
    seed = nominal.copy()

    for angle_deg in args.angles:
        tip = Rotation.from_rotvec(np.asarray([np.deg2rad(angle_deg), 0.0, 0.0]))
        rotation = tip * object_rotation0
        position = pivot + tip.apply(object_position0 - pivot)
        support_points = pivot + tip.apply(settled_support_points - pivot)
        support_normal = tip.apply(np.asarray([0.0, 0.0, 1.0]))
        targets = support_points - 0.008 * support_normal
        normals = np.repeat(support_normal[None, :], 2, axis=0)
        if args.mode != "bilateral_under":
            under_index = 0 if args.mode == "left_under" else 1
            stabilizer_index = 1 - under_index
            stabilizer_settled = np.asarray(
                [
                    settled_support_points[stabilizer_index, 0],
                    bounds_max[1],
                    bounds_min[2] + 0.72 * (bounds_max[2] - bounds_min[2]),
                ]
            )
            front_normal = tip.apply(np.asarray([0.0, 1.0, 0.0]))
            targets[stabilizer_index] = (
                pivot + tip.apply(stabilizer_settled - pivot) + 0.008 * front_normal
            )
            normals[stabilizer_index] = front_normal
        desired_x = np.asarray([1.0, 0.0, 0.0])

        def set_state(
            values: np.ndarray,
            object_position: np.ndarray = position,
            object_rotation: Rotation = rotation,
        ) -> None:
            data.qpos[:] = qpos0
            data.qvel[:] = 0.0
            data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root0 + values[:3]
            root_rotation = Rotation.from_rotvec(values[3:6]) * root_rotation0
            data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = rotation_to_quat(
                root_rotation
            )
            data.qpos[engine.actuator_qpos] = values[6:]
            data.qpos[engine.object_qpos : engine.object_qpos + 3] = object_position
            data.qpos[engine.object_qpos + 3 : engine.object_qpos + 7] = rotation_to_quat(
                object_rotation
            )
            mujoco.mj_forward(model, data)

        def residual(
            values: np.ndarray,
            hand_targets: np.ndarray = targets,
            normal: np.ndarray = normals,
            longitudinal: np.ndarray = desired_x,
        ) -> np.ndarray:
            set_state(values)
            hand_positions = np.asarray(
                [data.geom_xpos[geom] for geom in engine.hand_geoms]
            )
            hand_rotations = np.asarray(
                [data.geom_xmat[geom].reshape(3, 3) for geom in engine.hand_geoms]
            )
            foot_positions = np.asarray(
                [data.xpos[body] for body in engine.foot_bodies]
            )
            foot_rotations = np.asarray(
                [data.xmat[body].reshape(3, 3) for body in engine.foot_bodies]
            )
            foot_orientation = np.concatenate(
                [
                    Rotation.from_matrix(
                        foot_rotations[index] @ foot_rotations0[index].T
                    ).as_rotvec()
                    for index in range(2)
                ]
            )
            torso_up = data.xmat[engine.torso_body].reshape(3, 3)[:, 2]
            root_up = data.xmat[generator.root_body].reshape(3, 3)[:, 2]
            torso_tilt = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
            root_tilt = np.arccos(np.clip(root_up[2], -1.0, 1.0))
            return np.r_[
                55.0 * (hand_positions - hand_targets).ravel(),
                9.0 * (hand_rotations[:, :, 1] - normal).ravel(),
                5.0 * (hand_rotations[:, :, 0] - longitudinal).ravel(),
                args.foot_weight * (foot_positions - foot_targets).ravel(),
                args.foot_orientation_weight * foot_orientation,
                7.0 * torso_up[:2],
                5.0 * root_up[:2],
                args.upright_weight * torso_tilt,
                0.6 * args.upright_weight * root_tilt,
                80.0 * max(0.0, torso_tilt - np.deg2rad(48.0)),
                0.05 * (values[6:] - joints0),
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
        hand_positions = np.asarray(
            [data.geom_xpos[geom] for geom in engine.hand_geoms]
        )
        hand_rotations = np.asarray(
            [data.geom_xmat[geom].reshape(3, 3) for geom in engine.hand_geoms]
        )
        contacts = []
        nonhand_object = []
        hand_object = [False, False]
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            record = {
                "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                "distance_m": float(contact.dist),
            }
            contacts.append(record)
            if first in object_geoms or second in object_geoms:
                other = second if first in object_geoms else first
                for hand_index, hand in enumerate(engine.hand_geoms):
                    if other == int(hand):
                        hand_object[hand_index] = True
                if (
                    int(model.geom_bodyid[other]) in engine.robot_bodies
                    and other not in hand_geoms
                ):
                    nonhand_object.append(record)
        physical = generator._physical(data)
        torso_tilt = float(np.rad2deg(physical["tilt"]))
        box_bounds = []
        for geom in engine.object_geoms:
            mesh = int(model.geom_dataid[geom])
            begin = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            local = model.mesh_vert[begin : begin + count]
            box_bounds.append(
                local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            )
        box_vertices = np.concatenate(box_bounds)
        report = {
            "angle_deg": angle_deg,
            "cost": float(solved.cost),
            "evaluations": int(solved.nfev),
            "hand_target_m": targets.tolist(),
            "hand_position_m": hand_positions.tolist(),
            "hand_position_error_m": np.linalg.norm(
                hand_positions - targets, axis=1
            ).tolist(),
            "hand_support_normal_alignment": np.sum(
                hand_rotations[:, :, 1] * normals, axis=1
            ).tolist(),
            "opening_height_m": float(np.min(support_points[:, 2])),
            "root_position_m": data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ].tolist(),
            "torso_tilt_deg": torso_tilt,
            "foot_position_error_m": np.linalg.norm(
                np.asarray([data.xpos[body] for body in engine.foot_bodies])
                - foot_targets,
                axis=1,
            ).tolist(),
            "box_minimum_height_m": float(box_vertices[:, 2].min()),
            "hand_object_contact": hand_object,
            "nonhand_object_contacts": nonhand_object,
            "forbidden_terrain": bool(physical["forbidden"]),
            "contacts": contacts,
        }
        key = f"angle_{int(angle_deg):02d}"
        results[key] = report
        solutions[key] = solved.x.copy()
        print(key, json.dumps(report), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **solutions)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "fixed-foot tilted-box flat-hand IK",
                "mode": args.mode,
                "foot_y_shift_m": args.foot_y_shift,
                "box_settled_bounds_m": [bounds_min.tolist(), bounds_max.tolist()],
                "tip_pivot_m": pivot.tolist(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
