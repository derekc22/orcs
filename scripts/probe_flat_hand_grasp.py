#!/usr/bin/env python3
"""Find collision-valid bilateral flat-hand poses and test physical lift.

This is an experimental manipulation-feasibility probe.  The box is fully
dynamic after reset; only legal G1 position-actuator targets are applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from orcs.cli.refine_diffusion_reference import _build_model
from orcs.tasks.uolm.dynamics_refinement import (
    DynamicsRefinementConfig,
    _DynamicRollout,
)
from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter
from orcs.tasks.uolm.physical_reference_generator import (
    PhysicalReferenceConfig,
    PhysicalReferenceGenerator,
)

ROOT = Path(__file__).resolve().parents[1]


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def build(
    robot_clearance_m: float = 0.60,
) -> tuple[_DynamicRollout, PhysicalReferenceGenerator]:
    trajectory = np.load(
        ROOT
        / "logs/direct_contact_dynamics_knot117/direct_contact_dynamics_start_000.npz"
    )["analytical_through140_trajectory"]
    reference = GeneratedReferenceAdapter(
        device=torch.device("cpu"), dt=0.02, fk_batch_size=256
    )(torch.as_tensor(trajectory))
    engine = _DynamicRollout(
        _build_model(),
        reference,
        DynamicsRefinementConfig(run_stage_b=False, seed=42),
    )
    stance = np.load(
        ROOT / "logs/unified_physical_reference/oriented_stance_start_000.npz"
    )
    generator = PhysicalReferenceGenerator(
        engine,
        PhysicalReferenceConfig(
            robot_clearance_m=robot_clearance_m,
            desired_stance_height_m=0.62,
            stance_parameters=tuple(stance["parameters"]),
            use_lqr_executor=False,
        ),
    )
    arms = np.setdiff1d(np.arange(engine.model.nu), generator.balance_actuators)
    generator.initial_data.qpos[engine.actuator_qpos[arms]] = stance["arm_pose"]
    mujoco.mj_forward(engine.model, generator.initial_data)
    return engine, generator


class WholeBodyHandIK:
    def __init__(self, engine: _DynamicRollout, generator: PhysicalReferenceGenerator):
        self.engine = engine
        self.generator = generator
        self.model = engine.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = generator.initial_data.qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.root0 = self.data.qpos[engine.robot_qpos : engine.robot_qpos + 3].copy()
        quaternion = self.data.qpos[
            engine.robot_qpos + 3 : engine.robot_qpos + 7
        ].copy()
        self.rotation0 = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        self.joint0 = self.data.qpos[engine.actuator_qpos].copy()
        self.foot_positions = np.asarray(
            [self.data.xpos[body].copy() for body in engine.foot_bodies]
        )
        self.foot_rotations = np.asarray(
            [self.data.xmat[body].reshape(3, 3).copy() for body in engine.foot_bodies]
        )
        self.object_position = self.data.qpos[
            engine.object_qpos : engine.object_qpos + 3
        ].copy()
        goal = engine.reference_object_pos[-1]
        self.forward = np.r_[goal[:2] - self.object_position[:2], 0.0]
        self.forward /= np.linalg.norm(self.forward)
        self.lateral = np.asarray([-self.forward[1], self.forward[0], 0.0])
        joint_ids = self.model.actuator_trnid[:, 0]
        self.lower = np.r_[
            [-0.05, -0.05, -0.20, -0.50, -0.50, -0.25],
            self.model.jnt_range[joint_ids, 0] + 0.005,
        ]
        self.upper = np.r_[
            [0.05, 0.05, 0.08, 0.50, 0.50, 0.25],
            self.model.jnt_range[joint_ids, 1] - 0.005,
        ]

    def set_state(self, values: np.ndarray) -> None:
        engine = self.engine
        self.data.qpos[engine.robot_qpos : engine.robot_qpos + 3] = (
            self.root0 + values[:3]
        )
        rotation = Rotation.from_rotvec(values[3:6]) * self.rotation0
        quaternion = rotation.as_quat()
        self.data.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = np.r_[
            quaternion[3], quaternion[:3]
        ]
        self.data.qpos[engine.actuator_qpos] = values[6:]
        mujoco.mj_forward(self.model, self.data)

    def solve(
        self,
        lateral_distance: float,
        height: float,
        initial: np.ndarray | None = None,
        *,
        upright_weight: float = 0.0,
        root_height_delta: float | None = None,
        max_nfev: int = 1200,
        lateral_shift: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, object]]:
        targets = np.asarray(
            [
                self.object_position
                + (lateral_distance + lateral_shift) * self.lateral,
                self.object_position
                - (lateral_distance - lateral_shift) * self.lateral,
            ]
        )
        targets[:, 2] = height
        normals = np.asarray([-self.lateral, self.lateral])
        longitudinal_axes = np.asarray([self.forward, self.forward])

        def residual(values: np.ndarray) -> np.ndarray:
            self.set_state(values)
            hand_positions = np.asarray(
                [self.data.geom_xpos[geom] for geom in self.engine.hand_geoms]
            )
            hand_rotations = np.asarray(
                [
                    self.data.geom_xmat[geom].reshape(3, 3)
                    for geom in self.engine.hand_geoms
                ]
            )
            foot_positions = np.asarray(
                [self.data.xpos[body] for body in self.engine.foot_bodies]
            )
            foot_rotations = np.asarray(
                [self.data.xmat[body].reshape(3, 3) for body in self.engine.foot_bodies]
            )
            foot_orientation = np.concatenate(
                [
                    Rotation.from_matrix(
                        foot_rotations[index] @ self.foot_rotations[index].T
                    ).as_rotvec()
                    for index in range(2)
                ]
            )
            root_rotation_error = Rotation.from_matrix(
                self.data.xmat[self.generator.root_body].reshape(3, 3)
                @ self.rotation0.as_matrix().T
            ).as_rotvec()
            torso_up = self.data.xmat[self.engine.torso_body].reshape(3, 3)[:, 2]
            posture = np.r_[
                upright_weight * root_rotation_error[:2],
                upright_weight * torso_up[:2],
            ]
            if root_height_delta is not None:
                posture = np.r_[
                    posture,
                    12.0
                    * (
                        self.data.qpos[self.engine.robot_qpos + 2]
                        - (self.root0[2] + root_height_delta)
                    ),
                ]
            return np.r_[
                35.0 * (hand_positions - targets).ravel(),
                6.0 * (hand_rotations[:, :, 1] - normals).ravel(),
                5.0 * (hand_rotations[:, :, 0] - longitudinal_axes).ravel(),
                60.0 * (foot_positions - self.foot_positions).ravel(),
                8.0 * foot_orientation,
                posture,
                0.03 * (values[6:] - self.joint0),
            ]

        seed = (
            np.r_[np.zeros(6), self.joint0]
            if initial is None
            else np.asarray(initial).copy()
        )
        solution = least_squares(
            residual,
            np.clip(seed, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=max_nfev,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        self.set_state(solution.x)
        hand_positions = np.asarray(
            [self.data.geom_xpos[geom] for geom in self.engine.hand_geoms]
        )
        hand_rotations = np.asarray(
            [self.data.geom_xmat[geom].reshape(3, 3) for geom in self.engine.hand_geoms]
        )
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            contacts.append(
                {
                    "geom1": mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, first
                    ),
                    "geom2": mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, second
                    ),
                    "distance_m": float(contact.dist),
                }
            )
        return solution.x, {
            "cost": float(solution.cost),
            "evaluations": int(solution.nfev),
            "root_position_m": self.data.qpos[
                self.engine.robot_qpos : self.engine.robot_qpos + 3
            ].tolist(),
            "hand_position_error_m": np.linalg.norm(
                hand_positions - targets, axis=1
            ).tolist(),
            "hand_normal_alignment": np.sum(
                hand_rotations[:, :, 1] * normals, axis=1
            ).tolist(),
            "hand_longitudinal_alignment": np.sum(
                hand_rotations[:, :, 0] * longitudinal_axes, axis=1
            ).tolist(),
            "root_world_tilt_deg": float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            self.data.xmat[self.generator.root_body].reshape(3, 3)[
                                2, 2
                            ],
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "torso_world_tilt_deg": float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            self.data.xmat[self.engine.torso_body].reshape(3, 3)[2, 2],
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "contacts": contacts,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulate-lift", action="store_true")
    parser.add_argument("--contact-height-offset", type=float, default=0.080)
    parser.add_argument("--upright-weight", type=float, default=0.0)
    parser.add_argument("--root-height-delta", type=float)
    parser.add_argument("--robot-clearance", type=float, default=0.60)
    parser.add_argument("--minimum-root-delta", type=float, default=-0.20)
    parser.add_argument(
        "--root-horizontal-bound",
        type=float,
        default=0.05,
        help="Symmetric root x/y search bound around the generated initial pose.",
    )
    parser.add_argument("--max-nfev", type=int, default=1200)
    args = parser.parse_args()
    engine, generator = build(args.robot_clearance)
    ik = WholeBodyHandIK(engine, generator)
    ik.lower[2] = args.minimum_root_delta
    ik.lower[:2] = -args.root_horizontal_bound
    ik.upper[:2] = args.root_horizontal_bound
    solutions: dict[str, np.ndarray] = {}
    reports: dict[str, object] = {}
    seed = None
    for distance in (
        0.30,
        0.27,
        0.25,
        0.235,
        0.225,
        0.215,
        0.205,
        0.195,
        0.185,
        0.175,
        0.165,
        0.155,
        0.145,
        0.135,
    ):
        values, report = ik.solve(
            distance,
            float(ik.object_position[2] + args.contact_height_offset),
            initial=seed,
            upright_weight=args.upright_weight,
            root_height_delta=args.root_height_delta,
            max_nfev=args.max_nfev,
        )
        seed = values
        key = f"distance_{distance:.3f}"
        solutions[key] = values
        reports[key] = report
        print(key, json.dumps(report), flush=True)
    lift_values, lift_report = ik.solve(
        0.212,
        float(ik.object_position[2] + args.contact_height_offset + 0.100),
        initial=solutions["distance_0.215"],
        upright_weight=args.upright_weight,
        root_height_delta=args.root_height_delta,
        max_nfev=args.max_nfev,
    )
    solutions["lift"] = lift_values
    reports["lift"] = lift_report
    print("lift", json.dumps(lift_report), flush=True)

    simulation_report: dict[str, object] | None = None
    simulation_arrays: dict[str, np.ndarray] = {}
    if args.simulate_lift:
        model = engine.model
        data = mujoco.MjData(model)
        ik.set_state(solutions["distance_0.270"])
        data.qpos[:] = ik.data.qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = data.qpos[engine.actuator_qpos]
        mujoco.mj_forward(model, data)
        targets = {
            "pregrasp": solutions["distance_0.270"][6:],
            "grasp": solutions["distance_0.215"][6:],
            "squeeze": solutions["distance_0.205"][6:],
            "lift": lift_values[6:],
        }
        group_impulse = {
            "left_hand": 0.0,
            "right_hand": 0.0,
            "wrist": 0.0,
            "feet": 0.0,
            "lower_legs": 0.0,
            "upper_legs": 0.0,
            "torso_pelvis": 0.0,
            "other_robot": 0.0,
        }
        trace = {
            "qpos": [],
            "qvel": [],
            "ctrl": [],
            "object_position_m": [],
            "hand_contact": [],
            "root_height_m": [],
            "torso_tilt_rad": [],
            "forbidden": [],
        }
        object_geoms = set(engine.object_geoms)

        def group(geom: int) -> str:
            if geom == int(engine.hand_geoms[0]):
                return "left_hand"
            if geom == int(engine.hand_geoms[1]):
                return "right_hand"
            body = int(model.geom_bodyid[geom])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if "wrist" in name:
                return "wrist"
            if "ankle" in name or "foot" in name:
                return "feet"
            if "knee" in name or "shin" in name:
                return "lower_legs"
            if "hip" in name or "thigh" in name:
                return "upper_legs"
            if any(token in name for token in ("torso", "pelvis", "waist")):
                return "torso_pelvis"
            return "other_robot"

        total_steps = 300 * engine.config.substeps
        for step in range(total_steps):
            frame = step / engine.config.substeps
            if frame < 30.0:
                target = targets["pregrasp"]
            elif frame < 70.0:
                amount = smoothstep((frame - 30.0) / 40.0)
                target = (1.0 - amount) * targets["pregrasp"] + amount * targets[
                    "grasp"
                ]
            elif frame < 100.0:
                amount = smoothstep((frame - 70.0) / 30.0)
                target = (1.0 - amount) * targets["grasp"] + amount * targets["squeeze"]
            elif frame < 160.0:
                amount = smoothstep((frame - 100.0) / 60.0)
                target = (1.0 - amount) * targets["squeeze"] + amount * targets["lift"]
            else:
                target = targets["lift"]
            data.ctrl[:] = np.clip(
                target,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(model, data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                pair = {first, second}
                if not pair & object_geoms:
                    continue
                other = second if first in object_geoms else first
                if int(model.geom_bodyid[other]) not in engine.robot_bodies:
                    continue
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, force)
                group_impulse[group(other)] += float(np.linalg.norm(force[:3])) * float(
                    model.opt.timestep
                )
            if step % engine.config.substeps == engine.config.substeps - 1:
                physical = generator._physical(data)
                trace["qpos"].append(data.qpos.copy())
                trace["qvel"].append(data.qvel.copy())
                trace["ctrl"].append(data.ctrl.copy())
                trace["object_position_m"].append(
                    data.qpos[engine.object_qpos : engine.object_qpos + 3].copy()
                )
                trace["hand_contact"].append(physical["hand_contact"].copy())
                trace["root_height_m"].append(float(data.qpos[engine.robot_qpos + 2]))
                trace["torso_tilt_rad"].append(float(physical["tilt"]))
                trace["forbidden"].append(bool(physical["forbidden"]))
        simulation_arrays = {key: np.asarray(value) for key, value in trace.items()}
        object_position = simulation_arrays["object_position_m"]
        baseline = float(np.median(object_position[10:30, 2]))
        lift = object_position[:, 2] - baseline
        lifted = lift >= 0.05
        longest = 0
        running = 0
        for value in lifted:
            running = running + 1 if value else 0
            longest = max(longest, running)
        manipulation_valid = (
            float(np.max(lift)) >= 0.05
            and longest >= 10
            and group_impulse["left_hand"] + group_impulse["right_hand"] > 1.0
            and sum(
                group_impulse[name]
                for name in (
                    "feet",
                    "lower_legs",
                    "upper_legs",
                    "torso_pelvis",
                    "other_robot",
                )
            )
            < 0.5
        )
        simulation_report = {
            "settled_object_height_m": baseline,
            "peak_lift_m": float(np.max(lift)),
            "sustained_lift_frames": int(longest),
            "hand_contact_frames": np.sum(
                np.any(simulation_arrays["hand_contact"], axis=1)
            ).item(),
            "object_impulse_ns": group_impulse,
            "minimum_root_height_m": float(np.min(simulation_arrays["root_height_m"])),
            "maximum_torso_tilt_deg": float(
                np.rad2deg(np.max(simulation_arrays["torso_tilt_rad"]))
            ),
            "forbidden_frames": int(np.sum(simulation_arrays["forbidden"])),
            "manipulation_probe_pass": manipulation_valid,
        }
        print("SIMULATION", json.dumps(simulation_report), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"), **solutions, **simulation_arrays
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "full-body fixed-foot bilateral flat-hand IK sweep",
                "object_position_m": ik.object_position.tolist(),
                "forward": ik.forward.tolist(),
                "lateral": ik.lateral.tolist(),
                "solutions": reports,
                "simulation": simulation_report,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
