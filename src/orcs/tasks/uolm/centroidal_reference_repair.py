"""Centroidal/contact-feasible repair of the early generated G1 reference.

The optimizer changes a desired floating-base/lower-body trajectory.  It never
writes a physical root during rollout.  MuJoCo is used here for kinematics,
mass properties, collision geometry, and centroidal quantities only; dynamic
validation remains a separate free-root rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch

from orcs.tasks.uolm.dynamics_refinement import _LEG_WAIST_JOINTS, _np
from orcs.tasks.uolm.generated_reference import GeneratedReference
from orcs.tasks.uolm.reference_repair import (
    _quat_multiply,
    _quat_to_columns6d,
    _roll_pitch_quat,
)
from orcs.tasks.uolm.support_diagnostics import _convex_hull, _polygon_margin

__all__ = [
    "CentroidalRepairConfig",
    "CentroidalRepairMetrics",
    "CentroidalRepairResult",
    "analyze_centroidal_reference",
    "extend_centroidal_reference",
    "repair_centroidal_reference",
]


@dataclass(frozen=True, slots=True)
class CentroidalRepairWeights:
    root_position: float = 20.0
    root_orientation: float = 12.0
    lower_body: float = 6.0
    velocity: float = 1.0
    acceleration: float = 0.08

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CentroidalRepairConfig:
    dynamics_end_frame: int = 40
    blend_end_frame: int = 60
    knot_frames: tuple[int, ...] = (0, 30, 60)
    repair_start_frame: int = 0
    optimize_first_knot: bool = True
    initialize_from_baseline: bool = False
    overlap_preservation_end_frame: int = 0
    overlap_preservation_weight: float = 200.0
    horizontal_projection_iterations: int = 1
    final_horizontal_projection_iterations: int | None = None
    contact_distance_m: float = 0.002
    maximum_foot_penetration_m: float = 0.008
    wrist_clearance_margin_m: float = 0.005
    cop_target_margin_m: float = 0.002
    cop_numerical_tolerance_m: float = 0.001
    friction_coefficient: float = 0.60
    unsupported_force_tolerance_n: float = 10.0
    unsupported_moment_tolerance_nm: float = 3.0
    maximum_vertical_force_weight: float = 2.5
    root_horizontal_bound_m: float = 0.06
    maximum_projected_horizontal_correction_m: float = 0.20
    root_vertical_bound_m: float = 0.08
    root_roll_pitch_bound_rad: float = 0.12
    lower_body_bound_rad: float = 0.25
    population: int = 40
    elite_count: int = 8
    generations: int = 12
    initial_std_fraction: float = 0.16
    minimum_std_fraction: float = 0.025
    seed: int = 242
    weights: CentroidalRepairWeights = field(default_factory=CentroidalRepairWeights)


@dataclass(slots=True)
class CentroidalRepairMetrics:
    feasible: bool
    support_state_by_frame: list[str]
    first_left_support_frame: int | None
    first_right_support_frame: int | None
    unsupported_frames: list[int]
    unsupported_force_violation_frames: list[int]
    minimum_required_vertical_force_n: float
    maximum_required_vertical_force_n: float
    minimum_cop_margin_m: float
    cop_outside_time_s: float
    minimum_roll_pitch_moment_margin_nm: float
    maximum_required_roll_pitch_moment_nm: float
    minimum_friction_margin_n: float
    friction_violation_time_s: float
    minimum_left_wrist_clearance_m: float
    minimum_right_wrist_clearance_m: float
    first_forbidden_terrain_contact_frame: int | None
    forbidden_terrain_contact_bodies: dict[str, list[int]]
    maximum_foot_penetration_m: float
    maximum_joint_limit_violation_rad: float
    minimum_root_height_m: float
    maximum_torso_tilt_deg: float
    root_linear_velocity_frame0_m_s: list[float]
    root_angular_velocity_frame0_rad_s: list[float]
    com_velocity_frame0_m_s: list[float]
    vertical_momentum_frame0_kg_m_s: float
    vertical_momentum_by_frame_kg_m_s: dict[int, float]
    required_vertical_force_by_frame_n: dict[int, float]
    cop_margin_by_frame_m: dict[int, float | None]
    friction_margin_by_frame_n: dict[int, float | None]
    root_height_by_frame_m: dict[int, float]
    correction_rms: float
    correction_max: float
    prior_objective: float
    violation: float

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(slots=True)
class CentroidalRepairResult:
    trajectory: torch.Tensor
    raw: CentroidalRepairMetrics
    kinematic: CentroidalRepairMetrics
    initial: CentroidalRepairMetrics
    repaired: CentroidalRepairMetrics
    correction: np.ndarray
    parameters: np.ndarray
    history: list[dict[str, Any]]
    joint_names: tuple[str, ...]
    optimizer: dict[str, Any]
    analytical_data: dict[str, np.ndarray]


@dataclass(slots=True)
class _Evaluation:
    metrics: CentroidalRepairMetrics
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_position: np.ndarray
    correction: np.ndarray
    data: dict[str, np.ndarray]


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def _smootherstep(value: np.ndarray) -> np.ndarray:
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _piecewise_correction(
    knots: np.ndarray, length: int, frames: tuple[int, ...] | None = None
) -> np.ndarray:
    """C2 interpolation through arbitrary temporal knots."""
    frames = frames or (0, 30, length - 1)
    if len(frames) != len(knots) or frames[0] < 0 or frames[-1] >= length:
        raise ValueError("knot frames do not match the correction array")
    if any(upper <= lower for lower, upper in zip(frames, frames[1:], strict=False)):
        raise ValueError("knot frames must be strictly increasing")
    result = np.zeros((length, knots.shape[1]), dtype=np.float64)
    for segment in range(len(frames) - 1):
        lower, upper = frames[segment : segment + 2]
        sample = np.arange(lower, upper + 1)
        phase = (sample - lower) / (upper - lower)
        blend = _smootherstep(phase)[:, None]
        result[sample] = (1.0 - blend) * knots[segment] + blend * knots[segment + 1]
    return result


def _differentiate(value: np.ndarray, dt: float) -> np.ndarray:
    result = np.empty_like(value)
    result[0] = (value[1] - value[0]) / dt
    result[1:-1] = (value[2:] - value[:-2]) / (2.0 * dt)
    result[-1] = (value[-1] - value[-2]) / dt
    return result


def _angular_velocity(quaternion: np.ndarray, dt: float) -> np.ndarray:
    result = np.empty((len(quaternion), 3), dtype=np.float64)
    conjugate = np.empty(4, dtype=np.float64)
    relative = np.empty(4, dtype=np.float64)
    for frame in range(len(quaternion)):
        if frame <= 1:
            lower, upper = 0, min(2, len(quaternion) - 1)
        elif frame == len(quaternion) - 1:
            lower, upper = max(0, len(quaternion) - 3), len(quaternion) - 1
        else:
            lower, upper = frame - 1, frame + 1
        duration = max((upper - lower) * dt, dt)
        mujoco.mju_negQuat(conjugate, quaternion[lower])
        mujoco.mju_mulQuat(relative, quaternion[upper], conjugate)
        mujoco.mju_quat2Vel(result[frame], relative, duration)
    return result


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


class _CentroidalRepair:
    def __init__(
        self,
        model: mujoco.MjModel,
        raw_trajectory: torch.Tensor,
        raw_reference: GeneratedReference,
        kinematic_trajectory: torch.Tensor,
        kinematic_reference: GeneratedReference,
        config: CentroidalRepairConfig,
    ) -> None:
        self.model = model
        self.raw_trajectory = raw_trajectory
        self.raw_reference = raw_reference
        self.kinematic_trajectory = kinematic_trajectory
        self.kinematic_reference = kinematic_reference
        self.config = config
        self.dt = float(raw_reference.dt)
        self.length = config.blend_end_frame + 1
        self.raw_root_position = _np(raw_reference.root_pos_local[0])
        self.raw_root_quaternion = _np(raw_reference.root_quat_wxyz[0])
        self.raw_joint_position = _np(raw_reference.joint_pos[0])
        self.kinematic_root_position = _np(kinematic_reference.root_pos_local[0])
        self.kinematic_root_quaternion = _np(kinematic_reference.root_quat_wxyz[0])
        self.kinematic_joint_position = _np(kinematic_reference.joint_pos[0])

        self.robot_free_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint"
        )
        self.robot_qpos = int(model.jnt_qposadr[self.robot_free_joint])
        self.robot_dof = int(model.jnt_dofadr[self.robot_free_joint])
        hinge_ids = sorted(
            (
                index
                for index in range(model.njnt)
                if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
                and _name(model, mujoco.mjtObj.mjOBJ_JOINT, index).startswith("robot/")
            ),
            key=lambda index: int(model.jnt_qposadr[index]),
        )
        self.joint_ids = np.asarray(hinge_ids, dtype=np.int32)
        self.joint_qpos = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.joint_dof = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.joint_limits = model.jnt_range[self.joint_ids].copy()
        self.joint_names = tuple(
            _name(model, mujoco.mjtObj.mjOBJ_JOINT, index).split("/", 1)[-1]
            for index in hinge_ids
        )
        self.balance_indexes = np.asarray(
            [self.joint_names.index(name) for name in _LEG_WAIST_JOINTS],
            dtype=np.int32,
        )
        self.robot_root_body = int(model.jnt_bodyid[self.robot_free_joint])
        self.mass = float(model.body_subtreemass[self.robot_root_body])
        self.gravity = np.asarray(model.opt.gravity, dtype=np.float64).copy()
        self.terrain_geoms = {
            index
            for index in range(model.ngeom)
            if _name(model, mujoco.mjtObj.mjOBJ_GEOM, index) == "terrain"
            or _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index]))
            == "terrain"
        }
        self.robot_bodies = {
            index
            for index in range(model.nbody)
            if _name(model, mujoco.mjtObj.mjOBJ_BODY, index).startswith("robot/")
        }
        self.foot_bodies = (
            self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link"),
            self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link"),
        )
        self.allowed_terrain_bodies = set(self.foot_bodies)
        self.foot_geoms = tuple(
            tuple(
                geom
                for geom in range(model.ngeom)
                if int(model.geom_bodyid[geom]) == body
                and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
            )
            for body in self.foot_bodies
        )
        self.sole_geoms = tuple(
            tuple(
                geom
                for geom in geoms
                if "foot" in _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                and "collision" in _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            )
            for geoms in self.foot_geoms
        )
        self.wrist_geoms = tuple(
            tuple(
                geom
                for geom in range(model.ngeom)
                if side
                in _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
                and any(
                    token
                    in _name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[geom]),
                    )
                    for token in ("wrist", "hand")
                )
                and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
            )
            for side in ("left_", "right_")
        )
        self.torso_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        self.data = mujoco.MjData(model)
        self.fromto = np.empty(6, dtype=np.float64)
        self.data.qpos[self.robot_qpos : self.robot_qpos + 3] = self.raw_root_position[
            0
        ]
        self.data.qpos[self.robot_qpos + 3 : self.robot_qpos + 7] = (
            self.raw_root_quaternion[0]
        )
        self.data.qpos[self.joint_qpos] = self.raw_joint_position[0]
        mujoco.mj_forward(model, self.data)
        initial_torso = self.data.xmat[self.torso_body].reshape(3, 3)
        self.torso_local_up = initial_torso.T @ np.asarray([0.0, 0.0, 1.0])
        self.base_root_position, self.base_root_quaternion, self.base_joint_position = (
            self._make_initial_reference()
        )

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"compiled model is missing {name!r}")
        return index

    def _terrain_distance(self, geoms: tuple[int, ...]) -> float:
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model, self.data, geom, terrain, 1.0, self.fromto
                )
            )
            for geom in geoms
            for terrain in self.terrain_geoms
        )

    @staticmethod
    def _interpolate(values: np.ndarray, source: np.ndarray) -> np.ndarray:
        frames = np.arange(len(values), dtype=np.float64)
        return np.stack(
            [
                np.interp(source, frames, values[:, axis])
                for axis in range(values.shape[1])
            ],
            axis=1,
        )

    def _make_initial_reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retimed collision repair used only to initialize the constrained search."""
        if self.config.initialize_from_baseline:
            return (
                self.kinematic_root_position.copy(),
                self.kinematic_root_quaternion.copy(),
                self.kinematic_joint_position.copy(),
            )
        root_position = self.kinematic_root_position.copy()
        root_quaternion = self.kinematic_root_quaternion.copy()
        joint_position = self.kinematic_joint_position.copy()
        end = self.config.dynamics_end_frame
        frame = np.arange(end + 1, dtype=np.float64)
        phase = np.clip((frame - 1.0) / (end - 1.0), 0.0, 1.0)
        source = end * (2.0 * phase**2 - phase**3)
        source[:2] = 0.0
        root_position[: end + 1] = self._interpolate(
            self.kinematic_root_position, source
        )
        joint_position[: end + 1, self.balance_indexes] = self._interpolate(
            self.kinematic_joint_position[:, self.balance_indexes], source
        )
        # Quaternion nlerp is sufficient here because adjacent repaired poses are close.
        lower = np.floor(source).astype(int)
        upper = np.minimum(lower + 1, len(root_quaternion) - 1)
        fraction = (source - lower)[:, None]
        first = self.kinematic_root_quaternion[lower]
        second = self.kinematic_root_quaternion[upper].copy()
        second[np.sum(first * second, axis=1) < 0.0] *= -1.0
        blended = (1.0 - fraction) * first + fraction * second
        root_quaternion[: end + 1] = blended / np.linalg.norm(
            blended, axis=1, keepdims=True
        )

        # Establish the earliest available sole contact with the smallest vertical shift.
        support_shift = np.zeros(self.length, dtype=np.float64)
        for index in range(end + 1):
            self._set_pose(
                root_position[index], root_quaternion[index], joint_position[index]
            )
            nearest = min(self._terrain_distance(geoms) for geoms in self.foot_geoms)
            support_shift[index] = -(nearest + 0.0005)
        root_position[: end + 1, 2] += support_shift[: end + 1]
        start_shift = support_shift[end]
        blend_frame = np.arange(end + 1, self.config.blend_end_frame + 1)
        blend_phase = (blend_frame - end) / (self.config.blend_end_frame - end)
        root_position[blend_frame, 2] += start_shift * (
            1.0 - _smootherstep(blend_phase)
        )
        return root_position, root_quaternion, joint_position

    def _set_pose(
        self, root_position: np.ndarray, root_quaternion: np.ndarray, joint: np.ndarray
    ) -> None:
        self.data.qpos[self.robot_qpos : self.robot_qpos + 3] = root_position
        self.data.qpos[self.robot_qpos + 3 : self.robot_qpos + 7] = root_quaternion
        self.data.qpos[self.joint_qpos] = joint
        mujoco.mj_forward(self.model, self.data)

    def decode(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        variables = 5 + len(self.balance_indexes)
        free_indexes = list(range(len(self.config.knot_frames) - 1))
        if not self.config.optimize_first_knot:
            free_indexes = free_indexes[1:]
        free = parameters.reshape(len(free_indexes), variables)
        knots = np.zeros((len(self.config.knot_frames), variables), dtype=np.float64)
        knots[free_indexes] = free
        correction = _piecewise_correction(knots, self.length, self.config.knot_frames)
        root_position = self.base_root_position.copy()
        root_quaternion = self.base_root_quaternion.copy()
        joint_position = self.base_joint_position.copy()
        root_position[: self.length] += correction[:, :3]
        for frame in range(self.length):
            delta = _roll_pitch_quat(correction[frame, 3], correction[frame, 4])
            root_quaternion[frame] = _quat_multiply(
                delta, self.base_root_quaternion[frame]
            )
            root_quaternion[frame] /= np.linalg.norm(root_quaternion[frame])
        joint_position[: self.length, self.balance_indexes] += correction[:, 5:]
        # Contact is an equality constraint on floating height.  Project to the
        # least vertical shift that establishes one sole contact, or both when
        # their height mismatch fits inside the penetration allowance.
        projection_start = self.config.repair_start_frame
        if projection_start:
            projection_start += 1
        projection_end = min(
            self.config.dynamics_end_frame + 2, self.config.blend_end_frame
        )
        for frame in range(projection_start, projection_end + 1):
            self._set_pose(
                root_position[frame], root_quaternion[frame], joint_position[frame]
            )
            distance = np.asarray(
                [self._terrain_distance(geoms) for geoms in self.foot_geoms]
            )
            if np.ptp(distance) <= self.config.maximum_foot_penetration_m - 0.0005:
                shift = -(float(np.max(distance)) + 0.0005)
            else:
                shift = -(float(np.min(distance)) + 0.0005)
            root_position[frame, 2] += shift
            correction[frame, 2] = (
                root_position[frame, 2] - self.base_root_position[frame, 2]
            )
        return root_position, root_quaternion, joint_position, correction

    def _footprint(self, active: np.ndarray) -> np.ndarray:
        points: list[np.ndarray] = []
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        for side, enabled in enumerate(active):
            if not enabled:
                continue
            for geom in self.sole_geoms[side]:
                center = self.data.geom_xpos[geom]
                rotation = self.data.geom_xmat[geom].reshape(3, 3)
                radius = float(self.model.geom_size[geom, 0])
                half_length = float(self.model.geom_size[geom, 1])
                axis = rotation[:, 2]
                for sign in (-1.0, 1.0):
                    endpoint = center + sign * half_length * axis
                    points.extend(endpoint[:2] + radius * circle)
        return _convex_hull(np.asarray(points)) if points else np.empty((0, 2))

    def evaluate_arrays(
        self,
        root_position: np.ndarray,
        root_quaternion: np.ndarray,
        joint_position: np.ndarray,
        correction: np.ndarray,
    ) -> _Evaluation:
        cfg = self.config
        length = self.length
        root_velocity = _differentiate(root_position[:length], self.dt)
        root_angular_velocity = _angular_velocity(root_quaternion[:length], self.dt)
        joint_velocity = _differentiate(joint_position[:length], self.dt)
        com_position = np.empty((length, 3), dtype=np.float64)
        com_velocity = np.empty_like(com_position)
        angular_momentum = np.empty_like(com_position)
        feet = np.empty((length, 2), dtype=np.float64)
        wrists = np.empty((length, 2), dtype=np.float64)
        support = np.zeros((length, 2), dtype=bool)
        torso_tilt = np.empty(length, dtype=np.float64)
        forbidden: dict[str, list[int]] = {}
        polygons: list[np.ndarray] = []
        support_center = np.full((length, 2), np.nan, dtype=np.float64)

        for frame in range(length):
            data = self.data
            data.qpos[self.robot_qpos : self.robot_qpos + 3] = root_position[frame]
            data.qpos[self.robot_qpos + 3 : self.robot_qpos + 7] = root_quaternion[
                frame
            ]
            data.qpos[self.joint_qpos] = joint_position[frame]
            data.qvel[self.robot_dof : self.robot_dof + 3] = root_velocity[frame]
            rotation = _rotation_matrix(root_quaternion[frame])
            data.qvel[self.robot_dof + 3 : self.robot_dof + 6] = (
                rotation.T @ root_angular_velocity[frame]
            )
            data.qvel[self.joint_dof] = joint_velocity[frame]
            mujoco.mj_forward(self.model, data)
            com_position[frame] = data.subtree_com[self.robot_root_body]
            com_velocity[frame] = data.subtree_linvel[self.robot_root_body]
            angular_momentum[frame] = data.subtree_angmom[self.robot_root_body]
            feet[frame] = [self._terrain_distance(geoms) for geoms in self.foot_geoms]
            wrists[frame] = [
                self._terrain_distance(geoms) for geoms in self.wrist_geoms
            ]
            support[frame] = (feet[frame] <= cfg.contact_distance_m) & (
                feet[frame] >= -cfg.maximum_foot_penetration_m
            )
            polygons.append(self._footprint(support[frame]))
            if len(polygons[-1]) >= 3:
                support_center[frame] = np.mean(polygons[-1], axis=0)
            torso_up = data.xmat[self.torso_body].reshape(3, 3) @ self.torso_local_up
            torso_tilt[frame] = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if first not in self.terrain_geoms and second not in self.terrain_geoms:
                    continue
                other = second if first in self.terrain_geoms else first
                body = int(self.model.geom_bodyid[other])
                if (
                    body in self.robot_bodies
                    and body not in self.allowed_terrain_bodies
                ):
                    name = _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                    forbidden.setdefault(name, []).append(frame)

        com_acceleration = _differentiate(com_velocity, self.dt)
        angular_momentum_rate = _differentiate(angular_momentum, self.dt)
        required_force = self.mass * (com_acceleration - self.gravity)
        required_support_moment = angular_momentum_rate + np.cross(
            com_position, required_force
        )
        cop = np.full((length, 2), np.nan, dtype=np.float64)
        cop_margin = np.full(length, np.nan, dtype=np.float64)
        friction_margin = np.full(length, np.nan, dtype=np.float64)
        supported = np.any(support, axis=1)
        for frame in range(length):
            if not supported[frame] or required_force[frame, 2] <= 1.0e-6:
                continue
            force = required_force[frame]
            com = com_position[frame]
            hdot = angular_momentum_rate[frame]
            cop[frame, 0] = com[0] - com[2] * force[0] / force[2] - hdot[1] / force[2]
            cop[frame, 1] = com[1] - com[2] * force[1] / force[2] + hdot[0] / force[2]
            cop_margin[frame] = _polygon_margin(cop[frame], polygons[frame])
            friction_margin[frame] = cfg.friction_coefficient * force[
                2
            ] - np.linalg.norm(force[:2])

        active = slice(0, cfg.dynamics_end_frame + 1)
        unsupported = np.flatnonzero(~supported[active])
        unsupported_force_norm = np.linalg.norm(required_force[active], axis=1)
        unsupported_moment_norm = np.linalg.norm(angular_momentum_rate[active], axis=1)
        unsupported_bad = np.flatnonzero(
            (~supported[active])
            & (
                (unsupported_force_norm > cfg.unsupported_force_tolerance_n)
                | (unsupported_moment_norm > cfg.unsupported_moment_tolerance_nm)
            )
        )
        supported_active = supported[active]
        force_z = required_force[active, 2]
        force_low = supported_active & (force_z < 0.0)
        force_high = supported_active & (
            force_z
            > cfg.maximum_vertical_force_weight * self.mass * abs(self.gravity[2])
        )
        cop_bad = supported_active & (
            ~np.isfinite(cop_margin[active])
            | (cop_margin[active] < -cfg.cop_numerical_tolerance_m)
        )
        friction_bad = supported_active & (
            ~np.isfinite(friction_margin[active]) | (friction_margin[active] < 0.0)
        )
        below = np.maximum(self.joint_limits[:, 0] - joint_position[:length], 0.0)
        above = np.maximum(joint_position[:length] - self.joint_limits[:, 1], 0.0)
        joint_violation = float(np.max(below + above))
        contact_frames = sorted({f for frames in forbidden.values() for f in frames})
        penetration = np.maximum(-feet - cfg.maximum_foot_penetration_m, 0.0)
        wrist_deficit = np.maximum(cfg.wrist_clearance_margin_m - wrists, 0.0)
        projected_horizontal_excess = np.maximum(
            np.abs(root_position[:length, :2] - self.raw_root_position[:length, :2])
            - cfg.maximum_projected_horizontal_correction_m,
            0.0,
        )
        second_foot_gap = np.maximum(
            np.max(feet[active], axis=1) - cfg.contact_distance_m, 0.0
        )
        cop_deficit = np.maximum(cfg.cop_target_margin_m - cop_margin[active], 0.0)
        cop_deficit[~np.isfinite(cop_deficit)] = 1.0
        friction_deficit = np.maximum(-friction_margin[active], 0.0)
        friction_deficit[~np.isfinite(friction_deficit)] = self.mass * abs(
            self.gravity[2]
        )
        violation = float(
            1.0e5 * len(unsupported_bad)
            + 1.0e5 * np.sum(force_low)
            + 1.0e5 * np.sum(force_high)
            + 1.0e4 * np.sum(cop_bad)
            + 1.0e4 * np.sum(friction_bad)
            + 2.0e3 * np.sum(cop_deficit[supported_active])
            + 2.0 * np.sum(friction_deficit[supported_active])
            + 2.0e4 * np.sum(penetration)
            + 5.0e4 * np.sum(second_foot_gap)
            + 5.0e4 * np.sum(wrist_deficit)
            + 1.0e5 * np.sum(projected_horizontal_excess)
            + 100.0 * len(contact_frames)
            + 1.0e5 * joint_violation
        )
        feasible = bool(
            not len(unsupported_bad)
            and not np.any(force_low)
            and not np.any(force_high)
            and not np.any(cop_bad)
            and not np.any(friction_bad)
            and not np.any(penetration > 0.0)
            and np.min(wrists) >= cfg.wrist_clearance_margin_m
            and not np.any(projected_horizontal_excess > 0.0)
            and not contact_frames
            and joint_violation <= 1.0e-9
        )

        root_delta = root_position[:length] - self.raw_root_position[:length]
        joint_delta = (
            joint_position[:length, self.balance_indexes]
            - self.raw_joint_position[:length, self.balance_indexes]
        )
        raw_root_velocity = _differentiate(self.raw_root_position[:length], self.dt)
        raw_joint_velocity = _differentiate(
            self.raw_joint_position[:length, self.balance_indexes], self.dt
        )
        weights = cfg.weights
        prior = float(
            weights.root_position * np.mean(np.square(root_delta))
            + weights.lower_body * np.mean(np.square(joint_delta))
            + weights.velocity
            * (
                np.mean(np.square(root_velocity - raw_root_velocity))
                + np.mean(
                    np.square(
                        joint_velocity[:, self.balance_indexes] - raw_joint_velocity
                    )
                )
            )
            + weights.acceleration
            * (
                np.mean(np.square(_differentiate(root_velocity, self.dt)))
                + np.mean(
                    np.square(
                        _differentiate(joint_velocity[:, self.balance_indexes], self.dt)
                    )
                )
            )
            + weights.root_orientation * np.mean(np.square(correction[:, 3:5]))
        )
        if cfg.overlap_preservation_end_frame > cfg.repair_start_frame:
            overlap = slice(
                cfg.repair_start_frame,
                cfg.overlap_preservation_end_frame + 1,
            )
            prior += cfg.overlap_preservation_weight * float(
                np.mean(np.square(correction[overlap]))
            )
        diagnostic_frames = (0, 5, 10, 15, 20, 25, 30, 34, 36, 40)
        support_state = [
            "both"
            if left and right
            else "left"
            if left
            else "right"
            if right
            else "none"
            for left, right in support[active]
        ]
        left_frames = np.flatnonzero(support[active, 0])
        right_frames = np.flatnonzero(support[active, 1])
        finite_cop = cop_margin[active][np.isfinite(cop_margin[active])]
        moment_margin = cop_margin[active] * np.maximum(force_z, 0.0)
        finite_moment_margin = moment_margin[np.isfinite(moment_margin)]
        finite_friction = friction_margin[active][np.isfinite(friction_margin[active])]
        vertical_momentum = self.mass * com_velocity[:, 2]
        metrics = CentroidalRepairMetrics(
            feasible=feasible,
            support_state_by_frame=support_state,
            first_left_support_frame=(
                int(left_frames[0]) if len(left_frames) else None
            ),
            first_right_support_frame=(
                int(right_frames[0]) if len(right_frames) else None
            ),
            unsupported_frames=unsupported.astype(int).tolist(),
            unsupported_force_violation_frames=unsupported_bad.astype(int).tolist(),
            minimum_required_vertical_force_n=float(np.min(force_z)),
            maximum_required_vertical_force_n=float(np.max(force_z)),
            minimum_cop_margin_m=(
                float(np.min(finite_cop)) if len(finite_cop) else float("nan")
            ),
            cop_outside_time_s=float(np.sum(cop_bad) * self.dt),
            minimum_roll_pitch_moment_margin_nm=(
                float(np.min(finite_moment_margin))
                if len(finite_moment_margin)
                else float("nan")
            ),
            maximum_required_roll_pitch_moment_nm=float(
                np.max(np.linalg.norm(required_support_moment[active, :2], axis=1))
            ),
            minimum_friction_margin_n=(
                float(np.min(finite_friction)) if len(finite_friction) else float("nan")
            ),
            friction_violation_time_s=float(np.sum(friction_bad) * self.dt),
            minimum_left_wrist_clearance_m=float(np.min(wrists[:, 0])),
            minimum_right_wrist_clearance_m=float(np.min(wrists[:, 1])),
            first_forbidden_terrain_contact_frame=(
                contact_frames[0] if contact_frames else None
            ),
            forbidden_terrain_contact_bodies={
                name: sorted(set(frames)) for name, frames in sorted(forbidden.items())
            },
            maximum_foot_penetration_m=float(max(0.0, -np.min(feet))),
            maximum_joint_limit_violation_rad=joint_violation,
            minimum_root_height_m=float(np.min(root_position[:length, 2])),
            maximum_torso_tilt_deg=float(np.rad2deg(np.max(torso_tilt))),
            root_linear_velocity_frame0_m_s=root_velocity[0].tolist(),
            root_angular_velocity_frame0_rad_s=root_angular_velocity[0].tolist(),
            com_velocity_frame0_m_s=com_velocity[0].tolist(),
            vertical_momentum_frame0_kg_m_s=float(vertical_momentum[0]),
            vertical_momentum_by_frame_kg_m_s={
                frame: float(vertical_momentum[frame]) for frame in diagnostic_frames
            },
            required_vertical_force_by_frame_n={
                frame: float(required_force[frame, 2]) for frame in diagnostic_frames
            },
            cop_margin_by_frame_m={
                frame: (
                    float(cop_margin[frame]) if np.isfinite(cop_margin[frame]) else None
                )
                for frame in diagnostic_frames
            },
            friction_margin_by_frame_n={
                frame: (
                    float(friction_margin[frame])
                    if np.isfinite(friction_margin[frame])
                    else None
                )
                for frame in diagnostic_frames
            },
            root_height_by_frame_m={
                frame: float(root_position[frame, 2]) for frame in diagnostic_frames
            },
            correction_rms=float(np.sqrt(np.mean(np.square(correction)))),
            correction_max=float(np.max(np.abs(correction))),
            prior_objective=prior,
            violation=violation,
        )
        data_out = {
            "root_position_m": root_position[:length].copy(),
            "root_quaternion_wxyz": root_quaternion[:length].copy(),
            "joint_position_rad": joint_position[:length].copy(),
            "root_linear_velocity_m_s": root_velocity,
            "root_angular_velocity_rad_s": root_angular_velocity,
            "com_position_m": com_position,
            "com_velocity_m_s": com_velocity,
            "com_acceleration_m_s2": com_acceleration,
            "vertical_momentum_kg_m_s": vertical_momentum,
            "centroidal_angular_momentum_kg_m2_s": angular_momentum,
            "centroidal_angular_momentum_rate_nm": angular_momentum_rate,
            "required_force_world_n": required_force,
            "required_support_moment_world_nm": required_support_moment,
            "required_cop_xy_m": cop,
            "cop_margin_m": cop_margin,
            "friction_margin_n": friction_margin,
            "foot_terrain_distance_m": feet,
            "foot_support": support,
            "support_center_xy_m": support_center,
            "wrist_terrain_clearance_m": wrists,
        }
        return _Evaluation(
            metrics=metrics,
            root_position=root_position,
            root_quaternion=root_quaternion,
            joint_position=joint_position,
            correction=correction,
            data=data_out,
        )

    def evaluate(self, parameters: np.ndarray) -> _Evaluation:
        decoded = self.decode(parameters)
        evaluation = self.evaluate_arrays(*decoded)
        for _ in range(self.config.horizontal_projection_iterations):
            evaluation = self.evaluate_arrays(*self._project_horizontal(evaluation))
        return evaluation

    def _project_horizontal(
        self, evaluation: _Evaluation
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project root x/y so the required COP targets support-polygon center."""
        root = evaluation.root_position.copy()
        quaternion = evaluation.root_quaternion.copy()
        joint = evaluation.joint_position.copy()
        correction = evaluation.correction.copy()
        data = evaluation.data
        end = self.config.dynamics_end_frame
        center = data["support_center_xy_m"][: end + 1]
        if not np.all(np.isfinite(center)):
            return root, quaternion, joint, correction
        com = data["com_position_m"][: end + 1]
        force = data["required_force_world_n"][: end + 1]
        hdot = data["centroidal_angular_momentum_rate_nm"][: end + 1]
        current_root_velocity = _differentiate(root[: end + 1, :2], self.dt)
        current_root_acceleration = _differentiate(current_root_velocity, self.dt)
        com_acceleration = data["com_acceleration_m_s2"][: end + 1, :2]
        relative_acceleration = com_acceleration - current_root_acceleration
        desired_com_acceleration = np.empty_like(relative_acceleration)
        safe_fz = np.maximum(force[:, 2], 1.0)
        safe_height = np.maximum(com[:, 2], 0.1)
        desired_com_acceleration[:, 0] = (
            safe_fz * (com[:, 0] - center[:, 0]) - hdot[:, 1]
        ) / (self.mass * safe_height)
        desired_com_acceleration[:, 1] = (
            safe_fz * (com[:, 1] - center[:, 1]) + hdot[:, 0]
        ) / (self.mass * safe_height)
        desired_root_acceleration = desired_com_acceleration - relative_acceleration

        start = self.config.repair_start_frame
        projection_end = min(end + 2, self.config.blend_end_frame)
        extended_acceleration = np.vstack(
            [
                desired_root_acceleration,
                np.repeat(desired_root_acceleration[-1:], projection_end - end, axis=0),
            ]
        )
        for axis in range(2):
            if start:
                # Integrate only an acceleration correction.  Zero displacement
                # and velocity at the overlap anchor preserve the established
                # early reference with C1 continuity.
                delta_acceleration = extended_acceleration[:, axis] - _differentiate(
                    _differentiate(
                        evaluation.root_position[: projection_end + 1, axis],
                        self.dt,
                    ),
                    self.dt,
                )
                delta = np.zeros(projection_end + 1, dtype=np.float64)
                delta_velocity = 0.0
                for frame in range(start + 1, projection_end + 1):
                    delta_velocity += delta_acceleration[frame - 1] * self.dt
                    delta[frame] = delta[frame - 1] + delta_velocity * self.dt
                root[start : projection_end + 1, axis] += delta[
                    start : projection_end + 1
                ]
                continue
            zero_velocity_path = np.empty(projection_end + 1, dtype=np.float64)
            zero_velocity_path[0] = root[0, axis]
            zero_velocity_path[1] = (
                zero_velocity_path[0]
                + 0.5 * extended_acceleration[0, axis] * self.dt**2
            )
            for frame in range(1, projection_end):
                zero_velocity_path[frame + 1] = (
                    2.0 * zero_velocity_path[frame]
                    - zero_velocity_path[frame - 1]
                    + extended_acceleration[frame, axis] * self.dt**2
                )
            time = np.arange(projection_end + 1) * self.dt
            target = evaluation.root_position[: projection_end + 1, axis]
            initial_velocity = float(
                np.dot(time, target - zero_velocity_path)
                / max(np.dot(time, time), self.dt**2)
            )
            root[: projection_end + 1, axis] = (
                zero_velocity_path + initial_velocity * time
            )

        delta_end = (
            root[projection_end, :2] - evaluation.root_position[projection_end, :2]
        )
        blend_frames = np.arange(projection_end + 1, self.config.blend_end_frame + 1)
        blend_phase = (blend_frames - projection_end) / (
            self.config.blend_end_frame - projection_end
        )
        root[blend_frames, :2] += (
            delta_end * (1.0 - _smootherstep(blend_phase))[:, None]
        )
        correction[:, :2] = (
            root[: self.length, :2] - self.base_root_position[: self.length, :2]
        )
        return root, quaternion, joint, correction

    def evaluate_reference(self, reference: GeneratedReference) -> _Evaluation:
        root = _np(reference.root_pos_local[0])
        quat = _np(reference.root_quat_wxyz[0])
        joint = _np(reference.joint_pos[0])
        correction = np.zeros((self.length, 20), dtype=np.float64)
        return self.evaluate_arrays(root, quat, joint, correction)


def _rank(evaluation: _Evaluation) -> tuple[int, float, float]:
    return (
        0 if evaluation.metrics.feasible else 1,
        evaluation.metrics.violation,
        evaluation.metrics.prior_objective,
    )


def analyze_centroidal_reference(
    model: mujoco.MjModel,
    raw_trajectory: torch.Tensor,
    raw_reference: GeneratedReference,
    kinematic_trajectory: torch.Tensor,
    kinematic_reference: GeneratedReference,
    reference: GeneratedReference,
    config: CentroidalRepairConfig | None = None,
) -> CentroidalRepairMetrics:
    engine = _CentroidalRepair(
        model,
        raw_trajectory,
        raw_reference,
        kinematic_trajectory,
        kinematic_reference,
        config or CentroidalRepairConfig(),
    )
    return engine.evaluate_reference(reference).metrics


def repair_centroidal_reference(
    model: mujoco.MjModel,
    raw_trajectory: torch.Tensor,
    raw_reference: GeneratedReference,
    kinematic_trajectory: torch.Tensor,
    kinematic_reference: GeneratedReference,
    config: CentroidalRepairConfig | None = None,
) -> CentroidalRepairResult:
    cfg = config or CentroidalRepairConfig()
    return _optimize_centroidal_reference(
        model,
        raw_trajectory,
        raw_reference,
        kinematic_trajectory,
        kinematic_reference,
        cfg,
    )


def extend_centroidal_reference(
    model: mujoco.MjModel,
    raw_trajectory: torch.Tensor,
    raw_reference: GeneratedReference,
    baseline_trajectory: torch.Tensor,
    baseline_reference: GeneratedReference,
    config: CentroidalRepairConfig,
) -> CentroidalRepairResult:
    """Repair a later window while preserving an established early baseline."""
    if config.repair_start_frame <= 0 or config.optimize_first_knot:
        raise ValueError("an extension requires a fixed nonzero first knot")
    if not config.initialize_from_baseline:
        raise ValueError("an extension must initialize from its repaired baseline")
    return _optimize_centroidal_reference(
        model,
        raw_trajectory,
        raw_reference,
        baseline_trajectory,
        baseline_reference,
        config,
    )


def _optimize_centroidal_reference(
    model: mujoco.MjModel,
    raw_trajectory: torch.Tensor,
    raw_reference: GeneratedReference,
    baseline_trajectory: torch.Tensor,
    baseline_reference: GeneratedReference,
    cfg: CentroidalRepairConfig,
) -> CentroidalRepairResult:
    if cfg.knot_frames[0] != cfg.repair_start_frame:
        raise ValueError("the first knot must equal repair_start_frame")
    if cfg.knot_frames[-1] != cfg.blend_end_frame:
        raise ValueError("the final knot must equal blend_end_frame")
    if cfg.dynamics_end_frame >= cfg.blend_end_frame:
        raise ValueError("blend_end_frame must follow dynamics_end_frame")
    if cfg.population < 2 or not 1 <= cfg.elite_count <= cfg.population:
        raise ValueError("invalid CEM population/elite count")
    engine = _CentroidalRepair(
        model,
        raw_trajectory,
        raw_reference,
        baseline_trajectory,
        baseline_reference,
        cfg,
    )
    variables = 5 + len(engine.balance_indexes)
    free_knots = len(cfg.knot_frames) - 1 - int(not cfg.optimize_first_knot)
    dimensions = free_knots * variables
    single_bound = np.asarray(
        [
            cfg.root_horizontal_bound_m,
            cfg.root_horizontal_bound_m,
            cfg.root_vertical_bound_m,
            cfg.root_roll_pitch_bound_rad,
            cfg.root_roll_pitch_bound_rad,
        ]
        + [cfg.lower_body_bound_rad] * len(engine.balance_indexes),
        dtype=np.float64,
    )
    bound = np.tile(single_bound, free_knots)
    lower, upper = -bound, bound
    mean = np.zeros(dimensions, dtype=np.float64)
    std = cfg.initial_std_fraction * (upper - lower)
    minimum_std = cfg.minimum_std_fraction * (upper - lower)
    rng = np.random.default_rng(cfg.seed)
    zero = np.zeros(dimensions, dtype=np.float64)
    raw = engine.evaluate_reference(raw_reference)
    kinematic = engine.evaluate_reference(baseline_reference)
    initial = engine.evaluate(zero)
    best_parameters = zero.copy()
    best = initial
    history: list[dict[str, Any]] = []
    for generation in range(cfg.generations):
        samples = np.clip(
            rng.normal(mean, std, size=(cfg.population, dimensions)), lower, upper
        )
        samples[0] = best_parameters
        samples[1] = zero
        evaluations = [engine.evaluate(sample) for sample in samples]
        order = sorted(
            range(cfg.population), key=lambda index: _rank(evaluations[index])
        )
        elites = samples[order[: cfg.elite_count]]
        mean = np.clip(0.25 * mean + 0.75 * elites.mean(axis=0), lower, upper)
        std = np.maximum(minimum_std, 0.25 * std + 0.75 * elites.std(axis=0))
        candidate = evaluations[order[0]]
        if _rank(candidate) < _rank(best):
            best = candidate
            best_parameters = samples[order[0]].copy()
        history.append(
            {
                "generation": generation,
                "feasible": best.metrics.feasible,
                "violation": best.metrics.violation,
                "prior_objective": best.metrics.prior_objective,
                "minimum_cop_margin_m": best.metrics.minimum_cop_margin_m,
                "cop_outside_time_s": best.metrics.cop_outside_time_s,
                "minimum_friction_margin_n": best.metrics.minimum_friction_margin_n,
                "unsupported_force_violation_count": len(
                    best.metrics.unsupported_force_violation_frames
                ),
                "minimum_wrist_clearance_m": min(
                    best.metrics.minimum_left_wrist_clearance_m,
                    best.metrics.minimum_right_wrist_clearance_m,
                ),
            }
        )

    final_iterations = (
        cfg.horizontal_projection_iterations
        if cfg.final_horizontal_projection_iterations is None
        else cfg.final_horizontal_projection_iterations
    )
    if final_iterations < cfg.horizontal_projection_iterations:
        raise ValueError("final horizontal projection cannot undo search projections")
    for _ in range(final_iterations - cfg.horizontal_projection_iterations):
        best = engine.evaluate_arrays(*engine._project_horizontal(best))

    repaired = baseline_trajectory.clone()
    length = cfg.blend_end_frame + 1
    repaired[0, :length, 0:3] = torch.as_tensor(
        best.root_position[:length], dtype=repaired.dtype, device=repaired.device
    )
    encoded = np.stack(
        [_quat_to_columns6d(quat) for quat in best.root_quaternion[:length]]
    )
    repaired[0, :length, 3:9] = torch.as_tensor(
        encoded, dtype=repaired.dtype, device=repaired.device
    )
    repaired[0, :length, 9:38] = torch.as_tensor(
        best.joint_position[:length], dtype=repaired.dtype, device=repaired.device
    )
    if cfg.repair_start_frame:
        # The overlap anchor is fixed in the optimizer.  Preserve the source
        # tensor literally as well, avoiding quaternion decode/re-encode noise
        # in an already validated prefix.
        repaired[:, : cfg.repair_start_frame + 1] = baseline_trajectory[
            :, : cfg.repair_start_frame + 1
        ]
    arm_indexes = [
        index
        for index, name in enumerate(engine.joint_names)
        if name not in _LEG_WAIST_JOINTS
    ]
    if not torch.equal(repaired[..., 38:47], raw_trajectory[..., 38:47]):
        raise AssertionError("centroidal repair changed the object trajectory")
    for index in arm_indexes:
        if not torch.equal(repaired[..., 9 + index], raw_trajectory[..., 9 + index]):
            raise AssertionError("centroidal repair changed a non-balance joint")
    return CentroidalRepairResult(
        trajectory=repaired,
        raw=raw.metrics,
        kinematic=kinematic.metrics,
        initial=initial.metrics,
        repaired=best.metrics,
        correction=best.correction,
        parameters=best_parameters,
        history=history,
        joint_names=engine.joint_names,
        optimizer={
            "method": "cross_entropy",
            "population": cfg.population,
            "elite_count": cfg.elite_count,
            "generations": cfg.generations,
            "dimensions": dimensions,
            "knot_frames": list(cfg.knot_frames),
            "free_knots": free_knots,
            "variables_per_knot": ("root xyz + root roll/pitch + 12 leg + 3 waist"),
            "weights": cfg.weights.as_dict(),
            "bounds": {
                "root_horizontal_m": cfg.root_horizontal_bound_m,
                "root_vertical_m": cfg.root_vertical_bound_m,
                "root_roll_pitch_rad": cfg.root_roll_pitch_bound_rad,
                "lower_body_rad": cfg.lower_body_bound_rad,
            },
            "initialization": (
                "existing centroidal-repaired baseline"
                if cfg.initialize_from_baseline
                else "local C2 retiming of the collision-feasible repair with "
                "frames 0-1 held and minimum vertical shift to first sole contact"
            ),
        },
        analytical_data=best.data,
    )
