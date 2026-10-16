"""Kinematic support/clearance repair for the early diffusion reference.

The repaired floating-base path is a *reference only*.  Dynamic validation is
performed separately with a free MuJoCo root; this module never integrates or
teleports the physical robot or object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch

from orcs.tasks.uolm.dynamics_refinement import _LEG_WAIST_JOINTS, _np
from orcs.tasks.uolm.generated_reference import GeneratedReference

__all__ = [
    "EarlyReferenceRepairConfig",
    "EarlyReferenceRepairResult",
    "KinematicRepairMetrics",
    "repair_early_reference",
]


@dataclass(frozen=True, slots=True)
class EarlyReferenceRepairWeights:
    root_position: float = 25.0
    root_orientation: float = 20.0
    lower_body: float = 8.0
    velocity: float = 0.8
    acceleration: float = 0.15
    support_projection: float = 40.0
    foot_height: float = 20.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class EarlyReferenceRepairConfig:
    start_frame: int = 0
    end_frame: int = 60
    knot_frames: tuple[int, ...] = (0, 20, 40, 60)
    wrist_clearance_margin_m: float = 0.005
    support_foot_distance_m: float = 0.015
    maximum_foot_penetration_m: float = 0.008
    support_projection_margin_m: float = 0.12
    minimum_root_height_m: float = 0.439
    maximum_torso_tilt_deg: float = 50.0
    root_horizontal_bound_m: float = 0.08
    root_vertical_lower_bound_m: float = -0.02
    root_vertical_upper_bound_m: float = 0.20
    root_roll_pitch_bound_rad: float = 0.15
    lower_body_bound_rad: float = 0.35
    population: int = 48
    elite_count: int = 8
    generations: int = 20
    minimum_std_fraction: float = 0.05
    seed: int = 142
    weights: EarlyReferenceRepairWeights = field(
        default_factory=EarlyReferenceRepairWeights
    )


@dataclass(slots=True)
class KinematicRepairMetrics:
    feasible: bool
    first_forbidden_terrain_contact_frame: int | None
    forbidden_terrain_contact_bodies: dict[str, list[int]]
    minimum_left_wrist_clearance_m: float
    minimum_right_wrist_clearance_m: float
    minimum_root_height_m: float
    maximum_torso_tilt_deg: float
    maximum_joint_limit_violation_rad: float
    minimum_left_foot_distance_m: float
    minimum_right_foot_distance_m: float
    maximum_left_foot_distance_m: float
    maximum_right_foot_distance_m: float
    unsupported_frames: list[int]
    minimum_support_projection_margin_m: float
    correction_rms: float
    correction_max: float
    root_position_correction_rms_m: float
    root_position_correction_max_m: float
    root_orientation_correction_rms_rad: float
    root_orientation_correction_max_rad: float
    lower_body_correction_rms_rad: float
    lower_body_correction_max_rad: float
    root_height_correction_by_frame_m: dict[int, float]
    root_height_by_frame_m: dict[int, float]
    foot_support_frames: dict[str, list[int]]
    objective: float
    violation: float

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(slots=True)
class EarlyReferenceRepairResult:
    trajectory: torch.Tensor
    correction: np.ndarray
    raw: KinematicRepairMetrics
    repaired: KinematicRepairMetrics
    history: list[dict[str, Any]]
    parameters: np.ndarray
    joint_names: tuple[str, ...]
    balance_joint_names: tuple[str, ...]
    optimizer: dict[str, Any]


@dataclass(slots=True)
class _Evaluation:
    metrics: KinematicRepairMetrics
    correction: np.ndarray


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(out, left, right)
    return out


def _roll_pitch_quat(roll: float, pitch: float) -> np.ndarray:
    half_roll, half_pitch = 0.5 * roll, 0.5 * pitch
    q_roll = np.asarray(
        [np.cos(half_roll), np.sin(half_roll), 0.0, 0.0], dtype=np.float64
    )
    q_pitch = np.asarray(
        [np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0], dtype=np.float64
    )
    return _quat_multiply(q_pitch, q_roll)


def _quat_to_columns6d(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)[:, :2].T.reshape(6)


def _smooth_correction(parameters: np.ndarray, length: int) -> np.ndarray:
    """Two interior controls with C1-zero splice boundaries."""
    controls = parameters.reshape(2, 20)
    t = np.linspace(0.0, 1.0, length)[:, None]
    blend = np.clip(3.0 * t * t - 2.0 * t * t * t, 0.0, 1.0)
    interior = (1.0 - blend) * controls[0] + blend * controls[1]
    envelope = 16.0 * t * t * (1.0 - t) * (1.0 - t)
    correction = envelope * interior
    correction[0] = 0.0
    correction[-1] = 0.0
    return correction


class _KinematicRepair:
    def __init__(
        self,
        model: mujoco.MjModel,
        reference: GeneratedReference,
        config: EarlyReferenceRepairConfig,
    ) -> None:
        self.model = model
        self.reference = reference
        self.config = config
        self.root_pos = _np(reference.root_pos_local[0])
        self.root_quat = _np(reference.root_quat_wxyz[0])
        self.joint = _np(reference.joint_pos[0])
        if config.start_frame != 0:
            raise ValueError("the first support-repair prototype must begin at frame 0")
        if config.end_frame >= len(self.joint):
            raise ValueError("repair interval exceeds reference horizon")
        if config.knot_frames != (0, 20, 40, config.end_frame):
            raise ValueError("knot_frames must match the four-knot repair interval")

        robot_free = self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
        self.robot_qpos = int(model.jnt_qposadr[robot_free])
        hinge_ids = sorted(
            (
                index
                for index in range(model.njnt)
                if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
                and self._name(mujoco.mjtObj.mjOBJ_JOINT, index).startswith("robot/")
            ),
            key=lambda index: int(model.jnt_qposadr[index]),
        )
        self.joint_ids = np.asarray(hinge_ids, dtype=np.int32)
        self.joint_qpos = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.joint_limits = model.jnt_range[self.joint_ids].copy()
        self.joint_names = tuple(
            self._name(mujoco.mjtObj.mjOBJ_JOINT, index).split("/", 1)[-1]
            for index in hinge_ids
        )
        self.balance_indexes = np.asarray(
            [self.joint_names.index(name) for name in _LEG_WAIST_JOINTS],
            dtype=np.int32,
        )
        self.terrain_geoms = tuple(
            index
            for index in range(model.ngeom)
            if self._name(mujoco.mjtObj.mjOBJ_GEOM, index) == "terrain"
            or self._name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index]))
            == "terrain"
        )
        self.robot_geoms = tuple(
            index
            for index in range(model.ngeom)
            if self._name(
                mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])
            ).startswith("robot/")
            and (int(model.geom_contype[index]) or int(model.geom_conaffinity[index]))
        )
        self.foot_bodies = (
            self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link"),
            self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link"),
        )
        self.allowed_bodies = set(self.foot_bodies)
        self.foot_geoms = tuple(
            tuple(g for g in self.robot_geoms if int(model.geom_bodyid[g]) == body)
            for body in self.foot_bodies
        )
        self.wrist_geoms = tuple(
            tuple(
                geom
                for geom in self.robot_geoms
                if side
                in self._name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
                and any(
                    token
                    in self._name(
                        mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
                    )
                    for token in ("wrist", "hand")
                )
            )
            for side in ("left_", "right_")
        )
        if any(not geoms for geoms in self.wrist_geoms):
            raise ValueError("compiled model has no wrist/hand collision geometry")
        self.torso_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        self.pelvis_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        self.data = mujoco.MjData(model)
        self.fromto = np.empty(6, dtype=np.float64)
        self.data.qpos[self.robot_qpos : self.robot_qpos + 3] = self.root_pos[0]
        self.data.qpos[self.robot_qpos + 3 : self.robot_qpos + 7] = self.root_quat[0]
        self.data.qpos[self.joint_qpos] = self.joint[0]
        mujoco.mj_forward(self.model, self.data)
        initial_torso = self.data.xmat[self.torso_body].reshape(3, 3)
        self.torso_local_up = initial_torso.T @ np.asarray([0.0, 0.0, 1.0])

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"compiled model is missing {name!r}")
        return index

    def _name(self, kind: mujoco.mjtObj, index: int) -> str:
        return _name(self.model, kind, index)

    def _terrain_distance(self, geoms: tuple[int, ...]) -> float:
        if not geoms:
            return float("inf")
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model, self.data, geom, terrain, 1.0, self.fromto
                )
            )
            for geom in geoms
            for terrain in self.terrain_geoms
        )

    def decode(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        length = self.config.end_frame + 1
        correction = _smooth_correction(parameters, length)
        root_pos = self.root_pos.copy()
        root_quat = self.root_quat.copy()
        joint = self.joint.copy()
        root_pos[:length] += correction[:, :3]
        for frame in range(length):
            delta = _roll_pitch_quat(correction[frame, 3], correction[frame, 4])
            root_quat[frame] = _quat_multiply(delta, self.root_quat[frame])
            root_quat[frame] /= np.linalg.norm(root_quat[frame])
        joint[:length, self.balance_indexes] += correction[:, 5:]
        return root_pos, root_quat, joint, correction

    def evaluate(self, parameters: np.ndarray) -> _Evaluation:
        cfg = self.config
        root_pos, root_quat, joint, correction = self.decode(parameters)
        length = cfg.end_frame + 1
        forbidden: dict[str, list[int]] = {}
        wrist = np.full((length, 2), np.inf, dtype=np.float64)
        feet = np.full((length, 2), np.inf, dtype=np.float64)
        torso_tilt = np.zeros(length, dtype=np.float64)
        support_margin = np.full(length, np.inf, dtype=np.float64)
        unsupported: list[int] = []

        for frame in range(length):
            data = self.data
            data.qpos[self.robot_qpos : self.robot_qpos + 3] = root_pos[frame]
            data.qpos[self.robot_qpos + 3 : self.robot_qpos + 7] = root_quat[frame]
            data.qpos[self.joint_qpos] = joint[frame]
            mujoco.mj_forward(self.model, data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if first not in self.terrain_geoms and second not in self.terrain_geoms:
                    continue
                other = second if first in self.terrain_geoms else first
                body = int(self.model.geom_bodyid[other])
                name = self._name(mujoco.mjtObj.mjOBJ_BODY, body)
                if name.startswith("robot/") and body not in self.allowed_bodies:
                    forbidden.setdefault(name, []).append(frame)
            wrist[frame] = [self._terrain_distance(geoms) for geoms in self.wrist_geoms]
            feet[frame] = [self._terrain_distance(geoms) for geoms in self.foot_geoms]
            torso_up = data.xmat[self.torso_body].reshape(3, 3) @ self.torso_local_up
            torso_tilt[frame] = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
            if frame >= 3 and float(np.min(feet[frame])) > cfg.support_foot_distance_m:
                unsupported.append(frame)
            foot_xy = data.xpos[np.asarray(self.foot_bodies), :2]
            lower = foot_xy.min(axis=0) - cfg.support_projection_margin_m
            upper = foot_xy.max(axis=0) + cfg.support_projection_margin_m
            com_xy = data.subtree_com[self.pelvis_body, :2]
            support_margin[frame] = float(
                np.min(np.minimum(com_xy - lower, upper - com_xy))
            )

        below = np.maximum(self.joint_limits[:, 0] - joint[:length], 0.0)
        above = np.maximum(joint[:length] - self.joint_limits[:, 1], 0.0)
        joint_violation = float(np.max(below + above))
        contact_frames = sorted({f for frames in forbidden.values() for f in frames})
        wrist_deficit = np.maximum(cfg.wrist_clearance_margin_m - wrist, 0.0)
        root_deficit = np.maximum(cfg.minimum_root_height_m - root_pos[:length, 2], 0.0)
        tilt_deficit = np.maximum(
            torso_tilt - np.deg2rad(cfg.maximum_torso_tilt_deg), 0.0
        )
        support_deficit = np.maximum(
            np.min(feet[3:], axis=1) - cfg.support_foot_distance_m, 0.0
        )
        foot_penetration_deficit = np.maximum(
            -cfg.maximum_foot_penetration_m - feet, 0.0
        )
        projection_deficit = np.maximum(-support_margin, 0.0)
        violation = (
            100.0 * len(contact_frames)
            + 5.0e4 * float(np.sum(wrist_deficit))
            + 1.0e4 * float(np.sum(support_deficit))
            + 1.0e4 * float(np.sum(foot_penetration_deficit))
            + 1.0e4 * float(np.sum(root_deficit))
            + 1.0e3 * float(np.sum(tilt_deficit))
            + 1.0e3 * float(np.sum(projection_deficit))
            + 1.0e5 * joint_violation
        )
        feasible = bool(
            not contact_frames
            and np.min(wrist) >= cfg.wrist_clearance_margin_m
            and not unsupported
            and np.min(feet) >= -cfg.maximum_foot_penetration_m
            and np.min(root_pos[:length, 2]) >= cfg.minimum_root_height_m
            and np.max(torso_tilt) <= np.deg2rad(cfg.maximum_torso_tilt_deg)
            and joint_violation <= 1.0e-9
            and np.min(support_margin) >= 0.0
        )

        weights = cfg.weights
        root_delta = correction[:, :3]
        orientation_delta = correction[:, 3:5]
        joint_delta = correction[:, 5:]
        velocity = np.diff(correction, axis=0)
        acceleration = np.diff(correction, n=2, axis=0)
        foot_excess = np.maximum(
            np.min(feet, axis=1) - cfg.support_foot_distance_m, 0.0
        )
        objective = float(
            weights.root_position * np.mean(np.square(root_delta))
            + weights.root_orientation * np.mean(np.square(orientation_delta))
            + weights.lower_body * np.mean(np.square(joint_delta))
            + weights.velocity * np.mean(np.square(velocity))
            + weights.acceleration * np.mean(np.square(acceleration))
            + weights.support_projection * np.mean(np.square(projection_deficit))
            + weights.foot_height * np.mean(np.square(foot_excess))
        )
        first_contact = contact_frames[0] if contact_frames else None
        diagnostic_frames = (0, 10, 20, 26, 30, 35, 40, 50, cfg.end_frame)
        metrics = KinematicRepairMetrics(
            feasible=feasible,
            first_forbidden_terrain_contact_frame=first_contact,
            forbidden_terrain_contact_bodies={
                body: sorted(set(frames)) for body, frames in sorted(forbidden.items())
            },
            minimum_left_wrist_clearance_m=float(np.min(wrist[:, 0])),
            minimum_right_wrist_clearance_m=float(np.min(wrist[:, 1])),
            minimum_root_height_m=float(np.min(root_pos[:length, 2])),
            maximum_torso_tilt_deg=float(np.rad2deg(np.max(torso_tilt))),
            maximum_joint_limit_violation_rad=joint_violation,
            minimum_left_foot_distance_m=float(np.min(feet[:, 0])),
            minimum_right_foot_distance_m=float(np.min(feet[:, 1])),
            maximum_left_foot_distance_m=float(np.max(feet[:, 0])),
            maximum_right_foot_distance_m=float(np.max(feet[:, 1])),
            unsupported_frames=unsupported,
            minimum_support_projection_margin_m=float(np.min(support_margin)),
            correction_rms=float(np.sqrt(np.mean(np.square(correction)))),
            correction_max=float(np.max(np.abs(correction))),
            root_position_correction_rms_m=float(
                np.sqrt(np.mean(np.square(root_delta)))
            ),
            root_position_correction_max_m=float(np.max(np.abs(root_delta))),
            root_orientation_correction_rms_rad=float(
                np.sqrt(np.mean(np.square(orientation_delta)))
            ),
            root_orientation_correction_max_rad=float(
                np.max(np.abs(orientation_delta))
            ),
            lower_body_correction_rms_rad=float(
                np.sqrt(np.mean(np.square(joint_delta)))
            ),
            lower_body_correction_max_rad=float(np.max(np.abs(joint_delta))),
            root_height_correction_by_frame_m={
                frame: float(correction[frame, 2])
                for frame in diagnostic_frames
                if frame < length
            },
            root_height_by_frame_m={
                frame: float(root_pos[frame, 2])
                for frame in diagnostic_frames
                if frame < length
            },
            foot_support_frames={
                "left": np.flatnonzero(feet[:, 0] <= cfg.support_foot_distance_m)
                .astype(int)
                .tolist(),
                "right": np.flatnonzero(feet[:, 1] <= cfg.support_foot_distance_m)
                .astype(int)
                .tolist(),
            },
            objective=objective,
            violation=float(violation),
        )
        return _Evaluation(metrics=metrics, correction=correction)


def _rank(evaluation: _Evaluation) -> tuple[int, float, float]:
    metrics = evaluation.metrics
    return (0 if metrics.feasible else 1, metrics.violation, metrics.objective)


def _validate(config: EarlyReferenceRepairConfig) -> None:
    if config.population < 2 or not 1 <= config.elite_count <= config.population:
        raise ValueError("invalid CEM population/elite count")
    if config.generations < 1:
        raise ValueError("generations must be positive")
    if config.end_frame <= 3:
        raise ValueError("repair interval is too short")


def repair_early_reference(
    model: mujoco.MjModel,
    trajectory: torch.Tensor,
    reference: GeneratedReference,
    config: EarlyReferenceRepairConfig | None = None,
) -> EarlyReferenceRepairResult:
    """Repair frames 0..60 while preserving arms/object and splice endpoints."""
    cfg = config or EarlyReferenceRepairConfig()
    _validate(cfg)
    if trajectory.shape != (1, 300, 47):
        raise ValueError(f"expected one [300,47] trajectory, got {trajectory.shape}")
    engine = _KinematicRepair(model, reference, cfg)
    dimensions = 2 * 20
    lower_one = np.asarray(
        [
            -cfg.root_horizontal_bound_m,
            -cfg.root_horizontal_bound_m,
            cfg.root_vertical_lower_bound_m,
            -cfg.root_roll_pitch_bound_rad,
            -cfg.root_roll_pitch_bound_rad,
        ]
        + [-cfg.lower_body_bound_rad] * 15,
        dtype=np.float64,
    )
    upper_one = np.asarray(
        [
            cfg.root_horizontal_bound_m,
            cfg.root_horizontal_bound_m,
            cfg.root_vertical_upper_bound_m,
            cfg.root_roll_pitch_bound_rad,
            cfg.root_roll_pitch_bound_rad,
        ]
        + [cfg.lower_body_bound_rad] * 15,
        dtype=np.float64,
    )
    lower = np.tile(lower_one, 2)
    upper = np.tile(upper_one, 2)
    span = upper - lower
    mean = np.zeros(dimensions, dtype=np.float64)
    mean[[2, 22]] = 0.06
    std = 0.22 * span
    minimum_std = cfg.minimum_std_fraction * span
    rng = np.random.default_rng(cfg.seed)
    zero = np.zeros(dimensions, dtype=np.float64)
    raw = engine.evaluate(zero)
    best_parameters = mean.copy()
    best = engine.evaluate(best_parameters)
    if _rank(raw) < _rank(best):
        best_parameters = zero.copy()
        best = raw
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
        mean = np.clip(0.2 * mean + 0.8 * elites.mean(axis=0), lower, upper)
        std = np.maximum(minimum_std, 0.2 * std + 0.8 * elites.std(axis=0))
        candidate_index = order[0]
        if _rank(evaluations[candidate_index]) < _rank(best):
            best = evaluations[candidate_index]
            best_parameters = samples[candidate_index].copy()
        history.append(
            {
                "generation": generation,
                "feasible": best.metrics.feasible,
                "violation": best.metrics.violation,
                "objective": best.metrics.objective,
                "first_forbidden_terrain_contact_frame": (
                    best.metrics.first_forbidden_terrain_contact_frame
                ),
                "minimum_left_wrist_clearance_m": (
                    best.metrics.minimum_left_wrist_clearance_m
                ),
                "minimum_right_wrist_clearance_m": (
                    best.metrics.minimum_right_wrist_clearance_m
                ),
                "unsupported_frame_count": len(best.metrics.unsupported_frames),
            }
        )

    root_pos, root_quat, joint, correction = engine.decode(best_parameters)
    repaired = trajectory.clone()
    length = cfg.end_frame + 1
    repaired[0, :length, 0:3] = torch.as_tensor(
        root_pos[:length], dtype=repaired.dtype, device=repaired.device
    )
    root_6d = np.stack([_quat_to_columns6d(quat) for quat in root_quat[:length]])
    repaired[0, :length, 3:9] = torch.as_tensor(
        root_6d, dtype=repaired.dtype, device=repaired.device
    )
    repaired[0, :length, 9:38] = torch.as_tensor(
        joint[:length], dtype=repaired.dtype, device=repaired.device
    )
    # Enforce the two invariants explicitly: arms and object are byte-identical.
    arm_indexes = [
        engine.joint_names.index(name)
        for name in (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
    ]
    if not torch.equal(repaired[..., 38:47], trajectory[..., 38:47]):
        raise AssertionError("reference repair changed the object trajectory")
    for index in arm_indexes:
        if not torch.equal(repaired[..., 9 + index], trajectory[..., 9 + index]):
            raise AssertionError("reference repair changed an arm trajectory")

    return EarlyReferenceRepairResult(
        trajectory=repaired,
        correction=correction,
        raw=raw.metrics,
        repaired=best.metrics,
        history=history,
        parameters=best_parameters,
        joint_names=engine.joint_names,
        balance_joint_names=_LEG_WAIST_JOINTS,
        optimizer={
            "method": "cross_entropy",
            "population": cfg.population,
            "elite_count": cfg.elite_count,
            "generations": cfg.generations,
            "dimensions": dimensions,
            "knot_frames": list(cfg.knot_frames),
            "free_interior_knots": 2,
            "weights": cfg.weights.as_dict(),
            "bounds": {
                "root_horizontal_m": cfg.root_horizontal_bound_m,
                "root_vertical_m": [
                    cfg.root_vertical_lower_bound_m,
                    cfg.root_vertical_upper_bound_m,
                ],
                "root_roll_pitch_rad": cfg.root_roll_pitch_bound_rad,
                "lower_body_rad": cfg.lower_body_bound_rad,
            },
            "splice": "zero value and zero first derivative at frames 0 and 60",
        },
    )
