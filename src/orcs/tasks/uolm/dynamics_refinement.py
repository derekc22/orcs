"""Staged simulator-backed refinement of a diffusion UOLM reference.

Stage A searches only compact leg/waist corrections and must produce a valid
300-frame free-root execution. Stage B optimizes the existing arm spline only
after Stage A passes, and every manipulation candidate is hard rejected if it
violates the same balance gates. The object is written once at initialization
and is thereafter governed exclusively by MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco
import numpy as np

from orcs.tasks.uolm.generated_reference import GeneratedReference

__all__ = [
    "DynamicsRefinementConfig",
    "DynamicsRefinementResult",
    "RolloutResult",
    "detect_pickup_window",
    "refine_generated_reference",
]

_LEG_WAIST_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

_ARM_JOINTS = (
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


@dataclass(frozen=True, slots=True)
class BalanceWeights:
    joint_tracking: float = 5.0
    root_tracking: float = 20.0
    diffusion_prior: float = 10.0
    initial_correction: float = 100.0
    correction_velocity: float = 0.02
    correction_acceleration: float = 0.002
    actuator_saturation: float = 20.0
    unintended_terrain_contact: float = 10.0
    joint_limits: float = 200.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ManipulationWeights:
    object_position: float = 4.0
    object_orientation: float = 0.5
    final_position: float = 80.0
    final_orientation: float = 4.0
    arm_prior: float = 0.5
    correction_velocity: float = 0.03
    correction_acceleration: float = 0.01
    joint_limits: float = 200.0
    actuator_saturation: float = 12.0
    unintended_collision: float = 8.0
    penetration: float = 400.0

    def with_arm_prior(self, value: float) -> "ManipulationWeights":
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["arm_prior"] = value
        return ManipulationWeights(**values)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class DynamicsRefinementConfig:
    control_dt: float = 0.02
    physics_dt: float = 0.005
    min_window_frames: int = 60
    max_window_frames: int = 120
    approach_frames: int = 15
    lift_threshold_m: float = 0.04
    displacement_threshold_m: float = 0.04

    balance_num_knots: int = 5
    balance_optimize_initial_knot: bool = True
    balance_initial_knot_bound_rad: float = 0.15
    balance_initial_knot_std_rad: float = 0.05
    balance_initial_knot_minimum_std_rad: float = 0.01
    balance_correction_bound_rad: float = 0.45
    balance_initial_std_rad: float = 0.12
    balance_minimum_std_rad: float = 0.015
    balance_population: int = 24
    balance_elite_count: int = 6
    balance_generations: int = 14
    balance_weights: BalanceWeights = field(default_factory=BalanceWeights)

    arm_num_knots: int = 5
    arm_correction_bound_rad: float = 0.65
    arm_initial_std_rad: float = 0.16
    arm_minimum_std_rad: float = 0.025
    arm_population: int = 24
    arm_elite_count: int = 6
    arm_generations: int = 12
    manipulation_weights: ManipulationWeights = field(
        default_factory=ManipulationWeights
    )
    loose_arm_prior: float = 0.05

    root_height_fraction: float = 0.55
    max_torso_tilt_deg: float = 50.0
    hard_penetration_m: float = 0.02
    penetration_allowance_m: float = 0.003
    joint_limit_margin_rad: float = 0.03
    severe_saturation_usage: float = 0.98
    severe_saturation_actuator_fraction: float = 0.20
    severe_saturation_duration_s: float = 0.30
    sustained_nonfoot_terrain_contact_s: float = 0.10
    maximum_balance_rms_correction_rad: float = 0.25
    invalid_score: float = 1.0e6
    run_stage_b: bool = True
    seed: int = 42

    @property
    def substeps(self) -> int:
        ratio = self.control_dt / self.physics_dt
        rounded = round(ratio)
        if abs(ratio - rounded) > 1.0e-9 or rounded < 1:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        return rounded


@dataclass(slots=True)
class RolloutResult:
    object_pos: np.ndarray
    object_quat_wxyz: np.ndarray
    robot_joint_pos: np.ndarray
    root_pos: np.ndarray
    torso_tilt_rad: np.ndarray
    hand_contact: np.ndarray
    foot_contact: np.ndarray
    metrics: dict[str, Any]
    valid: bool
    invalid_reason: str | None
    first_invalid_frame: int | None
    objective: float = float("inf")
    objective_terms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicsRefinementResult:
    window: tuple[int, int]
    joint_names: tuple[str, ...]
    balance_joint_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    raw: RolloutResult
    balance: RolloutResult
    manipulation: RolloutResult | None
    normal_prior_manipulation: RolloutResult | None
    loose_prior_manipulation: RolloutResult | None
    balance_joint_target: np.ndarray
    candidate_joint_target: np.ndarray | None
    status: str
    stage_a_passed: bool
    selected_arm_prior: float | None
    optimizer: dict[str, Any]
    refined_object_pos: np.ndarray | None = None
    refined_object_quat_wxyz: np.ndarray | None = None


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def detect_pickup_window(
    object_pos: np.ndarray,
    *,
    min_frames: int = 60,
    max_frames: int = 120,
    approach_frames: int = 15,
    lift_threshold_m: float = 0.04,
    displacement_threshold_m: float = 0.04,
) -> tuple[int, int]:
    """Return an inclusive approach/lift/initial-transport frame interval."""
    pos = np.asarray(object_pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 2:
        raise ValueError(f"object_pos must have shape [T,3], got {pos.shape}")
    if not (2 <= min_frames <= max_frames <= len(pos)):
        raise ValueError("invalid pickup-window bounds")
    anchor_count = min(30, max(3, len(pos) // 10))
    anchor = np.median(pos[:anchor_count], axis=0)
    lift = pos[:, 2] - anchor[2]
    displacement = np.linalg.norm(pos - anchor, axis=1)
    intended = (lift >= lift_threshold_m) | (displacement >= displacement_threshold_m)
    intended[:anchor_count] = False
    sustained = np.convolve(intended.astype(np.int32), np.ones(5), "same") >= 4
    indexes = np.flatnonzero(sustained)
    first = int(indexes[0]) if len(indexes) else int(np.argmax(displacement))
    start = max(0, first - approach_frames)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    end = min(len(pos) - 1, start + max_frames - 1)
    earliest = min(len(pos) - 1, start + min_frames - 1)
    if end > earliest:
        quiet = np.convolve((speed < 0.002).astype(np.int32), np.ones(8), "same")
        settled = np.flatnonzero((np.arange(len(speed)) >= earliest) & (quiet >= 7))
        if len(settled):
            end = min(end, int(settled[0]) + 1)
    return start, end


def _catmull_rom_basis(length: int, knots: int) -> np.ndarray:
    """Interpolating C1 cubic basis for evenly spaced temporal knots."""
    if knots < 4 or length < 2:
        raise ValueError("at least four knots and two frames are required")
    basis = np.empty((length, knots), dtype=np.float64)
    grid = np.linspace(0.0, knots - 1, length)
    for row, value in enumerate(grid):
        segment = min(int(np.floor(value)), knots - 2)
        t = value - segment
        t2, t3 = t * t, t * t * t
        weights = np.asarray(
            (
                -0.5 * t + t2 - 0.5 * t3,
                1.0 - 2.5 * t2 + 1.5 * t3,
                0.5 * t + 2.0 * t2 - 1.5 * t3,
                -0.5 * t2 + 0.5 * t3,
            )
        )
        basis[row] = 0.0
        for index, weight in zip(
            (segment - 1, segment, segment + 1, segment + 2), weights, strict=True
        ):
            basis[row, min(max(index, 0), knots - 1)] += weight
    basis[0] = 0.0
    basis[0, 0] = 1.0
    basis[-1] = 0.0
    basis[-1, -1] = 1.0
    return basis


def _spline_correction(
    length: int,
    num_joints: int,
    indexes: np.ndarray,
    num_knots: int,
    parameters: np.ndarray,
    *,
    optimize_first: bool = False,
    optimize_last: bool = False,
) -> np.ndarray:
    knots = np.zeros((num_knots, len(indexes)), dtype=np.float64)
    free_knots = list(range(num_knots))
    if not optimize_first:
        free_knots.pop(0)
    if not optimize_last:
        free_knots.pop()
    knots[free_knots] = parameters.reshape(len(free_knots), len(indexes))
    correction = np.zeros((length, num_joints), dtype=np.float64)
    correction[:, indexes] = _catmull_rom_basis(length, num_knots) @ knots
    return correction


class _DynamicRollout:
    def __init__(
        self,
        model: mujoco.MjModel,
        reference: GeneratedReference,
        config: DynamicsRefinementConfig,
    ) -> None:
        self.model = model
        self.reference = reference
        self.config = config
        model.opt.timestep = config.physics_dt
        self.reference_joint = _np(reference.joint_pos[0])
        self.reference_root = _np(reference.root_pos_local[0])
        self.reference_object_pos = _np(reference.object_pos_local[0])
        self.reference_object_quat = _np(reference.object_quat_wxyz[0])
        self.reference_mj_qvel = _np(reference.mujoco_qvel[0])
        self.reference_object_lin_vel = _np(reference.object_lin_vel_w[0])
        self.reference_object_ang_vel = _np(reference.object_ang_vel_w[0])
        self.horizon = self.reference_joint.shape[0]

        self.robot_free_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint"
        )
        self.object_free_joint = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, "object/floating_base_joint"
        )
        self.robot_qpos = int(model.jnt_qposadr[self.robot_free_joint])
        self.robot_dof = int(model.jnt_dofadr[self.robot_free_joint])
        self.object_qpos = int(model.jnt_qposadr[self.object_free_joint])
        self.object_dof = int(model.jnt_dofadr[self.object_free_joint])
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
        self.joint_names = tuple(
            self._name(mujoco.mjtObj.mjOBJ_JOINT, index).split("/", 1)[-1]
            for index in hinge_ids
        )
        if len(self.joint_names) != self.reference_joint.shape[1]:
            raise ValueError("compiled model and generated reference joint mismatch")
        self.joint_qpos = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.joint_dof = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.joint_limits = model.jnt_range[self.joint_ids].copy()
        self.balance_indexes = np.asarray(
            [self.joint_names.index(name) for name in _LEG_WAIST_JOINTS]
        )
        self.arm_indexes = np.asarray(
            [self.joint_names.index(name) for name in _ARM_JOINTS]
        )
        self.actuator_qpos = np.asarray(
            [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        )
        self.actuator_joint_indexes = np.asarray(
            [np.flatnonzero(self.joint_qpos == qpos)[0] for qpos in self.actuator_qpos]
        )
        self.force_scale = np.maximum(
            np.max(np.abs(model.actuator_forcerange), axis=1), 1.0e-9
        )

        self.object_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "object/object")
        self.torso_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
        self.hand_geoms = np.asarray(
            [
                self._id(mujoco.mjtObj.mjOBJ_GEOM, "robot/left_hand_collision"),
                self._id(mujoco.mjtObj.mjOBJ_GEOM, "robot/right_hand_collision"),
            ]
        )
        self.object_geoms = {
            index
            for index in range(model.ngeom)
            if int(model.geom_bodyid[index]) == self.object_body
            and int(model.geom_contype[index]) != 0
        }
        self.terrain_geoms = {
            index
            for index in range(model.ngeom)
            if self._name(mujoco.mjtObj.mjOBJ_GEOM, index) == "terrain"
            or self._name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index]))
            == "terrain"
        }
        self.robot_bodies = {
            index
            for index in range(model.nbody)
            if self._name(mujoco.mjtObj.mjOBJ_BODY, index).startswith("robot/")
        }
        self.wrist_terrain_geoms = tuple(
            tuple(
                geom
                for geom in range(model.ngeom)
                if side
                in self._name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
                and any(
                    token
                    in self._name(
                        mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
                    )
                    for token in ("wrist", "hand")
                )
                and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
            )
            for side in ("left_", "right_")
        )
        self.foot_bodies = (
            self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/left_ankle_roll_link"),
            self._id(mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link"),
        )
        self.allowed_terrain_bodies = set(self.foot_bodies)
        self.fall_terrain_bodies = {
            body
            for body in self.robot_bodies
            if any(
                token in self._name(mujoco.mjtObj.mjOBJ_BODY, body)
                for token in (
                    "pelvis",
                    "torso",
                    "head",
                    "hip_",
                    "knee_",
                    "ankle_pitch",
                )
            )
        }
        self.minimum_root_height = (
            config.root_height_fraction * self.reference_root[0, 2]
        )
        self.maximum_torso_tilt = np.deg2rad(config.max_torso_tilt_deg)
        self.severe_saturation_steps = max(
            1, round(config.severe_saturation_duration_s / config.physics_dt)
        )
        self.sustained_terrain_steps = max(
            1,
            round(config.sustained_nonfoot_terrain_contact_s / config.physics_dt),
        )

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"compiled refinement model is missing {name!r}")
        return index

    def _name(self, kind: mujoco.mjtObj, index: int) -> str:
        return mujoco.mj_id2name(self.model, kind, index) or ""

    @staticmethod
    def _local_omega(quat_wxyz: np.ndarray, omega_world: np.ndarray) -> np.ndarray:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quat_wxyz)
        return rotation.reshape(3, 3).T @ omega_world

    def _new_initialized_data(self) -> mujoco.MjData:
        """Create the exact generated frame-0 robot/object physical state."""
        data = mujoco.MjData(self.model)
        data.qpos[self.robot_qpos : self.robot_qpos + 3] = self.reference_root[0]
        data.qpos[self.robot_qpos + 3 : self.robot_qpos + 7] = _np(
            self.reference.root_quat_wxyz[0, 0]
        )
        data.qpos[self.joint_qpos] = self.reference_joint[0]
        data.qvel[self.robot_dof : self.robot_dof + 35] = self.reference_mj_qvel[0]
        data.qpos[self.object_qpos : self.object_qpos + 3] = self.reference_object_pos[
            0
        ]
        data.qpos[self.object_qpos + 3 : self.object_qpos + 7] = (
            self.reference_object_quat[0]
        )
        data.qvel[self.object_dof : self.object_dof + 3] = (
            self.reference_object_lin_vel[0]
        )
        data.qvel[self.object_dof + 3 : self.object_dof + 6] = self._local_omega(
            self.reference_object_quat[0], self.reference_object_ang_vel[0]
        )
        mujoco.mj_forward(self.model, data)
        return data

    def rollout(
        self, joint_target: np.ndarray, *, stop_on_invalid: bool = False
    ) -> RolloutResult:
        if joint_target.shape != self.reference_joint.shape:
            raise ValueError("joint target has the wrong shape")
        data = self._new_initialized_data()
        initial_torso_rotation = data.xmat[self.torso_body].reshape(3, 3).copy()
        torso_local_up = initial_torso_rotation.T @ np.asarray([0.0, 0.0, 1.0])
        initial_foot_heights = {
            self._name(mujoco.mjtObj.mjOBJ_BODY, body): float(data.xpos[body, 2])
            for body in sorted(self.allowed_terrain_bodies)
        }
        initial_foot_contacts: set[str] = set()
        initial_nonfoot_terrain_contacts: set[str] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            pair = {first, second}
            if not pair & self.terrain_geoms:
                continue
            other = second if first in self.terrain_geoms else first
            body = int(self.model.geom_bodyid[other])
            if body not in self.robot_bodies:
                continue
            name = self._name(mujoco.mjtObj.mjOBJ_BODY, body)
            if body in self.allowed_terrain_bodies:
                initial_foot_contacts.add(name)
            else:
                initial_nonfoot_terrain_contacts.add(name)

        count = self.horizon
        object_pos = np.empty((count, 3), dtype=np.float64)
        object_quat = np.empty((count, 4), dtype=np.float64)
        actual_joint = np.empty_like(joint_target)
        root_pos = np.empty((count, 3), dtype=np.float64)
        torso_tilt = np.empty(count, dtype=np.float64)
        hand_contact = np.zeros((count, 2), dtype=bool)
        foot_contact = np.zeros((count, 2), dtype=bool)
        terrain_contact_steps: dict[str, int] = {}
        hand_contact_steps = np.zeros(2, dtype=np.int64)
        other_object_contact_steps = 0
        penetration_steps = 0
        max_penetration = 0.0
        max_penetration_pair = ("", "")
        max_usage = 0.0
        minimum_wrist_clearance = np.full(2, np.inf, dtype=np.float64)
        saturation_sq = 0.0
        severe_run = 0
        maximum_severe_run = 0
        forbidden_terrain_steps = 0
        forbidden_terrain_run = 0
        maximum_forbidden_terrain_run = 0
        first_invalid_frame: int | None = None
        invalid_reason: str | None = None
        invalid_terrain_bodies: list[str] = []
        invalid_root_height = float("nan")
        invalid_torso_tilt = float("nan")
        invalid_penetration = float("nan")
        frames_simulated = 1

        target_below = np.maximum(self.joint_limits[:, 0] - joint_target, 0.0)
        target_above = np.maximum(joint_target - self.joint_limits[:, 1], 0.0)
        max_target_violation = float(np.max(target_below + target_above))
        if max_target_violation > self.config.joint_limit_margin_rad:
            first_invalid_frame = 0
            invalid_reason = "catastrophic_joint_target_limit"

        def record(frame: int) -> None:
            object_pos[frame] = data.qpos[self.object_qpos : self.object_qpos + 3]
            object_quat[frame] = data.qpos[self.object_qpos + 3 : self.object_qpos + 7]
            actual_joint[frame] = data.qpos[self.joint_qpos]
            root_pos[frame] = data.qpos[self.robot_qpos : self.robot_qpos + 3]
            rotation = data.xmat[self.torso_body].reshape(3, 3)
            torso_up = rotation @ torso_local_up
            torso_tilt[frame] = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
            fromto = np.empty(6, dtype=np.float64)
            for side, geoms in enumerate(self.wrist_terrain_geoms):
                if geoms:
                    clearance = min(
                        float(
                            mujoco.mj_geomDistance(
                                self.model, data, geom, terrain, 1.0, fromto
                            )
                        )
                        for geom in geoms
                        for terrain in self.terrain_geoms
                    )
                    minimum_wrist_clearance[side] = min(
                        minimum_wrist_clearance[side], clearance
                    )

        record(0)
        if first_invalid_frame is not None and stop_on_invalid:
            for frame in range(1, count):
                record(frame)
        else:
            halted = False
            for frame in range(1, count):
                target = joint_target[frame, self.actuator_joint_indexes]
                data.ctrl[:] = np.clip(
                    target,
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
                for _ in range(self.config.substeps):
                    mujoco.mj_step(self.model, data)
                    active_hands = np.zeros(2, dtype=bool)
                    active_other_object = False
                    active_terrain_bodies: set[int] = set()
                    penetrating = False
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        first, second = int(contact.geom1), int(contact.geom2)
                        pair = {first, second}
                        if pair & self.object_geoms:
                            other = second if first in self.object_geoms else first
                            if other in self.hand_geoms:
                                side = int(np.flatnonzero(self.hand_geoms == other)[0])
                                active_hands[side] = True
                            elif (
                                int(self.model.geom_bodyid[other]) in self.robot_bodies
                            ):
                                active_other_object = True
                        if pair & self.terrain_geoms:
                            other = second if first in self.terrain_geoms else first
                            body = int(self.model.geom_bodyid[other])
                            if body in self.robot_bodies:
                                active_terrain_bodies.add(body)
                        depth = max(0.0, -float(contact.dist))
                        if depth > max_penetration:
                            max_penetration = depth
                            max_penetration_pair = (
                                self._name(mujoco.mjtObj.mjOBJ_GEOM, first),
                                self._name(mujoco.mjtObj.mjOBJ_GEOM, second),
                            )
                        penetrating |= depth > self.config.penetration_allowance_m
                    hand_contact_steps += active_hands.astype(np.int64)
                    other_object_contact_steps += int(active_other_object)
                    penetration_steps += int(penetrating)
                    hand_contact[frame] |= active_hands
                    for side, body in enumerate(self.foot_bodies):
                        foot_contact[frame, side] |= body in active_terrain_bodies
                    for body in active_terrain_bodies:
                        name = self._name(mujoco.mjtObj.mjOBJ_BODY, body)
                        terrain_contact_steps[name] = (
                            terrain_contact_steps.get(name, 0) + 1
                        )

                    usage = np.abs(data.actuator_force) / self.force_scale
                    max_usage = max(max_usage, float(np.max(usage)))
                    saturation_sq += float(
                        np.square(np.maximum(usage - 0.9, 0.0)).mean()
                    )
                    severe = (
                        np.mean(usage >= self.config.severe_saturation_usage)
                        >= self.config.severe_saturation_actuator_fraction
                    )
                    severe_run = severe_run + 1 if severe else 0
                    maximum_severe_run = max(maximum_severe_run, severe_run)

                    rotation = data.xmat[self.torso_body].reshape(3, 3)
                    torso_up = rotation @ torso_local_up
                    tilt = np.arccos(np.clip(torso_up[2], -1.0, 1.0))
                    forbidden = active_terrain_bodies - self.allowed_terrain_bodies
                    forbidden_terrain_steps += int(bool(forbidden))
                    forbidden_terrain_run = (
                        forbidden_terrain_run + 1 if forbidden else 0
                    )
                    maximum_forbidden_terrain_run = max(
                        maximum_forbidden_terrain_run, forbidden_terrain_run
                    )
                    fall_terrain = forbidden & self.fall_terrain_bodies
                    reason = None
                    if data.qpos[self.robot_qpos + 2] < self.minimum_root_height:
                        reason = "root_height"
                    elif tilt > self.maximum_torso_tilt:
                        reason = "torso_tilt"
                    elif fall_terrain:
                        reason = "fall_body_terrain_contact"
                    elif forbidden_terrain_run >= self.sustained_terrain_steps:
                        reason = "sustained_nonfoot_terrain_contact"
                    elif max_penetration > self.config.hard_penetration_m:
                        reason = "pathological_penetration"
                    elif severe_run >= self.severe_saturation_steps:
                        reason = "sustained_actuator_saturation"
                    if reason is not None and first_invalid_frame is None:
                        first_invalid_frame = frame
                        invalid_reason = reason
                        invalid_terrain_bodies = sorted(
                            self._name(mujoco.mjtObj.mjOBJ_BODY, body)
                            for body in forbidden
                        )
                        invalid_root_height = float(data.qpos[self.robot_qpos + 2])
                        invalid_torso_tilt = float(tilt)
                        invalid_penetration = float(max_penetration)
                        if stop_on_invalid:
                            halted = True
                            break
                record(frame)
                frames_simulated = frame + 1
                if halted:
                    for remaining in range(frame + 1, count):
                        record(remaining)
                    break

        valid = first_invalid_frame is None
        goal_pos = self.reference_object_pos[-1]
        goal_quat = self.reference_object_quat[-1]
        lift = object_pos[:, 2] - object_pos[0, 2]
        tracking = actual_joint - joint_target
        diagnostic_frame = min(26, count - 1)
        denominator = max(1, (frames_simulated - 1) * self.config.substeps)
        terrain_durations = {
            name: steps * self.config.physics_dt
            for name, steps in sorted(terrain_contact_steps.items())
        }
        metrics: dict[str, Any] = {
            "minimum_root_height_m": float(np.min(root_pos[:frames_simulated, 2])),
            "final_root_height_m": float(root_pos[-1, 2]),
            "maximum_torso_tilt_rad": float(np.max(torso_tilt[:frames_simulated])),
            "maximum_torso_tilt_deg": float(
                np.rad2deg(np.max(torso_tilt[:frames_simulated]))
            ),
            "terrain_contacts_by_body_s": terrain_durations,
            "pelvis_or_torso_terrain_contact": bool(
                any(
                    name in terrain_durations
                    for name in ("robot/pelvis", "robot/torso_link")
                )
            ),
            "fall": not valid,
            "full_horizon_upright": valid,
            "invalid_reason": invalid_reason,
            "first_invalid_frame": first_invalid_frame,
            "first_invalid_root_height_m": invalid_root_height,
            "first_invalid_torso_tilt_deg": float(np.rad2deg(invalid_torso_tilt)),
            "first_invalid_terrain_bodies": invalid_terrain_bodies,
            "first_invalid_penetration_m": invalid_penetration,
            "frames_simulated": frames_simulated,
            "root_height_at_frame_26_m": float(root_pos[diagnostic_frame, 2]),
            "torso_tilt_at_frame_26_deg": float(
                np.rad2deg(torso_tilt[diagnostic_frame])
            ),
            "initial_foot_heights_m": initial_foot_heights,
            "initial_foot_contact_bodies": sorted(initial_foot_contacts),
            "initial_nonfoot_terrain_contact_bodies": sorted(
                initial_nonfoot_terrain_contacts
            ),
            "max_actuator_usage": float(max_usage),
            "minimum_left_wrist_clearance_m": float(minimum_wrist_clearance[0]),
            "minimum_right_wrist_clearance_m": float(minimum_wrist_clearance[1]),
            "max_severe_saturation_duration_s": float(
                maximum_severe_run * self.config.physics_dt
            ),
            "nonfoot_terrain_contact_duration_s": float(
                forbidden_terrain_steps * self.config.physics_dt
            ),
            "max_sustained_nonfoot_terrain_contact_s": float(
                maximum_forbidden_terrain_run * self.config.physics_dt
            ),
            "max_box_lift_m": float(max(0.0, np.max(lift))),
            "max_object_displacement_m": float(
                np.max(np.linalg.norm(object_pos - object_pos[0], axis=1))
            ),
            "final_goal_position_error_m": float(
                np.linalg.norm(object_pos[-1] - goal_pos)
            ),
            "final_goal_orientation_error_rad": float(
                _quat_angle(object_quat[-1:], goal_quat[None])[0]
            ),
            "left_hand_contact_duration_s": float(
                hand_contact_steps[0] * self.config.physics_dt
            ),
            "right_hand_contact_duration_s": float(
                hand_contact_steps[1] * self.config.physics_dt
            ),
            "other_robot_object_contact_duration_s": float(
                other_object_contact_steps * self.config.physics_dt
            ),
            "max_penetration_m": float(max_penetration),
            "max_penetration_geoms": list(max_penetration_pair),
            "penetrating_contact_duration_s": float(
                penetration_steps * self.config.physics_dt
            ),
            "robot_joint_tracking_rms_rad": float(
                np.sqrt(np.mean(np.square(tracking[:frames_simulated])))
            ),
            "root_tracking_rms_m": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            root_pos[:frames_simulated]
                            - self.reference_root[:frames_simulated]
                        )
                    )
                )
            ),
            "maximum_target_joint_limit_violation_rad": max_target_violation,
            "_saturation_sq": float(saturation_sq / denominator),
        }
        return RolloutResult(
            object_pos=object_pos,
            object_quat_wxyz=object_quat,
            robot_joint_pos=actual_joint,
            root_pos=root_pos,
            torso_tilt_rad=torso_tilt,
            hand_contact=hand_contact,
            foot_contact=foot_contact,
            metrics=metrics,
            valid=valid,
            invalid_reason=invalid_reason,
            first_invalid_frame=first_invalid_frame,
        )


def _correction_statistics(
    rollout: RolloutResult,
    correction: np.ndarray,
    indexes: np.ndarray,
    prefix: str,
) -> None:
    selected = correction[:, indexes]
    rollout.metrics[f"{prefix}_rms_correction_rad"] = float(
        np.sqrt(np.mean(np.square(selected)))
    )
    rollout.metrics[f"{prefix}_max_correction_rad"] = float(np.max(np.abs(selected)))


def _smoothness(correction: np.ndarray, dt: float) -> tuple[float, float]:
    if len(correction) < 2:
        return 0.0, 0.0
    velocity = np.diff(correction, axis=0) / dt
    if len(velocity) < 2:
        return float(np.square(velocity).mean()), 0.0
    acceleration = np.diff(velocity, axis=0) / dt
    return float(np.square(velocity).mean()), float(np.square(acceleration).mean())


def _invalid_penalty(
    rollout: RolloutResult, horizon: int, config: DynamicsRefinementConfig
) -> float:
    if rollout.valid:
        return 0.0
    frame = rollout.first_invalid_frame or 0
    return config.invalid_score + (horizon - frame) * 1.0e3


def _balance_objective(
    engine: _DynamicRollout,
    rollout: RolloutResult,
    correction: np.ndarray,
    target: np.ndarray,
) -> float:
    cfg, weights = engine.config, engine.config.balance_weights
    frames = int(rollout.metrics["frames_simulated"])
    selected = correction[:frames, engine.balance_indexes]
    velocity_sq, acceleration_sq = _smoothness(selected, cfg.control_dt)
    below = np.maximum(engine.joint_limits[:, 0] - target, 0.0)
    above = np.maximum(target - engine.joint_limits[:, 1], 0.0)
    terms = {
        "joint_tracking": weights.joint_tracking
        * float(rollout.metrics["robot_joint_tracking_rms_rad"]) ** 2,
        "root_tracking": weights.root_tracking
        * float(rollout.metrics["root_tracking_rms_m"]) ** 2,
        "diffusion_prior": weights.diffusion_prior * float(np.square(selected).mean()),
        "initial_correction": weights.initial_correction
        * float(np.square(correction[0, engine.balance_indexes]).mean()),
        "correction_velocity": weights.correction_velocity * velocity_sq,
        "correction_acceleration": weights.correction_acceleration * acceleration_sq,
        "actuator_saturation": weights.actuator_saturation
        * float(rollout.metrics["_saturation_sq"]),
        "unintended_terrain_contact": weights.unintended_terrain_contact
        * float(rollout.metrics["nonfoot_terrain_contact_duration_s"]),
        "joint_limits": weights.joint_limits * float(np.square(below + above).mean()),
        "hard_invalid": _invalid_penalty(rollout, engine.horizon, cfg),
    }
    rollout.objective_terms = terms
    rollout.objective = float(sum(terms.values()))
    return rollout.objective


def _manipulation_objective(
    engine: _DynamicRollout,
    rollout: RolloutResult,
    arm_correction: np.ndarray,
    target: np.ndarray,
    window: tuple[int, int],
    weights: ManipulationWeights,
) -> float:
    cfg = engine.config
    start, end = window
    ref_pos = engine.reference_object_pos[start : end + 1]
    ref_quat = engine.reference_object_quat[start : end + 1]
    sim_pos = rollout.object_pos[start : end + 1]
    sim_quat = rollout.object_quat_wxyz[start : end + 1]
    selected = arm_correction[start : end + 1, engine.arm_indexes]
    velocity_sq, acceleration_sq = _smoothness(selected, cfg.control_dt)
    below = np.maximum(engine.joint_limits[:, 0] - target, 0.0)
    above = np.maximum(target - engine.joint_limits[:, 1], 0.0)
    penetration_excess = max(
        0.0,
        float(rollout.metrics["max_penetration_m"]) - cfg.penetration_allowance_m,
    )
    terms = {
        "object_position": weights.object_position
        * float(np.square(sim_pos - ref_pos).sum(axis=1).mean()),
        "object_orientation": weights.object_orientation
        * float(np.square(_quat_angle(sim_quat, ref_quat)).mean()),
        "final_position": weights.final_position
        * float(
            np.square(rollout.object_pos[-1] - engine.reference_object_pos[-1]).sum()
        ),
        "final_orientation": weights.final_orientation
        * float(
            _quat_angle(
                rollout.object_quat_wxyz[-1:], engine.reference_object_quat[-1:]
            )[0]
            ** 2
        ),
        "arm_prior": weights.arm_prior * float(np.square(selected).mean()),
        "correction_velocity": weights.correction_velocity * velocity_sq,
        "correction_acceleration": weights.correction_acceleration * acceleration_sq,
        "joint_limits": weights.joint_limits * float(np.square(below + above).mean()),
        "actuator_saturation": weights.actuator_saturation
        * float(rollout.metrics["_saturation_sq"]),
        "unintended_collision": weights.unintended_collision
        * (
            float(rollout.metrics["other_robot_object_contact_duration_s"])
            + float(rollout.metrics["nonfoot_terrain_contact_duration_s"])
        ),
        "penetration": weights.penetration * penetration_excess**2,
        "hard_invalid": _invalid_penalty(rollout, engine.horizon, cfg),
    }
    rollout.objective_terms = terms
    rollout.objective = float(sum(terms.values()))
    return rollout.objective


def _cem_search(
    *,
    dimensions: int,
    population: int,
    elite_count: int,
    generations: int,
    initial_std: float | np.ndarray,
    minimum_std: float | np.ndarray,
    bound: float | np.ndarray,
    seed: int,
    evaluate: Callable[[np.ndarray, bool], tuple[float, RolloutResult]],
    initial_parameters: np.ndarray | None = None,
    parameter_diagnostics: Callable[[np.ndarray, RolloutResult], dict[str, Any]]
    | None = None,
) -> tuple[np.ndarray, RolloutResult, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    mean = (
        np.zeros(dimensions, dtype=np.float64)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float64).copy()
    )
    std = np.broadcast_to(
        np.asarray(initial_std, dtype=np.float64), (dimensions,)
    ).copy()
    minimum_std_array = np.broadcast_to(
        np.asarray(minimum_std, dtype=np.float64), (dimensions,)
    )
    bound_array = np.broadcast_to(np.asarray(bound, dtype=np.float64), (dimensions,))
    best_parameters = mean.copy()
    best_cost, best_rollout = evaluate(best_parameters, True)
    history: list[dict[str, Any]] = []

    def rank(cost: float, rollout: RolloutResult) -> tuple[int, int, float]:
        if rollout.valid:
            return 0, 0, cost
        frame = rollout.first_invalid_frame or 0
        return 1, rollout.object_pos.shape[0] - frame, cost

    best_rank = rank(best_cost, best_rollout)
    for generation in range(generations):
        samples = np.clip(
            rng.normal(mean, std, size=(population, dimensions)),
            -bound_array,
            bound_array,
        )
        samples[0] = best_parameters
        costs = np.empty(population, dtype=np.float64)
        rollouts: list[RolloutResult] = []
        for index, sample in enumerate(samples):
            costs[index], rollout = evaluate(sample, True)
            rollouts.append(rollout)
        order = np.asarray(
            sorted(
                range(population), key=lambda index: rank(costs[index], rollouts[index])
            )
        )
        elite = samples[order[:elite_count]]
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(minimum_std_array, 0.25 * std + 0.75 * elite.std(axis=0))
        generation_rank = rank(costs[order[0]], rollouts[order[0]])
        if generation_rank < best_rank:
            best_cost = float(costs[order[0]])
            best_parameters = samples[order[0]].copy()
            best_rollout = rollouts[order[0]]
            best_rank = generation_rank
        best_frame = (
            best_rollout.first_invalid_frame
            if best_rollout.first_invalid_frame is not None
            else best_rollout.object_pos.shape[0]
        )
        entry: dict[str, Any] = {
            "generation": float(generation),
            "best_objective": float(best_cost),
            "median_objective": float(np.median(costs)),
            "valid_candidates": float(sum(result.valid for result in rollouts)),
            "best_first_invalid_frame": float(best_frame),
            "best_invalid_reason": best_rollout.invalid_reason,
            "mean_std_rad": float(std.mean()),
        }
        if parameter_diagnostics is not None:
            entry.update(parameter_diagnostics(best_parameters, best_rollout))
        history.append(entry)
    _, best_rollout = evaluate(best_parameters, False)
    return best_parameters, best_rollout, history


def _validate_config(config: DynamicsRefinementConfig) -> None:
    for population, elites, generations in (
        (
            config.balance_population,
            config.balance_elite_count,
            config.balance_generations,
        ),
        (config.arm_population, config.arm_elite_count, config.arm_generations),
    ):
        if population < 1 or not (1 <= elites <= population) or generations < 1:
            raise ValueError("invalid CEM settings")
    if config.balance_num_knots < 4 or config.arm_num_knots < 4:
        raise ValueError("cubic corrections require at least four knots")


def _classify(balance: RolloutResult, manipulation: RolloutResult | None) -> str:
    if not balance.valid:
        return "D"
    if manipulation is None or not manipulation.valid:
        return "C"
    base_error = float(balance.metrics["final_goal_position_error_m"])
    error = float(manipulation.metrics["final_goal_position_error_m"])
    improvement = (base_error - error) / max(base_error, 1.0e-6)
    lift = float(manipulation.metrics["max_box_lift_m"])
    displacement = float(manipulation.metrics["max_object_displacement_m"])
    hand_contact = max(
        float(manipulation.metrics["left_hand_contact_duration_s"]),
        float(manipulation.metrics["right_hand_contact_duration_s"]),
    )
    if lift >= 0.10 and improvement >= 0.25 and hand_contact >= 0.15:
        return "A"
    if (
        improvement >= 0.10
        and hand_contact >= 0.08
        and (lift >= 0.025 or displacement >= 0.08)
    ):
        return "B"
    return "C"


def refine_generated_reference(
    model: mujoco.MjModel,
    generated_reference: GeneratedReference,
    config: DynamicsRefinementConfig | None = None,
) -> DynamicsRefinementResult:
    """Run balance refinement, then balance-gated manipulation refinement."""
    cfg = config or DynamicsRefinementConfig()
    _validate_config(cfg)
    if generated_reference.joint_pos.shape[0] != 1:
        raise ValueError("the staged prototype accepts exactly one reference")
    if abs(float(generated_reference.dt) - cfg.control_dt) > 1.0e-9:
        raise ValueError("reference and controller time steps differ")

    object_pos = _np(generated_reference.object_pos_local[0])
    window = detect_pickup_window(
        object_pos,
        min_frames=cfg.min_window_frames,
        max_frames=cfg.max_window_frames,
        approach_frames=cfg.approach_frames,
        lift_threshold_m=cfg.lift_threshold_m,
        displacement_threshold_m=cfg.displacement_threshold_m,
    )
    engine = _DynamicRollout(model, generated_reference, cfg)
    raw_target = engine.reference_joint.copy()
    raw = engine.rollout(raw_target, stop_on_invalid=False)
    raw_correction = np.zeros_like(raw_target)
    _balance_objective(engine, raw, raw_correction, raw_target)

    balance_free_knots = cfg.balance_num_knots - (
        1 if cfg.balance_optimize_initial_knot else 2
    )
    balance_dimensions = balance_free_knots * len(engine.balance_indexes)

    def balance_values(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correction = _spline_correction(
            engine.horizon,
            len(engine.joint_names),
            engine.balance_indexes,
            cfg.balance_num_knots,
            parameters,
            optimize_first=cfg.balance_optimize_initial_knot,
        )
        return raw_target + correction, correction

    def evaluate_balance(
        parameters: np.ndarray, stop_on_invalid: bool
    ) -> tuple[float, RolloutResult]:
        target, correction = balance_values(parameters)
        rollout = engine.rollout(target, stop_on_invalid=stop_on_invalid)
        return _balance_objective(engine, rollout, correction, target), rollout

    balance_bound = np.full(
        balance_dimensions, cfg.balance_correction_bound_rad, dtype=np.float64
    )
    balance_initial_std = np.full(
        balance_dimensions, cfg.balance_initial_std_rad, dtype=np.float64
    )
    balance_minimum_std = np.full(
        balance_dimensions, cfg.balance_minimum_std_rad, dtype=np.float64
    )
    if cfg.balance_optimize_initial_knot:
        initial_slice = slice(0, len(engine.balance_indexes))
        balance_bound[initial_slice] = cfg.balance_initial_knot_bound_rad
        balance_initial_std[initial_slice] = cfg.balance_initial_knot_std_rad
        balance_minimum_std[initial_slice] = cfg.balance_initial_knot_minimum_std_rad

    def balance_diagnostics(
        parameters: np.ndarray, rollout: RolloutResult
    ) -> dict[str, Any]:
        initial = (
            parameters[: len(engine.balance_indexes)]
            if cfg.balance_optimize_initial_knot
            else np.zeros(len(engine.balance_indexes), dtype=np.float64)
        )
        frame = min(26, len(rollout.root_pos) - 1)
        return {
            "frame_0_correction_l2_rad": float(np.linalg.norm(initial)),
            "frame_0_correction_max_rad": float(np.max(np.abs(initial))),
            "root_height_at_frame_26_m": float(rollout.root_pos[frame, 2]),
            "torso_tilt_at_frame_26_deg": float(
                np.rad2deg(rollout.torso_tilt_rad[frame])
            ),
        }

    balance_parameters, balance, balance_history = _cem_search(
        dimensions=balance_dimensions,
        population=cfg.balance_population,
        elite_count=cfg.balance_elite_count,
        generations=cfg.balance_generations,
        initial_std=balance_initial_std,
        minimum_std=balance_minimum_std,
        bound=balance_bound,
        seed=cfg.seed,
        evaluate=evaluate_balance,
        parameter_diagnostics=balance_diagnostics,
    )
    balance_target, balance_correction = balance_values(balance_parameters)
    _balance_objective(engine, balance, balance_correction, balance_target)
    _correction_statistics(
        balance, balance_correction, engine.balance_indexes, "balance"
    )
    initial_balance_correction = balance_correction[0, engine.balance_indexes]
    balance.metrics["frame_0_balance_correction_l2_rad"] = float(
        np.linalg.norm(initial_balance_correction)
    )
    balance.metrics["frame_0_balance_correction_max_rad"] = float(
        np.max(np.abs(initial_balance_correction))
    )
    balance.metrics["frame_0_balance_correction_by_joint_rad"] = {
        name: float(value)
        for name, value in zip(
            _LEG_WAIST_JOINTS, initial_balance_correction, strict=True
        )
    }
    for rollout in (raw, balance):
        rollout.metrics.setdefault("balance_rms_correction_rad", 0.0)
        rollout.metrics.setdefault("balance_max_correction_rad", 0.0)
        rollout.metrics["arm_rms_correction_rad"] = 0.0
        rollout.metrics["arm_max_correction_rad"] = 0.0
    raw.metrics["frame_0_balance_correction_l2_rad"] = 0.0
    raw.metrics["frame_0_balance_correction_max_rad"] = 0.0
    raw.metrics["frame_0_balance_correction_by_joint_rad"] = {
        name: 0.0 for name in _LEG_WAIST_JOINTS
    }
    stage_a_passed = bool(
        balance.valid
        and float(balance.metrics["balance_rms_correction_rad"])
        <= cfg.maximum_balance_rms_correction_rad
    )

    optimizer: dict[str, Any] = {
        "method": "cross_entropy",
        "balance": {
            "dimensions": balance_dimensions,
            "num_knots": cfg.balance_num_knots,
            "knot_frames": np.linspace(
                0.0, engine.horizon - 1, cfg.balance_num_knots
            ).tolist(),
            "optimize_initial_knot": cfg.balance_optimize_initial_knot,
            "initial_knot_bound_rad": cfg.balance_initial_knot_bound_rad,
            "later_knot_bound_rad": cfg.balance_correction_bound_rad,
            "final_knot_fixed_zero": True,
            "population": cfg.balance_population,
            "elite_count": cfg.balance_elite_count,
            "generations": cfg.balance_generations,
            "weights": cfg.balance_weights.as_dict(),
            "history": balance_history,
        },
        "manipulation": None,
    }
    if not stage_a_passed or not cfg.run_stage_b:
        return DynamicsRefinementResult(
            window=window,
            joint_names=engine.joint_names,
            balance_joint_names=_LEG_WAIST_JOINTS,
            arm_joint_names=_ARM_JOINTS,
            raw=raw,
            balance=balance,
            manipulation=None,
            normal_prior_manipulation=None,
            loose_prior_manipulation=None,
            balance_joint_target=balance_target,
            candidate_joint_target=None,
            status="stage_a_stable" if stage_a_passed else "D",
            stage_a_passed=stage_a_passed,
            selected_arm_prior=None,
            optimizer=optimizer,
        )

    start, end = window
    arm_dimensions = (cfg.arm_num_knots - 2) * len(engine.arm_indexes)

    def arm_values(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correction = np.zeros_like(raw_target)
        correction[start : end + 1] = _spline_correction(
            end - start + 1,
            len(engine.joint_names),
            engine.arm_indexes,
            cfg.arm_num_knots,
            parameters,
        )
        return balance_target + correction, correction

    def run_arm_search(
        weights: ManipulationWeights,
        seed: int,
        initial: np.ndarray | None = None,
    ) -> tuple[np.ndarray, RolloutResult, list[dict[str, float]]]:
        def evaluate(
            parameters: np.ndarray, stop_on_invalid: bool
        ) -> tuple[float, RolloutResult]:
            target, correction = arm_values(parameters)
            rollout = engine.rollout(target, stop_on_invalid=stop_on_invalid)
            objective = _manipulation_objective(
                engine, rollout, correction, target, window, weights
            )
            return objective, rollout

        return _cem_search(
            dimensions=arm_dimensions,
            population=cfg.arm_population,
            elite_count=cfg.arm_elite_count,
            generations=cfg.arm_generations,
            initial_std=cfg.arm_initial_std_rad,
            minimum_std=cfg.arm_minimum_std_rad,
            bound=cfg.arm_correction_bound_rad,
            seed=seed,
            evaluate=evaluate,
            initial_parameters=initial,
        )

    normal_parameters, normal, normal_history = run_arm_search(
        cfg.manipulation_weights, cfg.seed + 1
    )
    loose_weights = cfg.manipulation_weights.with_arm_prior(cfg.loose_arm_prior)
    loose_parameters, loose, loose_history = run_arm_search(
        loose_weights, cfg.seed + 2, normal_parameters
    )
    normal_target, normal_correction = arm_values(normal_parameters)
    loose_target, loose_correction = arm_values(loose_parameters)
    _manipulation_objective(
        engine,
        normal,
        normal_correction,
        normal_target,
        window,
        cfg.manipulation_weights,
    )
    _manipulation_objective(
        engine, loose, loose_correction, loose_target, window, loose_weights
    )
    for rollout, correction in ((normal, normal_correction), (loose, loose_correction)):
        _correction_statistics(
            rollout, balance_correction, engine.balance_indexes, "balance"
        )
        _correction_statistics(rollout, correction, engine.arm_indexes, "arm")

    candidates = [
        (normal, normal_target, cfg.manipulation_weights.arm_prior),
        (loose, loose_target, cfg.loose_arm_prior),
    ]
    valid_candidates = [candidate for candidate in candidates if candidate[0].valid]
    if valid_candidates:
        manipulation, candidate_target, selected_prior = min(
            valid_candidates,
            key=lambda candidate: float(
                candidate[0].metrics["final_goal_position_error_m"]
            ),
        )
    else:
        manipulation, candidate_target, selected_prior = min(
            candidates, key=lambda candidate: candidate[0].objective
        )
    status = _classify(balance, manipulation)
    optimizer["manipulation"] = {
        "dimensions": arm_dimensions,
        "num_knots": cfg.arm_num_knots,
        "population": cfg.arm_population,
        "elite_count": cfg.arm_elite_count,
        "generations": cfg.arm_generations,
        "normal_weights": cfg.manipulation_weights.as_dict(),
        "loose_arm_prior": cfg.loose_arm_prior,
        "normal_history": normal_history,
        "loose_history": loose_history,
    }
    physical_pos = manipulation.object_pos.copy() if status in ("A", "B") else None
    physical_quat = (
        manipulation.object_quat_wxyz.copy() if status in ("A", "B") else None
    )
    return DynamicsRefinementResult(
        window=window,
        joint_names=engine.joint_names,
        balance_joint_names=_LEG_WAIST_JOINTS,
        arm_joint_names=_ARM_JOINTS,
        raw=raw,
        balance=balance,
        manipulation=manipulation,
        normal_prior_manipulation=normal,
        loose_prior_manipulation=loose,
        balance_joint_target=balance_target,
        candidate_joint_target=candidate_target,
        status=status,
        stage_a_passed=True,
        selected_arm_prior=selected_prior,
        optimizer=optimizer,
        refined_object_pos=physical_pos,
        refined_object_quat_wxyz=physical_quat,
    )
