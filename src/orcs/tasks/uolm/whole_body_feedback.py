"""Contact-aware inverse-dynamics tracking for a frozen MuJoCo trajectory.

The controller deliberately keeps the compiled position actuators.  A desired
torque from the constrained whole-body solve is converted to the equivalent
position command of the existing affine position servo.  Contact forces are
optimization variables only; they are never written into MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from orcs.tasks.uolm.contact_dynamics_refinement import (
    _contact_state,
    _copy_data,
)
from orcs.tasks.uolm.dynamics_refinement import _DynamicRollout
from orcs.tasks.uolm.support_diagnostics import _contact_wrench

__all__ = [
    "WholeBodyFeedbackConfig",
    "WholeBodyFeedbackResult",
    "track_with_whole_body_feedback",
]


@dataclass(frozen=True, slots=True)
class WholeBodyFeedbackConfig:
    anchor_frame: int = 85
    evaluation_end_frame: int = 160
    friction: float = 0.60
    contact_on_force_n: float = 2.0
    contact_off_clearance_m: float = 0.005
    contact_off_samples: int = 4
    max_iterations: int = 60
    solver_ftol: float = 1.0e-7
    # Task-space acceleration feedback.  These are controller gains, not
    # modifications to the compiled actuator gains.
    com_kp: tuple[float, float, float] = (18.0, 18.0, 75.0)
    com_kd: tuple[float, float, float] = (8.0, 8.0, 17.0)
    torso_kp: tuple[float, float, float] = (75.0, 75.0, 4.0)
    torso_kd: tuple[float, float, float] = (14.0, 14.0, 2.0)
    momentum_gain: float = 0.8
    joint_kp: float = 35.0
    joint_kd: float = 9.0
    stance_position_kp: float = 180.0
    stance_position_kd: float = 28.0
    stance_orientation_kp: float = 70.0
    stance_orientation_kd: float = 13.0
    normal_position_gain: float = 300.0
    normal_velocity_gain: float = 18.0
    normal_acceleration_tolerance_m_s2: float = 2.0
    maximum_normal_target_acceleration_m_s2: float = 40.0
    normal_tracking_weight: float = 35.0
    tangential_velocity_gain: float = 12.0
    tangential_acceleration_weight: float = 18.0
    stance_rotation_weight: float = 8.0
    qdd_limit: float = 180.0


@dataclass(slots=True)
class WholeBodyFeedbackResult:
    valid: bool
    recoverable_terminal: bool
    first_invalid_frame: int | None
    first_invalid_sample: int | None
    invalid_reason: str | None
    data: dict[str, np.ndarray]
    summary: dict[str, Any]
    prefix: dict[str, Any]


@dataclass(slots=True)
class _ReferencePath:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    com_position: np.ndarray
    com_velocity: np.ndarray
    com_acceleration: np.ndarray
    angular_momentum: np.ndarray
    torso_rotation: np.ndarray
    torso_angular_velocity: np.ndarray
    torso_angular_acceleration: np.ndarray
    foot_position: np.ndarray
    foot_rotation: np.ndarray
    foot_linear_velocity: np.ndarray
    foot_angular_velocity: np.ndarray
    foot_contact: np.ndarray


@dataclass(slots=True)
class _ContactPoint:
    side: int
    body: int
    point: np.ndarray
    normal: np.ndarray
    tangents: np.ndarray
    penetration_m: float
    normal_force_n: float
    jacobian: np.ndarray
    velocity: np.ndarray
    normal_velocity_m_s: float


def _rotation_error(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Small-angle world-frame orientation error from actual to desired."""
    relative = desired @ actual.T
    return 0.5 * np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )


def _angular_velocity(rotations: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros((len(rotations), 3), dtype=np.float64)
    for index in range(1, len(rotations)):
        result[index] = _rotation_error(rotations[index], rotations[index - 1]) / dt
    if len(result) > 1:
        result[0] = result[1]
    return result


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values, dt, axis=0, edge_order=2)


def _foot_clearance(engine: _DynamicRollout, data: mujoco.MjData, body: int) -> float:
    fromto = np.empty(6, dtype=np.float64)
    geoms = [
        geom
        for geom in range(engine.model.ngeom)
        if int(engine.model.geom_bodyid[geom]) == body
        and (
            int(engine.model.geom_contype[geom])
            or int(engine.model.geom_conaffinity[geom])
        )
    ]
    if not geoms:
        return float("inf")
    return min(
        float(mujoco.mj_geomDistance(engine.model, data, geom, terrain, 1.0, fromto))
        for geom in geoms
        for terrain in engine.terrain_geoms
    )


def _reference_path(
    engine: _DynamicRollout,
    artifact: dict[str, np.ndarray],
    start_sample: int,
    end_sample: int,
) -> _ReferencePath:
    """Reconstruct body poses from the saved physically realized state path."""
    model = engine.model
    count = end_sample - start_sample + 1
    joint_position = np.asarray(
        artifact["actual_joint_position_rad"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    joint_velocity = np.asarray(
        artifact["actual_joint_velocity_rad_s"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    root_position = np.asarray(
        artifact["actual_root_position_m"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    root_quaternion = np.asarray(
        artifact["actual_root_quaternion_wxyz"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    com_position = np.asarray(
        artifact["actual_com_position_m"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    com_velocity = np.asarray(
        artifact["actual_com_velocity_m_s"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    angular_momentum = np.asarray(
        artifact["actual_centroidal_angular_momentum"][start_sample : end_sample + 1],
        dtype=np.float64,
    )
    foot_contact = np.asarray(
        artifact["actual_foot_contact"][start_sample : end_sample + 1], dtype=bool
    )
    torso_rotation = np.empty((count, 3, 3), dtype=np.float64)
    foot_position = np.empty((count, 2, 3), dtype=np.float64)
    foot_rotation = np.empty((count, 2, 3, 3), dtype=np.float64)
    scratch = mujoco.MjData(model)
    for sample in range(count):
        scratch.qpos[engine.robot_qpos : engine.robot_qpos + 3] = root_position[sample]
        scratch.qpos[engine.robot_qpos + 3 : engine.robot_qpos + 7] = root_quaternion[
            sample
        ]
        scratch.qpos[engine.actuator_qpos] = joint_position[sample]
        mujoco.mj_forward(model, scratch)
        torso_rotation[sample] = scratch.xmat[engine.torso_body].reshape(3, 3)
        for side, body in enumerate(engine.foot_bodies):
            foot_position[sample, side] = scratch.xpos[body]
            foot_rotation[sample, side] = scratch.xmat[body].reshape(3, 3)
    dt = engine.config.physics_dt
    torso_angular_velocity = _angular_velocity(torso_rotation, dt)
    foot_angular_velocity = np.stack(
        [_angular_velocity(foot_rotation[:, side], dt) for side in range(2)], axis=1
    )
    return _ReferencePath(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_acceleration=_finite_difference(joint_velocity, dt),
        root_position=root_position,
        root_quaternion=root_quaternion,
        com_position=com_position,
        com_velocity=com_velocity,
        com_acceleration=_finite_difference(com_velocity, dt),
        angular_momentum=angular_momentum,
        torso_rotation=torso_rotation,
        torso_angular_velocity=torso_angular_velocity,
        torso_angular_acceleration=_finite_difference(torso_angular_velocity, dt),
        foot_position=foot_position,
        foot_rotation=foot_rotation,
        foot_linear_velocity=_finite_difference(foot_position, dt),
        foot_angular_velocity=foot_angular_velocity,
        foot_contact=foot_contact,
    )


def _execute_prefix(
    engine: _DynamicRollout, target: np.ndarray, anchor: int
) -> tuple[mujoco.MjData, dict[str, Any], np.ndarray]:
    data = engine._new_initialized_data()
    initial_torso = data.xmat[engine.torso_body].reshape(3, 3).copy()
    torso_local_up = initial_torso.T @ np.asarray([0.0, 0.0, 1.0])
    data.ctrl[:] = np.clip(
        target[0, engine.actuator_joint_indexes],
        engine.model.actuator_ctrlrange[:, 0],
        engine.model.actuator_ctrlrange[:, 1],
    )
    for frame in range(1, anchor + 1):
        data.ctrl[:] = np.clip(
            target[frame, engine.actuator_joint_indexes],
            engine.model.actuator_ctrlrange[:, 0],
            engine.model.actuator_ctrlrange[:, 1],
        )
        for _ in range(engine.config.substeps):
            mujoco.mj_step(engine.model, data)
    # mj_step integrates qpos after its forward pass.  Refresh all kinematics
    # so the frame-85 Jacobians correspond to the saved integrated state.
    mujoco.mj_forward(engine.model, data)
    forces, _, normal, _, _, _ = _contact_wrench(engine, data)
    return (
        _copy_data(engine.model, data),
        {
            "frame": anchor,
            "root_position_m": data.qpos[
                engine.robot_qpos : engine.robot_qpos + 3
            ].copy(),
            "root_velocity": data.qvel[engine.robot_dof : engine.robot_dof + 6].copy(),
            "joint_position_rad": data.qpos[engine.actuator_qpos].copy(),
            "joint_velocity_rad_s": data.qvel[
                [
                    engine.model.jnt_dofadr[engine.model.actuator_trnid[i, 0]]
                    for i in range(engine.model.nu)
                ]
            ].copy(),
            "foot_normal_force_left_right_n": normal.copy(),
            "foot_force_left_right_n": forces.copy(),
        },
        torso_local_up,
    )


class _WholeBodyController:
    def __init__(
        self,
        engine: _DynamicRollout,
        reference: _ReferencePath,
        config: WholeBodyFeedbackConfig,
    ) -> None:
        self.engine = engine
        self.model = engine.model
        self.ref = reference
        self.cfg = config
        self.robot_dofs = np.arange(engine.robot_dof, engine.robot_dof + 35)
        self.actuator_dofs = np.asarray(
            [
                self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
                for i in range(self.model.nu)
            ],
            dtype=np.int32,
        )
        self.actuator_robot_dofs = self.actuator_dofs - engine.robot_dof
        self.force_limit = np.max(np.abs(self.model.actuator_forcerange), axis=1)
        self.kp = self.model.actuator_gainprm[:, 0].copy()
        self.kd = -self.model.actuator_biasprm[:, 2].copy()
        self.root_body = int(self.model.jnt_bodyid[engine.robot_free_joint])
        self.mass = float(self.model.body_subtreemass[self.root_body])
        self.active = np.zeros(2, dtype=bool)
        self.missing = np.zeros(2, dtype=np.int32)
        self.stance_position = np.zeros((2, 3), dtype=np.float64)
        self.stance_rotation = np.zeros((2, 3, 3), dtype=np.float64)
        self.previous_jacobians: dict[str, np.ndarray] = {}
        self.previous_tau = np.zeros(self.model.nu, dtype=np.float64)
        self.previous_solution: np.ndarray | None = None

    def _jacobians(
        self, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        nv = self.model.nv
        com = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(self.model, data, com, self.root_body)
        torso_p = np.zeros((3, nv), dtype=np.float64)
        torso_r = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, data, torso_p, torso_r, self.engine.torso_body)
        feet: list[tuple[np.ndarray, np.ndarray]] = []
        for body in self.engine.foot_bodies:
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, data, jacp, jacr, body)
            feet.append((jacp[:, self.robot_dofs], jacr[:, self.robot_dofs]))
        return com[:, self.robot_dofs], torso_r[:, self.robot_dofs], feet

    def _jdot_qdot(
        self, name: str, jacobian: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        previous = self.previous_jacobians.get(name)
        self.previous_jacobians[name] = jacobian.copy()
        if previous is None:
            return np.zeros(jacobian.shape[0], dtype=np.float64)
        return ((jacobian - previous) / self.engine.config.physics_dt) @ qvel

    def _update_contacts(
        self, data: mujoco.MjData, normal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        clearance = np.asarray(
            [
                _foot_clearance(self.engine, data, body)
                for body in self.engine.foot_bodies
            ]
        )
        for side, body in enumerate(self.engine.foot_bodies):
            observed = normal[side] >= self.cfg.contact_on_force_n
            if observed:
                self.missing[side] = 0
                if not self.active[side]:
                    self.active[side] = True
                    self.stance_position[side] = data.xpos[body]
                    self.stance_rotation[side] = data.xmat[body].reshape(3, 3)
            elif self.active[side]:
                self.missing[side] += 1
                if (
                    self.missing[side] >= self.cfg.contact_off_samples
                    and clearance[side] > self.cfg.contact_off_clearance_m
                ):
                    self.active[side] = False
        return self.active.copy(), clearance

    def _actual_contact_points(self, data: mujoco.MjData) -> list[_ContactPoint]:
        """Return force-bearing MuJoCo foot/terrain contact points."""
        points: list[_ContactPoint] = []
        contact_force = np.empty(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if (
                first not in self.engine.terrain_geoms
                and second not in self.engine.terrain_geoms
            ):
                continue
            robot_geom = second if first in self.engine.terrain_geoms else first
            body = int(self.model.geom_bodyid[robot_geom])
            if body not in self.engine.foot_bodies:
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, contact_force)
            if float(contact_force[0]) < self.cfg.contact_on_force_n:
                continue
            side = self.engine.foot_bodies.index(body)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            normal = frame[0].copy()
            if robot_geom == first:
                normal *= -1.0
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            tangent0 = frame[1].copy()
            tangent0 -= normal * float(np.dot(tangent0, normal))
            tangent0 /= max(float(np.linalg.norm(tangent0)), 1.0e-12)
            tangent1 = np.cross(normal, tangent0)
            point = np.asarray(contact.pos, dtype=np.float64).copy()
            jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(self.model, data, jacobian, None, point, body)
            jacobian = jacobian[:, self.robot_dofs]
            velocity = jacobian @ data.qvel[self.robot_dofs]
            points.append(
                _ContactPoint(
                    side=side,
                    body=body,
                    point=point,
                    normal=normal,
                    tangents=np.stack((tangent0, tangent1)),
                    penetration_m=max(0.0, -float(contact.dist)),
                    normal_force_n=float(contact_force[0]),
                    jacobian=jacobian,
                    velocity=velocity,
                    normal_velocity_m_s=float(np.dot(normal, velocity)),
                )
            )
        points.sort(key=lambda item: (item.side, *item.point.tolist()))
        return points

    def _contact_force_map(
        self, contacts: list[_ContactPoint]
    ) -> tuple[np.ndarray, list[int]]:
        """Map actual-point nonnegative friction rays to generalized force."""
        columns: list[np.ndarray] = []
        sides: list[int] = []
        mu = self.cfg.friction
        for contact in contacts:
            rays = (
                contact.normal + mu * contact.tangents[0],
                contact.normal - mu * contact.tangents[0],
                contact.normal + mu * contact.tangents[1],
                contact.normal - mu * contact.tangents[1],
            )
            for ray in rays:
                columns.append(contact.jacobian.T @ ray)
                sides.append(contact.side)
        if not columns:
            return np.zeros((35, 0), dtype=np.float64), sides
        return np.stack(columns, axis=1), sides

    def solve(
        self, data: mujoco.MjData, sample: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        started = perf_counter()
        model, cfg = self.model, self.cfg
        qvel = data.qvel[self.robot_dofs]
        forces, _, normal, slip, _, _ = _contact_wrench(self.engine, data)
        active, clearance = self._update_contacts(data, normal)
        jcom, jtorso, jfeet = self._jacobians(data)
        contacts = self._actual_contact_points(data)
        force_map, force_sides = self._contact_force_map(contacts)
        nrho = force_map.shape[1]
        nqdd, ntau = 35, model.nu
        tau_slice = slice(nqdd, nqdd + ntau)
        rho_slice = slice(nqdd + ntau, nqdd + ntau + nrho)
        dimension = nqdd + ntau + nrho

        mass_matrix_full = np.empty((model.nv, model.nv), dtype=np.float64)
        mujoco.mj_fullM(model, mass_matrix_full, data.qM)
        mass_matrix = mass_matrix_full[np.ix_(self.robot_dofs, self.robot_dofs)]
        dynamics = np.zeros((nqdd, dimension), dtype=np.float64)
        dynamics[:, :nqdd] = mass_matrix
        for actuator, dof in enumerate(self.actuator_robot_dofs):
            dynamics[dof, nqdd + actuator] = -float(data.actuator_moment[actuator])
        if nrho:
            dynamics[:, rho_slice] = -force_map
        bias = -data.qfrc_bias[self.robot_dofs].copy()

        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        normal_rows: list[np.ndarray] = []
        normal_lower: list[float] = []
        contact_kinematics: list[tuple[int, np.ndarray, float, float]] = []

        def task(
            matrix: np.ndarray, target: np.ndarray, weight: np.ndarray | float
        ) -> None:
            block = np.zeros((matrix.shape[0], dimension), dtype=np.float64)
            block[:, :nqdd] = matrix
            scale = np.sqrt(np.broadcast_to(weight, target.shape))
            rows.append(scale[:, None] * block)
            targets.append(scale * target)

        joint_error = (
            self.ref.joint_position[sample] - data.qpos[self.engine.actuator_qpos]
        )
        joint_velocity = data.qvel[self.actuator_dofs]
        desired_joint_acceleration = (
            self.ref.joint_acceleration[sample]
            + cfg.joint_kp * joint_error
            + cfg.joint_kd * (self.ref.joint_velocity[sample] - joint_velocity)
        )
        joint_matrix = np.zeros((model.nu, nqdd), dtype=np.float64)
        joint_matrix[np.arange(model.nu), self.actuator_robot_dofs] = 1.0
        joint_weights = np.full(model.nu, 0.8)
        balance_actuators = np.flatnonzero(
            np.isin(self.engine.actuator_joint_indexes, self.engine.balance_indexes)
        )
        joint_weights[balance_actuators] = 3.0
        task(joint_matrix, desired_joint_acceleration, joint_weights)

        com_velocity = data.subtree_linvel[self.root_body]
        desired_com_acceleration = (
            self.ref.com_acceleration[sample]
            + np.asarray(cfg.com_kp)
            * (self.ref.com_position[sample] - data.subtree_com[self.root_body])
            + np.asarray(cfg.com_kd) * (self.ref.com_velocity[sample] - com_velocity)
        )
        task(
            jcom,
            desired_com_acceleration - self._jdot_qdot("com", jcom, qvel),
            np.asarray([10.0, 10.0, 55.0]),
        )

        torso_rotation = data.xmat[self.engine.torso_body].reshape(3, 3)
        torso_omega = data.cvel[self.engine.torso_body, :3]
        momentum_error = (
            self.ref.angular_momentum[sample] - data.subtree_angmom[self.root_body]
        )
        desired_torso_acceleration = (
            self.ref.torso_angular_acceleration[sample]
            + np.asarray(cfg.torso_kp)
            * _rotation_error(self.ref.torso_rotation[sample], torso_rotation)
            + np.asarray(cfg.torso_kd)
            * (self.ref.torso_angular_velocity[sample] - torso_omega)
            + cfg.momentum_gain * momentum_error
        )
        task(
            jtorso,
            desired_torso_acceleration - self._jdot_qdot("torso", jtorso, qvel),
            np.asarray([45.0, 45.0, 2.0]),
        )

        contact_count = np.bincount(
            [contact.side for contact in contacts], minlength=2
        ).astype(np.int32)
        for contact_index, contact in enumerate(contacts):
            normal_matrix = contact.normal[None, :] @ contact.jacobian
            normal_bias = float(
                self._jdot_qdot(
                    f"contact_normal_{contact.side}_{contact_index}",
                    normal_matrix,
                    qvel,
                )[0]
            )
            normal_target = float(
                np.clip(
                    cfg.normal_position_gain * contact.penetration_m
                    - cfg.normal_velocity_gain * contact.normal_velocity_m_s,
                    -cfg.maximum_normal_target_acceleration_m_s2,
                    cfg.maximum_normal_target_acceleration_m_s2,
                )
            )
            normal_rows.append(normal_matrix[0])
            normal_lower.append(
                normal_target - normal_bias - cfg.normal_acceleration_tolerance_m_s2
            )
            task(
                normal_matrix,
                np.asarray([normal_target - normal_bias]),
                cfg.normal_tracking_weight,
            )
            tangent_matrix = contact.tangents @ contact.jacobian
            tangent_bias = self._jdot_qdot(
                f"contact_tangent_{contact.side}_{contact_index}",
                tangent_matrix,
                qvel,
            )
            tangent_velocity = contact.tangents @ contact.velocity
            tangent_target = (
                -cfg.tangential_velocity_gain * tangent_velocity - tangent_bias
            )
            task(
                tangent_matrix,
                tangent_target,
                cfg.tangential_acceleration_weight,
            )
            contact_kinematics.append(
                (
                    contact.side,
                    normal_matrix[0],
                    normal_bias,
                    normal_target,
                )
            )

        for side, (jacp, jacr) in enumerate(jfeet):
            body = self.engine.foot_bodies[side]
            spatial = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body, spatial, 0
            )
            omega, velocity = spatial[:3], spatial[3:]
            if contact_count[side]:
                angular_target = (
                    cfg.stance_orientation_kp
                    * _rotation_error(
                        self.stance_rotation[side], data.xmat[body].reshape(3, 3)
                    )
                    - cfg.stance_orientation_kd * omega
                )
                angular_bias = self._jdot_qdot(f"footr_stance{side}", jacr[:2], qvel)
                task(
                    jacr[:2],
                    angular_target[:2] - angular_bias,
                    cfg.stance_rotation_weight,
                )
            else:
                linear_target = 35.0 * (
                    self.ref.foot_position[sample, side] - data.xpos[body]
                ) + 9.0 * (self.ref.foot_linear_velocity[sample, side] - velocity)
                angular_target = 25.0 * _rotation_error(
                    self.ref.foot_rotation[sample, side],
                    data.xmat[body].reshape(3, 3),
                ) + 6.0 * (self.ref.foot_angular_velocity[sample, side] - omega)
                weight = 3.0
                task(
                    jacp,
                    linear_target - self._jdot_qdot(f"footp{side}", jacp, qvel),
                    weight,
                )
                task(
                    jacr,
                    angular_target - self._jdot_qdot(f"footr_swing{side}", jacr, qvel),
                    weight * 0.5,
                )

        # Small regularizers make the QP strictly convex and suppress torque chatter.
        acceleration_reg = np.zeros((nqdd, dimension), dtype=np.float64)
        acceleration_reg[:, :nqdd] = np.eye(nqdd) * np.sqrt(1.0e-4)
        rows.append(acceleration_reg)
        targets.append(np.zeros(nqdd))
        torque_reg = np.zeros((ntau, dimension), dtype=np.float64)
        torque_reg[:, tau_slice] = np.diag(np.sqrt(2.0e-3) / self.force_limit)
        rows.append(torque_reg)
        targets.append(np.sqrt(2.0e-3) * self.previous_tau / self.force_limit)
        if nrho:
            force_reg = np.zeros((nrho, dimension), dtype=np.float64)
            force_reg[:, rho_slice] = np.eye(nrho) * np.sqrt(2.0e-5)
            rows.append(force_reg)
            targets.append(np.zeros(nrho))
        objective_matrix = np.vstack(rows)
        objective_target = np.concatenate(targets)

        def objective(value: np.ndarray) -> float:
            residual = objective_matrix @ value - objective_target
            return 0.5 * float(residual @ residual)

        def gradient(value: np.ndarray) -> np.ndarray:
            residual = objective_matrix @ value - objective_target
            return objective_matrix.T @ residual

        equality_matrix = dynamics
        equality_value = bias
        constraints: list[LinearConstraint] = [
            LinearConstraint(equality_matrix, equality_value, equality_value)
        ]
        normal_constraint_matrix = np.zeros(
            (len(normal_rows), dimension), dtype=np.float64
        )
        if normal_rows:
            normal_constraint_matrix[:, :nqdd] = np.asarray(normal_rows)
            constraints.append(
                LinearConstraint(
                    normal_constraint_matrix,
                    np.asarray(normal_lower),
                    np.full(len(normal_rows), np.inf),
                )
            )

        lower = np.full(dimension, -np.inf, dtype=np.float64)
        upper = np.full(dimension, np.inf, dtype=np.float64)
        lower[:nqdd], upper[:nqdd] = -cfg.qdd_limit, cfg.qdd_limit
        lower[tau_slice], upper[tau_slice] = -self.force_limit, self.force_limit
        if nrho:
            lower[rho_slice] = 0.0
            upper[rho_slice] = self.mass * 9.81
        initial = np.zeros(dimension, dtype=np.float64)
        initial[tau_slice] = np.clip(
            self.previous_tau, -self.force_limit, self.force_limit
        )
        if nrho:
            initial[rho_slice] = self.mass * 9.81 / nrho
        if self.previous_solution is not None:
            shared = min(len(self.previous_solution), dimension)
            initial[:shared] = np.clip(
                self.previous_solution[:shared], lower[:shared], upper[:shared]
            )
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={
                "maxiter": cfg.max_iterations,
                "ftol": cfg.solver_ftol,
                "disp": False,
            },
        )
        value = result.x
        equality_residual = float(np.max(np.abs(dynamics @ value - bias)))
        normal_violation = (
            float(
                np.max(
                    np.maximum(
                        np.asarray(normal_lower) - normal_constraint_matrix @ value,
                        0.0,
                    )
                )
            )
            if normal_rows
            else 0.0
        )
        hard_residual = max(equality_residual, normal_violation)
        success = bool(result.success and hard_residual < 1.0e-3)
        tau = np.asarray(value[tau_slice], dtype=np.float64)
        self.previous_tau = tau.copy()
        self.previous_solution = value.copy()
        position_target = (
            data.qpos[self.engine.actuator_qpos]
            + (tau + self.kd * data.qvel[self.actuator_dofs]) / self.kp
        )
        position_target = np.clip(
            position_target,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        contact_normal_velocity = np.full(2, np.nan, dtype=np.float64)
        contact_normal_acceleration_residual = np.full(2, np.nan, dtype=np.float64)
        contact_tangential_speed = np.full(2, np.nan, dtype=np.float64)
        for side in range(2):
            selected = [contact for contact in contacts if contact.side == side]
            if selected:
                contact_normal_velocity[side] = min(
                    contact.normal_velocity_m_s for contact in selected
                )
                contact_tangential_speed[side] = max(
                    float(
                        np.linalg.norm(
                            contact.velocity
                            - contact.normal
                            * float(np.dot(contact.normal, contact.velocity))
                        )
                    )
                    for contact in selected
                )
            residuals = [
                float(row @ value[:nqdd] + bias_term - target)
                for point_side, row, bias_term, target in contact_kinematics
                if point_side == side
            ]
            if residuals:
                contact_normal_acceleration_residual[side] = max(residuals, key=abs)
        diagnostics = {
            "success": success,
            "status": int(result.status),
            "iterations": int(result.nit),
            "message": str(result.message),
            "solve_time_s": perf_counter() - started,
            "dimension": dimension,
            "equality_residual": equality_residual,
            "normal_constraint_violation": normal_violation,
            "hard_constraint_residual": hard_residual,
            "active_contact": active,
            "active_contact_point_count": contact_count,
            "contact_normal_velocity_m_s": contact_normal_velocity,
            "contact_normal_acceleration_residual_m_s2": contact_normal_acceleration_residual,
            "contact_tangential_speed_m_s": contact_tangential_speed,
            "foot_clearance_m": clearance,
            "foot_normal_force_n": normal,
            "foot_slip_m_s": slip,
            "desired_torque_nm": tau,
            "torque_usage": np.abs(tau) / self.force_limit,
            "torque_at_limit": np.abs(tau) >= 0.995 * self.force_limit,
            "joint_position_error": joint_error,
            "joint_velocity_error": self.ref.joint_velocity[sample] - joint_velocity,
            "com_error": self.ref.com_position[sample]
            - data.subtree_com[self.root_body],
            "com_velocity_error": self.ref.com_velocity[sample] - com_velocity,
            "torso_error": _rotation_error(
                self.ref.torso_rotation[sample], torso_rotation
            ),
            "torso_velocity_error": self.ref.torso_angular_velocity[sample]
            - torso_omega,
            "actual_angular_momentum": data.subtree_angmom[self.root_body].copy(),
            "reference_angular_momentum": self.ref.angular_momentum[sample].copy(),
            "actual_com": data.subtree_com[self.root_body].copy(),
            "actual_com_velocity": com_velocity.copy(),
            "forces": forces,
            "force_coefficients": value[rho_slice].copy(),
            "force_sides": np.asarray(force_sides, dtype=np.int32),
        }
        return position_target, diagnostics


def track_with_whole_body_feedback(
    engine: _DynamicRollout,
    nominal_target: np.ndarray,
    artifact: dict[str, np.ndarray],
    config: WholeBodyFeedbackConfig | None = None,
) -> WholeBodyFeedbackResult:
    """Track one frozen physical reference from its exact frame-85 anchor."""
    cfg = config or WholeBodyFeedbackConfig()
    if nominal_target.shape != engine.reference_joint.shape:
        raise ValueError("nominal target shape does not match the generated reference")
    start_sample = cfg.anchor_frame * engine.config.substeps
    end_sample = cfg.evaluation_end_frame * engine.config.substeps
    ref = _reference_path(engine, artifact, start_sample, end_sample)
    data, prefix, torso_local_up = _execute_prefix(
        engine, nominal_target, cfg.anchor_frame
    )
    anchor_errors = {
        "root_position_m": float(
            np.linalg.norm(
                data.qpos[engine.robot_qpos : engine.robot_qpos + 3]
                - ref.root_position[0]
            )
        ),
        "joint_position_rad": float(
            np.max(np.abs(data.qpos[engine.actuator_qpos] - ref.joint_position[0]))
        ),
        "joint_velocity_rad_s": float(
            np.max(
                np.abs(
                    data.qvel[
                        [
                            engine.model.jnt_dofadr[engine.model.actuator_trnid[i, 0]]
                            for i in range(engine.model.nu)
                        ]
                    ]
                    - ref.joint_velocity[0]
                )
            )
        ),
    }
    prefix["saved_reference_anchor_error"] = anchor_errors
    controller = _WholeBodyController(engine, ref, cfg)
    samples = end_sample - start_sample + 1
    nu = engine.model.nu
    keys_3 = (
        "actual_com",
        "actual_com_velocity",
        "reference_angular_momentum",
        "actual_angular_momentum",
        "com_error",
        "com_velocity_error",
        "torso_error",
        "torso_velocity_error",
    )
    recorded: dict[str, np.ndarray] = {
        key: np.full((samples, 3), np.nan, dtype=np.float64) for key in keys_3
    }
    recorded.update(
        {
            "joint_position": np.full((samples, nu), np.nan),
            "joint_velocity": np.full((samples, nu), np.nan),
            "joint_position_error": np.full((samples, nu), np.nan),
            "joint_velocity_error": np.full((samples, nu), np.nan),
            "desired_torque_nm": np.full((samples, nu), np.nan),
            "actual_torque_nm": np.full((samples, nu), np.nan),
            "torque_usage": np.full((samples, nu), np.nan),
            "torque_at_limit": np.zeros((samples, nu), dtype=bool),
            "root_position": np.full((samples, 3), np.nan),
            "root_velocity": np.full((samples, 3), np.nan),
            "torso_tilt_rad": np.full(samples, np.nan),
            "foot_contact": np.zeros((samples, 2), dtype=bool),
            "active_contact": np.zeros((samples, 2), dtype=bool),
            "foot_normal_force_n": np.zeros((samples, 2)),
            "foot_slip_m_s": np.zeros((samples, 2)),
            "foot_clearance_m": np.full((samples, 2), np.nan),
            "active_contact_point_count": np.zeros((samples, 2), dtype=np.int32),
            "contact_normal_velocity_m_s": np.full((samples, 2), np.nan),
            "contact_normal_acceleration_residual_m_s2": np.full((samples, 2), np.nan),
            "contact_tangential_speed_m_s": np.full((samples, 2), np.nan),
            "solve_time_s": np.full(samples, np.nan),
            "solver_success": np.zeros(samples, dtype=bool),
            "solver_status": np.full(samples, -1, dtype=np.int32),
            "solver_iterations": np.zeros(samples, dtype=np.int32),
            "equality_residual": np.full(samples, np.nan),
            "normal_constraint_violation": np.full(samples, np.nan),
            "hard_constraint_residual": np.full(samples, np.nan),
            "optimization_dimension": np.zeros(samples, dtype=np.int32),
            "nonfoot_terrain": np.zeros(samples, dtype=bool),
            "wrist_terrain": np.zeros(samples, dtype=bool),
            "arm_terrain_contact": np.zeros(samples, dtype=bool),
            "arm_object_contact": np.zeros(samples, dtype=bool),
            "maximum_penetration_m": np.zeros(samples),
            "control_target": np.full((samples, nu), np.nan),
        }
    )
    first_invalid_sample: int | None = None
    invalid_reason: str | None = None
    forbidden_run = 0
    severe_run = 0
    max_penetration = 0.0
    arm_bodies = {
        body
        for body in engine.robot_bodies
        if any(
            token in engine._name(mujoco.mjtObj.mjOBJ_BODY, body)
            for token in ("shoulder", "elbow", "wrist", "hand")
        )
    }

    def record(index: int, diagnostics: dict[str, Any]) -> None:
        nonlocal \
            forbidden_run, \
            severe_run, \
            max_penetration, \
            first_invalid_sample, \
            invalid_reason
        terrain, penetration, wrist = _contact_state(engine, data)
        max_penetration = max(max_penetration, penetration)
        forbidden = terrain - engine.allowed_terrain_bodies
        forbidden_run = forbidden_run + 1 if forbidden else 0
        actual_usage = np.abs(data.actuator_force) / engine.force_scale
        severe = (
            np.mean(actual_usage >= engine.config.severe_saturation_usage)
            >= engine.config.severe_saturation_actuator_fraction
        )
        severe_run = severe_run + 1 if severe else 0
        rotation = data.xmat[engine.torso_body].reshape(3, 3)
        torso_up = rotation @ torso_local_up
        tilt = float(np.arccos(np.clip(torso_up[2], -1.0, 1.0)))
        arm_object = False
        arm_terrain = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if {first, second} & engine.terrain_geoms:
                other = second if first in engine.terrain_geoms else first
                arm_terrain |= int(engine.model.geom_bodyid[other]) in arm_bodies
            if not ({first, second} & engine.object_geoms):
                continue
            other = second if first in engine.object_geoms else first
            arm_object |= int(engine.model.geom_bodyid[other]) in arm_bodies
        recorded["joint_position"][index] = data.qpos[engine.actuator_qpos]
        actuator_dofs = np.asarray(
            [
                engine.model.jnt_dofadr[engine.model.actuator_trnid[i, 0]]
                for i in range(nu)
            ]
        )
        recorded["joint_velocity"][index] = data.qvel[actuator_dofs]
        recorded["actual_torque_nm"][index] = data.actuator_force
        recorded["root_position"][index] = data.qpos[
            engine.robot_qpos : engine.robot_qpos + 3
        ]
        recorded["root_velocity"][index] = data.subtree_linvel[controller.root_body]
        recorded["torso_tilt_rad"][index] = tilt
        recorded["nonfoot_terrain"][index] = bool(forbidden)
        recorded["wrist_terrain"][index] = wrist
        recorded["arm_terrain_contact"][index] = arm_terrain
        recorded["arm_object_contact"][index] = arm_object
        recorded["maximum_penetration_m"][index] = max_penetration
        forces, _, normal, slip, _, _ = _contact_wrench(engine, data)
        recorded["foot_contact"][index] = normal > 1.0e-6
        recorded["foot_normal_force_n"][index] = normal
        recorded["foot_slip_m_s"][index] = slip
        for key in keys_3:
            recorded[key][index] = diagnostics[key]
        for key in (
            "joint_position_error",
            "joint_velocity_error",
            "desired_torque_nm",
            "torque_usage",
            "torque_at_limit",
            "active_contact",
            "foot_clearance_m",
            "active_contact_point_count",
            "contact_normal_velocity_m_s",
            "contact_normal_acceleration_residual_m_s2",
            "contact_tangential_speed_m_s",
            "solve_time_s",
            "equality_residual",
            "normal_constraint_violation",
            "hard_constraint_residual",
        ):
            recorded[key][index] = diagnostics[key]
        recorded["solver_success"][index] = diagnostics["success"]
        recorded["solver_status"][index] = diagnostics["status"]
        recorded["solver_iterations"][index] = diagnostics["iterations"]
        recorded["optimization_dimension"][index] = diagnostics["dimension"]
        reason = None
        if data.qpos[engine.robot_qpos + 2] < engine.minimum_root_height:
            reason = "root_height"
        elif tilt > engine.maximum_torso_tilt:
            reason = "torso_tilt"
        elif forbidden & engine.fall_terrain_bodies:
            reason = "fall_body_terrain_contact"
        elif forbidden_run >= engine.sustained_terrain_steps:
            reason = "sustained_nonfoot_terrain_contact"
        elif max_penetration > engine.config.hard_penetration_m:
            reason = "pathological_penetration"
        elif severe_run >= engine.severe_saturation_steps:
            reason = "sustained_actuator_saturation"
        if reason is not None and first_invalid_sample is None:
            first_invalid_sample = index
            invalid_reason = reason

    # Solve once at the anchor and then update at every 5 ms physics step.
    command, diagnostics = controller.solve(data, 0)
    data.ctrl[:] = command
    recorded["control_target"][0] = command
    record(0, diagnostics)
    for index in range(1, samples):
        mujoco.mj_step(engine.model, data)
        mujoco.mj_forward(engine.model, data)
        command, diagnostics = controller.solve(data, index)
        data.ctrl[:] = command
        recorded["control_target"][index] = command
        record(index, diagnostics)

    first_invalid_frame = (
        None
        if first_invalid_sample is None
        else cfg.anchor_frame
        + int(np.ceil(first_invalid_sample / engine.config.substeps))
    )
    final_window = slice(max(0, samples - 5), samples)
    recoverable = bool(
        first_invalid_sample is None
        and np.any(recorded["foot_contact"][final_window], axis=1).all()
        and np.min(recorded["root_position"][final_window, 2])
        >= engine.minimum_root_height
        and np.max(recorded["torso_tilt_rad"][final_window])
        <= engine.maximum_torso_tilt
        and np.mean(recorded["root_velocity"][final_window, 2]) > -0.35
    )
    valid = first_invalid_sample is None
    large_hard_residual = np.flatnonzero(recorded["hard_constraint_residual"] > 1.0e-3)
    tracking_end = min(samples, (140 - cfg.anchor_frame) * engine.config.substeps + 1)
    root_error = (
        recorded["root_position"][:tracking_end] - ref.root_position[:tracking_end]
    )
    summary = {
        "valid": valid,
        "recoverable_terminal": recoverable,
        "first_invalid_frame": first_invalid_frame,
        "first_invalid_sample": first_invalid_sample,
        "invalid_reason": invalid_reason,
        "anchor_error": anchor_errors,
        "solver_failures": int(np.sum(~recorded["solver_success"])),
        "solver_average_time_s": float(np.nanmean(recorded["solve_time_s"])),
        "solver_worst_time_s": float(np.nanmax(recorded["solve_time_s"])),
        "solver_average_iterations": float(np.mean(recorded["solver_iterations"])),
        "optimization_dimension_min_max": [
            int(np.min(recorded["optimization_dimension"])),
            int(np.max(recorded["optimization_dimension"])),
        ],
        "maximum_dynamics_equality_residual": float(
            np.nanmax(recorded["equality_residual"])
        ),
        "maximum_normal_constraint_violation": float(
            np.nanmax(recorded["normal_constraint_violation"])
        ),
        "maximum_hard_constraint_residual": float(
            np.nanmax(recorded["hard_constraint_residual"])
        ),
        "first_large_hard_residual_sample": (
            None if not len(large_hard_residual) else int(large_hard_residual[0])
        ),
        "first_large_hard_residual_frame": (
            None
            if not len(large_hard_residual)
            else cfg.anchor_frame
            + float(large_hard_residual[0]) / engine.config.substeps
        ),
        "maximum_joint_position_error_rad": float(
            np.nanmax(np.abs(recorded["joint_position_error"]))
        ),
        "maximum_joint_velocity_error_rad_s": float(
            np.nanmax(np.abs(recorded["joint_velocity_error"]))
        ),
        "maximum_torque_usage": float(np.nanmax(recorded["torque_usage"])),
        "torque_limit_active_samples": int(np.sum(recorded["torque_at_limit"])),
        "requested_torque_limit_violations": 0,
        "maximum_actual_torque_usage": float(
            np.nanmax(np.abs(recorded["actual_torque_nm"]) / engine.force_scale)
        ),
        "com_position_rms_error_m": float(
            np.sqrt(np.nanmean(np.square(recorded["com_error"][:tracking_end])))
        ),
        "com_velocity_rms_error_m_s": float(
            np.sqrt(
                np.nanmean(np.square(recorded["com_velocity_error"][:tracking_end]))
            )
        ),
        "root_position_rms_error_m": float(np.sqrt(np.nanmean(np.square(root_error)))),
        "root_height_rms_error_m": float(
            np.sqrt(np.nanmean(np.square(root_error[:, 2])))
        ),
        "torso_orientation_rms_error_rad": float(
            np.sqrt(np.nanmean(np.square(recorded["torso_error"][:tracking_end, :2])))
        ),
        "root_height_frame120_m": float(
            recorded["root_position"][
                (120 - cfg.anchor_frame) * engine.config.substeps, 2
            ]
        ),
        "vertical_momentum_frame120_kg_m_s": float(
            controller.mass
            * recorded["actual_com_velocity"][
                (120 - cfg.anchor_frame) * engine.config.substeps, 2
            ]
        ),
        "torso_tilt_frame120_deg": float(
            np.rad2deg(
                recorded["torso_tilt_rad"][
                    (120 - cfg.anchor_frame) * engine.config.substeps
                ]
            )
        ),
    }
    return WholeBodyFeedbackResult(
        valid=valid,
        recoverable_terminal=recoverable,
        first_invalid_frame=first_invalid_frame,
        first_invalid_sample=first_invalid_sample,
        invalid_reason=invalid_reason,
        data=recorded,
        summary=summary,
        prefix=prefix,
    )
