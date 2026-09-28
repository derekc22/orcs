"""Physical-space diffusion trajectory to native UOLM reference tensors.

This module deliberately starts *after* diffusion inference and denormalization.
It has no model, checkpoint, sampler, or diffusion-repository dependency.  The
input joint coordinates are already in the G1 MuJoCo order; the legacy
``IL2MJ`` clip-file permutation does not belong anywhere in this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_from_matrix,
)
from mocke.mdp.joint_maps import G1_TRACKED_BODIES
from mocke.sonic import profile

from orcs.assets.g1 import get_g1_flat_hand_cfg

__all__ = [
    "GeneratedReference",
    "GeneratedReferenceAdapter",
    "project_generated_joint_positions_",
]

_STATE_DIM = 47
_NUM_JOINTS = 29
_DEFAULT_DT = 0.02
_ROTATION_EPS = 1.0e-6


def project_generated_joint_positions_(
    trajectory: torch.Tensor, soft_joint_pos_limits: torch.Tensor
) -> torch.Tensor:
    """Project generated joints in-place onto the live robot's soft limits.

    ``soft_joint_pos_limits`` is the same per-environment tensor consumed by
    ``MotionCommand._write_reference_state_to_sim``. Projection happens before
    derivatives and FK so every generated robot field describes the state that
    the inherited reset writer will actually place in simulation.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] != _STATE_DIM:
        raise ValueError(
            f"trajectory must have shape [B, T, {_STATE_DIM}], got "
            f"{tuple(trajectory.shape)}"
        )
    expected_limits_shape = (trajectory.shape[0], _NUM_JOINTS, 2)
    if soft_joint_pos_limits.shape != expected_limits_shape:
        raise ValueError(
            "soft_joint_pos_limits must have shape "
            f"{expected_limits_shape}, got {tuple(soft_joint_pos_limits.shape)}"
        )
    if soft_joint_pos_limits.device != trajectory.device:
        raise ValueError(
            "soft joint limits and generated trajectory must be on the same "
            f"device, got {soft_joint_pos_limits.device} and {trajectory.device}"
        )

    lower = soft_joint_pos_limits[:, None, :, 0]
    upper = soft_joint_pos_limits[:, None, :, 1]
    joints = trajectory[..., 9:38]
    joints.copy_(torch.maximum(torch.minimum(joints, upper), lower))
    return trajectory


@dataclass(frozen=True, slots=True)
class GeneratedReference:
    """Complete native UOLM numerical reference, except authored contacts.

    Positions are environment-local.  Linear and angular velocities use world
    axes (the environment frames differ from world only by a translation).
    Body velocities are evaluated at the current MuJoCo link-frame origins.
    """

    root_pos_local: torch.Tensor
    root_quat_wxyz: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    body_pos_local: torch.Tensor
    body_quat_wxyz: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    object_pos_local: torch.Tensor
    object_quat_wxyz: torch.Tensor
    object_lin_vel_w: torch.Tensor
    object_ang_vel_w: torch.Tensor
    dt: float
    tracked_body_names: tuple[str, ...]
    bodywise_object_contact: torch.Tensor | None = None

    @property
    def mujoco_qvel(self) -> torch.Tensor:
        """Return ``[root v_world, root omega_body, joint velocity]``.

        Raw MuJoCo free-joint rotational velocity is root-local even though the
        public ORCS root and body velocity contract is world-frame.
        """
        root_rotation = matrix_from_quat(self.root_quat_wxyz)
        root_ang_vel_b = torch.matmul(
            root_rotation.transpose(-1, -2), self.root_ang_vel_w.unsqueeze(-1)
        ).squeeze(-1)
        return torch.cat(
            [self.root_lin_vel_w, root_ang_vel_b, self.joint_vel], dim=-1
        )

    def require_bodywise_object_contact(self) -> torch.Tensor:
        """Fail explicitly until a later integration supplies contact labels."""
        if self.bodywise_object_contact is None:
            raise RuntimeError(
                "bodywise object contact is unavailable from the 47D generated "
                "trajectory; an authored or separately derived schedule is required"
            )
        return self.bodywise_object_contact


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt decode of two stored rotation-matrix columns."""
    first = rotation_6d[..., 0:3]
    second = rotation_6d[..., 3:6]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    first_unit = F.normalize(first, dim=-1)
    second_orthogonal = second - (first_unit * second).sum(
        dim=-1, keepdim=True
    ) * first_unit
    second_norm = torch.linalg.vector_norm(
        second_orthogonal, dim=-1, keepdim=True
    )
    if bool((first_norm <= _ROTATION_EPS).any() or (second_norm <= _ROTATION_EPS).any()):
        raise ValueError("rotation 6D contains degenerate matrix columns")
    second_unit = F.normalize(second_orthogonal, dim=-1)
    third_unit = torch.cross(first_unit, second_unit, dim=-1)
    return torch.stack([first_unit, second_unit, third_unit], dim=-1)


def _position_derivative(value: torch.Tensor, dt: float) -> torch.Tensor:
    """Central interior and one-sided endpoint derivative along time axis 1."""
    velocity = torch.empty_like(value)
    velocity[:, 0] = (value[:, 1] - value[:, 0]) / dt
    if value.shape[1] > 2:
        velocity[:, 1:-1] = (value[:, 2:] - value[:, :-2]) / (2.0 * dt)
    velocity[:, -1] = (value[:, -1] - value[:, -2]) / dt
    return velocity


def _world_angular_velocity(rotation: torch.Tensor, dt: float) -> torch.Tensor:
    """Spatial SO(3) derivative with copied endpoints for three or more frames."""
    if rotation.shape[1] == 2:
        relative = rotation[:, 1] @ rotation[:, 0].transpose(-1, -2)
        one_step = axis_angle_from_quat(quat_from_matrix(relative)) / dt
        return torch.stack([one_step, one_step], dim=1)

    relative = rotation[:, 2:] @ rotation[:, :-2].transpose(-1, -2)
    interior = axis_angle_from_quat(quat_from_matrix(relative)) / (2.0 * dt)
    return torch.cat([interior[:, :1], interior, interior[:, -1:]], dim=1)


class _BatchedG1Kinematics:
    """Vectorized current-ORCS FK over flattened trajectory frames."""

    def __init__(self, *, capacity: int, dt: float, device: torch.device) -> None:
        self.capacity = capacity
        self.dt = dt
        self.device = device
        device_name = str(device)
        scene_cfg = SceneCfg(
            num_envs=capacity,
            env_spacing=0.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": profile.robot_cfg(base=get_g1_flat_hand_cfg())},
        )
        self.scene = Scene(scene_cfg, device=device_name)
        model = self.scene.compile()
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = dt
        self.sim = Simulation(
            num_envs=capacity,
            cfg=sim_cfg,
            model=model,
            device=device_name,
        )
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]

        if len(self.robot.joint_names) != _NUM_JOINTS:
            raise ValueError(
                f"ORCS G1 has {len(self.robot.joint_names)} joints; expected {_NUM_JOINTS}"
            )
        self.tracked_body_names = tuple(name for name, _ in G1_TRACKED_BODIES)
        missing = [
            name for name in self.tracked_body_names if name not in self.robot.body_names
        ]
        if missing:
            raise ValueError(f"tracked bodies absent from ORCS G1: {missing}")
        self.body_indexes = [
            self.robot.body_names.index(name) for name in self.tracked_body_names
        ]

    @torch.no_grad()
    def evaluate(
        self,
        root_pos_local: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = root_pos_local.shape[0]
        body_count = len(self.body_indexes)
        output_shape = (num_frames, body_count)
        body_pos = torch.empty((*output_shape, 3), device=self.device)
        body_quat = torch.empty((*output_shape, 4), device=self.device)
        body_lin_vel = torch.empty((*output_shape, 3), device=self.device)
        body_ang_vel = torch.empty((*output_shape, 3), device=self.device)
        origins = self.scene.env_origins

        for lower in range(0, num_frames, self.capacity):
            upper = min(lower + self.capacity, num_frames)
            count = upper - lower

            root_state = self.robot.data.default_root_state.clone()
            root_state[:count, 0:3] = root_pos_local[lower:upper] + origins[:count]
            root_state[:count, 3:7] = root_quat_wxyz[lower:upper]
            root_state[:count, 7:10] = root_lin_vel_w[lower:upper]
            root_state[:count, 10:13] = root_ang_vel_w[lower:upper]
            # This world-frame API owns omega_world -> omega_root for MuJoCo qvel.
            self.robot.write_root_state_to_sim(root_state)

            sim_joint_pos = self.robot.data.default_joint_pos.clone()
            sim_joint_vel = self.robot.data.default_joint_vel.clone()
            # Generated joints are already in robot.joint_names / MuJoCo order.
            # IL2MJ is only for legacy clip-file ordering and must not be used.
            sim_joint_pos[:count] = joint_pos[lower:upper]
            sim_joint_vel[:count] = joint_vel[lower:upper]
            self.robot.write_joint_state_to_sim(sim_joint_pos, sim_joint_vel)

            self.sim.forward()
            self.scene.update(self.dt)
            data = self.robot.data
            indexes = self.body_indexes
            body_pos[lower:upper] = (
                data.body_link_pos_w[:count, indexes] - origins[:count, None, :]
            )
            body_quat[lower:upper] = data.body_link_quat_w[:count, indexes]
            body_lin_vel[lower:upper] = data.body_link_lin_vel_w[:count, indexes]
            body_ang_vel[lower:upper] = data.body_link_ang_vel_w[:count, indexes]

            # These are current-ORCS world-axis velocities at each MuJoCo link
            # frame origin. Do not reproduce legacy staged body_lin_vel_w,
            # which was evaluated at URDF inertial/COM points.

        return body_pos, body_quat, body_lin_vel, body_ang_vel


class GeneratedReferenceAdapter:
    """Convert physical ``[B, T, 47]`` trajectories into UOLM references.

    ``fk_batch_size`` is the number of flattened trajectory frames evaluated by
    one vectorized MJLab forward call.  It is independent of public batch size
    ``B`` and avoids a single-environment conversion loop.
    """

    def __init__(
        self,
        *,
        device: str | torch.device,
        dt: float = _DEFAULT_DT,
        fk_batch_size: int = 256,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if fk_batch_size <= 0:
            raise ValueError(
                f"fk_batch_size must be positive, got {fk_batch_size}"
            )
        self.device = torch.device(device)
        self.dt = float(dt)
        self._kinematics = _BatchedG1Kinematics(
            capacity=fk_batch_size, dt=self.dt, device=self.device
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Authoritative input/output joint order: current robot.joint_names."""
        return tuple(self._kinematics.robot.joint_names)

    @property
    def tracked_body_names(self) -> tuple[str, ...]:
        return self._kinematics.tracked_body_names

    @torch.no_grad()
    def __call__(self, trajectory: torch.Tensor) -> GeneratedReference:
        if trajectory.ndim != 3 or trajectory.shape[-1] != _STATE_DIM:
            raise ValueError(
                f"trajectory must have shape [B, T, {_STATE_DIM}], got "
                f"{tuple(trajectory.shape)}"
            )
        if trajectory.shape[0] < 1 or trajectory.shape[1] < 2:
            raise ValueError(
                f"trajectory requires B >= 1 and T >= 2, got {tuple(trajectory.shape[:2])}"
            )
        if trajectory.device != self.device:
            raise ValueError(
                f"trajectory is on {trajectory.device}, adapter is on {self.device}"
            )
        if trajectory.dtype != torch.float32:
            raise TypeError(
                f"trajectory must be torch.float32 for the ORCS runtime, got "
                f"{trajectory.dtype}"
            )
        if not bool(torch.isfinite(trajectory).all()):
            raise ValueError("trajectory contains NaN or Inf")

        batch, horizon, _ = trajectory.shape
        root_pos = trajectory[..., 0:3].clone()
        root_rotation = _rotation_6d_to_matrix(trajectory[..., 3:9])
        root_quat = F.normalize(quat_from_matrix(root_rotation), dim=-1)
        joint_pos = trajectory[..., 9:38].clone()
        object_pos = trajectory[..., 38:41].clone()

        object_rotation_encoded = _rotation_6d_to_matrix(trajectory[..., 41:47])
        object_quat_encoded_wxyz = F.normalize(
            quat_from_matrix(object_rotation_encoded), dim=-1
        )
        # CHECKPOINT COMPATIBILITY — DO NOT "FIX" THIS TO A NORMAL REORDER.
        # The training data's physical wxyz components were historically fed to
        # an xyzw decoder.  Recovering the physical object rotation therefore
        # means taking the encoded quaternion's xyzw components and assigning
        # those four numbers directly to ORCS's wxyz field.
        object_quat = object_quat_encoded_wxyz.roll(-1, dims=-1)
        object_rotation_physical = matrix_from_quat(object_quat)

        root_lin_vel = _position_derivative(root_pos, self.dt)
        root_ang_vel = _world_angular_velocity(root_rotation, self.dt)
        joint_vel = _position_derivative(joint_pos, self.dt)
        object_lin_vel = _position_derivative(object_pos, self.dt)
        object_ang_vel = _world_angular_velocity(
            object_rotation_physical, self.dt
        )

        flat = lambda value: value.reshape(batch * horizon, *value.shape[2:])
        body_values = self._kinematics.evaluate(
            flat(root_pos),
            flat(root_quat),
            flat(root_lin_vel),
            flat(root_ang_vel),
            flat(joint_pos),
            flat(joint_vel),
        )
        body_pos, body_quat, body_lin_vel, body_ang_vel = (
            value.reshape(batch, horizon, *value.shape[1:]) for value in body_values
        )

        return GeneratedReference(
            root_pos_local=root_pos,
            root_quat_wxyz=root_quat,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_lin_vel_w=root_lin_vel,
            root_ang_vel_w=root_ang_vel,
            body_pos_local=body_pos,
            body_quat_wxyz=body_quat,
            body_lin_vel_w=body_lin_vel,
            body_ang_vel_w=body_ang_vel,
            object_pos_local=object_pos,
            object_quat_wxyz=object_quat,
            object_lin_vel_w=object_lin_vel,
            object_ang_vel_w=object_ang_vel,
            dt=self.dt,
            tracked_body_names=self.tracked_body_names,
            bodywise_object_contact=None,
        )
