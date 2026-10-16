"""Load a MuJoCo-realized trajectory as a native UOLM reference.

The experimental dynamics-aware pipeline writes the state reached by MuJoCo,
not the nominal controls used to produce it.  This module is the narrow bridge
from that artifact back to the standard :class:`GeneratedReference` contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from orcs.tasks.uolm.generated_reference import GeneratedReference


@dataclass(frozen=True, slots=True)
class RealizedReferenceArtifact:
    """A continuously simulated robot/object reference and its validation."""

    path: Path
    qpos: np.ndarray
    qvel: np.ndarray
    control: np.ndarray
    simulation_trace: np.ndarray
    diffusion_prior: np.ndarray
    goal_position: np.ndarray
    goal_quaternion: np.ndarray
    validation: dict[str, object]

    @property
    def is_task_success(self) -> bool:
        return bool(self.validation.get("full_valid")) and bool(
            self.validation.get("at_goal")
        )

    def to_generated_reference(
        self,
        *,
        device: torch.device | str = "cpu",
        dt: float = 0.02,
        fk_batch_size: int = 256,
    ) -> GeneratedReference:
        """Return the physical state sequence in the standard ORCS layout.

        Body poses are recomputed using the normal G1 FK path.  Root, object,
        and joint velocities are taken from MuJoCo rather than re-differenced.
        MuJoCo stores free-joint angular velocity in the local body frame, so
        it is rotated into the world-frame convention used by ORCS.
        """
        from dataclasses import replace

        import torch
        from mjlab.utils.lab_api.math import matrix_from_quat

        from orcs.tasks.uolm.generated_reference import GeneratedReferenceAdapter

        device = torch.device(device)
        qpos = torch.as_tensor(self.qpos, dtype=torch.float32, device=device)
        qvel = torch.as_tensor(self.qvel, dtype=torch.float32, device=device)
        if qpos.ndim != 2 or qpos.shape[1] != 43:
            raise ValueError(f"expected physical qpos shape (T, 43), got {qpos.shape}")
        if qvel.shape != (qpos.shape[0], 41):
            raise ValueError(f"expected physical qvel shape (T, 41), got {qvel.shape}")

        root_rotation = matrix_from_quat(qpos[:, 3:7])
        object_rotation = matrix_from_quat(qpos[:, 39:43])
        root_rotation_6d = torch.cat(
            [root_rotation[:, :, 0], root_rotation[:, :, 1]], dim=-1
        )
        object_rotation_6d = torch.cat(
            [object_rotation[:, :, 0], object_rotation[:, :, 1]], dim=-1
        )
        trajectory = torch.cat(
            [
                qpos[:, 0:3],
                root_rotation_6d,
                qpos[:, 7:36],
                qpos[:, 36:39],
                object_rotation_6d,
            ],
            dim=-1,
        ).unsqueeze(0)
        reference = GeneratedReferenceAdapter(
            device=device,
            dt=dt,
            fk_batch_size=fk_batch_size,
        )(trajectory)
        root_ang_world = torch.matmul(
            root_rotation, qvel[:, 3:6].unsqueeze(-1)
        ).squeeze(-1)
        object_ang_world = torch.matmul(
            object_rotation, qvel[:, 38:41].unsqueeze(-1)
        ).squeeze(-1)
        return replace(
            reference,
            root_lin_vel_w=qvel[None, :, 0:3],
            root_ang_vel_w=root_ang_world[None],
            joint_vel=qvel[None, :, 6:35],
            object_lin_vel_w=qvel[None, :, 35:38],
            object_ang_vel_w=object_ang_world[None],
        )


def load_realized_reference(path: str | Path) -> RealizedReferenceArtifact:
    """Load and validate a physical-reference artifact directory or NPZ path."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    npz_path = directory / "physical_reference.npz" if path.is_dir() else path
    report_path = directory / "validation.json"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    validation = json.loads(report_path.read_text())
    with np.load(npz_path) as data:
        artifact = RealizedReferenceArtifact(
            path=npz_path,
            qpos=data["qpos"].copy(),
            qvel=data["qvel"].copy(),
            control=data["ctrl"].copy(),
            simulation_trace=data["simulation_trace"].copy(),
            diffusion_prior=data["diffusion_prior"].copy(),
            goal_position=data["goal_position"].copy(),
            goal_quaternion=data["goal_quaternion"].copy(),
            validation=validation,
        )
    if artifact.qpos.shape[0] != 300 or artifact.qvel.shape[0] != 300:
        raise ValueError("a final physical reference must contain exactly 300 frames")
    if artifact.control.shape != (300, 29):
        raise ValueError(
            f"a final physical reference must contain controls with shape (300, 29), "
            f"got {artifact.control.shape}"
        )
    if not np.isfinite(artifact.qpos).all() or not np.isfinite(artifact.qvel).all():
        raise ValueError("physical reference contains non-finite state")
    if not np.isfinite(artifact.control).all():
        raise ValueError("physical reference contains non-finite controls")
    return artifact
