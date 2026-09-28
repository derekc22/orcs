from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_from_matrix,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mocke.mdp.joint_maps import G1_TRACKED_BODIES, IL2MJ

from orcs.assets.g1 import flat_hand_spec
from orcs.tasks.uolm.generated_reference import (
    GeneratedReferenceAdapter,
    project_generated_joint_positions_,
)

DT = 0.02
BATCH = 2
HORIZON = 4


def _rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)


def _decode_rotation_6d(rotation_6d: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(rotation_6d[..., :3], dim=-1)
    second = rotation_6d[..., 3:]
    second = second - (first * second).sum(-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    return torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-1)


def _angle_error(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    a_wxyz = a_wxyz / np.linalg.norm(a_wxyz)
    b_wxyz = b_wxyz / np.linalg.norm(b_wxyz)
    dot = float(np.clip(np.abs(np.dot(a_wxyz, b_wxyz)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)


def _hinge_names(model: mujoco.MjModel) -> list[str]:
    joint_ids = sorted(
        (
            index
            for index in range(model.njnt)
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
        ),
        key=lambda index: model.jnt_qposadr[index],
    )
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in joint_ids
    ]


def _trajectory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.arange(HORIZON, dtype=torch.float32) * DT
    trajectory = torch.zeros(BATCH, HORIZON, 47, dtype=torch.float32)
    root_rotations = []
    physical_object_quaternions = []

    for batch in range(BATCH):
        root_pos = torch.stack(
            [
                2.0 + 3.0 * batch + 0.4 * time + 0.1 * time.square(),
                -1.5 - batch + 0.2 * time,
                1.1 + 0.03 * time.square(),
            ],
            dim=-1,
        )
        root_quat = quat_from_euler_xyz(
            0.15 + 0.1 * time,
            -0.2 + 0.04 * time.square(),
            0.3 * batch + 0.35 * time,
        )
        root_rotation = matrix_from_quat(root_quat)

        joint_base = torch.linspace(-0.25, 0.3, 29)
        joint_slope = torch.linspace(-0.15, 0.2, 29)
        joint_pos = joint_base + 0.02 * batch + time[:, None] * joint_slope
        joint_pos += 0.03 * time[:, None].square() * torch.linspace(0.0, 1.0, 29)

        object_pos = torch.stack(
            [
                -0.7 + batch + 0.25 * time,
                0.8 - 0.1 * time.square(),
                0.65 + 0.05 * time,
            ],
            dim=-1,
        )
        object_quat_physical = quat_from_euler_xyz(
            -0.25 + 0.12 * time,
            0.35 + 0.08 * time,
            -0.4 * batch + 0.2 * time.square(),
        )
        # Historical preprocessing interpreted physical [w,x,y,z] components as
        # [x,y,z,w]. In a wxyz math API that encoded quaternion is [z,w,x,y].
        object_quat_encoded_wxyz = object_quat_physical.roll(1, dims=-1)
        object_rotation_encoded = matrix_from_quat(object_quat_encoded_wxyz)

        trajectory[batch, :, 0:3] = root_pos
        trajectory[batch, :, 3:9] = _rotation_6d(root_rotation)
        trajectory[batch, :, 9:38] = joint_pos
        trajectory[batch, :, 38:41] = object_pos
        trajectory[batch, :, 41:47] = _rotation_6d(object_rotation_encoded)
        root_rotations.append(root_rotation)
        physical_object_quaternions.append(object_quat_physical)

    return (
        trajectory,
        torch.stack(root_rotations),
        torch.stack(physical_object_quaternions),
    )


class GeneratedReferenceAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory, cls.root_rotation, cls.object_quat_physical = _trajectory()
        cls.adapter = GeneratedReferenceAdapter(
            device="cpu", dt=DT, fk_batch_size=BATCH * HORIZON
        )
        cls.reference = cls.adapter(cls.trajectory)

    def test_shape_batch_dtype_device_and_contacts(self) -> None:
        reference = self.reference
        expected_shapes = {
            "root_pos_local": (BATCH, HORIZON, 3),
            "root_quat_wxyz": (BATCH, HORIZON, 4),
            "joint_pos": (BATCH, HORIZON, 29),
            "joint_vel": (BATCH, HORIZON, 29),
            "root_lin_vel_w": (BATCH, HORIZON, 3),
            "root_ang_vel_w": (BATCH, HORIZON, 3),
            "body_pos_local": (BATCH, HORIZON, 14, 3),
            "body_quat_wxyz": (BATCH, HORIZON, 14, 4),
            "body_lin_vel_w": (BATCH, HORIZON, 14, 3),
            "body_ang_vel_w": (BATCH, HORIZON, 14, 3),
            "object_pos_local": (BATCH, HORIZON, 3),
            "object_quat_wxyz": (BATCH, HORIZON, 4),
            "object_lin_vel_w": (BATCH, HORIZON, 3),
            "object_ang_vel_w": (BATCH, HORIZON, 3),
        }
        for field, shape in expected_shapes.items():
            value = getattr(reference, field)
            self.assertEqual(value.shape, shape, field)
            self.assertEqual(value.dtype, torch.float32, field)
            self.assertEqual(value.device, self.trajectory.device, field)
        self.assertEqual(reference.mujoco_qvel.shape, (BATCH, HORIZON, 35))
        self.assertEqual(reference.dt, DT)
        self.assertEqual(len(reference.tracked_body_names), 14)
        self.assertIsNone(reference.bodywise_object_contact)
        with self.assertRaisesRegex(RuntimeError, "contact is unavailable"):
            reference.require_bodywise_object_contact()
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            self.adapter(self.trajectory.double())

    def test_joint_order_is_identity_and_never_il2mj(self) -> None:
        expected = self.trajectory[..., 9:38]
        torch.testing.assert_close(
            self.reference.joint_pos, expected, rtol=0.0, atol=0.0
        )
        self.assertEqual(
            self.adapter.joint_names, tuple(_hinge_names(flat_hand_spec().compile()))
        )
        self.assertFalse(torch.equal(self.reference.joint_pos, expected[..., IL2MJ]))

    def test_joint_limit_projection_precedes_derivatives_fk_and_reset(self) -> None:
        trajectory = self.trajectory.clone()
        limits = torch.empty(BATCH, 29, 2, dtype=trajectory.dtype)
        limits[..., 0] = -1.0
        limits[..., 1] = 1.0
        limits[:, 0, 0] = -0.1
        limits[:, 0, 1] = 0.1
        trajectory[:, :, 9] = torch.tensor(
            [[-0.4, -0.05, 0.2, 0.4], [0.5, 0.05, -0.2, -0.5]]
        )

        project_generated_joint_positions_(trajectory, limits)
        reference = self.adapter(trajectory)
        expected_joint = torch.tensor(
            [[-0.1, -0.05, 0.1, 0.1], [0.1, 0.05, -0.1, -0.1]]
        )
        torch.testing.assert_close(
            reference.joint_pos[:, :, 0], expected_joint, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            reference.joint_vel[:, 1:-1, 0],
            (expected_joint[:, 2:] - expected_joint[:, :-2]) / (2.0 * DT),
            rtol=0.0,
            atol=0.0,
        )

        # Independent MuJoCo FK for a projected frame must reproduce the stored
        # body reference, rather than the pose from the raw out-of-limit joint.
        model = flat_hand_spec().compile()
        data = mujoco.MjData(model)
        data.qpos[:3] = reference.root_pos_local[0, 0].numpy()
        data.qpos[3:7] = reference.root_quat_wxyz[0, 0].numpy()
        data.qpos[7:36] = reference.joint_pos[0, 0].numpy()
        mujoco.mj_forward(model, data)
        left_hip_roll = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_roll_link"
        )
        tracked_index = reference.tracked_body_names.index("left_hip_roll_link")
        np.testing.assert_allclose(
            data.xpos[left_hip_roll],
            reference.body_pos_local[0, 0, tracked_index].numpy(),
            rtol=0.0,
            atol=2e-6,
        )

        # Exercise the pinned inherited reset writer. Since storage was already
        # projected with these limits, its safety clamp must leave frame zero
        # unchanged.
        class FakeRobot:
            def __init__(self) -> None:
                self.data = type("Data", (), {"soft_joint_pos_limits": limits})()
                self.written_joint_pos = None

            def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids) -> None:
                self.written_joint_pos = joint_pos.clone()

            def write_root_state_to_sim(self, root_state, env_ids) -> None:
                pass

            def reset(self, env_ids) -> None:
                pass

        writer = type("Writer", (), {})()
        writer.robot = FakeRobot()
        env_ids = torch.arange(BATCH)
        MotionCommand._write_reference_state_to_sim(
            writer,
            env_ids,
            reference.root_pos_local[:, 0],
            reference.root_quat_wxyz[:, 0],
            reference.root_lin_vel_w[:, 0],
            reference.root_ang_vel_w[:, 0],
            reference.joint_pos[:, 0],
            reference.joint_vel[:, 0],
        )
        torch.testing.assert_close(
            writer.robot.written_joint_pos,
            reference.joint_pos[:, 0],
            rtol=0.0,
            atol=0.0,
        )

    def test_robot_and_checkpoint_specific_object_rotations(self) -> None:
        torch.testing.assert_close(
            matrix_from_quat(self.reference.root_quat_wxyz),
            self.root_rotation,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(self.reference.root_quat_wxyz, dim=-1),
            torch.ones(BATCH, HORIZON),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            matrix_from_quat(self.reference.object_quat_wxyz),
            matrix_from_quat(self.object_quat_physical),
            rtol=1e-5,
            atol=1e-6,
        )

        encoded_rotation = _decode_rotation_6d(self.trajectory[..., 41:47])
        encoded_wxyz = quat_from_matrix(encoded_rotation)
        # The checkpoint's encoded xyzw components become ORCS wxyz unchanged.
        torch.testing.assert_close(
            self.reference.object_quat_wxyz,
            encoded_wxyz.roll(-1, dims=-1),
            rtol=0.0,
            atol=1e-7,
        )
        ordinary_rotation = matrix_from_quat(encoded_wxyz)
        corrected_rotation = matrix_from_quat(self.reference.object_quat_wxyz)
        self.assertGreater(
            float(torch.max(torch.abs(ordinary_rotation - corrected_rotation))), 0.1
        )

    def test_derivatives_and_raw_mujoco_qvel(self) -> None:
        for value, velocity in (
            (self.trajectory[..., 0:3], self.reference.root_lin_vel_w),
            (self.trajectory[..., 9:38], self.reference.joint_vel),
            (self.trajectory[..., 38:41], self.reference.object_lin_vel_w),
        ):
            torch.testing.assert_close(
                velocity[:, 0],
                (value[:, 1] - value[:, 0]) / DT,
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                velocity[:, 1:-1],
                (value[:, 2:] - value[:, :-2]) / (2.0 * DT),
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                velocity[:, -1],
                (value[:, -1] - value[:, -2]) / DT,
                rtol=1e-6,
                atol=1e-6,
            )

        relative = self.root_rotation[:, 2:] @ self.root_rotation[
            :, :-2
        ].transpose(-1, -2)
        interior = axis_angle_from_quat(quat_from_matrix(relative)) / (2.0 * DT)
        torch.testing.assert_close(self.reference.root_ang_vel_w[:, 1:-1], interior)
        torch.testing.assert_close(self.reference.root_ang_vel_w[:, 0], interior[:, 0])
        torch.testing.assert_close(
            self.reference.root_ang_vel_w[:, -1], interior[:, -1]
        )

        expected_local = torch.matmul(
            self.root_rotation.transpose(-1, -2),
            self.reference.root_ang_vel_w.unsqueeze(-1),
        ).squeeze(-1)
        torch.testing.assert_close(
            self.reference.mujoco_qvel[..., 3:6], expected_local
        )
        torch.testing.assert_close(
            self.reference.mujoco_qvel[..., :3], self.reference.root_lin_vel_w
        )
        torch.testing.assert_close(
            self.reference.mujoco_qvel[..., 6:], self.reference.joint_vel
        )

    def test_two_frame_rotation_uses_one_step_velocity(self) -> None:
        reference = self.adapter(self.trajectory[:, :2])
        torch.testing.assert_close(
            reference.root_ang_vel_w[:, 0], reference.root_ang_vel_w[:, 1]
        )
        torch.testing.assert_close(
            reference.object_ang_vel_w[:, 0], reference.object_ang_vel_w[:, 1]
        )

    def test_current_orcs_link_fk_velocity_and_local_origin(self) -> None:
        model = flat_hand_spec().compile()
        data = mujoco.MjData(model)
        body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name, _ in G1_TRACKED_BODIES
        ]
        root_id = body_ids[0]
        maxima = {"pos": 0.0, "quat": 0.0, "lin": 0.0, "ang": 0.0}

        for batch in range(BATCH):
            for frame in range(HORIZON):
                data.qpos[:3] = self.reference.root_pos_local[batch, frame].numpy()
                data.qpos[3:7] = self.reference.root_quat_wxyz[batch, frame].numpy()
                data.qpos[7:36] = self.reference.joint_pos[batch, frame].numpy()
                data.qvel[:35] = self.reference.mujoco_qvel[batch, frame].numpy()
                mujoco.mj_forward(model, data)

                for index, body_id in enumerate(body_ids):
                    angular = data.cvel[body_id, :3]
                    linear = data.cvel[body_id, 3:6] - np.cross(
                        angular, data.subtree_com[root_id] - data.xpos[body_id]
                    )
                    maxima["pos"] = max(
                        maxima["pos"],
                        float(
                            np.linalg.norm(
                                data.xpos[body_id]
                                - self.reference.body_pos_local[
                                    batch, frame, index
                                ].numpy()
                            )
                        ),
                    )
                    maxima["quat"] = max(
                        maxima["quat"],
                        _angle_error(
                            data.xquat[body_id],
                            self.reference.body_quat_wxyz[
                                batch, frame, index
                            ].numpy(),
                        ),
                    )
                    maxima["lin"] = max(
                        maxima["lin"],
                        float(
                            np.linalg.norm(
                                linear
                                - self.reference.body_lin_vel_w[
                                    batch, frame, index
                                ].numpy()
                            )
                        ),
                    )
                    maxima["ang"] = max(
                        maxima["ang"],
                        float(
                            np.linalg.norm(
                                angular
                                - self.reference.body_ang_vel_w[
                                    batch, frame, index
                                ].numpy()
                            )
                        ),
                    )

        self.assertLess(maxima["pos"], 2e-6)
        # The adapter stores float32 quaternions; arccos magnifies their last-bit
        # normalization error into a sub-milliradian angular residual.
        self.assertLess(maxima["quat"], 1e-3)
        self.assertLess(maxima["lin"], 3e-6)
        self.assertLess(maxima["ang"], 3e-6)
        pelvis = self.reference.tracked_body_names.index("pelvis")
        torch.testing.assert_close(
            self.reference.body_pos_local[..., pelvis, :],
            self.reference.root_pos_local,
            rtol=1e-5,
            atol=2e-6,
        )
        self.assertGreater(float(self.reference.root_pos_local[1, 0, 0]), 4.0)


if __name__ == "__main__":
    unittest.main()
