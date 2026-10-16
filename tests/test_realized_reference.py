from __future__ import annotations

import json

import numpy as np
import pytest

from orcs.tasks.uolm.realized_reference import load_realized_reference


def test_load_realized_reference(tmp_path) -> None:
    qpos = np.zeros((300, 43))
    qvel = np.zeros((300, 41))
    qpos[:, 3] = 1.0
    qpos[:, 39] = 1.0
    np.savez_compressed(
        tmp_path / "physical_reference.npz",
        qpos=qpos,
        qvel=qvel,
        ctrl=np.zeros((300, 29)),
        simulation_trace=np.zeros((1196, 1)),
        diffusion_prior=np.zeros((1, 300, 47)),
        goal_position=np.zeros(3),
        goal_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    (tmp_path / "validation.json").write_text(
        json.dumps({"full_valid": True, "at_goal": True})
    )
    artifact = load_realized_reference(tmp_path)
    assert artifact.is_task_success
    assert artifact.qpos.shape == (300, 43)
    assert artifact.qvel.shape == (300, 41)


def test_load_realized_reference_rejects_wrong_horizon(tmp_path) -> None:
    np.savez_compressed(
        tmp_path / "physical_reference.npz",
        qpos=np.zeros((20, 43)),
        qvel=np.zeros((20, 41)),
        ctrl=np.zeros((20, 29)),
        simulation_trace=np.zeros((20, 1)),
        diffusion_prior=np.zeros((1, 300, 47)),
        goal_position=np.zeros(3),
        goal_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    (tmp_path / "validation.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly 300"):
        load_realized_reference(tmp_path)
