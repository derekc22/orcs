import numpy as np

from orcs.tasks.uolm.torque_shooting_mpc import (
    TorqueShootingMPCConfig,
    _decode_residual,
)


def test_torque_mpc_configuration_matches_bounded_experiment() -> None:
    config = TorqueShootingMPCConfig()
    assert config.horizon_frames == 12
    assert config.replan_interval_frames == 2
    assert config.future_knots == 2
    assert config.residual_limit_fraction == 0.25
    assert config.population == 24
    assert config.elite_count == 5
    assert config.generations == 2
    assert config.future_knots * 15 == 30


def test_full_body_configuration_changes_only_actuator_space() -> None:
    baseline = TorqueShootingMPCConfig()
    full_body = TorqueShootingMPCConfig(
        control_all_actuators=True,
        population=32,
        elite_count=6,
        generations=3,
    )
    assert full_body.horizon_frames == baseline.horizon_frames
    assert full_body.replan_interval_frames == baseline.replan_interval_frames
    assert full_body.future_knots == baseline.future_knots
    assert full_body.residual_limit_fraction == 0.25
    assert full_body.arm_residual_limit_fraction == 0.15
    assert full_body.future_knots * 29 == 58


def test_torque_residual_path_is_continuous_and_hits_knots() -> None:
    current = np.asarray([1.0, -1.0])
    parameters = np.asarray([2.0, -2.0, 3.0, -3.0])
    path = _decode_residual(parameters, current, 48, 2)
    np.testing.assert_allclose(path[0], current)
    np.testing.assert_allclose(path[24], parameters[:2])
    np.testing.assert_allclose(path[48], parameters[2:])
    np.testing.assert_allclose(np.diff(path, axis=0)[[0, 23, 24, 47]], 0.0, atol=0.02)
