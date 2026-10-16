import numpy as np

from orcs.tasks.uolm.whole_body_feedback import (
    WholeBodyFeedbackConfig,
    _rotation_error,
)


def test_rotation_error_is_zero_for_identical_frames() -> None:
    np.testing.assert_allclose(_rotation_error(np.eye(3), np.eye(3)), 0.0)


def test_rotation_error_has_expected_small_angle_direction() -> None:
    angle = 0.01
    desired = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        _rotation_error(desired, np.eye(3)), [0.0, 0.0, np.sin(angle)]
    )


def test_experiment_window_and_contact_friction_are_frozen() -> None:
    config = WholeBodyFeedbackConfig()
    assert (config.anchor_frame, config.evaluation_end_frame) == (85, 160)
    assert config.friction == 0.60
