import numpy as np

from orcs.tasks.uolm.wbc_reference_refinement import (
    WBCReferenceRefinementConfig,
    _residual_path,
)


def test_wbc_reference_space_is_compact_and_boundary_fixed() -> None:
    config = WBCReferenceRefinementConfig()
    assert config.knot_frames == (85, 95, 105, 120, 140, 160)
    assert (len(config.knot_frames) - 2) * 18 == 72
    assert config.wbc.anchor_frame == 85
    assert config.wbc.evaluation_end_frame == 160


def test_reference_residual_is_zero_at_fixed_boundaries() -> None:
    parameters = np.linspace(-0.1, 0.1, 4 * 18)
    knot_samples = (0, 40, 80, 140, 220, 300)
    residual = _residual_path(parameters, 301, 18, knot_samples)
    np.testing.assert_allclose(residual[0], 0.0)
    np.testing.assert_allclose(residual[-1], 0.0)
    np.testing.assert_allclose(residual[40], parameters[:18])
    np.testing.assert_allclose(residual[220], parameters[-18:])
