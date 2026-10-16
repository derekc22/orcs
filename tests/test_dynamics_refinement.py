import unittest

import numpy as np

from orcs.tasks.uolm.balance_mpc import BalanceMPCConfig, _decode_residual
from orcs.tasks.uolm.centroidal_reference_repair import _piecewise_correction
from orcs.tasks.uolm.contact_dynamics_refinement import _residual_path
from orcs.tasks.uolm.dynamics_refinement import (
    _catmull_rom_basis,
    _spline_correction,
    detect_pickup_window,
)
from orcs.tasks.uolm.reference_repair import _smooth_correction
from orcs.tasks.uolm.support_diagnostics import _convex_hull, _polygon_margin


class PickupWindowTest(unittest.TestCase):
    def test_ignores_initial_noise_and_bounds_pickup(self) -> None:
        rng = np.random.default_rng(7)
        position = np.zeros((300, 3), dtype=np.float64)
        position[:, 2] = 0.2
        position[:30] += rng.normal(0.0, 0.004, size=(30, 3))
        position[60:101, 2] += np.linspace(0.0, 0.4, 41)
        position[101:, 2] += 0.4
        position[101:, 1] = np.linspace(0.0, 0.8, 199)

        start, end = detect_pickup_window(position)

        self.assertGreaterEqual(start, 40)
        self.assertLess(start, 60)
        self.assertGreaterEqual(end - start + 1, 60)
        self.assertLessEqual(end - start + 1, 120)
        self.assertGreater(end, 101)

    def test_cubic_basis_has_exact_splice_boundaries(self) -> None:
        basis = _catmull_rom_basis(120, 5)
        np.testing.assert_array_equal(basis[0], [1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(basis[-1], [0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1.0e-12)

    def test_initial_knot_can_be_free_while_final_knot_is_zero(self) -> None:
        parameters = np.zeros((4, 2), dtype=np.float64)
        parameters[0] = [0.1, -0.15]

        correction = _spline_correction(
            300,
            3,
            np.asarray([0, 2]),
            5,
            parameters,
            optimize_first=True,
        )

        np.testing.assert_allclose(correction[0], [0.1, 0.0, -0.15])
        np.testing.assert_array_equal(correction[-1], np.zeros(3))

    def test_mpc_residual_is_bounded_and_rate_limited(self) -> None:
        config = BalanceMPCConfig(
            horizon_frames=12,
            residual_bound_rad=0.15,
            residual_rate_limit_rad_per_frame=0.01,
        )
        parameters = np.full(4, 0.15)

        path = _decode_residual(parameters, np.zeros(2), 12, config)

        self.assertEqual(path.shape, (13, 2))
        self.assertLessEqual(float(np.abs(path).max()), 0.15)
        self.assertLessEqual(float(np.abs(np.diff(path, axis=0)).max()), 0.01001)

    def test_reference_repair_splice_has_zero_value_and_slope(self) -> None:
        parameters = np.linspace(-0.2, 0.2, 40)
        correction = _smooth_correction(parameters, 61)

        np.testing.assert_allclose(correction[[0, -1]], 0.0)
        self.assertLess(float(np.abs(correction[1] - correction[0]).max()), 1.0e-3)
        self.assertLess(float(np.abs(correction[-1] - correction[-2]).max()), 1.0e-3)

    def test_support_polygon_margin_is_signed(self) -> None:
        points = np.asarray(
            [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]]
        )
        hull = _convex_hull(points)

        self.assertAlmostEqual(_polygon_margin(np.asarray([0.0, 0.0]), hull), 1.0)
        self.assertAlmostEqual(_polygon_margin(np.asarray([1.25, 0.0]), hull), -0.25)

    def test_centroidal_repair_knots_are_c2_and_end_at_zero(self) -> None:
        knots = np.asarray([[0.2, -0.1], [-0.1, 0.3], [0.0, 0.0]])
        correction = _piecewise_correction(knots, 61)

        np.testing.assert_allclose(correction[[0, 30, 60]], knots, atol=1.0e-12)
        np.testing.assert_allclose(correction[1] - correction[0], 0.0, atol=2.0e-4)
        np.testing.assert_allclose(correction[60] - correction[59], 0.0, atol=2.0e-4)

    def test_centroidal_extension_preserves_early_anchor(self) -> None:
        knots = np.asarray([[0.0], [0.15], [-0.10], [0.05], [0.0]], dtype=np.float64)
        correction = _piecewise_correction(knots, 111, (35, 55, 75, 95, 110))

        np.testing.assert_array_equal(correction[:36], np.zeros((36, 1)))
        np.testing.assert_allclose(correction[[35, 55, 75, 95, 110]], knots)
        self.assertLess(abs(float(correction[36, 0] - correction[35, 0])), 2.0e-4)

    def test_contact_dynamics_residual_preserves_anchor_and_exit(self) -> None:
        parameters = np.linspace(-0.12, 0.12, 8, dtype=np.float64)
        frames = (85, 95, 110, 125, 140, 160)

        correction = _residual_path(parameters, 300, 2, frames)

        np.testing.assert_array_equal(correction[:86], np.zeros((86, 2)))
        np.testing.assert_array_equal(correction[160:], np.zeros((140, 2)))
        np.testing.assert_allclose(
            correction[np.asarray(frames[1:-1])], parameters.reshape(4, 2)
        )
        self.assertLess(float(np.abs(correction[86] - correction[85]).max()), 1.1e-3)
        self.assertLess(float(np.abs(correction[160] - correction[159]).max()), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
