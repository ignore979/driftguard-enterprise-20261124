from __future__ import annotations

import copy
import unittest

import numpy as np

from zhixun.bearing_study import (
    bearing_gate,
    bearing_metrics,
    bearing_weights,
    fit_weighted_bearing_epoch,
)
from zhixun.data import Dataset
from zhixun.vendor.model import Model


class BearingStudyTests(unittest.TestCase):
    def test_weighted_mse_has_the_specified_device_mass(self):
        groups = np.array(["long", "long", "long", "short"])
        weights = bearing_weights(groups, .5)
        squared_errors = np.array([1., 4., 9., 16.])
        expected = .5 * squared_errors.mean() + .25 * (squared_errors[:3].mean() + 16)
        self.assertAlmostEqual(np.mean(weights * squared_errors), expected)
        self.assertAlmostEqual(weights.mean(), 1)

    def test_unit_weights_preserve_vendor_updates(self):
        x = np.arange(14, dtype=np.float64).reshape(7, 2)
        y = np.array([0., 40., 60., 30., 0., 10., 11.])
        ds = Dataset(tuple(str(i) for i in range(len(x))), x, y, frozenset())
        base = Model("regression", ["first", "second"], [], x, y, hidden=3, batch_size=3)
        base.meta["profile"] = "B"
        weighted = copy.deepcopy(base)
        expected_losses = base.fit(x, y, 2)
        actual_losses = [fit_weighted_bearing_epoch(weighted, ds, np.ones(len(x))) for _ in range(2)]
        np.testing.assert_array_equal(actual_losses, expected_losses)
        self.assertEqual(weighted.meta, base.meta)
        for field in ("params", "first", "second"):
            for name, expected in getattr(base, field).items():
                np.testing.assert_array_equal(getattr(weighted, field)[name], expected)

    def test_gate_rejects_improvement_on_only_the_long_bearing(self):
        groups = np.array(["long"] * 8 + ["short-a", "short-b"])
        y = np.zeros(len(groups))
        before = bearing_metrics(np.array([10.] * 8 + [8., 7.]), y, groups)
        single_win = bearing_metrics(np.array([5.] * 8 + [8., 7.]), y, groups)
        two_wins = bearing_metrics(np.array([5.] * 8 + [6., 7.]), y, groups)
        gate = bearing_gate(before, single_win)
        self.assertTrue(gate["checks"]["overall_rmse_improved"])
        self.assertFalse(gate["passes"])
        self.assertTrue(bearing_gate(before, two_wins)["passes"])
