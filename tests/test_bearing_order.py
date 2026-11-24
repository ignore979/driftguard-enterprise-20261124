from __future__ import annotations

import copy
import unittest

import numpy as np

from zhixun.bearing_order import fit_ordered, pairing_pools
from zhixun.bearing_study import bearing_weights, fit_weighted_bearing_epoch
from zhixun.data import Dataset
from zhixun.vendor.model import Model


class BearingOrderTests(unittest.TestCase):
    def setUp(self):
        x = np.arange(24, dtype=np.float64).reshape(12, 2)
        y = np.array([0., 15., 30., 45., 60., 75., 0., 15., 30., 45., 60., 75.])
        self.groups = np.repeat(["first", "second"], 6)
        self.data = Dataset(tuple(str(i) for i in range(len(x))), x, y, frozenset())
        self.base = Model("regression", ["first", "second"], [], x, y, hidden=3, batch_size=4)
        self.base.meta["profile"] = "B"
        self.weights = bearing_weights(self.groups, .5)
        self.pools = pairing_pools(y, self.groups)

    def test_pairs_keep_separated_stages_in_same_bearing(self):
        for high, low in self.pools:
            self.assertEqual(len(set(self.groups[np.r_[high, low]])), 1)
            self.assertGreater(self.data.y[high].min(), self.data.y[low].max())
        with self.assertRaises(ValueError):
            pairing_pools(np.zeros_like(self.data.y), self.groups)

    def test_zero_penalty_matches_reference_optimizer(self):
        reference = copy.deepcopy(self.base)
        reference_loss = fit_weighted_bearing_epoch(reference, self.data, self.weights)
        actual = fit_ordered(self.base, self.data, self.weights, self.pools, epochs=1, penalty=0.)
        self.assertEqual(actual, [{"weighted_mse": reference_loss, "order_violation_mse": 0.}])
        self.assertEqual(self.base.meta, reference.meta)
        for field in ("params", "first", "second"):
            for name, weights in getattr(reference, field).items():
                np.testing.assert_array_equal(getattr(self.base, field)[name], weights)

    def test_positive_penalty_changes_adam_update(self):
        reference = copy.deepcopy(self.base)
        fit_weighted_bearing_epoch(reference, self.data, self.weights)
        actual = fit_ordered(self.base, self.data, self.weights, self.pools, epochs=1, penalty=1.)
        self.assertGreater(actual[0]["order_violation_mse"], 0.)
        self.assertFalse(np.array_equal(self.base.params["w2"], reference.params["w2"]))
        self.assertEqual(self.base.meta["steps"], reference.meta["steps"])
