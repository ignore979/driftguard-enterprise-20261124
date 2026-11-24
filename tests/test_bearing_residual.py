from __future__ import annotations

import unittest

import numpy as np

from zhixun.b_residual import append_residual_basis, fit_residual_outputs
from zhixun.b_residual_tail import fit_asymmetric_residual
from zhixun.vendor.model import Model


class BearingResidualTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(41)
        self.x = rng.normal(size=(32, 2))
        self.y = np.linspace(0., 220., len(self.x))
        self.weights = np.ones_like(self.y)
        self.parent = Model("regression", ["first", "second"], [], self.x, self.y, hidden=3)
        self.parent.meta["profile"] = "B"

    def test_migration_preserves_predictions_and_parent_optimizer_state(self):
        self.parent.fit(self.x, self.y, 1)
        migrated = append_residual_basis(self.parent, self.x)
        np.testing.assert_allclose(migrated.predict(self.x), self.parent.predict(self.x), atol=1e-10, rtol=0)
        self.assertEqual(migrated.meta["steps"], self.parent.meta["steps"])
        self.assertEqual(migrated.meta["epochs"], self.parent.meta["epochs"])
        np.testing.assert_array_equal(migrated.first["w1"][:, :3], self.parent.first["w1"])
        np.testing.assert_array_equal(migrated.second["w2"][:3], self.parent.second["w2"])
        np.testing.assert_array_equal(migrated.params["w2"][3:], 0.)

    def test_unity_asymmetric_penalty_preserves_the_symmetric_solution(self):
        symmetric = append_residual_basis(self.parent, self.x)
        asymmetric = append_residual_basis(self.parent, self.x)
        options = {"parent_width": 3, "penalty": .03}
        fit_residual_outputs(symmetric, self.x, self.y, self.weights, **options)
        fit_asymmetric_residual(asymmetric, self.x, self.y, self.weights, multiplier=1, **options)
        np.testing.assert_array_equal(symmetric.params["w2"], asymmetric.params["w2"])
        trace = fit_asymmetric_residual(asymmetric, self.x, self.y, self.weights, multiplier=4, **options)
        self.assertGreater(len(trace), 1)
        self.assertLess(trace[-1], trace[0])
        np.testing.assert_array_equal(asymmetric.params["w2"][:3], self.parent.params["w2"])
