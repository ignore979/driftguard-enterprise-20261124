import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zhixun.c_round import measure_c, nasa_sum, quality_gate
from zhixun.c_round_model import continue_c, promote_c
from zhixun.contest_c_runtime import _guard_panel, _origin_for_panel, host_vote, parse_choice
from zhixun.vendor.model import Model


class CUpgradeTests(unittest.TestCase):
    def test_nasa_score_uses_asymmetric_scale(self):
        p, y = np.array([20., 0.]), np.array([10., 13.])
        self.assertAlmostEqual(nasa_sum(measure_c(p, y)), 2 * np.expm1(1))

    def test_scoring_missing_near_end_keeps_regression_guard(self):
        y = np.array([52., 55.])
        before, after = measure_c(np.array([65., 62.]), y), measure_c(np.array([61., 60.]), y)
        self.assertIsNone(before["near_end_rmse"])
        self.assertEqual(quality_gate(before, after), (True, []))

    def test_promotion_compares_candidates_to_parent_and_tails_to_panel_origin(self):
        y = np.array([4., 25., 100., 160.])
        origin = _guard_panel(np.array([42., 49., 105., 130.]), y)
        parent = _guard_panel(np.array([7., 30., 116., 136.]), y)
        improved = _guard_panel(np.array([5., 28., 110., 142.]), y)
        risky = _guard_panel(np.array([50., 25., 100., 161.]), y)
        self.assertTrue(promote_c(origin, parent, improved)["pass"])
        rejected = promote_c(origin, parent, risky)
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["checks"]["near_failure_p95_not_worse_vs_origin"])

    def test_state_origin_resets_on_new_projection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            state = Path(temporary)
            self.assertEqual(_origin_for_panel(state, "first", {"rmse": 5}), {"rmse": 5})
            self.assertEqual(_origin_for_panel(state, "first", {"rmse": 3}), {"rmse": 5})
            self.assertEqual(_origin_for_panel(state, "second", {"rmse": 2}), {"rmse": 2})
            self.assertEqual(json.loads((state / "public-panel.json").read_text())["revision"], "second")

    def test_model_response_and_host_metric_validation(self):
        self.assertEqual(parse_choice('{"select":"candidate","reason":"measured gain"}')[0], "candidate")
        self.assertEqual(parse_choice('{"select":"candidate","reason":[]}')[0], "parent")
        self.assertEqual(host_vote({"summary": {"metrics": {"rmse": 2.}}}, {"rmse": 3.}, {"rmse": 2.}), "accept")
        self.assertEqual(host_vote({"summary": {"metrics": {"rmse": 6.}}}, {"rmse": 3.}, {"rmse": 2.}), "unavailable")
        self.assertEqual(host_vote({"summary": {"metrics": {"rmse": 4.}}}, {"rmse": 3.}, {"rmse": 4.}), "reject")

    def test_projection_without_condition_uses_parent_adam_and_advances_each_time(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        y = np.arange(6, dtype=float) * 10
        parent = Model("regression", ["s1", "s2"], [], x, y, hidden=3, batch_size=3)
        parent.meta["profile"] = "C"
        next_model, losses, audit = continue_c(parent, x, y, epochs=1, learning_rate=.001, strategy="condition")
        last_model, _, _ = continue_c(next_model, x, y, epochs=1, learning_rate=.001, strategy="condition")
        self.assertEqual(audit["migration"], "missing_setting_standard_fallback")
        self.assertEqual(len(losses), 1)
        self.assertEqual((parent.meta["steps"], next_model.meta["steps"], last_model.meta["steps"]), (0, 2, 4))
        np.testing.assert_array_equal(last_model.mean, parent.mean)
