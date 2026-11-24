from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from zhixun.bearing_swanlab import PROJECT_PATH, PROJECT_URL, publish_run


class BearingPublicationTests(unittest.TestCase):
    def setUp(self):
        tests = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory(dir=tests)
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(tests))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "B2-balanced"
        self.local = self.folder / "swanlog/run-20261123_223008-w393xsyb"
        self.local.mkdir(parents=True)
        (self.local / "run-w393xsyb.swanlab").write_bytes(b"fixture")
        (self.folder / "history.jsonl").write_text(
            json.dumps({"step": 6, "metrics": {"validation/rmse": 664.75}}) + "\n",
            encoding="utf-8",
        )
        self.api = Mock()
        self.api.run.return_value = SimpleNamespace(
            name="B2-balanced", state="FINISHED", profile={"config": {"run": "B2-balanced"}},
            metrics=Mock(return_value={"list": [
                {"key": "validation/rmse", "metrics": [{"step": 6, "value": 664.75}]},
            ]}),
        )
        self.url = f"{PROJECT_URL}/runs/w393xsyb"
        self.result = subprocess.CompletedProcess([], 0, f"View synced run at {self.url}\n", "")
        runner = patch("zhixun.bearing_swanlab.subprocess.run", return_value=self.result)
        self.run_command = runner.start()
        self.addCleanup(runner.stop)
        sleeper = patch("zhixun.bearing_swanlab.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def publish(self):
        return publish_run(self.root, self.folder, {"run": "B2-balanced"}, self.api)

    def test_verifies_state_and_scalars_and_reuses_success_receipt(self):
        receipt = self.publish()
        self.assertEqual(receipt["cloud_url"], self.url)
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["checked_scalar_points"], 1)
        self.api.run.assert_called_once_with(f"{PROJECT_PATH}/w393xsyb")
        self.assertEqual(self.publish(), receipt)
        self.run_command.assert_called_once()
        self.assertEqual(self.run_command.call_args.args[0][-2:], ["--id", "w393xsyb"])

    def test_zero_exit_with_failed_server_finish_is_not_a_success(self):
        self.result.stdout += "Confirm sync finish failed: server error\n"
        self.assertEqual(self.publish()["status"], "sync_failed")
        self.api.run.assert_not_called()
        self.result.stdout = f"View synced run at {self.url}\n"
        self.assertEqual(self.publish()["status"], "verified")
        self.assertEqual(self.run_command.call_count, 2)
        self.assertEqual(self.run_command.call_args_list[0].args, self.run_command.call_args.args)

    def test_value_disagreement_leaves_readback_pending(self):
        self.api.run.return_value.metrics.return_value["list"][0]["metrics"][0]["value"] = 12.0
        receipt = self.publish()
        self.assertEqual(receipt["status"], "verification_pending")
        self.assertFalse(receipt["cloud_record_verified"])

