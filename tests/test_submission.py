import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from submit_evaluation import submit


class SubmissionTests(unittest.TestCase):
    def test_submission_does_not_wait_for_remote_result(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "job.json"
            function = Mock()
            function.spawn.return_value.object_id = "fc-test"
            with patch("submit_evaluation.modal.Function.from_name", return_value=function):
                result = submit(deployment="arc-recovery", run_id="eval-high-001",
                                source_run_id="high-001", output=output)
            function.spawn.assert_called_once_with("eval-high-001", "high-001")
            function.remote.assert_not_called()
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(result["function_call_id"], "fc-test")

    def test_existing_submission_is_rejected_before_sending(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "job.json"
            output.write_text("{}")
            with patch("submit_evaluation.modal.Function.from_name") as lookup:
                with self.assertRaises(FileExistsError):
                    submit(deployment="arc-recovery", run_id="eval-high-001",
                           source_run_id="high-001", output=output)
                lookup.assert_not_called()
