import json
from pathlib import Path
import tempfile
import unittest

from reference_runner import audit_data


class DataAuditTests(unittest.TestCase):
    def audit(self, challenges, solutions):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "challenges.json").write_text(json.dumps(challenges))
            (root / "solutions.json").write_text(json.dumps(solutions))
            return audit_data(root, root / "manifest.json")

    def test_demo_outputs_are_allowed_but_eval_test_outputs_are_hidden(self):
        manifest = self.audit({"eval": {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }}, {"eval": [[[3]]]})
        self.assertEqual(manifest["evaluation_tasks"], 1)
        self.assertEqual(manifest["eval_tasks_with_test_output_in_challenges"], 0)
        self.assertEqual(len(manifest["files"]["challenges.json"]["sha256"]), 64)

    def test_hidden_test_target_in_training_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exposed"):
            self.audit({"eval": {"test": [{"input": [[0]], "output": [[1]]}]}},
                       {"eval": [[[1]]]})

    def test_missing_eval_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            self.audit({}, {"eval": [[[1]]]})

    def test_empty_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.audit({}, {})
