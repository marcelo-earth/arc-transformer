import json
from pathlib import Path
import sys
import tempfile
import unittest

from run_support import run_logged, validate_run_id


class RunnerTests(unittest.TestCase):
    def run_child(self, source, *, timeout=3, require_score=True):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            commits = []
            result = run_logged(
                [sys.executable, "-u", "-c", source], cwd=folder, out_dir=output,
                timeout=timeout, metadata={"seed": 42},
                commit=lambda: commits.append(json.loads((output / "summary.json").read_text())),
                commit_interval=0.1, require_score=require_score,
            )
            saved = json.loads((output / "summary.json").read_text())
            log = (output / "log.txt").read_text()
            self.assertEqual(result, saved)
            self.assertEqual(commits[0]["status"], "running")
            return result, log, commits

    def test_failure_preserves_error_output(self):
        result, log, _ = self.run_child("raise RuntimeError('dtype mismatch')")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 1)
        self.assertIn("dtype mismatch", log)

    def test_timeout_keeps_live_log_and_final_summary(self):
        result, log, commits = self.run_child(
            "import time; print('Epoch 1/90', flush=True); time.sleep(30)", timeout=0.4,
        )
        self.assertEqual(result["status"], "timed_out")
        self.assertIn("Epoch 1/90", log)
        self.assertGreater(len(commits), 2)
        self.assertLess(result["seconds"], 5)

    def test_zero_exit_without_score_is_not_a_completed_baseline(self):
        result, _, _ = self.run_child("print('training stopped before evaluation')")
        self.assertEqual(result["status"], "missing_score")

    def test_success_keeps_structured_score(self):
        source = "from pathlib import Path; Path('run/artifacts/score.json').write_text('{\"percentage\": 12.5}')"
        result, _, _ = self.run_child(source)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["score"]["percentage"], 12.5)

    def test_probe_does_not_require_evaluation(self):
        result, _, _ = self.run_child("print('2 epochs completed')", require_score=False)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["score"])

    def test_reused_run_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run_logged([], cwd=folder, out_dir=folder, timeout=1, metadata={})

    def test_run_ids_cannot_escape_volume_directory(self):
        for run_id in ("../old", "x/y", "", "x" * 97):
            with self.assertRaises(ValueError):
                validate_run_id(run_id)
        self.assertEqual(validate_run_id("probe-h100-003"), "probe-h100-003")


if __name__ == "__main__":
    unittest.main()
