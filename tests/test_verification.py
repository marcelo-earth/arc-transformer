import unittest
from verify_submission import verify


class VerificationTests(unittest.TestCase):
    def test_partial_credit_is_averaged_by_task_not_by_test_pair(self):
        solutions = {"a": [[[1]], [[2]]], "b": [[[3]]]}
        submission = {"a": [{"attempt_2": [[1]]}, {"attempt_1": [[0]]}],
                      "b": [{"attempt_1": [[3]], "attempt_2": [[3]]}]}
        reported = {"score": 1.5, "max_score": 2, "percentage": 75,
                    "fully_solved_tasks": ["b"]}
        result = verify(submission, solutions, reported)
        self.assertEqual(result["percentage"], 75)
        self.assertEqual(result["fully_solved_tasks"], 1)
        self.assertEqual(result["test_pairs"], 3)

    def test_missing_task_is_rejected_even_if_reported_score_matches(self):
        with self.assertRaisesRegex(ValueError, "task IDs"):
            verify({}, {"a": [[[1]]]}, {})

    def test_missing_test_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pair count"):
            verify({"a": []}, {"a": [[[1]]]}, {})

    def test_incorrect_reported_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            verify({"a": [{"attempt_1": [[1]]}]}, {"a": [[[1]]]},
                   {"score": 0, "percentage": 0, "max_score": 1, "fully_solved_tasks": []})
