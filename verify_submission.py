"""Independently rescore a complete ARC submission against frozen scorer data."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(submission, solutions, reported):
    if not solutions:
        raise ValueError("No evaluation tasks.")
    if set(submission) != set(solutions):
        raise ValueError("Submission task IDs do not exactly match evaluation tasks.")
    total, pairs, fully = 0.0, 0, []
    for task, expected in solutions.items():
        attempts = submission[task]
        if not expected or len(attempts) != len(expected):
            raise ValueError(f"Test-pair count mismatch for {task}.")
        correct = sum(any(attempts[index].get(key) == truth
                          for key in ("attempt_1", "attempt_2"))
                      for index, truth in enumerate(expected))
        total += correct / len(expected)
        pairs += len(expected)
        if correct == len(expected):
            fully.append(task)
    percentage = total / len(solutions) * 100
    if (abs(total - reported["score"]) > 1e-10 or
        abs(percentage - reported["percentage"]) > 1e-10 or
        reported["max_score"] != len(solutions) or
        set(fully) != set(reported["fully_solved_tasks"])):
        raise ValueError("Independent score differs from the reported scorer output.")
    return {"tasks": len(solutions), "test_pairs": pairs,
            "task_averaged_pass_at_2_score": total, "percentage": percentage,
            "fully_solved_tasks": len(fully),
            "all_task_ids_and_test_pair_counts_match": True,
            "method": "Compare each expected grid with both attempts; average within each task, then across tasks."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--reported-score", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text())["files"]["solutions.json"]["sha256"]
    if hashlib.sha256(args.solutions.read_bytes()).hexdigest() != expected:
        raise ValueError("Scorer solutions do not match the frozen data manifest.")
    result = verify(json.loads(args.submission.read_text()),
                    json.loads(args.solutions.read_text()),
                    json.loads(args.reported_score.read_text()))
    result["solutions_sha256_matches_manifest"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
