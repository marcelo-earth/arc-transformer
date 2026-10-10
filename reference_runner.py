"""Run the pinned reference, adding observability without changing its model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import runpy
import sys
import time


class ProbeComplete(Exception):
    pass


def audit_data(root, destination):
    root = Path(root)
    challenges = json.loads((root / "challenges.json").read_text())
    solutions = json.loads((root / "solutions.json").read_text())
    if not solutions:
        raise ValueError("Evaluation solutions must contain at least one task.")
    missing = set(solutions) - set(challenges)
    if missing:
        raise ValueError(f"Evaluation tasks missing from challenges: {len(missing)}")
    exposed = [key for key in solutions if any(
        "output" in pair for pair in challenges[key].get("test", [])
    )]
    if exposed:
        raise ValueError(f"Evaluation test outputs exposed in challenges: {len(exposed)}")
    manifest = {
        "files": {name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                         "bytes": (root / name).stat().st_size}
                  for name in ("challenges.json", "solutions.json")},
        "challenge_tasks": len(challenges), "evaluation_tasks": len(solutions),
        "eval_tasks_with_test_output_in_challenges": 0,
    }
    Path(destination).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def logged_builder(original_build, *, probe=False):
    def build_with_logging(cfg, *build_args, **kwargs):
        if kwargs.get("is_eval", False):
            return original_build(cfg, *build_args, **kwargs)
        if probe:
            cfg.epochs = 2
        cfg.train_log_mode = "epoch"
        cfg.log_location = "both"
        cfg.checkpoint_epochs = list(range(10, cfg.epochs, 10))
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(cfg).items()}
        Path("runs/config.json").write_text(json.dumps(config, indent=2) + "\n")
        return original_build(cfg, *build_args, **kwargs)
    return build_with_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=("low", "medium"))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--evaluate-from", type=Path)
    args = parser.parse_args()
    audit_data("assets", "runs/data_manifest.json")
    sys.path.insert(0, str(Path.cwd() / "src"))
    import build
    import train
    import utils
    import torch

    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": importlib.metadata.version("flash-attn"),
        "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
    }
    Path("runs/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment), flush=True)

    if args.evaluate_from:
        import evaluate
        source = args.evaluate_from / "artifacts"
        config = json.loads((source / "config.json").read_text())
        for key in ("data_path", "train_log_file", "save_path", "checkpoint_path"):
            if config.get(key) is not None:
                config[key] = Path(config[key])
        cfg = argparse.Namespace(**config)
        checkpoint = source / "tiny.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Completed training checkpoint missing: {checkpoint}")
        print(f"Evaluating frozen checkpoint: {checkpoint}", flush=True)
        result = evaluate.run_evaluation(
            cfg, run_name="submission_eval", max_augments=cfg.max_augments,
            data_path=cfg.data_path, checkpoint_path=checkpoint, batch_size=100,
            splits=["test"], task_ids=None,
        )
        score = utils.score_arc_submission(Path("assets/solutions.json"),
                                            Path("runs") / result[0] / "submission.json")
        Path("runs/score.json").write_text(json.dumps(score, indent=2) + "\n")
        return

    original_build = build.build_model_and_data
    original_train = train.train_model
    original_score = utils.score_arc_submission
    original_epoch = train.train_one_epoch

    def epoch_with_timing(*epoch_args, **kwargs):
        started = time.monotonic()
        result = original_epoch(*epoch_args, **kwargs)
        record = {"epoch": kwargs["epoch"] + 1,
                  "seconds": round(time.monotonic() - started, 3)}
        with Path("runs/epoch_timings.jsonl").open("a") as timings:
            timings.write(json.dumps(record) + "\n")
        print(f"Completed epoch timing: {json.dumps(record)}", flush=True)
        return result

    def train_with_timing(*train_args, **kwargs):
        started = time.monotonic()
        result = original_train(*train_args, **kwargs)
        seconds = time.monotonic() - started
        Path("runs/training_summary.json").write_text(json.dumps({
            "seconds": seconds, "probe": args.probe,
            "epochs": train_args[0].epochs,
        }, indent=2) + "\n")
        if args.probe:
            print("Probe completed: 2 training epochs; evaluation not run.", flush=True)
            raise ProbeComplete
        return result

    def score_with_artifact(*score_args, **kwargs):
        result = original_score(*score_args, **kwargs)
        Path("runs/score.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    build.build_model_and_data = logged_builder(original_build, probe=args.probe)
    train.train_model = train_with_timing
    train.train_one_epoch = epoch_with_timing
    utils.score_arc_submission = score_with_artifact
    sys.argv = ["run_script.py", args.preset]
    try:
        runpy.run_path("run_script.py", run_name="__main__")
    except ProbeComplete:
        pass


if __name__ == "__main__":
    main()
