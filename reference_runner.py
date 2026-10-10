"""Run the pinned reference, adding observability without changing its model."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=("low", "medium"))
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd() / "src"))
    import build
    import train
    import utils

    original_build = build.build_model_and_data
    original_train = train.train_model
    original_score = utils.score_arc_submission

    def build_with_logging(cfg):
        if args.probe:
            cfg.epochs = 2
        cfg.train_log_mode = "epoch"
        cfg.log_location = "both"
        cfg.checkpoint_epochs = list(range(10, cfg.epochs, 10))
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(cfg).items()}
        Path("runs/config.json").write_text(json.dumps(config, indent=2) + "\n")
        return original_build(cfg)

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

    build.build_model_and_data = build_with_logging
    train.train_model = train_with_timing
    utils.score_arc_submission = score_with_artifact
    sys.argv = ["run_script.py", args.preset]
    try:
        runpy.run_path("run_script.py", run_name="__main__")
    except ProbeComplete:
        pass


if __name__ == "__main__":
    main()
