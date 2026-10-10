"""Submit checkpoint evaluation to a deployed app without a long-lived caller."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import modal
from run_support import validate_run_id


def submit(*, deployment, run_id, source_run_id, output):
    validate_run_id(run_id)
    validate_run_id(source_run_id)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Submission record already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    function = modal.Function.from_name(deployment, "evaluate_high_checkpoint")
    call = function.spawn(run_id, source_run_id)
    record = {"deployment": deployment, "function": "evaluate_high_checkpoint",
              "function_call_id": call.object_id, "run_id": run_id,
              "source_run_id": source_run_id,
              "submitted_utc": datetime.now(timezone.utc).isoformat()}
    with output.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(deployment=args.deployment, run_id=args.run_id,
                            source_run_id=args.source_run_id, output=args.output), indent=2))


if __name__ == "__main__":
    main()
