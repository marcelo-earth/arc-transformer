"""Stream subprocess evidence and preserve failure/timeout summaries."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone


def validate_run_id(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", run_id):
        raise ValueError("Run ID must be 1 to 96 letters, digits, hyphens or underscores.")
    return run_id


def run_logged(command, *, cwd, out_dir, timeout, metadata, commit=lambda: None,
               commit_interval=60, require_score=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "artifacts").mkdir()
    summary = dict(metadata, started_utc=datetime.now(timezone.utc).isoformat(),
                   status="running", timeout_seconds=timeout, score=None)
    summary_path = out_dir / "summary.json"

    def save():
        temporary = out_dir / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)
        commit()

    save()
    started = time.monotonic()
    process = None
    reader = None
    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                   start_new_session=True)

        def stream():
            with (out_dir / "log.txt").open("w", buffering=1) as log:
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        deadline = started + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                process.wait(timeout=min(commit_interval, remaining))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                save()
        summary["status"] = "completed" if process.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        summary["status"] = "timed_out"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if reader is not None:
                reader.join(timeout=15)
            process.stdout.close()
            summary["returncode"] = process.returncode
        summary["seconds"] = round(time.monotonic() - started, 1)
        score_file = out_dir / "artifacts" / "score.json"
        if score_file.exists():
            summary["score"] = json.loads(score_file.read_text())
        if require_score and summary["status"] == "completed" and summary["score"] is None:
            summary["status"] = "missing_score"
        save()
    return summary
