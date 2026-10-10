"""Run the mdlARC reference on Modal.

Usage:
    ARC_RUN=probe-h100-001 modal run modal_app.py --diagnostic # two training epochs, no evaluation
    ARC_RUN=low-h100-001 modal run modal_app.py --preset low   # reproduction run on an H100

ARC_RUN tags the app for experiment-hub's costs.py. Use a new value per run.

Each run streams logs and preserves artifacts in `arc-transformer-runs`.
Actual cost comes from Modal billing, not GPU-time estimates.
"""

import json
import os
from pathlib import Path

import modal
from run_support import run_logged, validate_run_id

MDLARC_REPO = "https://github.com/mvakde/mdlARC.git"
MDLARC_COMMIT = "8afc20d"
# torch 2.8 breaks mdlARC's compiled training step (Float vs BFloat16 addmm);
# the reference ran on CUDA 13, so match it with torch 2.9 + cu130.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

# Inner deadlines allow evidence to be committed before Modal's hard timeout.
# Check actual project spend with experiment-hub/costs.py before each launch.
TIMEOUT_SECONDS = {"probe": 7 * 60, "low": 30 * 60, "evaluation": 15 * 60}
RUN_ID = validate_run_id(os.environ.get("ARC_RUN", "untitled"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.9.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("numpy", "numba", "matplotlib", FLASH_ATTN_WHEEL)
    .run_commands(
        f"git clone {MDLARC_REPO} /mdlARC",
        f"cd /mdlARC && git checkout {MDLARC_COMMIT}",
        "cd /mdlARC/dataset_building_scripts && python download_and_group.py"
        " && python build_datasets.py arc1 --add-conceptarc --with-filtered",
    )
)
image = image.add_local_file(Path(__file__).with_name("reference_runner.py"),
                             "/reference_runner.py", copy=True)
image = image.add_local_python_source("run_support")

app = modal.App(
    "arc-transformer",
    image=image,
    tags={"project": "arc-transformer", "run": RUN_ID},
)
runs = modal.Volume.from_name("arc-transformer-runs", create_if_missing=True)


@app.function(gpu="H100", cpu=4, memory=16384, timeout=9 * 60,
              startup_timeout=120, retries=0, max_containers=1,
              volumes={"/runs": runs})
def probe(run_id: str) -> dict:
    """Complete two epochs before spending on the full low preset."""
    return _run("low", run_id, probe=True)


def _run(preset: str, run_id: str, probe: bool = False, evaluate_from: str = "") -> dict:
    run_id = validate_run_id(run_id)
    out_dir = Path("/runs") / run_id
    if out_dir.exists():
        raise ValueError(f"Run already exists: {run_id}; choose a new ARC_RUN.")
    # The reference writes to runs/; bind it directly to the persistent volume.
    work_runs = Path("/mdlARC/runs")
    if work_runs.is_symlink():
        work_runs.unlink()
    elif work_runs.exists():
        raise RuntimeError("Unexpected existing /mdlARC/runs; refusing to overwrite artifacts.")
    work_runs.symlink_to(out_dir / "artifacts", target_is_directory=True)
    command = ["python", "-u", "/reference_runner.py", preset]
    if probe:
        command.append("--probe")
    if evaluate_from:
        command.extend(["--evaluate-from", str(Path("/runs") / validate_run_id(evaluate_from))])
    mode = "evaluation" if evaluate_from else "probe" if probe else "baseline"
    summary = run_logged(
        command, cwd="/mdlARC", out_dir=out_dir,
        timeout=TIMEOUT_SECONDS[mode if mode != "baseline" else preset],
        metadata={"run_id": run_id, "preset": preset, "gpu": "H100",
                  "mdlarc_commit": MDLARC_COMMIT, "seed": 42,
                  "mode": mode, "source_run_id": evaluate_from or None,
                  "compute_cost_source": "modal billing report, project/run tags"},
        commit=runs.commit, require_score=not probe,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(gpu="H100", cpu=4, memory=16384, timeout=32 * 60,
              startup_timeout=120, retries=0, max_containers=1,
              volumes={"/runs": runs})
def run_low(run_id: str) -> dict:
    return _run("low", run_id)


@app.function(gpu="H100", cpu=4, memory=16384, timeout=17 * 60,
              startup_timeout=120, retries=0, max_containers=1,
              volumes={"/runs": runs})
def evaluate_checkpoint(run_id: str, source_run_id: str) -> dict:
    return _run("low", run_id, evaluate_from=source_run_id)


@app.local_entrypoint()
def main(preset: str = "low", diagnostic: bool = False, evaluate_from: str = ""):
    if RUN_ID == "untitled":
        raise ValueError("Set a unique ARC_RUN before launching compute.")
    if preset != "low":
        raise ValueError("Only low is enabled inside the current $5 project budget.")
    if diagnostic and evaluate_from:
        raise ValueError("Diagnostic and checkpoint evaluation are separate modes.")
    if evaluate_from:
        summary = evaluate_checkpoint.remote(RUN_ID, validate_run_id(evaluate_from))
    else:
        summary = (probe if diagnostic else run_low).remote(RUN_ID)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed":
        raise RuntimeError(f"Run {RUN_ID} ended with status {summary['status']}; evidence is saved.")
