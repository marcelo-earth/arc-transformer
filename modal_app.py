"""Run the mdlARC reference on Modal.

Usage:
    ARC_RUN=smoke-l4-001 modal run modal_app.py::smoke         # check flash-attn on a cheap GPU
    ARC_RUN=probe-l4-001 modal run modal_app.py::probe         # first epochs on an L4, to catch crashes
    ARC_RUN=low-h100-001 modal run modal_app.py --preset low   # reproduction run on an H100

ARC_RUN tags the app for experiment-hub's costs.py. Use a new value per run.

Each run writes its log and a cost summary to the `arc-transformer-runs` volume.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone

import modal

MDLARC_REPO = "https://github.com/mvakde/mdlARC.git"
MDLARC_COMMIT = "8afc20d"
# torch 2.8 breaks mdlARC's compiled training step (Float vs BFloat16 addmm);
# the reference ran on CUDA 13, so match it with torch 2.9 + cu130.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

# Modal list prices (USD per hour), checked 2026-10-05.
GPU_PRICE_PER_HOUR = {"H100": 3.949, "L4": 0.799}

# Hard caps so a stuck run cannot exceed the $5 project budget.
TIMEOUT_SECONDS = {"low": 40 * 60, "medium": 75 * 60}

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

app = modal.App(
    "arc-transformer",
    image=image,
    tags={"project": "arc-transformer", "run": os.environ.get("ARC_RUN", "untitled")},
)
runs = modal.Volume.from_name("arc-transformer-runs", create_if_missing=True)


@app.function(gpu="L4", timeout=10 * 60)
def smoke():
    import torch
    from flash_attn import flash_attn_varlen_qkvpacked_func

    qkv = torch.randn(64, 3, 12, 64, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu, 32)
    print(f"torch {torch.__version__}, gpu {torch.cuda.get_device_name()}, out {tuple(out.shape)}")


@app.function(gpu="L4", timeout=8 * 60)
def probe():
    """Run the low preset for a few minutes to check that training gets past epoch 1."""
    try:
        subprocess.run(["python", "run_script.py", "low"], cwd="/mdlARC", timeout=6 * 60)
    except subprocess.TimeoutExpired:
        print("probe: stopped after 6 minutes without crashing")


def _run(preset: str, gpu: str) -> dict:
    started = datetime.now(timezone.utc)
    t0 = time.time()
    proc = subprocess.run(
        ["python", "run_script.py", preset],
        cwd="/mdlARC",
        capture_output=True,
        text=True,
    )
    seconds = time.time() - t0
    log = proc.stdout + "\n--- stderr ---\n" + proc.stderr
    score_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("Official ARC style scoring")),
        None,
    )
    summary = {
        "preset": preset,
        "gpu": gpu,
        "mdlarc_commit": MDLARC_COMMIT,
        "started_utc": started.isoformat(),
        "seconds": round(seconds, 1),
        "gpu_hours": round(seconds / 3600, 4),
        "gpu_cost_usd": round(seconds / 3600 * GPU_PRICE_PER_HOUR[gpu], 3),
        "returncode": proc.returncode,
        "score": score_line,
    }
    out_dir = f"/runs/{started:%Y%m%d-%H%M%S}-{preset}-{gpu}"
    subprocess.run(["mkdir", "-p", out_dir], check=True)
    with open(f"{out_dir}/log.txt", "w") as f:
        f.write(log)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    subprocess.run(f"cp -r /mdlARC/runs/* {out_dir}/ 2>/dev/null || true", shell=True)
    runs.commit()
    print(log[-4000:])
    return summary


@app.function(gpu="H100", timeout=TIMEOUT_SECONDS["low"], volumes={"/runs": runs})
def run_low() -> dict:
    return _run("low", "H100")


@app.function(gpu="H100", timeout=TIMEOUT_SECONDS["medium"], volumes={"/runs": runs})
def run_medium() -> dict:
    return _run("medium", "H100")


@app.local_entrypoint()
def main(preset: str = "low"):
    fn = {"low": run_low, "medium": run_medium}[preset]
    print(json.dumps(fn.remote(), indent=2))
