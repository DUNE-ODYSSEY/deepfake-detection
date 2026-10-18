"""Run the Phase 5 and Phase 2 training/eval matrices back to back.

Each job is one run_phase1_matrix invocation (DF-only and FS-only, 2 seeds,
patience 2, all 4 test methods). The driver skips finished runs/evals and
resumes partial ones, so re-running this queue after a crash continues where
it stopped. A failed job is logged and the queue moves on.

Usage: python -u -m experiments.run_queue [--start-after-pid PID]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF = "C:/ffpp_data/ffpp_faces_ref"

GRID = ["--seeds", "1", "2", "--patience", "2"]
JOBS = [
    ("phase4_hfaug", "xception", REF, ["--seeds", "1", "2", "3", "--patience", "4", "--hf-aug"]),
    ("phase5_resnet50", "resnet50", REF, GRID),
    ("phase5_efficientnet_b4", "efficientnet_b4", REF, GRID),
    ("phase2_jpeg75", "xception", "C:/ffpp_data/ffpp_faces_ref_jpeg75", GRID),
    ("phase2_frames10", "xception", "C:/ffpp_data/ffpp_faces_ref_frames10", GRID),
    ("phase2_margin10", "xception", "C:/ffpp_data/ffpp_faces_ref_margin10", GRID),
    ("phase2_jpeg95", "xception", "C:/ffpp_data/ffpp_faces_ref_jpeg95", GRID),
]


def crops_ready(data_root):
    """Phase 2 crop sets are built by a separate CPU job; wait until complete."""
    return Path(data_root, "c23", ".complete").exists() if data_root != REF else True


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-after-pid", type=int, help="wait for this process (e.g. Phase 4) to exit first")
    p.add_argument("--only", nargs="*", help="run only these job names")
    args = p.parse_args()
    if args.start_after_pid:
        print(f"waiting for PID {args.start_after_pid} to exit", flush=True)
        while pid_alive(args.start_after_pid):
            time.sleep(60)
    for name, model, data_root, extra in JOBS:
        if args.only and name not in args.only:
            continue
        if not crops_ready(data_root):
            print(f"waiting for crop set {data_root} to finish building", flush=True)
            while not crops_ready(data_root):
                time.sleep(60)
        batch = 32
        if model not in ("xception", "cnn_vit"):
            bench = subprocess.run([sys.executable, "-m", "experiments.bench_batch", "--model", model,
                                    "--candidates", "32", "16"], cwd=REPO, capture_output=True, text=True)
            print(bench.stdout.strip(), flush=True)
            lines = bench.stdout.strip().splitlines()
            batch = int(lines[-1]) if bench.returncode == 0 and lines and lines[-1].isdigit() else 0
            if batch == 0:
                print(f"=== JOB {name} skipped: does not fit in GPU memory", flush=True)
                continue
        cmd = [sys.executable, "-u", "-m", "experiments.phase1.run_phase1_matrix",
               "--data-root", data_root, "--splits-dir", "splits", "--model", model,
               "--runs-root", f"runs/{name}", "--results-dir", f"results/{name}",
               *extra, "--batch-size", str(batch), "--execute"]
        print(f"=== JOB {name} start {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        print(f"=== JOB {name} end rc={rc} {time.strftime('%Y-%m-%d %H:%M')}", flush=True)
    print("=== QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
