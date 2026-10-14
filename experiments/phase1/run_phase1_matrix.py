"""Driver for Phase 1's training x evaluation matrix (reference pipeline).

Trains Xception on each of {Deepfakes, FaceSwap} (train-only methods per the
brief's Phase 1 scope), 3 seeds each = 6 training runs, then evaluates every
checkpoint against all 4 test methods = 24 eval cells.

Run-directory convention (ground rule 2):
  runs/reference_<model>_<trainmethod>_c23/seed<k>/
      config.json  (git commit, package versions, GPU name, seed, dataset
                    source -- written by src/train.py itself)
      DONE         (written only on successful completion; src/train.py
                    refuses to overwrite a directory containing it)
Results convention:
  <results-dir>/<model>_<trainmethod>_to_<testmethod>_c23_seed<k>.json
  <results-dir>/<model>_<trainmethod>_to_<testmethod>_c23_seed<k>.video_scores.csv
      (per-video scores, written by src/evaluate.py; needed to rerun
      bootstrap CIs / aggregate-seed analyses without retraining)

SAFETY: this script only *prints* the commands it would run unless --execute
is passed. The brief requires an explicit go-ahead before any run over
~1 GPU-hour, and the full matrix is estimated at double-digit GPU-hours (see
docs/PROGRESS.md "Phase 1" section) -- do not pass --execute without that
go-ahead.

Usage (dry run, default):
  python -m experiments.phase1.run_phase1_matrix \
      --data-root C:/ffpp_data/ffpp_faces_ref --splits-dir splits

Usage (actually launch, only once authorized):
  python -m experiments.phase1.run_phase1_matrix \
      --data-root C:/ffpp_data/ffpp_faces_ref --splits-dir splits --execute
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def train_cmd(args, train_method: str, seed: int, run_dir: Path):
    return [sys.executable, "-m", "src.train",
            "--data-root", args.data_root, "--splits-dir", args.splits_dir,
            "--model", args.model, "--methods", train_method,
            "--compression", args.compression,
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--patience", str(args.patience), "--seed", str(seed),
            "--dataset-source", args.dataset_source,
            "--out", str(run_dir), "--resume", *(["--hf-aug"] if args.hf_aug else [])]


def eval_cmd(args, run_dir: Path, test_method: str, out_json: Path):
    return [sys.executable, "-m", "src.evaluate",
            "--checkpoint", str(run_dir / "best.pt"),
            "--data-root", args.data_root, "--splits-dir", args.splits_dir,
            "--methods", test_method, "--compression", args.compression,
            "--split", "test", "--n-resamples", str(args.n_resamples),
            "--out", str(out_json)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="Reference-pipeline crop root, e.g. C:/ffpp_data/ffpp_faces_ref")
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--results-dir", default="results/reference")
    p.add_argument("--model", default="xception", choices=["xception", "cnn_vit"],
                   help="Phase 1 trains Xception only per the brief; cnn_vit "
                        "left available for a future architecture-independence pass")
    p.add_argument("--train-methods", nargs="+", default=["Deepfakes", "FaceSwap"])
    p.add_argument("--test-methods", nargs="+", default=TEST_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--compression", default="c23")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--dataset-source", default="mirror", choices=["mirror", "official"])
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--hf-aug", action="store_true",
                   help="train with random blur/sharpen (Phase 4 intervention); use "
                        "with a separate --runs-root/--results-dir")
    p.add_argument("--execute", action="store_true",
                   help="Actually run the commands. Without this flag, only "
                        "prints the plan (dry run). Requires an explicit "
                        "go-ahead per the brief's >1-GPU-hour rule.")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    results_dir = Path(args.results_dir)
    plan = []  # list of (kind, cmd, note)

    for train_method in args.train_methods:
        for seed in args.seeds:
            run_dir = runs_root / f"reference_{args.model}_{train_method}_{args.compression}" / f"seed{seed}"
            done = run_dir / "DONE"
            if done.exists():
                plan.append(("train-skip", None,
                             f"{run_dir} already DONE, would skip training"))
            else:
                plan.append(("train", train_cmd(args, train_method, seed, run_dir),
                             f"train {args.model} on {train_method} seed={seed} -> {run_dir}"))

            for test_method in args.test_methods:
                out_json = results_dir / (
                    f"{args.model}_{train_method}_to_{test_method}_"
                    f"{args.compression}_seed{seed}.json")
                if out_json.exists():
                    plan.append(("eval-skip", None,
                                 f"{out_json} already exists, would skip eval"))
                else:
                    plan.append(("eval", eval_cmd(args, run_dir, test_method, out_json),
                                 f"eval {train_method}(seed={seed}) -> {test_method}: {out_json}"))

    n_train = sum(1 for k, *_ in plan if k == "train")
    n_eval = sum(1 for k, *_ in plan if k == "eval")
    print(f"Plan: {n_train} training run(s), {n_eval} eval cell(s) "
          f"(of {len(args.train_methods) * len(args.seeds)} train x "
          f"{len(args.test_methods)} test full matrix).")
    for kind, cmd, note in plan:
        print(f"  [{kind}] {note}")
        if cmd:
            print(f"        $ {' '.join(cmd)}")

    if not args.execute:
        print("\nDry run only (no --execute passed). Nothing was launched.")
        return

    print("\n--execute passed: running the plan now.")
    results_dir.mkdir(parents=True, exist_ok=True)
    for kind, cmd, note in plan:
        if cmd is None:
            print(f"skip: {note}")
            continue
        print(f"running: {note}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
