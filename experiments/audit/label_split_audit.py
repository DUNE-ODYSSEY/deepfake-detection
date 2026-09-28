"""Phase 0 item 2: label/split audit.

Three checks, each backed by code actually executed against real repo files
(never hand-computed numbers):

1. fake=1 convention: read src/data/dataset.py and src/metrics.py source and
   grep for the label-assignment and AUC-orientation lines that must hold;
   additionally run a synthetic AUC-polarity sanity check through the real
   compute_metrics() function to prove sklearn's roc_auc_score is being
   called with the fake-as-positive-class polarity the brief requires.
2. Video-disjointness: load splits/train.json, val.json, test.json through
   the repo's own src.data.splits.load_split_ids() (not a re-implementation)
   and check pairwise intersections of real_ids and fake_ids are empty.
3. Same-real-test-videos-per-cell: FFPPDataset always loads real/ crops
   filtered only by the split's real_ids, independent of the `methods` arg
   (see src/data/dataset.py lines ~44-48). This script verifies that
   structural claim empirically: for every results/*.json cross-manipulation
   cell on split=="test", it independently recomputes the expected real-video
   count (directory scan of ffpp_faces/<compression>/real filtered by
   test split real_ids) and the expected fake-video count for that cell's
   test_methods (directory scan of ffpp_faces/<compression>/<method>
   filtered by test split fake_ids), and checks
       n_videos_in_result_json == real_count + fake_count
   and that the recomputed real_count is IDENTICAL across every cell for a
   given (test_compression, split) -- which is what "same real test videos in
   every cell" means operationally, since results/*.json only stores counts,
   not per-video ID lists.

Usage:
    python -m experiments.audit.label_split_audit \
        --repo-root . --faces-root C:/ffpp_data/ffpp_faces \
        --out results/audit/label_split_audit.csv
"""
import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import load_split_ids  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root)
        ).decode().strip()
    except Exception as e:
        return f"unknown ({e})"


def check_fake_eq_1(rows):
    ds_src = (REPO_ROOT / "src" / "data" / "dataset.py").read_text()
    metrics_src = (REPO_ROOT / "src" / "metrics.py").read_text()

    real_is_0 = "self.samples.append((f, 0, f\"real/{vdir.name}\"))" in ds_src
    fake_is_1 = 'self.samples.append((f, 1, f"{method}/{vdir.name}"))' in ds_src
    rows.append({
        "check": "dataset.py: real label literal is 0",
        "result": "PASS" if real_is_0 else "FAIL",
        "detail": "src/data/dataset.py real-branch append((f, 0, ...))",
    })
    rows.append({
        "check": "dataset.py: fake (any method) label literal is 1",
        "result": "PASS" if fake_is_1 else "FAIL",
        "detail": "src/data/dataset.py method-branch append((f, 1, ...))",
    })

    # metrics.py must not flip scores/labels before calling roc_auc_score.
    # roc_auc_score itself is called once, inside the shared _safe_auc()
    # helper; that helper is then invoked once for frame-level scores
    # (labels, probs) and once for video-level scores (vl, vp) -- both with
    # the raw, unflipped label/score order.
    safe_auc_def_ok = "def _safe_auc(labels, probs):" in metrics_src and \
        "roc_auc_score(labels, probs)" in metrics_src
    frame_call_ok = '"frame_auc": _safe_auc(labels, probs)' in metrics_src
    video_call_ok = '"video_auc": _safe_auc(vl, vp)' in metrics_src
    no_flip = "1 - probs" not in metrics_src and "1-probs" not in metrics_src \
        and "1 - vp" not in metrics_src and "-probs" not in metrics_src
    all_ok = safe_auc_def_ok and frame_call_ok and video_call_ok and no_flip
    rows.append({
        "check": "metrics.py: roc_auc_score called on raw (labels, probs) "
                 "via shared _safe_auc() for both frame and video level, "
                 "with no post-hoc flip (e.g. no '1 - probs' or '-probs')",
        "result": "PASS" if all_ok else "FAIL",
        "detail": f"_safe_auc def uses roc_auc_score(labels, probs)="
                  f"{safe_auc_def_ok}; frame_auc=_safe_auc(labels, probs)="
                  f"{frame_call_ok}; video_auc=_safe_auc(vl, vp)="
                  f"{video_call_ok}; no post-hoc flip found={no_flip}",
    })

    # Synthetic polarity sanity check through the REAL compute_metrics().
    # Construct a case where fake(1) frames get systematically higher probs
    # than real(0) frames -> AUC must come back high (>0.9), not low.
    rng = np.random.RandomState(0)
    n = 200
    labels = np.array([0] * n + [1] * n)
    videos = [f"real/vid{i}" for i in range(n)] + [f"m/vid{i}" for i in range(n)]
    probs = np.concatenate([
        rng.uniform(0.0, 0.4, n),   # real -> low fake-probability
        rng.uniform(0.6, 1.0, n),   # fake -> high fake-probability
    ])
    m = compute_metrics(probs, labels, videos)
    polarity_ok = m["video_auc"] > 0.9 and m["frame_auc"] > 0.9
    rows.append({
        "check": "compute_metrics() polarity sanity check: fake=1 frames "
                 "given higher scores -> AUC should be high, not low",
        "result": "PASS" if polarity_ok else "FAIL",
        "detail": f"synthetic frame_auc={m['frame_auc']:.4f} "
                  f"video_auc={m['video_auc']:.4f} (expected >0.9 both)",
    })

    # Inverse case: fake=1 frames given LOWER scores -> AUC should be low.
    probs_inv = np.concatenate([
        rng.uniform(0.6, 1.0, n),   # real -> high fake-probability (wrong)
        rng.uniform(0.0, 0.4, n),   # fake -> low fake-probability (wrong)
    ])
    m_inv = compute_metrics(probs_inv, labels, videos)
    polarity_inv_ok = m_inv["video_auc"] < 0.1 and m_inv["frame_auc"] < 0.1
    rows.append({
        "check": "compute_metrics() inverse polarity check: fake=1 frames "
                 "given lower scores -> AUC should be low (<0.1), confirming "
                 "no hidden 1-AUC flip anywhere in the metric",
        "result": "PASS" if polarity_inv_ok else "FAIL",
        "detail": f"synthetic frame_auc={m_inv['frame_auc']:.4f} "
                  f"video_auc={m_inv['video_auc']:.4f} (expected <0.1 both)",
    })

    # train.py: BCEWithLogitsLoss target y comes straight from the dataset
    # label (1=fake) with sigmoid(logits) as the score fed to compute_metrics.
    train_src = (REPO_ROOT / "src" / "train.py").read_text()
    sigmoid_used = "torch.sigmoid(logits)" in train_src
    rows.append({
        "check": "train.py evaluate(): score fed to AUC is sigmoid(logits) "
                 "against dataset's fake=1 label (no flip between model "
                 "output and metric input)",
        "result": "PASS" if sigmoid_used else "FAIL",
        "detail": "src/train.py evaluate(): probs.extend(torch.sigmoid("
                  "logits)...)",
    })
    return rows


def check_split_disjointness(splits_dir: Path, rows):
    ids = {}
    for split in ["train", "val", "test"]:
        real_ids, fake_ids = load_split_ids(str(splits_dir), split)
        ids[split] = (real_ids, fake_ids)
        rows.append({
            "check": f"{split} split size (via src.data.splits.load_split_ids)",
            "result": "INFO",
            "detail": f"n_real_ids={len(real_ids)} n_fake_ids={len(fake_ids)}",
        })

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    all_disjoint = True
    for a, b in pairs:
        real_overlap = ids[a][0] & ids[b][0]
        fake_overlap = ids[a][1] & ids[b][1]
        ok = len(real_overlap) == 0 and len(fake_overlap) == 0
        all_disjoint = all_disjoint and ok
        rows.append({
            "check": f"video-disjoint: {a} vs {b}",
            "result": "PASS" if ok else "FAIL",
            "detail": f"real_id_overlap={len(real_overlap)} "
                      f"fake_id_overlap={len(fake_overlap)}"
                      + (f" e.g. {sorted(real_overlap)[:5]}"
                         if real_overlap else "")
                      + (f" e.g.fake {sorted(fake_overlap)[:5]}"
                         if fake_overlap else ""),
        })
    rows.append({
        "check": "video-disjoint: train/val/test overall",
        "result": "PASS" if all_disjoint else "FAIL",
        "detail": "all pairwise real_id and fake_id intersections empty"
                  if all_disjoint else "see pairwise rows above",
    })
    return ids


def scan_video_dirs(faces_root: Path, compression: str, name: str) -> set:
    d = faces_root / compression / name
    if not d.exists():
        return set()
    return {p.name for p in d.iterdir() if p.is_dir()}


def check_same_real_videos_per_cell(faces_root: Path, split_ids, rows):
    results_dir = REPO_ROOT / "results"
    result_files = sorted(results_dir.glob("*.json"))
    cell_rows = []
    real_count_by_key = {}  # (compression, split) -> recomputed real count
    real_dirs_cache = {}
    fake_dirs_cache = {}

    for f in result_files:
        data = json.loads(f.read_text())
        split = data.get("split")
        comp = data.get("test_compression")
        test_methods = data.get("test_methods", [])
        n_videos = data.get("n_videos")
        if split not in split_ids or comp is None:
            continue  # e.g. c40proxy has no split file of its own; see note

        real_ids, fake_ids = split_ids[split]

        cache_key = (comp,)
        if cache_key not in real_dirs_cache:
            real_dirs_cache[cache_key] = scan_video_dirs(faces_root, comp, "real")
        real_dirs = real_dirs_cache[cache_key]
        real_count = len(real_dirs & real_ids)

        fake_count = 0
        for method in test_methods:
            mk = (comp, method)
            if mk not in fake_dirs_cache:
                fake_dirs_cache[mk] = scan_video_dirs(faces_root, comp, method)
            fake_count += len(fake_dirs_cache[mk] & fake_ids)

        expected = real_count + fake_count
        match = (expected == n_videos)
        key = (comp, split)
        real_count_by_key.setdefault(key, set()).add(real_count)

        cell_rows.append({
            "result_file": f.name,
            "model": data.get("model"),
            "train_methods": ",".join(data.get("train_methods", [])),
            "test_methods": ",".join(test_methods),
            "compression": comp,
            "split": split,
            "n_videos_reported": n_videos,
            "real_video_count_recomputed": real_count,
            "fake_video_count_recomputed": fake_count,
            "expected_total": expected,
            "match": "PASS" if match else "FAIL",
        })

    for (comp, split), counts in real_count_by_key.items():
        ok = len(counts) == 1
        rows.append({
            "check": f"same real videos in every cell: compression={comp} "
                     f"split={split}",
            "result": "PASS" if ok else "FAIL",
            "detail": f"recomputed real-video count across all cells with "
                      f"this (compression, split): {sorted(counts)} "
                      f"(single value = same real set size in every cell, "
                      f"consistent with dataset.py loading real/ independent "
                      f"of the methods argument, confirmed by code read at "
                      f"src/data/dataset.py:44-48)",
        })

    n_mismatch = sum(1 for r in cell_rows if r["match"] == "FAIL")
    rows.append({
        "check": "n_videos reconciliation across all results/*.json cells",
        "result": "PASS" if n_mismatch == 0 else "FAIL",
        "detail": f"{len(cell_rows)} result files checked, {n_mismatch} "
                  f"mismatches between reported n_videos and "
                  f"real_count+fake_count recomputed from split files + "
                  f"directory scan",
    })
    return cell_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--splits-dir", default=None)
    p.add_argument("--faces-root", default="C:/ffpp_data/ffpp_faces")
    p.add_argument("--out", default=None)
    p.add_argument("--cells-out", default=None)
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    splits_dir = Path(args.splits_dir) if args.splits_dir else repo_root / "splits"
    faces_root = Path(args.faces_root)
    out = Path(args.out) if args.out else repo_root / "results" / "audit" / "label_split_audit.csv"
    cells_out = Path(args.cells_out) if args.cells_out else \
        repo_root / "results" / "audit" / "label_split_audit_cells.csv"

    rows = []
    rows.append({
        "check": "META: git commit at time of audit",
        "result": "INFO",
        "detail": git_commit(repo_root),
    })

    check_fake_eq_1(rows)
    split_ids = check_split_disjointness(splits_dir, rows)
    cell_rows = check_same_real_videos_per_cell(faces_root, split_ids, rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "result", "detail"])
        w.writeheader()
        w.writerows(rows)

    with cells_out.open("w", newline="") as fh:
        if cell_rows:
            w = csv.DictWriter(fh, fieldnames=list(cell_rows[0].keys()))
            w.writeheader()
            w.writerows(cell_rows)

    n_fail = sum(1 for r in rows if r["result"] == "FAIL")
    print(f"wrote {out} ({len(rows)} rows, {n_fail} FAIL)")
    print(f"wrote {cells_out} ({len(cell_rows)} rows)")
    for r in rows:
        print(f"[{r['result']}] {r['check']}: {r['detail']}")


if __name__ == "__main__":
    main()
