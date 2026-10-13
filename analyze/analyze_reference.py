"""Aggregate Phase 1 (reference-pipeline) result JSONs into a train x test
matrix with cross-seed mean/std and bootstrap CIs, and diff it cell-by-cell
against the current (MTCNN/JPEG) pipeline's existing matrix -- this is the
GATE 1 comparison from RESEARCH_BRIEF.md ("Compare against the current
pipeline cell by cell").

Expects result JSONs named <model>_<trainmethod>_to_<testmethod>_<comp>_seed<k>.json
(as written by experiments/phase1/run_phase1_matrix.py -> src/evaluate.py),
each with a "video_auc" field and (if evaluate.py's --bootstrap wasn't
disabled) "video_auc_ci95_lo"/"video_auc_ci95_hi".

Outputs (in --out-dir):
  reference_matrix_<model>.csv        train-method x test-method mean video AUC
                                       across seeds, "mean (std, n_seeds)" cells
  reference_matrix_<model>_raw.csv    one row per (train, test, seed) -- the
                                       underlying numbers behind the summary
  reference_vs_current_<model>.csv    cell-by-cell delta vs. the existing
                                       current-pipeline matrix (GATE 1 input)

Usage:
  python -m analyze.analyze_reference --results-dir results/reference \
      --out-dir analysis/reference --current-csv analysis/cross_manipulation_xception.csv
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SHORT = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS",
         "NeuralTextures": "NT"}
NAME_RE = re.compile(
    r"^(?P<model>[a-zA-Z_]+)_(?P<train>[A-Za-z0-9]+)_to_(?P<test>[A-Za-z0-9]+)_"
    r"(?P<comp>[a-zA-Z0-9]+)_seed(?P<seed>\d+)\.json$")


def load_results(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        m = NAME_RE.match(f.name)
        if not m:
            print(f"[warn] skipping file that doesn't match the naming "
                  f"convention: {f.name}")
            continue
        data = json.loads(f.read_text())
        rows.append({
            "model": m["model"], "train_method": m["train"],
            "test_method": m["test"], "compression": m["comp"],
            "seed": int(m["seed"]), "video_auc": data.get("video_auc"),
            "ci_lo": data.get("video_auc_ci95_lo"),
            "ci_hi": data.get("video_auc_ci95_hi"),
            "n_videos": data.get("n_videos"), "file": str(f),
        })
    return pd.DataFrame(rows)


def summarize(df, model, out_dir):
    sub = df[df.model == model]
    if sub.empty:
        print(f"[warn] no results for model={model}")
        return None
    sub.to_csv(out_dir / f"reference_matrix_{model}_raw.csv", index=False)

    train_methods = sorted(sub.train_method.unique(), key=lambda m: SHORT.get(m, m))
    test_methods = sorted(sub.test_method.unique(), key=lambda m: SHORT.get(m, m))
    cell_text = pd.DataFrame(index=[SHORT.get(m, m) for m in train_methods],
                             columns=[SHORT.get(m, m) for m in test_methods])
    mean_mat = pd.DataFrame(index=train_methods, columns=test_methods, dtype=float)

    for tr in train_methods:
        for te in test_methods:
            cell = sub[(sub.train_method == tr) & (sub.test_method == te)]
            if cell.empty:
                continue
            aucs = cell.video_auc.dropna().to_numpy()
            mean_mat.loc[tr, te] = aucs.mean() if len(aucs) else np.nan
            n = len(aucs)
            txt = f"{aucs.mean():.4f}" if n else "NA"
            if n > 1:
                txt += f" (sd={aucs.std(ddof=1):.4f}, n={n})"
            elif n == 1:
                txt += " (n=1 -- not yet 3 seeds)"
            cell_text.loc[SHORT.get(tr, tr), SHORT.get(te, te)] = txt

    cell_text.to_csv(out_dir / f"reference_matrix_{model}.csv")
    return mean_mat


def compare_to_current(mean_mat, current_csv, model, out_dir):
    if mean_mat is None or not Path(current_csv).exists():
        print(f"[warn] skipping current-pipeline comparison "
              f"(matrix empty or {current_csv} missing)")
        return
    current = pd.read_csv(current_csv, index_col=0)
    rows = []
    for tr in mean_mat.index:
        tr_short = SHORT.get(tr, tr)
        if tr_short not in current.index:
            continue
        for te in mean_mat.columns:
            te_short = SHORT.get(te, te)
            if te_short not in current.columns:
                continue
            ref_auc = mean_mat.loc[tr, te]
            cur_auc = current.loc[tr_short, te_short]
            rows.append({
                "train_method": tr_short, "test_method": te_short,
                "current_pipeline_auc": cur_auc,
                "reference_pipeline_auc": ref_auc,
                "delta": (ref_auc - cur_auc) if pd.notna(ref_auc) else np.nan,
                "current_below_chance": bool(cur_auc < 0.5),
                "reference_below_chance": (bool(ref_auc < 0.5)
                                           if pd.notna(ref_auc) else None),
            })
    pd.DataFrame(rows).to_csv(out_dir / f"reference_vs_current_{model}.csv",
                              index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/reference")
    p.add_argument("--out-dir", default="analysis/reference")
    p.add_argument("--current-csv", default="analysis/cross_manipulation_xception.csv",
                   help="Current-pipeline matrix to diff against (GATE 1)")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    if df.empty:
        raise SystemExit(f"No matching result JSONs in {args.results_dir} "
                         f"(expected <model>_<train>_to_<test>_<comp>_seed<k>.json)")

    for model in sorted(df.model.unique()):
        mean_mat = summarize(df, model, out_dir)
        compare_to_current(mean_mat, args.current_csv, model, out_dir)
    print(f"wrote tables to {out_dir}/")


if __name__ == "__main__":
    main()
