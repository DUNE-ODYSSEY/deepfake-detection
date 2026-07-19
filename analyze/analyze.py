"""Aggregate result JSONs into tables and figures.

Outputs (in --out-dir):
  baseline_comparison.csv        Experiment 1 table
  cross_manipulation_<model>.csv 4x4 generalization matrix (video AUC)
  cross_manipulation_heatmaps.png  side-by-side heatmaps for both models
  generalization_gap.csv         same-manip vs cross-manip average drop
  compression_robustness.csv     Experiment 3 table

Usage: python -m analyze.analyze --results-dir results --out-dir analysis
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
SHORT = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS",
         "NeuralTextures": "NT"}


def load_results(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        rows.append(json.loads(f.read_text()))
    if not rows:
        raise SystemExit(f"No result JSONs in {results_dir}")
    df = pd.DataFrame(rows)
    df["train_key"] = df["train_methods"].apply(
        lambda m: "all" if len(m) == 4 else m[0])
    df["test_key"] = df["test_methods"].apply(
        lambda m: "all" if len(m) == 4 else m[0])
    return df


def baseline_table(df, out_dir):
    sel = df[(df.train_key == "all") & (df.test_key == "all")
             & (df.train_compression == "c23") & (df.test_compression == "c23")]
    cols = ["model", "video_acc", "video_f1", "video_auc",
            "frame_acc", "frame_f1", "frame_auc"]
    if not sel.empty:
        sel[cols].round(4).to_csv(out_dir / "baseline_comparison.csv", index=False)


def cross_manip(df, out_dir):
    models = sorted(df.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5),
                             squeeze=False)
    gap_rows = []
    for ax, model in zip(axes[0], models):
        mat = np.full((4, 4), np.nan)
        for i, tr in enumerate(METHODS):
            for j, te in enumerate(METHODS):
                sel = df[(df.model == model) & (df.train_key == tr)
                         & (df.test_key == te) & (df.test_compression == "c23")]
                if not sel.empty:
                    mat[i, j] = sel.iloc[0]["video_auc"]
        pd.DataFrame(mat, index=[SHORT[m] for m in METHODS],
                     columns=[SHORT[m] for m in METHODS]).round(4).to_csv(
            out_dir / f"cross_manipulation_{model}.csv")

        same = np.nanmean(np.diag(mat))
        off = mat.copy()
        np.fill_diagonal(off, np.nan)
        cross = np.nanmean(off)
        gap_rows.append({"model": model, "same_manip_auc": round(same, 4),
                         "cross_manip_auc": round(cross, 4),
                         "generalization_gap": round(same - cross, 4)})

        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(4), [SHORT[m] for m in METHODS])
        ax.set_yticks(range(4), [SHORT[m] for m in METHODS])
        ax.set_xlabel("Tested on")
        ax.set_ylabel("Trained on")
        ax.set_title(f"{model} — video AUC")
        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                            fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_manipulation_heatmaps.png", dpi=200)
    pd.DataFrame(gap_rows).to_csv(out_dir / "generalization_gap.csv", index=False)


def compression_table(df, out_dir):
    sel = df[(df.train_key == "all") & (df.test_key == "all")]
    if sel.empty:
        return
    cols = ["model", "train_compression", "test_compression",
            "video_acc", "video_f1", "video_auc"]
    (sel[cols].sort_values(["model", "train_compression", "test_compression"])
     .round(4).to_csv(out_dir / "compression_robustness.csv", index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    baseline_table(df, out_dir)
    cross_manip(df, out_dir)
    compression_table(df, out_dir)
    print(f"wrote tables and figures to {out_dir}/")


if __name__ == "__main__":
    main()
