"""ROC curves from saved per-video scores -> analysis/figures/F6_roc_curves.png.

Left: course baseline (trained and tested on all four methods, original
pipeline) for XceptionNet and the CNN-ViT hybrid. Scores come from
results/roc/, produced by re-running src.evaluate on the baseline checkpoints
with per-video score saving on.

Right: Xception trained on Deepfakes (reference pipeline, seed 1), tested on
each manipulation. A curve under the diagonal is a below-chance cell.

Also writes analysis/roc_auc.csv with the AUC of every plotted curve.

Usage: python -m analysis.roc_curves
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

OUT_FIG = Path("analysis/figures/F6_roc_curves.png")
OUT_CSV = Path("analysis/roc_auc.csv")

BASELINE = [
    ("XceptionNet", "results/roc/xception_all_to_all_c23.video_scores.csv", "#1f4e79"),
    ("CNN-ViT hybrid", "results/roc/cnn_vit_all_to_all_c23.video_scores.csv", "#0e9f9a"),
]
CROSS = [
    ("Deepfakes (seen)", "Deepfakes", "#1f4e79"),
    ("Face2Face", "Face2Face", "#7d8b99"),
    ("NeuralTextures", "NeuralTextures", "#e0a020"),
    ("FaceSwap", "FaceSwap", "#c0392b"),
]
CROSS_PATH = "results/reference/xception_Deepfakes_to_{}_c23_seed1.video_scores.csv"


def curve(path):
    df = pd.read_csv(path)
    fpr, tpr, _ = roc_curve(df["label"], df["video_prob"])
    return fpr, tpr, roc_auc_score(df["label"], df["video_prob"]), len(df)


def panel(ax, title):
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", label="chance (AUC 0.50)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (real flagged as fake)")
    ax.set_ylabel("True positive rate (fakes caught)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def main():
    rows = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.6))

    panel(a, "Baseline: trained and tested on all 4 methods")
    for label, path, colour in BASELINE:
        fpr, tpr, auc, n = curve(path)
        a.plot(fpr, tpr, lw=2.2, color=colour, label=f"{label} (AUC {auc:.3f})")
        rows.append({"panel": "baseline", "curve": label, "auc": auc, "n_videos": n})
    a.legend(loc="lower right", fontsize=10)

    panel(b, "Xception trained on Deepfakes, tested on each method")
    for label, method, colour in CROSS:
        fpr, tpr, auc, n = curve(CROSS_PATH.format(method))
        b.plot(fpr, tpr, lw=2.2, color=colour, label=f"{label} (AUC {auc:.2f})")
        rows.append({"panel": "cross_DF_seed1", "curve": method, "auc": auc, "n_videos": n})
    b.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {OUT_FIG} and {OUT_CSV}")


if __name__ == "__main__":
    main()
