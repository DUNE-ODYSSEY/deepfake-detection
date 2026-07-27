"""Box-plot (IQR) outlier detection on FF++ per-video metadata.

Restricted to the 5 folders this project actually uses (original + the 4
manipulation methods in src/data/preprocess.py's METHODS list) — the Kaggle
mirror's csv/ also covers DeepFakeDetection and FaceShifter, which this study
doesn't train or evaluate on, so they're excluded here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

CSV_DIR = r"C:\ffpp_data\raw_kaggle\FaceForensics++_C23\csv"
USED_PREFIXES = ("original/", "Deepfakes/", "Face2Face/", "FaceSwap/", "NeuralTextures/")

# palette matching docs/mid_review_update.pptx
NAVY = "#0F172A"
TEXT = "#1A202C"
MUTED = "#475569"
GRID = "#E2E8F0"
CYAN = "#22B8CF"
DANGER = "#C2410C"

if any("Calibri" in f.name for f in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Calibri"
else:
    plt.rcParams["font.family"] = "sans-serif"

df = pd.read_csv(f"{CSV_DIR}/FF++_Metadata.csv")
df = df[df["File Path"].str.startswith(USED_PREFIXES)].copy()
total_n = len(df)

def iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, q1, q3

metrics = [("Frame Count", "Frame Count"), ("File Size(MB)", "File Size (MB)")]
summary = []
outlier_mask = pd.Series(False, index=df.index)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=200)

for ax, (col, label) in zip(axes, metrics):
    lo, hi, q1, q3 = iqr_bounds(df[col])
    is_out = (df[col] < lo) | (df[col] > hi)
    outlier_mask |= is_out
    summary.append({
        "metric": label, "q1": q1, "q3": q3,
        "lower_fence": max(lo, 0), "upper_fence": hi,
        "n_outliers": int(is_out.sum()),
        "pct_outliers": 100 * is_out.sum() / total_n,
    })

    bp = ax.boxplot(
        df[col], orientation="vertical", widths=0.45, patch_artist=True,
        flierprops=dict(marker="D", markerfacecolor=DANGER, markeredgecolor=DANGER,
                         markersize=4, alpha=0.75, linewidth=0),
        boxprops=dict(facecolor=CYAN, alpha=0.25, edgecolor=CYAN, linewidth=1.6),
        medianprops=dict(color=NAVY, linewidth=2),
        whiskerprops=dict(color=MUTED, linewidth=1.3),
        capprops=dict(color=MUTED, linewidth=1.3),
    )
    ax.set_title(label, fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.96, f"{is_out.sum()} outliers flagged", fontsize=8.5, color=DANGER,
            va="top", ha="right", fontweight="bold", transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

fig.patch.set_facecolor("white")
plt.tight_layout(pad=1.2)
fig.savefig("analyze/outputs/outlier_boxplots.png", facecolor="white", bbox_inches="tight")

summary_df = pd.DataFrame(summary)
print(f"Total videos analyzed (original + 4 manipulation methods): {total_n}")
print(summary_df.to_string(index=False))
print(f"\nUnion of outliers across both metrics: {int(outlier_mask.sum())} "
      f"({100*outlier_mask.sum()/total_n:.2f}% of dataset)")
print(f"Clean videos remaining if excluded: {total_n - int(outlier_mask.sum())}")

# per-method breakdown of flagged outliers
df["method"] = df["File Path"].str.split("/").str[0]
df["is_outlier"] = outlier_mask
breakdown = df.groupby("method")["is_outlier"].sum().astype(int)
print("\nOutliers by method:")
print(breakdown.to_string())
