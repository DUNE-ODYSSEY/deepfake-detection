"""Paper figures (RESEARCH_BRIEF.md Phase 7), all drawn from script outputs.

  F1  transfer matrices: current pipeline vs. reference pipeline (Xception)
  F3  hf_ratio effect size per method vs. real, both crop sets, with 95% CIs
  F4  predicted (d_A * d_B, hf_ratio) vs. observed cross-manipulation AUC
  F6  Xception vs. CNN-ViT off-diagonal AUCs (architecture independence)

Colours: the dataviz reference palette (categorical slots 1-3, which validate
all-pairs; diverging blue <-> red with a gray midpoint at AUC 0.5). Writes
PNG (300 dpi) and PDF to analysis/figures/.

Usage: python -m analysis.make_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

OUT = Path("analysis/figures")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list(
    "auc", [(0.0, "#a3302f"), (0.25, "#e34948"), (0.5, "#f0efec"), (0.75, "#5598e7"), (1.0, "#184f95")])
ORDER = ["DF", "F2F", "FS", "NT"]
NAMES = {"DF": "Deepfakes", "F2F": "Face2Face", "FS": "FaceSwap", "NT": "NeuralTextures"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": "white", "axes.titlesize": 9, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png/.pdf")


def ink_for(v):
    return "white" if abs(v - 0.5) > 0.32 else INK


def heatmap(ax, values, rows, cols, labels, below):
    ax.imshow(values, cmap=DIVERGING, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if below[i][j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor="white", linewidth=0, alpha=0.55))
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7,
                    color=ink_for(values[i][j]), fontweight="semibold" if below[i][j] else "normal")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Trained on")


def fig1():
    cur = pd.read_csv("analysis/cross_manipulation_xception.csv", index_col=0).loc[ORDER, ORDER]
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv").set_index(["train", "test"])
    ref_rows = ["DF", "FS"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.05]})
    v = cur.to_numpy(float)
    heatmap(axes[0], v, ORDER, ORDER, [[f"{x:.2f}" for x in r] for r in v],
            [[x < 0.5 and abs(x - 0.5) >= 0.03 for x in r] for r in v])
    axes[0].set_title("Current pipeline (MTCNN, JPEG, 1 seed)", loc="left")
    rv = np.array([[g.loc[(r, c), "seed_mean_auc"] for c in ORDER] for r in ref_rows])
    rl = [[f"{g.loc[(r, c), 'seed_mean_auc']:.2f}\n[{g.loc[(r, c), 'ci95_lo']:.2f}, {g.loc[(r, c), 'ci95_hi']:.2f}]"
           for c in ORDER] for r in ref_rows]
    rb = [[g.loc[(r, c), "reference_call"] == "below chance" for c in ORDER] for r in ref_rows]
    heatmap(axes[1], rv, ref_rows, ORDER, rl, rb)
    axes[1].set_title("Reference pipeline (dlib + align, PNG, 3 seeds)", loc="left")
    fig.text(0.01, -0.04, "Video-level AUC, Xception, FF++ c23 (Kaggle mirror). Hatched = below chance "
             "(reference: whole 95% CI < 0.5; current: point estimate < 0.47). Red < 0.5 < blue.",
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "F1_transfer_matrices")


def fig3():
    e = pd.read_csv("analysis/phase3/effect_sizes.csv")
    e = e[e["stat"] == "hf_ratio"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    y = np.arange(len(ORDER))[::-1]
    for k, (pl, col, lab, off) in enumerate([("current", S1, "Current crops", 0.13),
                                             ("reference", S2, "Reference crops", -0.13)]):
        sub = e[e["pipeline"] == pl].set_index("method").loc[ORDER]
        ax.errorbar(sub["d"], y + off, xerr=[sub["d"] - sub["d_ci_lo"], sub["d_ci_hi"] - sub["d"]],
                    fmt="o", color=col, ms=5, elinewidth=1.4, capsize=0, label=lab)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, [NAMES[m] for m in ORDER])
    ax.tick_params(axis="y", length=0, labelcolor=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cohen's d, fakes vs. real  (high-frequency energy)")
    ax.text(-0.02, 1.02, "← blurrier than real", transform=ax.transAxes, fontsize=6.5, color=MUTED)
    ax.text(1.0, 1.02, "sharper than real →", transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="right")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "F3_hf_effect_sizes")


def fig4():
    e = pd.read_csv("analysis/phase3/effect_sizes.csv")
    d = {pl: e[(e["stat"] == "hf_ratio") & (e["pipeline"] == pl)].set_index("method")["d"]
         for pl in ["current", "reference"]}
    pts = []
    for name, path, col in [("Xception, current", "analysis/cross_manipulation_xception.csv", S1),
                            ("CNN-ViT, current", "analysis/cross_manipulation_cnn_vit.csv", S3)]:
        m = pd.read_csv(path, index_col=0)
        pts += [(name, col, a, b, d["current"][a] * d["current"][b], float(m.loc[a, b]))
                for a in ORDER for b in ORDER if a != b]
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv")
    pts += [("Xception, reference", S2, r.train, r.test, d["reference"][r.train] * d["reference"][r.test],
             r.seed_mean_auc) for r in g[g.train != g.test].itertuples()]
    df = pd.DataFrame(pts, columns=["series", "color", "a", "b", "x", "auc"])

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.axhline(0.5, color=AXIS, linewidth=1)
    ax.axvline(0, color=AXIS, linewidth=1)
    for name, sub in df.groupby("series", sort=False):
        ax.scatter(sub["x"], sub["auc"], s=30, color=sub["color"].iloc[0], edgecolor=SURFACE,
                   linewidth=1, label=name, zorder=3)
    below = df[df["auc"] < 0.47]
    assert all("FS" in (a, b) and {a, b} != {"FS", "F2F"} for a, b in zip(below["a"], below["b"]))
    ax.text(0.98, 0.12, "Every cell below chance pairs FaceSwap\nwith Deepfakes or NeuralTextures",
            transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    rho_x = spearmanr(df[df.series == "Xception, current"]["x"], df[df.series == "Xception, current"]["auc"]).statistic
    ax.set_xlabel("Predicted: d(train) × d(test), high-frequency energy")
    ax.set_ylabel("Observed cross-manipulation AUC")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.04, f"Spearman ρ = {rho_x:.2f} (Xception, current)", transform=ax.transAxes,
            ha="right", fontsize=7, color=INK2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "F4_sign_prediction")


def fig6():
    xc = pd.read_csv("analysis/cross_manipulation_xception.csv", index_col=0)
    cv = pd.read_csv("analysis/cross_manipulation_cnn_vit.csv", index_col=0)
    cells = [(a, b) for a in ORDER for b in ORDER if a != b]
    x = np.array([xc.loc[a, b] for a, b in cells], float)
    y = np.array([cv.loc[a, b] for a, b in cells], float)
    r = pearsonr(x, y).statistic
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    ax.axhline(0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.scatter(x, y, s=32, color=S1, edgecolor=SURFACE, linewidth=1, zorder=3)
    for (a, b), xi, yi in zip(cells, x, y):
        if min(xi, yi) < 0.45:
            left = (a, b) == ("FS", "NT")
            ax.annotate(f"{a}→{b}", (xi, yi), xytext=(-5 if left else 5, -3),
                        textcoords="offset points", ha="right" if left else "left",
                        fontsize=6.5, color=INK2)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Xception AUC")
    ax.set_ylabel("CNN-ViT AUC")
    ax.text(0.04, 0.95, f"Pearson r = {r:.2f}, 12 off-diagonal cells", transform=ax.transAxes,
            fontsize=7, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "F6_architecture_agreement")


if __name__ == "__main__":
    fig1()
    fig3()
    fig4()
    fig6()
