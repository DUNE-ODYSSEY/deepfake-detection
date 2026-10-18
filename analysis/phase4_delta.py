"""Phase 4 vs. Phase 1: paired change in cross-manipulation AUC per cell.

Both phases evaluate the same test videos, so each bootstrap resample draws
one set of video indices and recomputes the 3-seed mean AUC under both
conditions; the statistic is the difference (Phase 4 - Phase 1).

Brief's success criterion (Phase 4): cells below chance in Phase 1 rise with
the Phase 4 CI excluding the Phase 1 mean, and in-domain AUC drops by no more
than 0.02. Writes analysis/phase4/delta_vs_phase1.csv and figure F5.

Usage: python -m analysis.phase4_delta
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.make_figures import AXIS, GRID, INK2, MUTED, S1, S2, save

SHORT = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS", "NeuralTextures": "NT"}
NAME_RE = re.compile(r"^xception_(?P<train>[A-Za-z0-9]+)_to_(?P<test>[A-Za-z0-9]+)_c23_seed(?P<seed>\d+)"
                     r"\.video_scores\.csv$")


def load(results_dir):
    out = {}
    for f in Path(results_dir).glob("*.video_scores.csv"):
        m = NAME_RE.match(f.name)
        if m:
            s = pd.read_csv(f).set_index("video_id")
            out.setdefault((SHORT[m["train"]], SHORT[m["test"]]), {})[int(m["seed"])] = s
    return out


def seed_mean_auc(frames, idx, ids):
    return float(np.mean([roc_auc_score(fr.loc[ids, "label"].to_numpy()[idx],
                                        fr.loc[ids, "video_prob"].to_numpy()[idx]) for fr in frames]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/reference")
    p.add_argument("--intervention", default="results/phase4_hfaug")
    p.add_argument("--out", default="analysis/phase4/delta_vs_phase1.csv")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    base, intv = load(args.baseline), load(args.intervention)
    rows = []
    for cell in sorted(set(base) & set(intv)):
        bf = [base[cell][k] for k in sorted(base[cell])]
        inf = [intv[cell][k] for k in sorted(intv[cell])]
        ids = bf[0].index
        if any(not f.index.sort_values().equals(ids.sort_values()) for f in bf + inf):
            raise SystemExit(f"test videos differ between phases for {cell}")
        labels = bf[0].loc[ids, "label"].to_numpy()
        full = np.arange(len(ids))
        b0, i0 = seed_mean_auc(bf, full, ids), seed_mean_auc(inf, full, ids)
        bs, ins, ds = [], [], []
        while len(ds) < args.n_resamples:
            idx = rng.integers(0, len(ids), len(ids))
            if len(np.unique(labels[idx])) < 2:
                continue
            b, i = seed_mean_auc(bf, idx, ids), seed_mean_auc(inf, idx, ids)
            bs.append(b); ins.append(i); ds.append(i - b)
        lo, hi = np.percentile(ins, [2.5, 97.5])
        dlo, dhi = np.percentile(ds, [2.5, 97.5])
        blo, bhi = np.percentile(bs, [2.5, 97.5])
        train, test = cell
        rows.append({
            "train": train, "test": test, "n_seeds_phase1": len(bf), "n_seeds_phase4": len(inf),
            "phase1_auc": round(b0, 4), "phase1_ci_lo": round(blo, 4), "phase1_ci_hi": round(bhi, 4),
            "phase4_auc": round(i0, 4), "phase4_ci_lo": round(lo, 4), "phase4_ci_hi": round(hi, 4),
            "delta": round(i0 - b0, 4), "delta_ci_lo": round(dlo, 4), "delta_ci_hi": round(dhi, 4),
            "in_domain": train == test,
            "phase1_below_chance": bhi < 0.5,
            "phase4_below_chance": hi < 0.5,
            "criterion_ci_excludes_phase1_mean": (lo > b0) if bhi < 0.5 else None,
            "criterion_in_domain_drop_le_0.02": (b0 - i0 <= 0.02) if train == test else None,
            "data_source": "mirror (preliminary)",
        })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df[["train", "test", "n_seeds_phase4", "phase1_auc", "phase4_auc", "phase4_ci_lo",
              "phase4_ci_hi", "delta", "delta_ci_lo", "delta_ci_hi"]].to_string(index=False))

    off = df[~df["in_domain"]].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(off))[::-1]
    for col, pref, lab, dy in [(S1, "phase1", "Phase 1 (no augmentation)", 0.13),
                               (S2, "phase4", "Phase 4 (blur/sharpen aug.)", -0.13)]:
        ax.errorbar(off[f"{pref}_auc"], y + dy,
                    xerr=[off[f"{pref}_auc"] - off[f"{pref}_ci_lo"], off[f"{pref}_ci_hi"] - off[f"{pref}_auc"]],
                    fmt="o", color=col, ms=5, elinewidth=1.4, capsize=0, label=lab)
    ax.axvline(0.5, color=AXIS, linewidth=1)
    ax.text(0.5, 0.01, " chance", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=6.5, color=MUTED)
    ax.set_yticks(y, [f"{a}→{b}" for a, b in zip(off["train"], off["test"])])
    ax.tick_params(axis="y", length=0, labelcolor=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cross-manipulation video AUC (3-seed mean, 95% CI)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.tight_layout()
    save(fig, "F5_intervention")


if __name__ == "__main__":
    main()
