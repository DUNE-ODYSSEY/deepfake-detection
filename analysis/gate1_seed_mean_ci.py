"""GATE 1 table: seed-mean video AUC per train x test cell with a 95% CI from
bootstrapping over test videos (RESEARCH_BRIEF.md section 5).

Every seed scores the same test videos, so each resample draws one set of
video indices and recomputes every seed's AUC on it; the statistic is the mean
across seeds. A cell is called below/above chance only when the whole CI is on
one side of 0.5 -- a mean under 0.5 with a CI spanning it is "at chance".

Reads results/reference/*_c23_seed<k>.video_scores.csv (written by
src/evaluate.py) and the current pipeline's analysis/cross_manipulation_xception.csv.

Usage: python -m analysis.gate1_seed_mean_ci
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SHORT = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS", "NeuralTextures": "NT"}
NAME_RE = re.compile(r"^(?P<model>[a-zA-Z_]+)_(?P<train>[A-Za-z0-9]+)_to_(?P<test>[A-Za-z0-9]+)_"
                     r"(?P<comp>[a-zA-Z0-9]+)_seed(?P<seed>\d+)\.video_scores\.csv$")


def classify(lo, hi):
    if hi < 0.5:
        return "below chance"
    if lo > 0.5:
        return "above chance"
    return "at chance"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/reference")
    p.add_argument("--current-csv", default="analysis/cross_manipulation_xception.csv")
    p.add_argument("--out", default="analysis/reference/gate1_seed_mean_ci.csv")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    groups = {}
    for f in sorted(Path(args.results_dir).glob("*.video_scores.csv")):
        m = NAME_RE.match(f.name)
        if not m:
            raise SystemExit(f"unexpected file name: {f.name}")
        key = (m["model"], m["train"], m["test"], m["comp"])
        groups.setdefault(key, {})[int(m["seed"])] = f

    current = pd.read_csv(args.current_csv, index_col=0)
    rng = np.random.default_rng(args.seed)
    rows = []
    for (model, train, test, comp), seed_files in sorted(groups.items()):
        frames = [pd.read_csv(f).set_index("video_id") for _, f in sorted(seed_files.items())]
        ids = frames[0].index
        for fr in frames[1:]:
            if not fr.index.sort_values().equals(ids.sort_values()):
                raise SystemExit(f"seeds score different test videos for {train}->{test}")
        labels = frames[0].loc[ids, "label"].to_numpy()
        probs = np.stack([fr.loc[ids, "video_prob"].to_numpy() for fr in frames])

        per_seed = [roc_auc_score(labels, pr) for pr in probs]
        boot = []
        n = len(ids)
        while len(boot) < args.n_resamples:
            idx = rng.integers(0, n, n)
            if len(np.unique(labels[idx])) < 2:
                continue
            boot.append(np.mean([roc_auc_score(labels[idx], pr[idx]) for pr in probs]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        mean = float(np.mean(per_seed))
        tr, te = SHORT.get(train, train), SHORT.get(test, test)
        rows.append({
            "model": model, "train": tr, "test": te, "n_seeds": len(per_seed),
            "n_videos": n, "seed_mean_auc": round(mean, 4),
            "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4),
            "seed_sd": round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else None,
            "per_seed_auc": ";".join(f"{a:.4f}" for a in per_seed),
            "reference_call": classify(lo, hi),
            "current_pipeline_auc": float(current.loc[tr, te]) if tr in current.index else None,
            "data_source": "mirror (preliminary)",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df[["train", "test", "n_seeds", "seed_mean_auc", "ci95_lo", "ci95_hi",
              "reference_call", "current_pipeline_auc"]].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
