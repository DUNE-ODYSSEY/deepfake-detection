"""Phase 3.2-3.4: effect sizes, sign-prediction test, and model reliance.

Design fixed before looking at any statistic (RESEARCH_BRIEF.md Phase 3):

3.2  Effect size per (pipeline, statistic, method): Cohen's d of per-video
     means, method fakes vs. all real test videos (positive = fakes higher),
     95% CI from 1000 bootstrap resamples over videos. Also reported: paired
     d (fake a_b minus its target real a) and the AUC of the raw statistic as
     a fake score.
3.3  Sign prediction for train A -> test B (A != B): predicted transfer sign
     sign(d_A) * sign(d_B) (+1 above chance, -1 inverted), continuous score
     d_A * d_B. Scored against observed off-diagonal AUCs:
       - accuracy of the below/above-0.5 call. Reference-pipeline cells use
         the CI-based call ("at chance" cells excluded); current-pipeline cells
         have no CIs, so cells with |AUC - 0.5| < 0.03 are excluded;
       - Spearman rho between d_A * d_B and the observed AUC (all cells).
     Null: all 24 relabelings of the four methods' d values; p = share of
     relabelings scoring >= the observed one (identity included, so the
     smallest attainable p is 1/24 ~ 0.042 -- the test is coarse by design).
3.4  Reliance (reference pipeline only; per-video scores exist there): for
     each trained model, Spearman between its fake probability (mean over 3
     seeds) and each statistic among the 140 real test videos. A model relying
     on statistic s should score reals higher when s moves the way its
     training fakes moved: predicted sign(rho) = sign(d_train).

Usage: python -m experiments.phase3.sign_prediction
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.phase3.crop_statistics import STATS

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
SHORT = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS", "NeuralTextures": "NT"}
LONG = {v: k for k, v in SHORT.items()}
AT_CHANCE_BAND = 0.03


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else 0.0


def effect_sizes(stats, pipeline, rng, n_boot):
    rows = []
    for stat in [c for c in stats.columns if c in STATS or c == "box_area_frac"]:
        s = stats[stats[stat].notna()]
        real = s[s["class"] == "real"]
        r = real[stat].to_numpy()
        real_by_video = real.groupby("video")[stat].mean()
        for m in METHODS:
            fk = s[s["class"] == m]
            f = fk[stat].to_numpy()
            if len(f) < 5 or len(r) < 5:
                continue
            d = cohens_d(f, r)
            boots = [cohens_d(rng.choice(f, len(f)), rng.choice(r, len(r))) for _ in range(n_boot)]
            paired = pd.Series(dtype=float)
            if "target_real" in fk.columns and fk["target_real"].notna().all():
                paired = pd.Series(f - real_by_video.reindex(fk["target_real"]).to_numpy()).dropna()
            rows.append({
                "pipeline": pipeline, "stat": stat, "method": SHORT[m], "n_fake": len(f),
                "n_real": len(r), "d": d, "d_ci_lo": np.percentile(boots, 2.5),
                "d_ci_hi": np.percentile(boots, 97.5),
                "d_paired": paired.mean() / paired.std(ddof=1) if len(paired) > 2 else np.nan,
                "auc_stat": roc_auc_score(np.r_[np.ones(len(f)), np.zeros(len(r))], np.r_[f, r]),
            })
    return pd.DataFrame(rows)


def load_matrices():
    mats = {}
    for name, path in [("current_xception", "analysis/cross_manipulation_xception.csv"),
                       ("current_cnn_vit", "analysis/cross_manipulation_cnn_vit.csv")]:
        m = pd.read_csv(path, index_col=0)
        cells = []
        for a in m.index:
            for b in m.columns:
                if a != b:
                    auc = float(m.loc[a, b])
                    call = 0 if abs(auc - 0.5) < AT_CHANCE_BAND else int(np.sign(auc - 0.5))
                    cells.append((a, b, auc, call))
        mats[name] = ("current", cells)
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv")
    cells = []
    for _, r in g[g["train"] != g["test"]].iterrows():
        call = {"below chance": -1, "above chance": 1}.get(r["reference_call"], 0)
        cells.append((r["train"], r["test"], float(r["seed_mean_auc"]), call))
    mats["reference_xception"] = ("reference", cells)
    return mats


def score(dvec, cells):
    pred = np.array([np.sign(dvec[a] * dvec[b]) for a, b, _, _ in cells])
    cont = np.array([dvec[a] * dvec[b] for a, b, _, _ in cells])
    obs_auc = np.array([c[2] for c in cells])
    calls = np.array([c[3] for c in cells])
    keep = calls != 0
    acc = float((pred[keep] == calls[keep]).mean()) if keep.any() else np.nan
    rho = spearmanr(cont, obs_auc).statistic if len(set(cont)) > 1 else np.nan
    return acc, rho, int(keep.sum()), int((pred[keep] == calls[keep]).sum())


def sign_prediction(eff, mats):
    rows = []
    shorts = [SHORT[m] for m in METHODS]
    for mname, (pipeline, cells) in mats.items():
        sub = eff[eff["pipeline"] == pipeline]
        for stat, g in sub.groupby("stat"):
            d = dict(zip(g["method"], g["d"]))
            if set(d) != set(shorts):
                continue
            acc, rho, n_cells, n_ok = score(d, cells)
            null = [score(dict(zip(shorts, [d[s] for s in perm])), cells)[:2]
                    for perm in itertools.permutations(shorts)]
            accs, rhos = np.array([n[0] for n in null]), np.array([n[1] for n in null])
            rows.append({
                "matrix": mname, "stat": stat, "cells_scored": n_cells, "cells_correct": n_ok,
                "accuracy": acc, "perm_p_accuracy": float((accs >= acc - 1e-12).mean()),
                "spearman_rho": rho,
                "perm_p_rho": float((rhos >= rho - 1e-12).mean()) if not np.isnan(rho) else np.nan,
                "d_signs": " ".join(f"{s}{'+' if d[s] > 0 else '-'}" for s in shorts),
            })
    return pd.DataFrame(rows).sort_values(["matrix", "perm_p_rho", "perm_p_accuracy"])


def reliance(eff, results_dir, stats_ref):
    real_stats = stats_ref[stats_ref["class"] == "real"].set_index("video")
    rows = []
    for train in ["Deepfakes", "FaceSwap"]:
        probs = []
        for f in sorted(Path(results_dir).glob(f"xception_{train}_to_{train}_c23_seed*.video_scores.csv")):
            s = pd.read_csv(f)
            s = s[s["label"] == 0].copy()
            s["video"] = s["video_id"].str.split("/").str[-1]
            probs.append(s.set_index("video")["video_prob"])
        if not probs:
            continue
        p = pd.concat(probs, axis=1).mean(axis=1)
        for stat in STATS:
            x = real_stats[stat].reindex(p.index)
            ok = x.notna()
            rho, pval = spearmanr(p[ok], x[ok])
            d = eff[(eff["pipeline"] == "reference") & (eff["stat"] == stat)
                    & (eff["method"] == SHORT[train])]["d"]
            d = float(d.iloc[0]) if len(d) else np.nan
            rows.append({"train": SHORT[train], "stat": stat, "n_real_videos": int(ok.sum()),
                         "rho_prob_vs_stat_on_reals": rho, "p_value": pval, "d_train": d,
                         "sign_matches_d_train": bool(np.sign(rho) == np.sign(d))})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats-dir", default="analysis/phase3")
    p.add_argument("--box-csv", default="results/audit/crop_audit_box_sizes.csv")
    p.add_argument("--results-dir", default="results/reference")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.stats_dir)

    ids = {"video": str, "target_real": str}
    stats = {pl: pd.read_csv(out / f"crop_stats_{pl}.csv", dtype=ids) for pl in ["current", "reference"]}
    box = pd.read_csv(args.box_csv)
    box = box[box["detected"]]
    box = box.assign(box_area_frac=box["box_area"] / (box["frame_h"] * box["frame_w"]),
                     video=box["video"].astype(str))[["class", "video", "box_area_frac"]]
    stats["current"] = pd.concat([stats["current"], box], ignore_index=True)

    eff = pd.concat([effect_sizes(stats[pl], pl, rng, args.n_boot) for pl in stats], ignore_index=True)
    eff.to_csv(out / "effect_sizes.csv", index=False)
    sp = sign_prediction(eff, load_matrices())
    sp.to_csv(out / "sign_prediction.csv", index=False)
    rel = reliance(eff, args.results_dir, stats["reference"])
    rel.to_csv(out / "reliance_reference.csv", index=False)

    pd.set_option("display.width", 200)
    print(eff.pivot_table(index=["pipeline", "stat"], columns="method", values="d").round(2))
    print(sp.round(3).to_string(index=False))
    print(rel.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
