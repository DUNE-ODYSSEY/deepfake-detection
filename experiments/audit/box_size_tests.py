"""Statistical comparison for the crop audit's raw face-box-size sample.

Reads results/audit/crop_audit_box_sizes.csv (written by crop_audit.py) and
runs the same real-vs-method KS + Mann-Whitney U comparison used in
encoding_audit_summary.py, on box_w, box_h, box_area, det_prob.

Usage:
    python -m experiments.audit.box_size_tests \
        --in results/audit/crop_audit_box_sizes.csv \
        --out results/audit/crop_audit_box_size_tests.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

FIELDS = ["box_w", "box_h", "box_area", "det_prob"]
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/audit/crop_audit_box_sizes.csv")
    p.add_argument("--out", default="results/audit/crop_audit_box_size_tests.csv")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    real = df[df["class"] == "real"]

    rows = []
    for method in METHODS:
        sub = df[df["class"] == method]
        for field in FIELDS:
            r = real[field].dropna().to_numpy()
            m = sub[field].dropna().to_numpy()
            if len(r) < 2 or len(m) < 2:
                continue
            ks_stat, ks_p = ks_2samp(r, m)
            mwu_stat, mwu_p = mannwhitneyu(r, m, alternative="two-sided")
            flag = (ks_p < 0.05) and (mwu_p < 0.05)
            rows.append({
                "method": method, "field": field,
                "n_real": len(r), "n_method": len(m),
                "real_median": float(np.median(r)),
                "method_median": float(np.median(m)),
                "median_shift": float(np.median(m) - np.median(r)),
                "real_mean": float(np.mean(r)), "method_mean": float(np.mean(m)),
                "ks_p": float(ks_p), "mwu_p": float(mwu_p),
                "flag_systematic_mismatch": "YES" if flag else "no",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print(f"[{r['flag_systematic_mismatch']:>3}] real vs {r['method']:15s} "
              f"{r['field']:10s} real_median={r['real_median']:.1f} "
              f"method_median={r['method_median']:.1f} "
              f"shift={r['median_shift']:+.1f} ks_p={r['ks_p']:.4f} "
              f"mwu_p={r['mwu_p']:.4f}")


if __name__ == "__main__":
    main()
