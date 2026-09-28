"""Phase 0 item 3 (continued): statistical comparison of the encoding audit.

Reads results/audit/encoding_audit.csv (written by encoding_audit.py) and,
for each numeric field (width, height, fps, bit_rate) and each manipulation
method, runs a two-sample Kolmogorov-Smirnov test and a Mann-Whitney U test
against the "real" sample from the same random draw. Flags a field as a
systematic mismatch when both tests reject equality at alpha=0.05 AND the
effect direction is consistent (median shift), rather than on a single
p-value, since with n=60 per class a lone borderline test is not strong
evidence of a confound.

Usage:
    python -m experiments.audit.encoding_audit_summary \
        --in results/audit/encoding_audit.csv \
        --out results/audit/encoding_audit_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

FIELDS = ["width", "height", "fps", "bit_rate"]
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/audit/encoding_audit.csv")
    p.add_argument("--out", default="results/audit/encoding_audit_summary.csv")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    real = df[df["class"] == "real"]

    rows = []
    # Descriptive stats per class per field, plus crf and categorical fields.
    for cls in ["real"] + METHODS:
        sub = df[df["class"] == cls]
        for field in FIELDS:
            vals = sub[field].dropna().to_numpy()
            rows.append({
                "class": cls, "field": field, "n": len(vals),
                "mean": np.mean(vals) if len(vals) else np.nan,
                "median": np.median(vals) if len(vals) else np.nan,
                "std": np.std(vals) if len(vals) else np.nan,
                "min": np.min(vals) if len(vals) else np.nan,
                "max": np.max(vals) if len(vals) else np.nan,
                "vs_real_ks_stat": np.nan, "vs_real_ks_p": np.nan,
                "vs_real_mwu_p": np.nan, "median_shift_vs_real": np.nan,
                "flag_systematic_mismatch": "",
            })
        rows.append({
            "class": cls, "field": "x264_crf",
            "n": sub["x264_crf"].notna().sum(),
            "mean": sub["x264_crf"].mean(), "median": sub["x264_crf"].median(),
            "std": sub["x264_crf"].std(), "min": sub["x264_crf"].min(),
            "max": sub["x264_crf"].max(),
            "vs_real_ks_stat": np.nan, "vs_real_ks_p": np.nan,
            "vs_real_mwu_p": np.nan, "median_shift_vs_real": np.nan,
            "flag_systematic_mismatch": "",
        })
        rows.append({
            "class": cls, "field": "pix_fmt",
            "n": len(sub),
            "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan,
            "max": np.nan, "vs_real_ks_stat": np.nan, "vs_real_ks_p": np.nan,
            "vs_real_mwu_p": np.nan, "median_shift_vs_real": np.nan,
            "flag_systematic_mismatch":
                ",".join(f"{k}:{v}" for k, v in
                         sub["pix_fmt"].value_counts().items()),
        })
        rows.append({
            "class": cls, "field": "profile",
            "n": len(sub),
            "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan,
            "max": np.nan, "vs_real_ks_stat": np.nan, "vs_real_ks_p": np.nan,
            "vs_real_mwu_p": np.nan, "median_shift_vs_real": np.nan,
            "flag_systematic_mismatch":
                ",".join(f"{k}:{v}" for k, v in
                         sub["profile"].value_counts().items()),
        })

    # Statistical comparisons: each method vs real, per numeric field.
    comparisons = []
    for method in METHODS:
        sub = df[df["class"] == method]
        for field in FIELDS:
            r = real[field].dropna().to_numpy()
            m = sub[field].dropna().to_numpy()
            if len(r) < 2 or len(m) < 2:
                continue
            ks_stat, ks_p = ks_2samp(r, m)
            mwu_stat, mwu_p = mannwhitneyu(r, m, alternative="two-sided")
            shift = float(np.median(m) - np.median(r))
            flag = (ks_p < 0.05) and (mwu_p < 0.05)
            comparisons.append({
                "method": method, "field": field,
                "real_median": float(np.median(r)),
                "method_median": float(np.median(m)),
                "median_shift": shift,
                "real_mean": float(np.mean(r)),
                "method_mean": float(np.mean(m)),
                "ks_stat": float(ks_stat), "ks_p": float(ks_p),
                "mwu_p": float(mwu_p),
                "flag_systematic_mismatch": "YES" if flag else "no",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    comp_out = out.with_name(out.stem + "_tests.csv")
    pd.DataFrame(comparisons).to_csv(comp_out, index=False)

    print(f"wrote {out} ({len(rows)} rows)")
    print(f"wrote {comp_out} ({len(comparisons)} rows)")
    print()
    for c in comparisons:
        print(f"[{c['flag_systematic_mismatch']:>3}] real vs {c['method']:15s} "
              f"{c['field']:10s} real_median={c['real_median']:.1f} "
              f"{c['method']}_median={c['method_median']:.1f} "
              f"shift={c['median_shift']:+.1f} ks_p={c['ks_p']:.4f} "
              f"mwu_p={c['mwu_p']:.4f}")


if __name__ == "__main__":
    main()
