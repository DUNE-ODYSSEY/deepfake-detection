"""Correlate the two architectures' off-diagonal cross-manipulation AUC cells.

Tests whether cross-manipulation transfer failure tracks the training
manipulation method (architecture-independent) rather than the architecture
itself. Reads the existing per-architecture 4x4 matrices in analysis/ and
writes analysis/architecture_correlation.json.

Usage: python -m analysis.architecture_correlation
"""
import json
from pathlib import Path

import pandas as pd
from scipy import stats

XCEPTION_CSV = "analysis/cross_manipulation_xception.csv"
CNN_VIT_CSV = "analysis/cross_manipulation_cnn_vit.csv"
OUT = "analysis/architecture_correlation.json"


def main():
    xc = pd.read_csv(XCEPTION_CSV, index_col=0)
    cv = pd.read_csv(CNN_VIT_CSV, index_col=0)
    methods = xc.index.tolist()

    off_xc, off_cv = [], []
    for r in methods:
        for c in methods:
            if r == c:
                continue
            off_xc.append(xc.loc[r, c])
            off_cv.append(cv.loc[r, c])

    pear = stats.pearsonr(off_xc, off_cv)
    spear = stats.spearmanr(off_xc, off_cv)
    out = {
        "n_cells": len(off_xc),
        "pearson_r": round(float(pear.statistic), 4),
        "pearson_p": float(pear.pvalue),
        "spearman_r": round(float(spear.statistic), 4),
        "spearman_p": float(spear.pvalue),
        "source_files": [XCEPTION_CSV, CNN_VIT_CSV],
    }
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
