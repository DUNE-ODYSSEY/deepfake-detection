"""Phase 2 x Phase 3: high-frequency effect size per method for every crop variant.

For the reference crops and each one-factor variant, computes hf_ratio (same
definition as Phase 3) on 8 evenly spaced frames of every test-split video and
reports Cohen's d of each method vs. real. Run before the variant models are
evaluated, so the predicted direction of change in each transfer cell is
fixed in advance. Writes analysis/phase2/hf_d_by_variant.csv.

Usage: python -m experiments.phase2.hf_by_variant
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.phase3.crop_statistics import hf_ratio
from experiments.phase3.sign_prediction import cohens_d
from src.data.splits import load_split_ids

VARIANTS = {"reference": "C:/ffpp_data/ffpp_faces_ref/c23",
            "jpeg95": "C:/ffpp_data/ffpp_faces_ref_jpeg95/c23",
            "jpeg75": "C:/ffpp_data/ffpp_faces_ref_jpeg75/c23",
            "frames10": "C:/ffpp_data/ffpp_faces_ref_frames10/c23",
            "margin10": "C:/ffpp_data/ffpp_faces_ref_margin10/c23"}
CLASSES = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
SHORT = {"Deepfakes": "DF", "Face2Face": "F2F", "FaceSwap": "FS", "NeuralTextures": "NT"}


def video_hf(job):
    variant, cls, vdir, k = job
    frames = sorted(p for p in Path(vdir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if not frames:
        return None
    idx = np.unique(np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int))
    vals = [hf_ratio(np.asarray(Image.open(frames[i]).convert("L"))) for i in idx]
    return variant, cls, float(np.mean(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default="analysis/phase2/hf_d_by_variant.csv")
    args = p.parse_args()
    real_ids, fake_ids = load_split_ids("splits", "test")
    jobs = []
    for variant, root in VARIANTS.items():
        for cls in CLASSES:
            ids = real_ids if cls == "real" else fake_ids
            for vid in sorted(ids):
                vdir = Path(root) / cls / vid
                if vdir.is_dir():
                    jobs.append((variant, cls, str(vdir), args.frames))
    with Pool(args.workers) as pool:
        recs = [r for r in pool.imap_unordered(video_hf, jobs, chunksize=8) if r]
    df = pd.DataFrame(recs, columns=["variant", "class", "hf_ratio"])
    rows = []
    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        real = sub[sub["class"] == "real"]["hf_ratio"].to_numpy()
        for m in CLASSES[1:]:
            f = sub[sub["class"] == m]["hf_ratio"].to_numpy()
            rows.append({"variant": variant, "method": SHORT[m], "d": round(cohens_d(f, real), 4),
                         "mean_fake": float(f.mean()), "mean_real": float(real.mean()),
                         "n_fake": len(f), "n_real": len(real)})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.pivot(index="variant", columns="method", values="d").round(3))


if __name__ == "__main__":
    main()
