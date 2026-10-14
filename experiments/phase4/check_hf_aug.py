"""Does RandomBlurSharpen remove the high-frequency cue Phase 3 found?

Samples one crop per train-split video (real, Deepfakes, FaceSwap), resizes to
the model input size, and compares Cohen's d of hf_ratio (fake vs. real) with
and without the augmentation. Writes analysis/phase4/hf_aug_check.csv.

Usage: python -m experiments.phase4.check_hf_aug
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.phase3.crop_statistics import hf_ratio
from experiments.phase3.sign_prediction import cohens_d
from src.data.dataset import RandomBlurSharpen, _list_frames
from src.data.splits import load_split_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-root", default="C:/ffpp_data/ffpp_faces_ref/c23")
    p.add_argument("--img-size", type=int, default=299)
    p.add_argument("--videos-per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="analysis/phase4/hf_aug_check.csv")
    args = p.parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    real_ids, fake_ids = load_split_ids("splits", "train")
    aug = RandomBlurSharpen()
    vals = {}
    for cls, ids in [("real", real_ids), ("Deepfakes", fake_ids), ("FaceSwap", fake_ids)]:
        vids = sorted(v for v in ids if (Path(args.crop_root) / cls / v).is_dir())
        vids = rng.choice(vids, min(args.videos_per_class, len(vids)), replace=False)
        plain, augmented = [], []
        for v in vids:
            frames = _list_frames(Path(args.crop_root) / cls / v)
            img = Image.open(frames[len(frames) // 2]).convert("RGB").resize((args.img_size,) * 2)
            plain.append(hf_ratio(np.asarray(img.convert("L"))))
            augmented.append(hf_ratio(np.asarray(aug(img).convert("L"))))
        vals[cls] = (np.array(plain), np.array(augmented))

    rows = []
    for m in ["Deepfakes", "FaceSwap"]:
        for i, cond in enumerate(["no augmentation", "RandomBlurSharpen"]):
            rows.append({"method": m, "condition": cond,
                         "d_hf_ratio_vs_real": round(cohens_d(vals[m][i], vals["real"][i]), 3),
                         "n_videos": len(vals[m][i])})
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
