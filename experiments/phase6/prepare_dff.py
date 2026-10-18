"""Phase 6: reference-pipeline face crops for a sample of DeepFakeFace (DFF).

DFF (OpenRL/DeepFakeFace on Hugging Face, OpenRAIL licence; real images from
IMDB-WIKI, research use only -- never redistribute) has 30k images per class:
wiki (real), inpainting (SD inpainting), insight (InsightFace swap), text2img
(SD v1.5). Samples --per-class images per class with a fixed seed, reads them
straight from the zips, and applies exactly the reference crop used for FF++
(dlib HOG, upsample 1, largest face, 68 landmarks, eye-line alignment, 1.3x
margin, 256 px, PNG). Writes <out>/<class>/<name>.png and a manifest CSV with
every sampled image and whether a face was found.

Usage: python -m experiments.phase6.prepare_dff --per-class 1000
"""
import argparse
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.preprocess_reference import (DEFAULT_PREDICTOR_PATH, align_and_crop,
                                           get_detector_and_predictor, largest_rect)

CLASSES = ["wiki", "inpainting", "insight", "text2img"]
_det = _pred = None


def _init():
    global _det, _pred
    cv2.setNumThreads(1)
    _det, _pred = get_detector_and_predictor(DEFAULT_PREDICTOR_PATH)


def _job(job):
    zpath, member, out_path = job
    with zipfile.ZipFile(zpath) as z:
        buf = np.frombuffer(z.read(member), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return member, False, "decode"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rect = largest_rect(list(_det(rgb, 1)))
    if rect is None:
        return member, False, "no_face"
    pts = np.array([[p.x, p.y] for p in _pred(rgb, rect).parts()], dtype=np.float64)
    crop = align_and_crop(img, pts, 256, 1.3)
    if crop is None:
        return member, False, "empty_crop"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return member, True, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", default="C:/ffpp_data/deepfakeface")
    p.add_argument("--out", default="C:/ffpp_data/dff_ref")
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    jobs, meta = [], []
    for cls in CLASSES:
        zpath = Path(args.zip_dir) / f"{cls}.zip"
        with zipfile.ZipFile(zpath) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
        pick = rng.choice(len(names), min(args.per_class, len(names)), replace=False)
        for i in sorted(pick):
            member = names[i]
            stem = member.replace("/", "__").rsplit(".", 1)[0]
            out_path = Path(args.out) / cls / f"{stem}.png"
            jobs.append((str(zpath), member, str(out_path)))
            meta.append((cls, member, str(out_path)))
        print(f"{cls}: {len(names)} images in zip, sampled {len(pick)}", flush=True)
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_job, jobs, chunksize=16)
    df = pd.DataFrame([{"class": c, "member": m, "crop": o, "face_found": ok, "reason": why}
                       for (c, m, o), (_, ok, why) in zip(meta, results)])
    df.to_csv(Path(args.out) / "manifest.csv", index=False)
    print(df.groupby("class")["face_found"].agg(["sum", "count", "mean"]).round(3))


if __name__ == "__main__":
    main()
