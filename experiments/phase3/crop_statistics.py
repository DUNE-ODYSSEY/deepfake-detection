"""Phase 3.1: low-level image statistics per face crop, aggregated per video.

For every test-split video of every class (real + 4 methods) in one crop set,
samples --frames-per-video evenly spaced crops and computes:

  lap_var            variance of the Laplacian (sharpness)
  hf_ratio           share of spectral energy above 0.25 cycles/px (Hann-windowed
                     grayscale FFT, DC and r < 0.02 excluded)
  noise_std          std of grayscale minus its 3x3 median filter
  blockiness         mean |horizontal+vertical diff| on the 8-px grid / off-grid
  color_dE           Lab distance between mean inner-face and mean border-ring
  lum_diff           L*(inner face) - L*(border ring), signed
  boundary_grad      mean Sobel magnitude in a +-3 px band on the face-hull edge
  boundary_grad_rel  boundary_grad / mean Sobel magnitude inside the face

The face region is the convex hull of dlib's 68 landmarks on the crop; when
dlib finds no face, the central box implied by the 1.3x crop margin is used
(counted in landmark_fallback_frac). Writes one row per video.

Usage:
  python -m experiments.phase3.crop_statistics --crop-root C:/ffpp_data/ffpp_faces/c23 \
      --pipeline current --out analysis/phase3/crop_stats_current.csv
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.splits import load_split_ids

CLASSES = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
STATS = ["lap_var", "hf_ratio", "noise_std", "blockiness", "color_dE", "lum_diff",
         "boundary_grad", "boundary_grad_rel"]
LANDMARK_MODEL = "C:/ffpp_data/tools/shape_predictor_68_face_landmarks.dat"
MARGIN = 1.3

_detector = _predictor = None


def _init_worker():
    global _detector, _predictor
    import dlib
    cv2.setNumThreads(1)
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(LANDMARK_MODEL)


def landmarks(gray):
    import dlib
    h, w = gray.shape
    dets = _detector(gray, 0)
    if dets:
        rect, fallback = max(dets, key=lambda r: r.area()), False
    else:
        mx, my = int(w * (1 - 1 / MARGIN) / 2), int(h * (1 - 1 / MARGIN) / 2)
        rect, fallback = dlib.rectangle(mx, my, w - mx, h - my), True
    shape = _predictor(gray, rect)
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], np.int32)
    return np.clip(pts, 0, [w - 1, h - 1]), fallback


def hf_ratio(gray):
    g = gray.astype(np.float64)
    g -= g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    p = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(g.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(g.shape[1]))
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    band = r > 0.02
    return p[r > 0.25].sum() / p[band].sum()


def blockiness(gray):
    g = gray.astype(np.float64)
    dh, dv = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    on_h = (np.arange(dh.shape[1]) % 8) == 7
    on_v = (np.arange(dv.shape[0]) % 8) == 7
    on = dh[:, on_h].mean() + dv[on_v, :].mean()
    off = dh[:, ~on_h].mean() + dv[~on_v, :].mean()
    return on / off


def frame_stats(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out = {
        "lap_var": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "hf_ratio": hf_ratio(gray),
        "noise_std": (gray.astype(np.float64) - cv2.medianBlur(gray, 3)).std(),
        "blockiness": blockiness(gray),
    }
    pts, fallback = landmarks(gray)
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    k = max(3, int(0.04 * w)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(hull, kernel).astype(bool)
    ring = (cv2.dilate(hull, kernel, iterations=2) - hull).astype(bool)
    band = (cv2.dilate(hull, np.ones((7, 7), np.uint8)) - cv2.erode(hull, np.ones((7, 7), np.uint8))).astype(bool)

    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    if inner.sum() > 50 and ring.sum() > 50:
        mi, mr = lab[inner].mean(0), lab[ring].mean(0)
        out["color_dE"] = float(np.linalg.norm(mi - mr))
        out["lum_diff"] = float(mi[0] - mr[0])
    else:
        out["color_dE"] = out["lum_diff"] = np.nan
    gm = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    out["boundary_grad"] = gm[band].mean()
    out["boundary_grad_rel"] = gm[band].mean() / max(gm[inner].mean(), 1e-6) if inner.any() else np.nan
    out["fallback"] = fallback
    return out


def video_stats(job):
    pipeline, cls, vid, vdir, k = job
    frames = sorted(p for p in Path(vdir).iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if not frames:
        return None
    idx = np.unique(np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int))
    rows = [frame_stats(frames[i]) for i in idx]
    df = pd.DataFrame(rows)
    rec = {"pipeline": pipeline, "class": cls, "video": vid,
           "target_real": vid.split("_")[0], "n_frames": len(rows),
           "landmark_fallback_frac": float(df["fallback"].mean())}
    rec.update({s: float(df[s].mean()) for s in STATS})
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-root", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--split", default="test")
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    real_ids, fake_ids = load_split_ids(args.splits_dir, args.split)
    root = Path(args.crop_root)
    jobs = []
    for cls in CLASSES:
        ids = real_ids if cls == "real" else fake_ids
        for vid in sorted(ids):
            vdir = root / cls / vid
            if vdir.is_dir():
                jobs.append((args.pipeline, cls, vid, str(vdir), args.frames_per_video))
    print(f"{len(jobs)} videos, {args.frames_per_video} frames each, {args.workers} workers")

    with Pool(args.workers, initializer=_init_worker) as pool:
        recs = [r for r in pool.imap_unordered(video_stats, jobs, chunksize=4) if r]
    df = pd.DataFrame(recs).sort_values(["class", "video"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("class")[["n_frames", "landmark_fallback_frac"]].mean().round(3))
    print(f"wrote {out} ({len(df)} videos)")


if __name__ == "__main__":
    main()
