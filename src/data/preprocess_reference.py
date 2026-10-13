"""Reference-pipeline face extraction for FaceForensics++ (Phase 1).

Standard-benchmark reference pipeline, as distinct from `preprocess.py`'s
MTCNN-based pipeline (kept untouched for comparison):
  - dlib frontal (HOG) face detector + 68-point landmark predictor
  - alignment: rotate so the eye line is horizontal
  - 1.3x margin around the aligned landmark bounding box
  - 256x256 output
  - 32 frames per video
  - PNG storage (lossless; tests whether JPEG re-compression itself is a
    confound in the original pipeline's results)

Requires dlib's 68-point landmark model (download once, ~100MB):
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Decompress the .bz2 and point --predictor-path at the resulting .dat file.

Expected FF++ layout (same as preprocess.py):
  <ffpp_root>/original_sequences/youtube/<comp>/videos/*.mp4
  <ffpp_root>/manipulated_sequences/<Method>/<comp>/videos/*.mp4

Output layout:
  <out_root>/<comp>/<real|Method>/<video_id>/frame_xxxxx.png

Optional scoping (for Phase 1's partial crop sets and for smoke tests):
  --filter-split {train,val,test}  restrict to videos whose id is in that
      official split (requires --splits-dir, default "splits")
  --limit N                        cap the number of videos per method
      (post count/filter), in video-id sorted order

Per-video timing is logged to --timing-log (CSV: video_id,method,n_frames_
sampled,n_frames_saved,seconds) so wall-clock can be extrapolated honestly
from a small sample without guessing.

Parallelization (added after the initial smoke test, which was single-
threaded): --workers N runs N worker processes via
multiprocessing.Pool, each with its OWN dlib detector + predictor instance
(dlib detector objects cannot be pickled/shared across a process
boundary, so each worker constructs them once in a pool initializer and
reuses them for every video that lands on that worker -- this is the
standard CPU-bound-multiprocessing pattern and does not change dlib's
per-frame detection behavior at all; each video is still processed exactly
as before, just more of them run concurrently on the 20 available cores).
--workers 1 (the default) preserves the original single-threaded code path
exactly (no Pool is created), so nothing changes for anyone relying on the
old behavior.

Usage (smoke test: 5 real + 5 Deepfakes + 5 FaceSwap videos from train split):
  python -m src.data.preprocess_reference \
      --ffpp-root C:/ffpp_data/ffpp_root \
      --out-root C:/ffpp_data/ffpp_faces_ref \
      --compressions c23 --methods real Deepfakes FaceSwap \
      --filter-split train --limit 5 --frames-per-video 32 \
      --timing-log experiments/phase1/smoke_preprocess_timing.csv

Usage (full Phase 1 scope, run per method group with the appropriate split
filter -- NOT launched by this script automatically, since the brief
requires an explicit go-ahead before any run over ~1 GPU/CPU-hour):
  # real, Deepfakes, FaceSwap: all videos (train+val+test -> no --filter-split)
  python -m src.data.preprocess_reference --methods real --compressions c23 ...
  python -m src.data.preprocess_reference --methods Deepfakes --compressions c23 ...
  python -m src.data.preprocess_reference --methods FaceSwap --compressions c23 ...
  # Face2Face, NeuralTextures: test split only
  python -m src.data.preprocess_reference --methods Face2Face --filter-split test ...
  python -m src.data.preprocess_reference --methods NeuralTextures --filter-split test ...
"""
import argparse
import csv
import multiprocessing
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .preprocess import METHODS, sample_frame_indices
from .splits import load_split_ids

DEFAULT_PREDICTOR_PATH = "C:/ffpp_data/tools/shape_predictor_68_face_landmarks.dat"

# dlib 68-point scheme: 36-41 right eye, 42-47 left eye (subject's right/left,
# i.e. the points at 36-41 sit on the LEFT side of a front-facing image and
# 42-47 on the right -- this matters for the alignment angle sign below).
RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX = list(range(42, 48))


def get_detector_and_predictor(predictor_path: str):
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


def largest_rect(rects):
    if not rects:
        return None
    return max(rects, key=lambda r: r.width() * r.height())


def align_and_crop(frame_bgr, landmarks_xy, out_size, margin):
    """landmarks_xy: (68, 2) float array in frame_bgr's pixel coordinates."""
    h, w = frame_bgr.shape[:2]
    left_eye = landmarks_xy[LEFT_EYE_IDX].mean(axis=0)
    right_eye = landmarks_xy[RIGHT_EYE_IDX].mean(axis=0)
    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    # right_eye sits left-of / right_eye.x < left_eye.x on a level, upright,
    # front-facing image, so the raw atan2 here is ~180 deg off upright;
    # correct it so angle==0 means "already level" (pyimagesearch convention).
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    eye_center = tuple(((left_eye + right_eye) / 2).tolist())

    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(frame_bgr, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

    ones = np.ones((landmarks_xy.shape[0], 1))
    pts_h = np.hstack([landmarks_xy, ones])
    pts_rot = (rot_mat @ pts_h.T).T  # (68, 2), landmark positions post-rotation

    x1, y1 = pts_rot.min(axis=0)
    x2, y2 = pts_rot.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * margin / 2
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def process_video(video_path: Path, out_dir: Path, detector, predictor,
                   frames_per_video: int, out_size: int, margin: float,
                   upsample: int):
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= frames_per_video // 2:
        return "skipped", 0, 0
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, frames_per_video)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector(rgb, upsample)
        rect = largest_rect(list(rects))
        if rect is None:
            continue
        shape = predictor(rgb, rect)
        pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)
        crop = align_and_crop(frame, pts, out_size, margin)
        if crop is None:
            continue
        cv2.imwrite(str(out_dir / f"frame_{idx:05d}.png"), crop)
        saved += 1
    cap.release()
    return f"{saved} frames", len(indices), saved


# Per-worker globals, set once by _init_worker (one call per pool process).
_worker_detector = None
_worker_predictor = None
_worker_upsample = None


def _init_worker(predictor_path: str, upsample: int):
    global _worker_detector, _worker_predictor, _worker_upsample
    _worker_detector, _worker_predictor = get_detector_and_predictor(predictor_path)
    _worker_upsample = upsample


def _process_job(job):
    """Top-level (picklable) per-video worker function for multiprocessing.Pool.
    Uses this process's own detector/predictor (set by _init_worker) -- never
    passes dlib objects across the process boundary."""
    method, vp, out_dir, comp, frames_per_video, out_size, margin = job
    t0 = time.time()
    try:
        status, n_sampled, n_saved = process_video(
            vp, out_dir, _worker_detector, _worker_predictor, frames_per_video,
            out_size, margin, _worker_upsample)
    except Exception as e:  # keep going on corrupt videos
        status, n_sampled, n_saved = f"error: {e}", 0, 0
        print(f"[error] {vp}: {e}")
    elapsed = time.time() - t0
    return {
        "video_id": vp.stem, "method": method, "compression": comp,
        "status": status, "n_frames_sampled": n_sampled,
        "n_frames_saved": n_saved, "seconds": round(elapsed, 3),
    }


def iter_jobs(ffpp_root: Path, comp: str, methods, splits_dir: str, filter_split: str,
              limit: int):
    real_ids, fake_ids = (None, None)
    if filter_split:
        real_ids, fake_ids = load_split_ids(splits_dir, filter_split)

    for method in methods:
        if method == "real":
            vdir = ffpp_root / "original_sequences" / "youtube" / comp / "videos"
        else:
            vdir = ffpp_root / "manipulated_sequences" / method / comp / "videos"
        if not vdir.exists():
            print(f"[warn] missing: {vdir}")
            continue
        vpaths = sorted(vdir.glob("*.mp4"))
        if filter_split:
            allowed = real_ids if method == "real" else fake_ids
            vpaths = [vp for vp in vpaths if vp.stem in allowed]
        if limit is not None:
            vpaths = vpaths[:limit]
        for vp in vpaths:
            yield method, vp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ffpp-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--compressions", nargs="+", default=["c23"],
                   choices=["c0", "c23", "c40", "c40proxy"])
    p.add_argument("--methods", nargs="+", default=METHODS + ["real"],
                   help="Subset of methods plus 'real' for originals")
    p.add_argument("--frames-per-video", type=int, default=32)
    p.add_argument("--out-size", type=int, default=256)
    p.add_argument("--margin", type=float, default=1.3)
    p.add_argument("--upsample", type=int, default=1,
                   help="dlib detector upsample factor; higher finds smaller "
                        "faces but is slower")
    p.add_argument("--predictor-path", default=DEFAULT_PREDICTOR_PATH,
                   help="Path to dlib's shape_predictor_68_face_landmarks.dat")
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--filter-split", default=None, choices=["train", "val", "test"],
                   help="Restrict to videos in this official split only "
                        "(e.g. Face2Face/NeuralTextures test-split-only scoping)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap videos processed per method (smoke testing)")
    p.add_argument("--timing-log", default=None,
                   help="CSV path to append per-video timing rows to")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (multiprocessing.Pool). "
                        "1 (default) = original single-threaded path, no Pool "
                        "created. Each worker builds its own dlib "
                        "detector+predictor (see module docstring).")
    args = p.parse_args()

    if not Path(args.predictor_path).exists():
        raise FileNotFoundError(
            f"dlib landmark predictor not found at {args.predictor_path}. "
            "Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
            "and decompress it, or pass --predictor-path.")
    ffpp = Path(args.ffpp_root)

    timing_rows = []
    for comp in args.compressions:
        jobs = [(method, vp) for method, vp in
                 iter_jobs(ffpp, comp, args.methods, args.splits_dir,
                           args.filter_split, args.limit)]
        print(f"compression={comp}: {len(jobs)} videos to process "
              f"(dlib, CPU, workers={args.workers})")

        if args.workers <= 1:
            detector, predictor = get_detector_and_predictor(args.predictor_path)
            for method, vp in tqdm(jobs, desc=comp):
                out_dir = Path(args.out_root) / comp / method / vp.stem
                t0 = time.time()
                try:
                    status, n_sampled, n_saved = process_video(
                        vp, out_dir, detector, predictor, args.frames_per_video,
                        args.out_size, args.margin, args.upsample)
                except Exception as e:  # keep going on corrupt videos
                    print(f"[error] {vp}: {e}")
                    status, n_sampled, n_saved = f"error: {e}", 0, 0
                elapsed = time.time() - t0
                timing_rows.append({
                    "video_id": vp.stem, "method": method, "compression": comp,
                    "status": status, "n_frames_sampled": n_sampled,
                    "n_frames_saved": n_saved, "seconds": round(elapsed, 3),
                })
        else:
            pool_jobs = [
                (method, vp, Path(args.out_root) / comp / method / vp.stem, comp,
                 args.frames_per_video, args.out_size, args.margin)
                for method, vp in jobs
            ]
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                    processes=args.workers, initializer=_init_worker,
                    initargs=(args.predictor_path, args.upsample)) as pool:
                for row in tqdm(pool.imap_unordered(_process_job, pool_jobs),
                                 total=len(pool_jobs), desc=comp):
                    timing_rows.append(row)

    if args.timing_log:
        log_path = Path(args.timing_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(timing_rows)
        print(f"wrote timing log ({len(timing_rows)} rows) -> {log_path}")


if __name__ == "__main__":
    main()
