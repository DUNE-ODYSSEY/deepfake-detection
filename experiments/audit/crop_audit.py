"""Phase 0 item 4: crop audit.

Three parts, run against the actual preprocessed crops in
C:/ffpp_data/ffpp_faces and (for the box-size part only) the raw videos in
C:/ffpp_data/ffpp_root:

1. Frames kept per video / MTCNN failure rate (exhaustive, all 5000 video
   dirs at c23 -- real + 4 methods x 1000 videos each). src/data/preprocess.py
   does not log per-video detection failures anywhere (checked
   logs/preprocess_c23.log: zero "[error]"/"[warn]" lines), and saved crop
   filenames carry no bbox/detection metadata, so the only way to recover the
   failure rate is: preprocess.py requests up to 20 frame indices per video
   (src/data/preprocess.py sample_frame_indices) and only writes a jpg when
   MTCNN finds a face on that sampled frame (process_video). So
   frames_kept = count of frame_*.jpg actually on disk, and
   failure_rate = 1 - frames_kept / frames_requested (20, or fewer if the
   video decoded fewer frames than that -- we do not have the raw
   frames_requested per video without opening the video, so we report against
   the fixed target of 20 as the brief specifies "20 frames/video").

2. Raw face-box size before resize (recovered by re-running MTCNN once per
   sampled video, at CPU, on a small random sample -- NOT the full ~100k
   crops/5000 videos, per the task's instruction to sample rather than run
   GPU-heavy re-detection).

3. JPEG quantization tables of saved crops, read cheaply from JPEG headers
   via PIL (Image.quantization) on a random sample of saved crops per class.

Usage:
    python -m experiments.audit.crop_audit \
        --faces-root C:/ffpp_data/ffpp_faces --ffpp-root C:/ffpp_data/ffpp_root \
        --compression c23 --box-sample-per-class 40 --qtable-sample-per-class 300 \
        --seed 42 --out-dir results/audit
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
CLASSES = ["real"] + METHODS
FRAMES_PER_VIDEO_TARGET = 20


def raw_video_dir(ffpp_root: Path, cls: str, compression: str) -> Path:
    if cls == "real":
        return ffpp_root / "original_sequences" / "youtube" / compression / "videos"
    return ffpp_root / "manipulated_sequences" / cls / compression / "videos"


def audit_frame_counts(faces_root: Path, compression: str, out_csv: Path):
    rows = []
    for cls in CLASSES:
        cdir = faces_root / compression / cls
        if not cdir.exists():
            print(f"[warn] missing {cdir}")
            continue
        for vdir in sorted(cdir.iterdir()):
            if not vdir.is_dir():
                continue
            n = len(list(vdir.glob("*.jpg")))
            rows.append({"class": cls, "video": vdir.name, "frames_kept": n})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "video", "frames_kept"])
        w.writeheader()
        w.writerows(rows)

    summary = []
    for cls in CLASSES:
        vals = np.array([r["frames_kept"] for r in rows if r["class"] == cls])
        if len(vals) == 0:
            continue
        failure_rate = 1.0 - vals.sum() / (FRAMES_PER_VIDEO_TARGET * len(vals))
        summary.append({
            "class": cls,
            "n_videos": len(vals),
            "mean_frames_kept": float(np.mean(vals)),
            "median_frames_kept": float(np.median(vals)),
            "min_frames_kept": int(np.min(vals)),
            "n_videos_below_target": int(np.sum(vals < FRAMES_PER_VIDEO_TARGET)),
            "pct_videos_below_target": float(
                np.mean(vals < FRAMES_PER_VIDEO_TARGET) * 100),
            "frame_level_failure_rate_pct": float(failure_rate * 100),
        })
    return rows, summary


def audit_box_sizes(ffpp_root: Path, compression: str, n_per_class: int,
                     seed: int, device: str):
    import cv2
    import torch
    from facenet_pytorch import MTCNN

    detector = MTCNN(select_largest=True, post_process=False, device=device)
    rng = random.Random(seed)
    rows = []
    for cls in CLASSES:
        vdir = raw_video_dir(ffpp_root, cls, compression)
        videos = sorted(vdir.glob("*.mp4"))
        if not videos:
            print(f"[warn] no videos in {vdir}")
            continue
        sample = rng.sample(videos, min(n_per_class, len(videos)))
        n_ok = 0
        for vp in sample:
            cap = cv2.VideoCapture(str(vp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue
            mid = total // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = detector.detect(rgb)
            if boxes is None or len(boxes) == 0:
                rows.append({"class": cls, "video": vp.stem,
                             "detected": False, "box_w": None, "box_h": None,
                             "box_area": None, "frame_w": frame.shape[1],
                             "frame_h": frame.shape[0], "det_prob": None})
                continue
            x1, y1, x2, y2 = boxes[0]
            w, h = float(x2 - x1), float(y2 - y1)
            rows.append({"class": cls, "video": vp.stem, "detected": True,
                         "box_w": w, "box_h": h, "box_area": w * h,
                         "frame_w": frame.shape[1], "frame_h": frame.shape[0],
                         "det_prob": float(probs[0]) if probs is not None else None})
            n_ok += 1
        print(f"box-size sample: {cls}: {n_ok}/{len(sample)} detected a face "
              f"on the sampled frame")
    return rows


def audit_jpeg_qtables(faces_root: Path, compression: str, n_per_class: int,
                        seed: int):
    from PIL import Image

    rng = random.Random(seed)
    rows = []
    for cls in CLASSES:
        cdir = faces_root / compression / cls
        if not cdir.exists():
            continue
        all_jpgs = []
        for vdir in cdir.iterdir():
            if vdir.is_dir():
                all_jpgs.extend(vdir.glob("*.jpg"))
        if not all_jpgs:
            continue
        sample = rng.sample(all_jpgs, min(n_per_class, len(all_jpgs)))
        for jp in sample:
            try:
                with Image.open(jp) as im:
                    im.load()
                    qtables = im.quantization  # dict {table_id: [64 ints]}
                    size = im.size
                    # Summarize each table by its DC coefficient (index 0,
                    # the standard "quality proxy") and mean of the table.
                    qsummary = {
                        str(k): {"dc": v[0], "mean": float(np.mean(v))}
                        for k, v in (qtables or {}).items()
                    }
            except Exception as e:
                rows.append({"class": cls, "video": jp.parent.name,
                             "frame": jp.name, "error": str(e)})
                continue
            rows.append({
                "class": cls, "video": jp.parent.name, "frame": jp.name,
                "width": size[0], "height": size[1],
                "n_qtables": len(qtables) if qtables else 0,
                "qtables_json": json.dumps(qsummary),
                "luma_dc": qsummary.get("0", {}).get("dc"),
                "luma_mean": qsummary.get("0", {}).get("mean"),
                "chroma_dc": qsummary.get("1", {}).get("dc"),
                "chroma_mean": qsummary.get("1", {}).get("mean"),
            })
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted(set(k for r in rows for k in r.keys()))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--faces-root", default="C:/ffpp_data/ffpp_faces")
    p.add_argument("--ffpp-root", default="C:/ffpp_data/ffpp_root")
    p.add_argument("--compression", default="c23")
    p.add_argument("--box-sample-per-class", type=int, default=40)
    p.add_argument("--qtable-sample-per-class", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                    help="MTCNN device for the box-size re-detection sample; "
                         "cpu is plenty for a few hundred frames")
    p.add_argument("--out-dir", default="results/audit")
    args = p.parse_args()

    faces_root = Path(args.faces_root)
    ffpp_root = Path(args.ffpp_root)
    out_dir = Path(args.out_dir)

    print("=== 1/3: frame-count / MTCNN-failure-rate audit (exhaustive) ===")
    frame_rows, frame_summary = audit_frame_counts(
        faces_root, args.compression, out_dir / "crop_audit_frame_counts.csv")
    write_csv(frame_summary, out_dir / "crop_audit_frame_summary.csv")
    for s in frame_summary:
        print(s)

    print("\n=== 2/3: raw face-box size before resize (sampled) ===")
    box_rows = audit_box_sizes(ffpp_root, args.compression,
                                args.box_sample_per_class, args.seed, args.device)
    write_csv(box_rows, out_dir / "crop_audit_box_sizes.csv")

    print("\n=== 3/3: JPEG quantization tables of saved crops (sampled) ===")
    qtable_rows = audit_jpeg_qtables(faces_root, args.compression,
                                      args.qtable_sample_per_class, args.seed)
    write_csv(qtable_rows, out_dir / "crop_audit_jpeg_qtables.csv")

    print(f"\nwrote crop_audit_frame_counts.csv ({len(frame_rows)} rows), "
          f"crop_audit_frame_summary.csv ({len(frame_summary)} rows), "
          f"crop_audit_box_sizes.csv ({len(box_rows)} rows), "
          f"crop_audit_jpeg_qtables.csv ({len(qtable_rows)} rows) under {out_dir}")


if __name__ == "__main__":
    main()
