"""Extract face crops from FaceForensics++ videos.

Expected FF++ layout (as downloaded by the official script):
  <ffpp_root>/original_sequences/youtube/<comp>/videos/*.mp4
  <ffpp_root>/manipulated_sequences/<Method>/<comp>/videos/*.mp4
where <comp> is c0 (raw), c23, or c40 and <Method> is one of
Deepfakes, Face2Face, FaceSwap, NeuralTextures.

Output layout:
  <out_root>/<comp>/<real|Method>/<video_id>/frame_0001.jpg ...

Usage:
  python -m src.data.preprocess --ffpp-root /data/FFpp --out-root /data/ffpp_faces \
      --compressions c23 c40 --frames-per-video 20
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
MARGIN_SCALE = 1.3  # enlarge face bbox to include context artifacts (blending edges)


def get_detector(device):
    from facenet_pytorch import MTCNN
    return MTCNN(select_largest=True, post_process=False, device=device)


def sample_frame_indices(total, n):
    if total <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, min(n, total), dtype=int).tolist()))


def crop_face(frame_bgr, box, out_size):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * MARGIN_SCALE / 2
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def process_video(video_path: Path, out_dir: Path, detector, frames_per_video: int,
                  out_size: int):
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= frames_per_video // 2:
        return "skipped"
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
        boxes, _ = detector.detect(rgb)
        if boxes is None or len(boxes) == 0:
            continue
        crop = crop_face(frame, boxes[0], out_size)
        if crop is None:
            continue
        cv2.imwrite(str(out_dir / f"frame_{idx:05d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    return f"{saved} frames"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ffpp-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--compressions", nargs="+", default=["c23"],
                   choices=["c0", "c23", "c40", "c40proxy"],
                   help="c40proxy is a locally re-transcoded (c23->crf40) stand-in "
                        "for official c40, used when official access is pending")
    p.add_argument("--methods", nargs="+", default=METHODS + ["real"],
                   help="Subset of methods plus 'real' for originals")
    p.add_argument("--frames-per-video", type=int, default=20)
    p.add_argument("--out-size", type=int, default=299,
                   help="Crop size; 299 fits Xception, downscaled to 224 for the hybrid")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = get_detector(device)
    ffpp = Path(args.ffpp_root)

    jobs = []
    for comp in args.compressions:
        for method in args.methods:
            if method == "real":
                vdir = ffpp / "original_sequences" / "youtube" / comp / "videos"
            else:
                vdir = ffpp / "manipulated_sequences" / method / comp / "videos"
            if not vdir.exists():
                print(f"[warn] missing: {vdir}")
                continue
            for vp in sorted(vdir.glob("*.mp4")):
                out_dir = Path(args.out_root) / comp / method / vp.stem
                jobs.append((vp, out_dir))

    print(f"{len(jobs)} videos to process on {device}")
    for vp, out_dir in tqdm(jobs):
        try:
            process_video(vp, out_dir, detector, args.frames_per_video, args.out_size)
        except Exception as e:  # keep going on corrupt videos
            print(f"[error] {vp}: {e}")


if __name__ == "__main__":
    main()
