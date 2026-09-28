"""Phase 0 item 3: encoding audit.

Runs ffprobe over a random sample of raw FF++ videos (real + each of the four
c23 manipulation methods) from C:/ffpp_data/ffpp_root and extracts codec,
profile, resolution, fps, bitrate, pixel format, and (via a raw-byte scan for
the x264 SEI text, since ffprobe does not surface it) the x264 `crf=` used at
encode time. Compares real vs each method's distributions and flags any
systematic mismatch -- a candidate confound for the cross-manipulation AUC
inversion described in RESEARCH_BRIEF.md.

Tooling note: no ffprobe/ffmpeg was on PATH and imageio-ffmpeg (already in
the venv) bundles ffmpeg but not ffprobe. Obtained a static Windows build
(gyan.dev "ffmpeg-release-essentials", ffmpeg 9.0.2, ~110MB, well under the
1GB ground-rule limit) and copied ffprobe.exe/ffmpeg.exe to
C:/ffpp_data/tools/ (outside the repo, so nothing large lands under git).
The x264 encoder-settings string (containing crf=) is not exposed by
ffprobe's -show_streams/-show_format on an mp4 container -- x264 writes it as
plain ASCII inside an H.264 SEI NAL unit, which this script recovers by
regex-searching the raw video bytes directly (no ffmpeg/ffprobe needed for
that part).

Usage:
    python -m experiments.audit.encoding_audit \
        --ffpp-root C:/ffpp_data/ffpp_root --ffprobe C:/ffpp_data/tools/ffprobe.exe \
        --n-per-class 60 --seed 42 --out results/audit/encoding_audit.csv
"""
import argparse
import csv
import json
import random
import re
import subprocess
from pathlib import Path

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
X264_RE = re.compile(rb"x264 - core[^\x00]{0,600}")
CRF_RE = re.compile(r"crf=([0-9.]+)")


def class_video_dir(ffpp_root: Path, cls: str, compression: str) -> Path:
    if cls == "real":
        return ffpp_root / "original_sequences" / "youtube" / compression / "videos"
    return ffpp_root / "manipulated_sequences" / cls / compression / "videos"


def extract_x264_options(video_path: Path):
    """Regex-scan raw bytes for the x264 SEI text; return (raw_string, crf) or (None, None)."""
    try:
        data = video_path.read_bytes()
    except Exception:
        return None, None
    m = X264_RE.search(data)
    if not m:
        return None, None
    text = m.group(0).decode("latin1", errors="replace")
    # Trim at first run of >=2 consecutive non-printable bytes (end of SEI text).
    cut = re.search(r"[\x00-\x08\x0b-\x1f\x7f-\xff]{2,}", text)
    if cut:
        text = text[:cut.start()]
    crf_m = CRF_RE.search(text)
    crf = float(crf_m.group(1)) if crf_m else None
    return text, crf


def probe_video(ffprobe: str, video_path: Path):
    cmd = [ffprobe, "-v", "error", "-show_format", "-show_streams",
           "-print_format", "json", str(video_path)]
    out = subprocess.run(cmd, capture_output=True, text=False, timeout=60)
    if out.returncode != 0:
        return None
    data = json.loads(out.stdout.decode("utf-8", errors="replace"))
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        return None
    s = streams[0]
    fmt = data.get("format", {})

    def fps_of(s):
        for key in ("avg_frame_rate", "r_frame_rate"):
            v = s.get(key)
            if v and v != "0/0":
                try:
                    num, den = v.split("/")
                    den = float(den)
                    return float(num) / den if den else None
                except Exception:
                    continue
        return None

    bit_rate = s.get("bit_rate") or fmt.get("bit_rate")
    return {
        "codec_name": s.get("codec_name"),
        "profile": s.get("profile"),
        "width": s.get("width"),
        "height": s.get("height"),
        "pix_fmt": s.get("pix_fmt"),
        "fps": fps_of(s),
        "bit_rate": int(bit_rate) if bit_rate is not None else None,
        "duration_s": float(fmt.get("duration")) if fmt.get("duration") else None,
        "nb_frames": int(s["nb_frames"]) if s.get("nb_frames") else None,
        "file_size_bytes": int(fmt.get("size")) if fmt.get("size") else None,
        "format_name": fmt.get("format_name"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ffpp-root", default="C:/ffpp_data/ffpp_root")
    p.add_argument("--ffprobe", default="C:/ffpp_data/tools/ffprobe.exe")
    p.add_argument("--compression", default="c23")
    p.add_argument("--n-per-class", type=int, default=60,
                    help="random sample size per class; brief requires >=50")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/audit/encoding_audit.csv")
    args = p.parse_args()

    ffpp_root = Path(args.ffpp_root)
    rng = random.Random(args.seed)
    classes = ["real"] + METHODS

    rows = []
    for cls in classes:
        vdir = class_video_dir(ffpp_root, cls, args.compression)
        videos = sorted(vdir.glob("*.mp4"))
        if not videos:
            print(f"[warn] no videos found in {vdir}")
            continue
        sample = rng.sample(videos, min(args.n_per_class, len(videos)))
        for vp in sample:
            info = probe_video(args.ffprobe, vp)
            if info is None:
                print(f"[warn] ffprobe failed on {vp}")
                continue
            x264_str, crf = extract_x264_options(vp)
            row = {
                "class": cls,
                "compression": args.compression,
                "video": vp.stem,
                "path": str(vp),
                **info,
                "x264_crf": crf,
                "x264_options_present": x264_str is not None,
            }
            rows.append(row)
        print(f"{cls}: probed {len(sample)}/{len(videos)} available")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
