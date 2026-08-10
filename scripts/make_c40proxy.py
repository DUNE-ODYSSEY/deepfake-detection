"""Build a local proxy for FF++'s official c40 compression level.

Official c40 doesn't exist on the Kaggle mirror this project uses, and
FaceForensics++ access-form approval (which would unlock the real c40) is
still pending. As a stand-in for the compression-robustness experiment, this
re-transcodes the c23 videos actually needed for evaluation (the test-split
real + manipulated videos) at crf=40 with ffmpeg, mirroring the official
FF++ layout under a "c40proxy" compression folder.

Caveat (documented, not hidden): this is c23 re-compressed a second time,
not raw re-compressed once like official c40, so it's a harsher/different
degradation than the real thing. Only videos needed for the test split are
built (proxy-c40 is eval-only here, never trained on).

Usage:
  python -m scripts.make_c40proxy --ffpp-root C:/ffpp_data/ffpp_root \
      --splits-dir splits --out-root C:/ffpp_data/ffpp_root_proxy
"""
import argparse
import subprocess
import time
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

from src.data.preprocess import METHODS
from src.data.splits import load_split_ids

FFMPEG = get_ffmpeg_exe()


def transcode(src: Path, dst: Path):
    if dst.exists():
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-crf", "40", "-preset", "fast", "-an", str(dst)],
        check=True,
    )
    return "done"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ffpp-root", required=True)
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--out-root", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    ffpp = Path(args.ffpp_root)
    out = Path(args.out_root)
    real_ids, fake_ids = load_split_ids(args.splits_dir, args.split)

    jobs = []
    real_dir = ffpp / "original_sequences" / "youtube" / "c23" / "videos"
    for vid in sorted(real_ids):
        src = real_dir / f"{vid}.mp4"
        if src.exists():
            jobs.append((src, out / "original_sequences" / "youtube" / "c40proxy" / "videos" / src.name))

    for method in METHODS:
        mdir = ffpp / "manipulated_sequences" / method / "c23" / "videos"
        for vid in sorted(fake_ids):
            src = mdir / f"{vid}.mp4"
            if src.exists():
                jobs.append((src, out / "manipulated_sequences" / method / "c40proxy" / "videos" / src.name))

    print(f"{len(jobs)} videos to transcode to proxy-c40 (crf=40)")
    t0 = time.time()
    done = skipped = 0
    for i, (src, dst) in enumerate(jobs, 1):
        r = transcode(src, dst)
        done += r == "done"
        skipped += r == "skipped"
        if i % 50 == 0 or i == len(jobs):
            elapsed = time.time() - t0
            print(f"{i}/{len(jobs)} ({done} done, {skipped} skipped) "
                  f"[{elapsed:.0f}s elapsed]")

    print(f"finished: {done} transcoded, {skipped} already present -> {out}")


if __name__ == "__main__":
    main()
