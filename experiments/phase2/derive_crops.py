"""Phase 2: derive one-factor-at-a-time crop sets from the reference crops.

Every variant changes exactly one factor of the reference pipeline (dlib +
alignment, 1.3x margin, 256 px, 32 frames/video, PNG) and keeps the same
videos, frames and directory layout:

  jpeg95   re-encode each PNG crop as JPEG quality 95
  jpeg75   re-encode each PNG crop as JPEG quality 75
  margin10 keep the central 1/1.3 of each crop (the aligned landmark box, i.e.
           a 1.0x margin) and resize back to 256 px with bicubic, PNG
  frames10 10 evenly spaced frames out of the 32 (hard links, no new pixels)

Detector, alignment and data source are not derivable from existing crops;
Phase 1 already varied detector + alignment jointly against the original
pipeline.

Usage: python -m experiments.phase2.derive_crops --variant jpeg75
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("C:/ffpp_data/ffpp_faces_ref/c23")
MARGIN = 1.3


def out_root(variant):
    return Path(f"C:/ffpp_data/ffpp_faces_ref_{variant}/c23")


def _ok(path):
    """True if an earlier (possibly interrupted) run left a fully decodable file."""
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def do_video(job):
    variant, vdir = job
    vdir = Path(vdir)
    rel = vdir.relative_to(SRC)
    dst = out_root(variant) / rel
    dst.mkdir(parents=True, exist_ok=True)
    frames = sorted(vdir.glob("*.png"))
    if variant == "frames10":
        keep = np.unique(np.linspace(0, len(frames) - 1, min(10, len(frames))).round().astype(int))
        for i in keep:
            target = dst / frames[i].name
            if not target.exists():
                os.link(frames[i], target)
        return len(keep)
    for f in frames:
        img = Image.open(f).convert("RGB")
        if variant.startswith("jpeg"):
            target = dst / (f.stem + ".jpg")
            if not _ok(target):
                img.save(target, "JPEG", quality=int(variant[4:]))
        elif variant == "margin10":
            target = dst / f.name
            if not _ok(target):
                w, h = img.size
                cw, ch = round(w / MARGIN), round(h / MARGIN)
                l, t = (w - cw) // 2, (h - ch) // 2
                img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BICUBIC).save(target)
        else:
            raise ValueError(variant)
    return len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["jpeg95", "jpeg75", "margin10", "frames10"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    vdirs = sorted(d for cls in SRC.iterdir() for d in cls.iterdir() if d.is_dir())
    with Pool(args.workers) as pool:
        n = sum(pool.imap_unordered(do_video, [(args.variant, str(d)) for d in vdirs], chunksize=8))
    (out_root(args.variant) / ".complete").write_text(f"{len(vdirs)} videos, {n} frames\n")
    print(f"{args.variant}: {len(vdirs)} videos, {n} frames -> {out_root(args.variant)}", flush=True)


if __name__ == "__main__":
    main()
