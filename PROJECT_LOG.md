# Project log

Running notes on execution status, separate from the README (which documents
how to run things). Update this as the project progresses.

## Status as of setup session

Code (src/, analyze/, scripts/, README.md) was written first; this log starts
at the point of actually getting it running.

### Environment

- System Python was 3.13, too new for `facenet-pytorch`'s pinned
  `torch<2.3,torchvision<0.18` (predates Python 3.13 wheel support) and for
  `numpy<2.0` (no prebuilt wheel, and no C compiler on this machine to build
  from source).
- Fix: installed Python 3.11 side by side (via winget), rebuilt `.venv` on it.
- `pip install torch` alone gives a **CPU-only** build on Windows. Had to
  install from PyTorch's CUDA index explicitly:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- Confirmed working: torch 2.6.0+cu124, CUDA available, GPU = RTX 3060 Laptop
  (6GB VRAM).

### GPU capacity (RTX 3060 Laptop, 6GB)

Timed forward+backward+optimizer-step at increasing batch sizes for both
models (random tensors, matching train.py's AMP loop):

| batch | xception ms/it | xception img/s | cnn_vit ms/it | cnn_vit img/s |
|---|---|---|---|---|
| 8  | 65.5   | 122.1 | 55.1  | 145.1 |
| 16 | 115.7  | 138.2 | 114.0 | 140.4 |
| 24 | 165.3  | 145.2 | 170.7 | 140.6 |
| 32 | 217.1  | 147.4 | 204.8 | 156.2 |
| 48 | 2545.7 | 18.9  | 349.3 | 137.4 |
| 64 | 10802.1| 5.9   | 446.0 | 143.5 |

**Finding:** Xception collapses in throughput above batch 32 — not an OOM
crash, but Windows silently paging GPU memory into system RAM once
allocation exceeds the card's real 6.4GB (peak_alloc goes to 6.6GB+ at
batch 48). It still "runs," just 10-25x slower. cnn_vit stays under the
physical VRAM ceiling even at batch 64, so it doesn't hit this.

**Action:** keep `--batch-size 32` (already `train.py`'s default) for both
models. Do not raise it for xception. Full 12-run/40-eval matrix at this
batch size is estimated at roughly 15-25 GPU-hours total (not the
"several GPU-days" the README's generic estimate assumed), well within a
3-week window — no need to cut the experiment matrix down.

### Dataset

- Official FF++ access form (github.com/ondyari/FaceForensics) submitted;
  approval typically takes a few days.
- Bridging with Kaggle mirrors for c23 in the meantime:
  `xdxd003/ff-c23`, `fatimahirshad/faceforensics-extracted-dataset-c23`.
  No credible c40 mirror found on Kaggle — c40 (needed only for the
  compression-robustness experiment) waits on official approval.
- Folder layout and whether a given Kaggle mirror is raw video vs.
  pre-extracted frames needs manual verification against what
  `src/data/preprocess.py` expects before running preprocessing.
- `train.json`/`val.json`/`test.json` (official video-level splits) go in
  `splits/` — required, since random frame-level splits leak identity
  between train/test and inflate accuracy.

### Backup plan

`notebooks/colab_backup.ipynb` mirrors the local train/evaluate commands
against Drive paths, in case local training stalls. Preprocessing should
still happen locally/once — never re-run MTCNN extraction per Colab session.

## Next up

1. Verify downloaded Kaggle data's folder layout against `preprocess.py`'s
   expectations.
2. Run `src/data/preprocess.py` on a handful of videos first as a dry run.
3. Then the full experiment matrix via `scripts/run_all_experiments.sh`.
