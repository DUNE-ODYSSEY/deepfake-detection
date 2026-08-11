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

## Status as of full-run session (2026-08-10 -> 2026-08-11)

### Preprocessing

- Kaggle mirror (`xdxd003/ff-c23`-style, `FaceForensics++_C23`) arranged into
  the official layout at `C:/ffpp_data/ffpp_root`; matched `preprocess.py`'s
  expectations exactly, no changes needed.
- Ran `src/data/preprocess.py` on all 5,000 videos (real + 4 methods), c23,
  20 frames/video -> ~99,987 face crops. Took ~5.5 hours of actual GPU time,
  though wall-clock was much longer (see "sleep problem" below).

### Official c40/c0 access never came through -> proxy-c40 workaround

Official FaceForensics++ approval was still pending, and no c40/c0 Kaggle
mirror exists. Rather than dropping the compression-robustness experiment
entirely, added `scripts/make_c40proxy.py`: re-transcodes just the test-split
videos (real + 4 methods, 700 videos) from c23 down to crf=40 with ffmpeg
(via `imageio-ffmpeg`'s bundled binary), mirroring the official layout under
a `c40proxy` label. **Caveat, not hidden in the results:** this is c23
re-compressed a second time, not raw compressed once like official c40, so
degradation is harsher/different from the real thing. Eval-only, never
trained on.

### Two bugs found and fixed

1. **`.gitignore`'s unanchored `data/` rule was matching `src/data/` too** —
   `preprocess.py`, `dataset.py`, `splits.py` had never actually been
   committed since the very first import, despite being core to the whole
   pipeline. Same issue silently excluded `splits/*.json`. Anchored to
   `/data/`, both recovered and committed.
2. **The laptop kept sleeping despite `powercfg standby-timeout-ac 0`** —
   went to sleep 3 separate times during the unattended run (confirmed via
   `Get-WinEvent` System log, ~2-2.5 hours dead time each), "Sleep Reason:
   System Idle," despite the timeout being correctly set to never. Likely
   Modern Standby (S0ix) on this Acer laptop has its own idle-sleep policy
   that ignores the classic `STANDBYIDLE` timer. Fixed with an app-level
   keep-awake (`scripts/keep_awake.ps1`, `SetThreadExecutionState` loop —
   the same mechanism video players use), which Windows honors even under
   Modern Standby. This is why the full run took ~14 hours of wall-clock
   despite only needing a few hours of actual GPU-busy time.

### Results (Experiments 1 & 2 full, Experiment 3 proxy)

Baseline (same-distribution), c23:

| model | video acc | video AUC |
|---|---|---|
| xception | 0.9729 | 0.9958 |
| cnn_vit | 0.9471 | 0.9836 |

Cross-manipulation generalization gap (same-manip AUC vs. cross-manip AUC):

| model | same-manip AUC | cross-manip AUC | gap |
|---|---|---|---|
| xception | 0.9947 | 0.5969 | 0.398 |
| cnn_vit | 0.9917 | 0.5710 | 0.421 |

Compression robustness (c23 -> c40proxy, caveated per above):

| model | c23 AUC | c40proxy AUC | drop |
|---|---|---|---|
| xception | 0.9958 | 0.8018 | 0.194 |
| cnn_vit | 0.9836 | 0.8172 | 0.166 |

**The original hypothesis (README: "the hybrid's off-diagonal AUC drops
less") did not hold.** Xception has both higher same-distribution accuracy
and a *smaller* generalization gap than the CNN-ViT hybrid in this run. The
hybrid does show a smaller compression-robustness drop, but that's on the
proxy-c40 set, not official. Worth reporting as a genuine (negative) result
on the main hypothesis, not glossed over — full breakdown in
`analysis/cross_manipulation_heatmaps.png` and the CSVs in `analysis/`.

## Next up

1. Live demo webapp (`webapp/`) exists but was built before training
   finished — point it at the now-trained checkpoints and verify end to end.
2. If official FF++ approval ever comes through, re-run Experiment 3 against
   real c0/c40 and compare against the proxy-c40 numbers above.
