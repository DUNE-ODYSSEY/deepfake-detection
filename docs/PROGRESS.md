# Progress log

Tracks execution against `RESEARCH_BRIEF.md`, phase by phase. Append, never rewrite past entries.

## Phase 0: Hygiene and audit

**Status: in progress.** Started 2026-09-29.

### Done so far (not yet committed to git)

1. Added `CLAUDE.md` pointing to `RESEARCH_BRIEF.md`.
2. Fixed the `--resume` bug in `src/train.py`: it unconditionally reset `history = []`
   on every run, so resuming a training run overwrote `history.json` and discarded all
   pre-resume epochs. Now loads existing `history.json` if present, and on `--resume`
   truncates any rows at/after the resume epoch (covers a crash mid-epoch after the
   history write but before the checkpoint write) before continuing to append.
3. Checked `--seed`: already present in `src/train.py` (default 42, applied via
   `torch.manual_seed`/`np.random.seed`) before this session started. The brief's
   "no seed argument" bug is stale — nothing to fix here. `src/evaluate.py` and
   `src/data/preprocess.py` have no seed argument, but eval has no stochastic ops
   (no shuffling, no augmentation) so this is not currently a correctness issue.
4. Checkpoints committed to git: confirmed `runs/xception_all_c23/best.pt` and
   `runs/cnn_vit_all_c23/best.pt` were tracked via `.gitignore` negation rules
   (`.git` was 196MB as a result). Per user decision: untracked going forward only
   (`git rm --cached`), history NOT rewritten (repo may have been cloned/forked
   already; rewriting would need a force-push). `.gitignore` simplified to ignore
   `runs/`, `results/`, `*.pt` outright, no exceptions.
5. Confirmed environment matches the brief's assumptions: GPU is RTX 3060 Laptop,
   6GB VRAM (`nvidia-smi`); FF++ data present locally at `C:/ffpp_data/ffpp_root`
   (Kaggle mirror `xdxd003/ff-c23`, arranged into official layout) and preprocessed
   crops at `C:/ffpp_data/ffpp_faces`.
6. **Data source confirmed with user: still Kaggle mirror only, official FF++ access
   has not come through.** All Phase 0/1 numbers below are mirror-derived and must
   be labeled as such per the brief's ground rule 5. Phase 2's mirror-vs-official arm
   and Phase 6 cross-dataset validity stay blocked until access arrives.

### Phase 0 items 2-5 (label/split, encoding, crop audits, GATE 0 report): done

See the "Phase 0 items 2-5: audit results" section below for the full writeup.

### Deviations from the brief

- None yet beyond the checkpoint-history decision above (untrack-only chosen over
  history rewrite, per explicit user instruction).

### Proposed next step

Phase 0 items 1-5 are complete and GATE 0 found no confirmed real-vs-method
confound (see below); per the brief's GATE 0 instruction ("continue regardless"),
proceed to Phase 1 (reference-pipeline replication: dlib + landmark alignment,
1.3x margin, 256x256, 32 frames/video, PNG storage) unless redirected.

## Phase 0 items 2-5: audit results

**Status: done.** 2026-09-29. Git commit at time of audit:
`4a180c0de8ef833788fa27da4ea0a92941cef74b` (working tree also carries the
uncommitted item-1 changes noted above; this audit did not touch git, per
instructions). Data provenance for every number below: **Kaggle mirror
(`xdxd003/ff-c23`), mirror/preliminary** — `C:/ffpp_data/ffpp_root` (raw
videos) and `C:/ffpp_data/ffpp_faces` (MTCNN crops).

New scripts under `experiments/audit/` (all outputs under `results/audit/`):
`label_split_audit.py`, `encoding_audit.py`, `encoding_audit_summary.py`,
`crop_audit.py`, `box_size_tests.py`. Full narrative in
`results/audit/SUMMARY.md`; this is the condensed version.

**1. Label/split audit — 17/17 checks PASS**
(`results/audit/label_split_audit.csv`, `label_split_audit_cells.csv`).
fake=1 confirmed in `src/data/dataset.py` (real->0, method->1),
`src/metrics.py` (`_safe_auc()` calls `roc_auc_score(labels, probs)`/`(vl,vp)`
directly, no post-hoc flip), and `src/train.py` (score = `sigmoid(logits)`
against the fake=1 target). A synthetic polarity check through the real
`compute_metrics()` gave AUC 1.0000 when fake frames scored higher and 0.0000
when they scored lower, confirming orientation is not silently invertible.
Train/val/test are video-disjoint via `src.data.splits.load_split_ids()`
(720/720, 140/140, 140/140 real/fake ids; all pairwise intersections empty).
Same-real-test-videos-per-cell verified operationally against all 36
`results/*.json` files: recomputed `real_count + fake_count` matches the
reported `n_videos` in every file (0 mismatches), and the recomputed real
count is a single constant (140) across every c23-test and c40proxy-test
cell — consistent with `FFPPDataset` loading `real/` independent of the
`methods` argument (`src/data/dataset.py:44-48`).

**2. Encoding audit — no confound.** 60 videos/class (real + 4 methods), c23,
seed 42 (`results/audit/encoding_audit.csv`, `..._summary.csv`,
`..._summary_tests.csv`). Tooling: no ffprobe was on PATH; downloaded the
static gyan.dev `ffmpeg-release-essentials` build (ffmpeg 9.0.2, ~110MB,
under the 1GB budget) to `C:/ffpp_data/tools/ffprobe.exe`/`ffmpeg.exe`
(outside the repo). The x264 `crf=` string isn't exposed by ffprobe on this
mp4 container, so it's recovered by regex-scanning raw video bytes for the
plain-ASCII x264 SEI text. Codec (h264), profile (High 4:4:4 Predictive),
pixel format (yuv444p), and crf (23.0, exact, zero variance) are identical
across all 300 sampled videos. Resolution/fps/bitrate: 0 of 16
Kolmogorov-Smirnov + Mann-Whitney U real-vs-method comparisons reach
significance at alpha=0.05 (all p >= 0.15).

**3. Crop audit — no confirmed confound; one soft signal.**
MTCNN failure rate (exhaustive over all 5000 c23 video dirs,
`crop_audit_frame_counts.csv`/`..._frame_summary.csv`): 0.00%-0.05% missing
frames out of the 20/video target across all 5 classes — negligible.
Raw face-box size before resize (sampled 40 videos/class, one re-detected
frame each via a fresh CPU MTCNN pass, 200/200 detected;
`crop_audit_box_sizes.csv`/`..._box_size_tests.csv`): real median box area
36569 px^2 vs method medians 21736-32382 (all four methods trend smaller),
but 0 of 16 KS/MWU tests significant at n=40 (all p >= 0.13), and the trend
does not single out FaceSwap over the other three methods. Flagged as a
low-confidence signal for a larger-sample re-check in Phase 1/3, not a
confirmed confound. JPEG quantization tables (sampled 300 crops/class,
`crop_audit_jpeg_qtables.csv`): bit-for-bit identical across all 5 classes
(zero variance) — expected, since `preprocess.py` re-saves every crop at a
fixed JPEG quality 95 regardless of source; not a confound.

**GATE 0 conclusion:** no confirmed systematic real-vs-method confound found
across label/split integrity, container encoding, MTCNN failure rate, or JPEG
re-quantization that would mechanically explain the FaceSwap-centred
below-chance cross-manipulation AUCs (e.g. Xception DF->FS 0.256). The one
candidate carried forward is the raw-box-size trend, currently underpowered
and not FaceSwap-specific. Per GATE 0, continuing to Phase 1 regardless.

## Phase 1: reference-pipeline implementation + smoke test

**Status: smoke test done, full run NOT launched (per explicit instruction —
stopping here for a go/no-go decision).** 2026-10-12. Starting git commit
`5801358caecffd85cdceeabfbb5958cfe4cae319` (working tree also carries the
uncommitted changes described below; nothing in this pass touched git).
Data provenance: **Kaggle mirror (`xdxd003/ff-c23`), mirror/preliminary**,
same as Phase 0.

### What was built

1. **`src/data/preprocess_reference.py`** — new, parallel to `preprocess.py`
   (which is untouched). Implements the brief's reference pipeline: dlib
   frontal (HOG) face detector + 68-point landmark predictor, alignment by
   rotating so the eye line is horizontal, 1.3x margin around the aligned
   landmark bounding box, 256x256 output, 32 frames/video, **PNG** (lossless)
   storage. CLI mirrors `preprocess.py`'s style, plus `--filter-split`
   (restrict to one official split's videos — used for Face2Face/
   NeuralTextures' test-only scoping) and `--limit` (smoke testing) and
   `--timing-log` (per-video CSV timing for honest extrapolation).
   dlib's landmark model
   (`shape_predictor_68_face_landmarks.dat`, ~95MB decompressed) was
   downloaded from dlib.net to `C:/ffpp_data/tools/` (under the 1GB budget).
2. **`experiments/phase1/run_phase1_matrix.py`** — driver for the 6-training
   x 24-eval matrix (Xception on DF-only/FS-only, 3 seeds, evaluated against
   all 4 methods). **Defaults to a dry run** (prints every `src.train`/
   `src.evaluate` command it would issue, launches nothing) and only
   executes with an explicit `--execute` flag, as a safety rail matching the
   brief's >1-GPU-hour go-ahead rule. Verified via dry run — see below.
3. **Run-directory reproducibility metadata (ground rule 2)** — extended
   `src/train.py` directly (chosen over a wrapper script since `config.json`
   writing already lived there and the diff stays smaller). `config.json`
   now contains: git commit (`git rev-parse HEAD`, best-effort), package
   versions (torch/torchvision/timm, the only ones actually imported),
   GPU name (`torch.cuda.get_device_name()`), seed, and a new
   `--dataset-source` flag (default `mirror`, choices `mirror`/`official`).
   Added a `DONE` marker file written only on successful completion;
   `src/train.py` now **refuses to overwrite a finished run** (raises
   `SystemExit`) unless `--resume` is passed — verified by re-running the
   smoke-test training command against its own completed output dir and
   confirming the refusal. Run-directory convention adopted:
   `runs/reference_<model>_<trainmethod>_<comp>/seed<k>/` (simpler than the
   brief's literal `<config_hash>_seed<k>`, but unique and self-describing
   per the brief's own allowance).
4. **Bootstrap video-AUC CI** (`src/metrics.py: bootstrap_video_auc_ci`) —
   didn't exist anywhere in the repo before this pass, despite being a
   ground-rule requirement ("bootstrap over videos, >=1000 resamples, 95% CI,
   for every reported AUC"). Resamples videos with replacement, 1000
   resamples by default, percentile CI; skips resamples that land on a
   single class (undefined AUC) and reports how many resamples were
   actually usable.
5. **Per-video score saving in `src/evaluate.py`** (ground rule 1: "save
   per-video scores so analyses rerun without retraining") — this was also
   missing before this pass; `src/evaluate.py` now writes a
   `<out>.video_scores.csv` sidecar (video_id, label, mean frame prob) next
   to every results JSON by default, and adds `video_auc_ci95_lo/hi` +
   `video_auc_bootstrap_n` to the JSON itself (both on by default, can be
   disabled). Also added `dataset_source`/`seed` fields to the result JSON
   for provenance, reading them from the checkpoint's own saved args.
6. **`analyze/analyze_reference.py`** — parallel to `analyze/analyze.py`
   (existing script assumes a square same-set train/test matrix, which
   Phase 1 isn't — it trains 2 methods but evaluates against 4). Aggregates
   `results/reference/*.json` across seeds into a train-method x test-method
   matrix (mean, std, n_seeds) and diffs it cell-by-cell against the current
   pipeline's existing `analysis/cross_manipulation_xception.csv` — this is
   the GATE 1 comparison input, ready to populate once real results exist.

### Two real bugs the smoke test caught (not hypothetical — both reproduced and fixed)

1. **Alignment sign error**: the first smoke-test crops showed the correct
   face region (right scale, right content) but rotated 180 degrees
   (upside down). Cause: the eye-center vector was computed as
   `right_eye - left_eye` and fed directly into `cv2.getRotationMatrix2D`;
   for dlib's 68-point numbering (points 36-41 = subject's right eye, which
   sits on the *left* side of a front-facing image) that vector points
   backwards, so a level face got "corrected" by a spurious 180-degree
   rotation. Fixed by subtracting 180 degrees from the computed angle
   (standard face-alignment convention). Re-verified visually after the fix
   — faces are upright, eyes horizontal, correctly centered (sample crops in
   `experiments/phase1/smoke_test_samples/`).
2. **`src/data/dataset.py` only globbed `*.jpg`**: `FFPPDataset` hardcoded
   JPEG globbing, so it silently found zero samples for the new PNG crops
   (`RuntimeError: No samples for split=train ...`). Fixed by adding a
   shared `_list_frames()` helper that globs both `*.jpg` and `*.png`, so
   the same dataset class now serves either pipeline's output. This is a
   small, backward-compatible change — `preprocess.py`'s existing JPEG
   output and all of Phase 0's `results/*.json` are unaffected.

### Smoke test: what ran and the result

- Environment: `dlib-bin` (prebuilt wheel, no C++ build needed — plain
  `pip install dlib-bin` succeeded in ~2 minutes) installed into `.venv`
  alongside the existing torch 2.6.0+cu124 / torchvision 0.21.0+cu124 /
  timm 1.0.28. **No dlib build blocker** — this was flagged as a possible
  risk going in and didn't materialize.
- Preprocessing: 5 real + 5 Deepfakes + 5 FaceSwap videos from the **train**
  split (so labels are meaningful), plus 2+2+2 from the **val** split (to
  let the smoke training run actually validate), all at the real reference
  settings (dlib+landmarks, 1.3x margin, 256px, 32 frames/video, PNG) ->
  `C:/ffpp_data/ffpp_faces_ref_smoke/`. **21/21 videos succeeded, 32/32
  frames detected per video (0% detection failure on this sample)**, no
  crashes. Sample crops for eyeballing (post-fix):
  `experiments/phase1/smoke_test_samples/{real_001,deepfakes_001_870,
  faceswap_001_870}_frame00000.png` — faces are clearly captured (not
  blank/full-frame), upright, well-centered. Timing logged to
  `experiments/phase1/smoke_preprocess_timing.csv`.
- Training: `python -m src.train --model xception --methods Deepfakes
  --compression c23 --epochs 2 --batch-size 16 --seed 42 --dataset-source
  mirror --out experiments/phase1/smoke_runs/reference_xception_Deepfakes_c23/seed42`
  against the smoke crop set (320 train frames, 128 val frames). **Ran
  cleanly end to end**: both epochs completed, `history.json` has correct
  per-epoch rows, `best.pt`/`last.pt` saved, `config.json` has every
  required metadata field (verified by inspection — git commit, torch/
  torchvision/timm versions, GPU name "NVIDIA GeForce RTX 3060 Laptop GPU",
  seed 42, dataset_source "mirror"), `DONE` marker written on completion,
  and re-running the identical command against the same `--out` without
  `--resume` was correctly refused.
- Eval: `python -m src.evaluate` against the trained smoke checkpoint, test
  method FaceSwap (cross-manipulation, DF->FS, matching the paper's single
  most important cell) on the **val**-split smoke crops (no FaceSwap
  test-split crops exist in the tiny smoke set). Metrics JSON wrote
  correctly with all expected fields plus the new `video_auc_ci95_lo/hi`,
  `video_auc_bootstrap_n`, `dataset_source`, `seed`; the `.video_scores.csv`
  sidecar wrote correctly (4 rows: video_id, label, prob). CI came out
  (0.0, 1.0) with only 880/1000 usable resamples — expected and correct
  given just 4 videos in this cell, not a bug. **Not a real AUC number —
  this is a 4-video correctness check, not a result.**
  `analyze/analyze_reference.py` was also smoke-tested against a correctly
  renamed copy of this one result file and produced sensible output
  (matrix CSV, raw CSV, and a vs-current-pipeline diff CSV showing, again,
  only as a mechanics check: this toy cell's AUC of 0.75 vs. the current
  pipeline's real DF->FS 0.2562 — meaningless with n=4 but confirms the
  script's logic is correct).

### Honest extrapolation to the full Phase 1 run (NOT launched)

**Preprocessing (CPU-bound, dlib HOG detector + landmark predictor, single-
threaded as currently implemented):**
- Measured: 21 videos, mean 22.1 s/video, median 15.4 s/video, range
  5.8-38.6 s/video (stdev 12.7s — high variance, apparently driven mostly by
  source video resolution/length rather than detection difficulty; 0%
  detection failures in the sample so this isn't about retries).
- Full scope: real (1000) + Deepfakes (1000) + FaceSwap (1000) fully, plus
  Face2Face (140, test-split only) + NeuralTextures (140, test-split only)
  = **3280 videos**, ~105,000 crops.
- **Extrapolated wall-clock, single-threaded: ~20 CPU-hours (point
  estimate), plausible range ~14-28 hours** given the observed variance;
  absolute bounds using the sample's min/max per-video time are 5-35 hours.
  This is markedly slower than the current pipeline's GPU-based MTCNN step
  (~0.2 s/frame -> ~4s/video for 20 frames; reference pipeline measured
  ~0.7 s/frame for 32 frames), consistent with the brief's own expectation
  that CPU-bound dlib detection wouldn't scale the same way as GPU MTCNN.
  **Not yet parallelized** — this machine has 20 logical CPUs and
  per-video processing is embarrassingly parallel (each video is
  independent), so a straightforward multiprocessing pool (e.g. 8-12
  workers) could plausibly cut this to **2-4 hours wall-clock** if that
  matters for the go/no-go decision; not implemented or measured in this
  pass since the task scope was "prove correctness + measure", not optimize.
- Disk: measured 42MB / 480 crops in the smoke set = ~87.5 KB/crop (PNG,
  256x256, lossless) vs. the current pipeline's ~26 KB/crop (JPEG q95,
  299x299, measured as 2.6GB / ~99,987 crops). **Extrapolated full
  reference-pipeline crop set: ~9.2 GB** (honest range ~7-12GB given
  content-dependent PNG compression variance) — bigger than the current
  2.6GB set as expected (lossless + larger crops), but not dramatically so.

**Training (GPU-bound, RTX 3060 Laptop 6GB):**
- 256px PNG crops need no architecture change: `FFPPDataset`'s transform
  pipeline already resizes every crop to the model's fixed `IMG_SIZE`
  (Xception = 299) via `torchvision.transforms.Resize`, regardless of the
  crop's native size, and `timm`'s `legacy_xception` already uses global
  average pooling (no fixed-size assumption beyond what `Resize` already
  guarantees). **No architecture/resizing-logic change needed** — one of
  the open questions in the task brief, now closed.
- Full DF-only or FS-only train set: 720 real + 720 method train-split
  videos x 32 frames/video = 46,080 frames/epoch (both DF-only and FS-only
  runs are the same size, since every method has the same number of paired
  videos). Using this project's own measured GPU throughput for Xception at
  batch 32 on this exact GPU (PROJECT_LOG.md: ~147 img/s, forward+backward+
  step) rather than the smoke test's own timing (smoke test used batch 16
  on a 320-frame set — too small and differently-batched to extrapolate
  reliably; the brief anticipated this and asked to sanity-check against
  PROJECT_LOG rather than trust the smoke number blindly) gives **~5.2
  minutes/epoch** compute time, plus a small validation-pass overhead
  (~1 minute/epoch, 280 val videos x 32 frames).
- With `--epochs 15 --patience 4` (train.py's defaults) and assuming
  convergence somewhere in the 6-15 epoch range (consistent with the
  original pipeline's behavior per PROJECT_LOG), **each of the 6 training
  runs is estimated at roughly 0.5-1.5 GPU-hours, point estimate ~0.9
  GPU-hours/run**: **total training ~5-6 GPU-hours (point estimate),
  honest range ~4-10 GPU-hours** depending on actual early-stopping epoch.
  This is a smaller total than PROJECT_LOG's original 12-run matrix
  (15-25 GPU-hours) because Phase 1 only has 6 runs (not 12) and all are
  single-method (not the much larger "train on all 4 methods combined"
  runs that dominated the original estimate), even though each Phase 1
  epoch processes 1.6x more frames (32 vs 20 frames/video).
- Eval matrix (24 cells x ~4480 test frames each, inference-only, plus
  bootstrap CI compute which is cheap CPU work) is negligible by
  comparison — well under 1 GPU-hour total for the whole matrix.
- **Total estimated compute for the full Phase 1 run: ~25-30 CPU-hours of
  preprocessing (single-threaded; ~2-4 hours if parallelized) + ~5-7
  GPU-hours of training/eval, well above the brief's 1-GPU-hour report-first
  threshold.** Confidence: moderate on the GPU-hour estimate (grounded in
  this project's own prior measured throughput on the same GPU/model), lower
  on the preprocessing wall-clock (grounded in only 21 timed videos with high
  per-video variance whose cause — resolution vs. length — wasn't isolated).

### Deviations from the brief / this task's instructions

- Found and fixed two bugs beyond what was asked (the alignment 180-degree
  flip and `dataset.py`'s JPEG-only glob) — both were blocking correctness,
  not scope creep; left undocumented they'd have silently produced garbage
  crops or a hard crash on the first real run.
- Added bootstrap-CI and per-video-score-saving machinery to
  `src/evaluate.py`/`src/metrics.py` that the brief's ground rules require
  but that didn't exist anywhere in the repo yet (not Phase-1-specific —
  it's now available to Phase 0's pipeline too, though Phase 0's existing
  `results/*.json` files were not regenerated/touched).
- Did not implement preprocessing multiprocessing/parallelization — flagged
  above as a lever available if the ~20-28 hour single-threaded wall-clock
  is unacceptable, but out of scope for "prove correctness + measure
  honestly."
- `preprocess_reference.py` currently targets `c23` only by default (matches
  Phase 1's scope); `c40`/`c0` reference-pipeline crops are not addressed
  here (official access still pending, same as Phase 0).

### Proposed next step

Do not launch the full preprocessing or training matrix without an explicit
go-ahead. Options for that decision:
1. **Proceed as scoped** (6 train runs, 24 eval cells, full 3280-video
   preprocessing) — ~20-28 CPU-hours preprocessing + ~5-7 GPU-hours
   train/eval, likely spanning more than one overnight unattended run (keep
   the existing `scripts/keep_awake.ps1` running, per the sleep issue noted
   in PROJECT_LOG.md).
2. **Narrow scope**: 1 seed instead of 3 (cuts training to ~1-2 GPU-hours,
   loses the brief's "at least 3 seeds" statistical standard — would need
   to be caveated), or DF/FS-only test-eval instead of all 4 methods (cuts
   eval but eval was already cheap, low value), or parallelize preprocessing
   first (implement a multiprocessing pool in `preprocess_reference.py`,
   re-measure, likely the highest-leverage change if wall-clock time is the
   binding constraint rather than GPU-hours).
3. **Investigate further**: the preprocessing timing variance (5.8-38.6
   s/video) wasn't root-caused (resolution vs. video length vs. detector
   upsample cost) — a slightly larger timed sample (~30-50 videos) would
   tighten the CPU-hour estimate before committing to an unattended
   20+ hour run.

## Phase 1: full run (in progress)

**Status: done (see "Phase 1 complete -- GATE 1 result" below).** Launched with explicit user go-ahead (parallelize
preprocessing, then the full scoped run). Data provenance: **Kaggle mirror,
mirror/preliminary.** Starting commit `5801358` plus the uncommitted Phase 1
code described above.

### Preprocessing (done)

- Parallelized `preprocess_reference.py` (process pool, ~9 workers after
  benchmarking 4/8/16; bench outputs in `C:/ffpp_data/ffpp_faces_ref_bench_w*`).
- Full run: **2026-10-12 22:30 -> 2026-10-13 10:56 IST, 12.4 h wall-clock**
  (from first/last crop file mtimes), producing 104,831 PNG crops (~7.8 GB+)
  in `C:/ffpp_data/ffpp_faces_ref`. This is well above the ~2-4 h
  parallelized estimate floated after the smoke test -- the estimate was
  wrong by ~3-6x; cause not isolated (dlib HOG detection cost and/or disk I/O
  contention are the likely candidates).

### Interruption and resume

- Training started 11:19. DF seeds 1 and 2 completed (5 and 7 epochs, DONE
  markers) and were evaluated on all 4 test methods (8/24 cells).
- DF seed 3: the laptop **slept ~13:55 -> ~20:02** mid-epoch (`keep_awake.ps1`
  was not running), so epoch 0 took 22,812 s; then the machine **restarted at
  20:24** and was later powered off from the Start menu (System log event
  1074, 21:34). The training process and the supervising agent both died.
- Resumed 2026-10-13 ~23:40 via `experiments/phase1/run_phase1_matrix.py
  --execute` launched as a detached process (log: `logs/phase1_resume.out`/
  `.err`), with `scripts/keep_awake.ps1` running detached alongside it.
  Driver skips DONE runs and existing eval JSONs; resumed throughput
  ~2.9 it/s (~8 min/epoch), remaining estimate ~4-5 h.

### Fixes made during resume

- `run_phase1_matrix.py`: training commands now pass `--resume`, so a
  partially-trained run continues from `last.pt` instead of restarting
  (DONE runs are still skipped by the driver before any command is built).
- `src/train.py`: `last.pt` was saved *before* `best_auc` was updated, so a
  resumed run restored a stale best AUC (seen live: "resumed at epoch 1 (best
  val AUC -1.0000)") and the early-stopping counter reset to 0. Now saves
  `last.pt` after the update and persists/restores `bad_epochs`. Affects only
  resumed runs; DF seed 3's resume used the old code (its epoch-0 best was
  0.99995, so the practical effect is negligible) -- noted for transparency.

### Preliminary GATE 1 signal (2 of 3 seeds, DF-trained row only -- NOT the gate decision)

| DF-trained Xception -> test | current pipeline | reference seed 1 [95% CI] | reference seed 2 [95% CI] |
|---|---|---|---|
| Deepfakes | 0.9999 | 0.9998 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| Face2Face | 0.7421 | 0.674 [0.606, 0.737] | 0.746 [0.690, 0.802] |
| **FaceSwap** | **0.2562** | **0.238 [0.184, 0.293]** | **0.227 [0.172, 0.285]** |
| NeuralTextures | 0.8123 | 0.807 [0.754, 0.856] | 0.737 [0.676, 0.792] |

Source: `results/reference/xception_Deepfakes_to_*_c23_seed{1,2}.json`
(video-level AUC, 1000 bootstrap resamples over videos). Under a reference
pipeline that changes detector, alignment, storage format, resolution and
frame count, DF->FS stays below chance with both CIs entirely under 0.5.
This points toward the brief's **C2 branch (inversion persists)**, but the
gate decision waits for seed 3 and the FaceSwap-trained row.

### Phase 1 complete -- GATE 1 result

**Status: done.** Run finished 2026-10-14 ~02:25 IST. All 6 training runs
DONE (DF seeds 1-3: 5/7/6 epochs incl. resume; FS seeds 1-3: 5 epochs each,
all early-stopped at val AUC 1.0), all 24 eval cells written, 0 tracebacks.
Data provenance: **Kaggle mirror, preliminary.**

Aggregation (both script-generated):
- `analysis/reference/reference_matrix_xception.csv` (+ `_raw.csv`,
  `reference_vs_current_xception.csv`) via `analyze/analyze_reference.py`.
- `analysis/reference/gate1_seed_mean_ci.csv` via
  `analysis/gate1_seed_mean_ci.py`: seed-mean video AUC with a 95% CI from
  1000 bootstrap resamples over the 280 test videos (all seeds score the same
  videos, so each resample recomputes every seed's AUC and averages). A cell is
  called below/above chance only if the **whole CI** is on one side of 0.5.

| Train -> test | Reference, 3-seed mean [95% CI] | Call | Current pipeline (1 seed, no CI) |
|---|---|---|---|
| DF -> DF | 0.9999 [0.9997, 1.0000] | above | 0.9999 |
| DF -> F2F | 0.694 [0.640, 0.745] | above | 0.742 |
| **DF -> FS** | **0.258 [0.213, 0.307]** | **below chance** | **0.256** |
| DF -> NT | 0.759 [0.710, 0.804] | above | 0.812 |
| FS -> DF | 0.469 [0.411, 0.534] | at chance | 0.501 |
| FS -> F2F | 0.678 [0.618, 0.734] | above | 0.700 |
| FS -> FS | 0.9996 [0.9990, 1.0000] | above | 1.000 |
| **FS -> NT** | **0.428 [0.365, 0.487]** | **below chance** | **0.395** |

**GATE 1 outcome: the inversion persists.** A reference pipeline that changes
the face detector (MTCNN -> dlib HOG), adds landmark alignment, changes storage
(JPEG q95 -> PNG), crop size (299 -> 256) and frames per video (20 -> 32)
reproduces the current pipeline's matrix almost cell for cell: max absolute
difference 0.054, DF->FS differs by 0.002, and both below-chance cells stay
below chance with CIs entirely under 0.5. Per RESEARCH_BRIEF.md: **lead claim
becomes C2 (inversion mechanism); Phase 2 (preprocessing grid) becomes a
robustness check, and its central question is already largely answered.**

Caveats:
- Mirror data only. The brief cites published Xception DF->FS ~0.70; this
  pipeline-independent 0.26 makes the mirror-vs-official question the main
  open threat to validity, not preprocessing.
- Phase 1 trained DF and FS only (brief's minimum); the NT->FS cell (current
  pipeline 0.320) was not re-tested here.
- Single architecture (Xception) in the reference pipeline.
- Small fixes made during this phase, all in this working tree: `--resume`
  in the driver, `last.pt` best-AUC/patience persistence in `src/train.py`,
  and a filename regex in `analyze/analyze_reference.py` that silently dropped
  every Face2Face file (caught by a pre-run test; the final tables include
  all 24 cells).

### Proposed next step

Phase 3 steps 3.1-3.3 (no training): per-crop statistics (Laplacian variance,
high-frequency energy, face-vs-border colour/luminance, mask-boundary gradient,
JPEG blockiness, noise residual, raw box size) on both crop sets, signed effect
size per method vs. real, and the sign-prediction test against the observed
below/above-chance pattern. Cheap (CPU only), and it decides whether C2 holds.
Separately: re-request official FF++ access.

## History rewrite: commit hash mapping (2026-10-14)

At the user's request, git history was rewritten to remove an AI
`Co-Authored-By` trailer from all 24 commit messages that carried it
(`git filter-branch --msg-filter`, then force-push). Only commit messages
changed: the file tree at HEAD, every commit's author, date and subject, and
the commit count (30) are identical. **Every commit hash changed**, so hashes
cited above, and the `git_commit` recorded in each `runs/*/config.json`,
refer to the pre-rewrite history. Old history is kept locally only, under tag
`backup/pre-trailer-strip-2026-10-14`.

Hashes cited in this log: `4a180c0` -> `a1d314e`, `5801358` -> `dfe5a78`
(the Phase 1 runs' recorded commit). Full mapping, old -> new:

```
1d94731 e7ba47a Phase 1: reference-pipeline replication -- inversion persists (GATE 1)
5801358 dfe5a78 Add project_report.docx: formal written project report
555bf06 425fda0 Add progress_update.pptx: supervisor update deck
043cf0f 7b579a0 Phase 0: hygiene fixes + audit (RESEARCH_BRIEF.md)
4a180c0 a1d314e Revert mid_review_update.pptx to scoped 9-slide version, add literature review
d9774f2 253117e Rebuild mid_review_update.pptx: formal Dean-review deck, full results
61fe361 039ef5b Add the concrete Kaggle dataset link, not just "search FaceForensics++"
b0dd3ef d4a86e4 Swap presenter order: Rithvik first, Vipin second
fcae1af c5381ed Remove end-review build scripts from the working tree too
92dd9bc 71b2a74 Remove end-review deck/script from repo, add talking points for mid-review
6d4074c 295e933 Split mid-review and end-review into separate decks
621571b b5b8e9e Add timed presenter script (talking_points.pdf) for the mid-review
0a3c60c 808bc5f Add the other 3 team members to the mid-review deck
7ce5ee2 45298b9 Condense mid-review deck from 23 to 11 slides
3225abb e250957 Rebuild mid-review deck to reflect completed project (23 slides)
94de132 afb33e2 Add Grad-CAM visualization explaining why cross-manip generalization fails
0f91f48 8c6daee Back up the two headline checkpoints via Git LFS
ebd0ed1 1b7bf9c Add scripts/keep_awake.ps1: app-level sleep prevention for long runs
1c8f917 f9e1e02 Add final study results: baseline, cross-manipulation, proxy-c40 tables
9d933d9 d59f417 Add live demo webapp (FastAPI backend + single-page frontend)
5d3fdc5 b6be1da Add scripts/run_study.sh: baseline + cross-manip + proxy-c40 experiment runner
2e57fcd 1a7913f Add scripts/make_c40proxy.py: local c40 stand-in via re-transcoding
560c1a6 1dadaac Add c40proxy as a recognized compression level for evaluation
e1ff433 3f1c957 Add official FF++ train/val/test splits (also previously gitignored)
01967e3 1efd376 Fix .gitignore accidentally excluding src/data/, add missing core modules
ba0694a ba0694a Add box-plot outlier analysis and update mid-review slides
56efd21 56efd21 Add mid-review status update slides
a0eb2d2 a0eb2d2 Add Colab fallback notebook
2dc53a3 2dc53a3 Add PROJECT_LOG.md: env setup, GPU capacity findings, dataset plan
fa2c827 fa2c827 Import deepfake detection project (Xception vs CNN-ViT on FF++)
```

## Phase 3: mechanism (C2) -- steps 3.1-3.4

**Status: done (Grad-CAM step 3.5 skipped; it's qualitative only).**
2026-10-14. No training. Data provenance: **Kaggle mirror, preliminary.**

### What ran

- `experiments/phase3/crop_statistics.py`: 8 frames from each of the 700
  test-split videos (real + 4 methods) in **both** crop sets (current MTCNN/JPEG
  and reference dlib/PNG); per-video means of lap_var, hf_ratio, noise_std,
  blockiness, color_dE, lum_diff, boundary_grad, boundary_grad_rel (face mask =
  convex hull of dlib 68 landmarks; landmark fallback used on <=0.4% of
  frames). Raw face-box size taken from the Phase 0 audit sample (current
  pipeline only). Outputs `analysis/phase3/crop_stats_{current,reference}.csv`.
- `experiments/phase3/sign_prediction.py`: design fixed in its docstring
  before any statistic was examined. Effect sizes (Cohen's d, fakes vs. all
  test reals, bootstrap CI; paired d vs. each fake's own target video; AUC
  of the raw statistic), sign-prediction test against all three observed
  matrices with an exact 24-relabeling permutation null, and a reliance check
  on the reference models' per-video scores. Outputs
  `analysis/phase3/effect_sizes.csv`, `sign_prediction.csv`,
  `reliance_reference.csv`, run log `sign_prediction_run.log`.

### Result: high-frequency energy flips sign between methods

`hf_ratio` (share of spectral energy above 0.25 cycles/px) is the only
statistic where FaceSwap moves opposite to Deepfakes/NeuralTextures with every
CI excluding zero, in both crop sets:

| Method vs. real | current d [95% CI] | reference d [95% CI] | paired d (cur / ref) |
|---|---|---|---|
| Deepfakes | -0.48 [-0.72, -0.26] | -0.36 [-0.60, -0.14] | -0.58 / -0.49 |
| NeuralTextures | -0.39 [-0.61, -0.15] | -0.30 [-0.52, -0.08] | -0.77 / -0.62 |
| FaceSwap | +0.37 [+0.11, +0.66] | +0.33 [+0.09, +0.58] | +0.45 / +0.49 |
| Face2Face | -0.01 [-0.23, +0.24] | -0.02 [-0.25, +0.21] | -0.03 / -0.06 |

DF and NT fakes are blurrier than real faces; FS fakes are sharper; F2F shows
no shift. Sign prediction (predicted transfer sign = sign(d_A) x sign(d_B)):

| Observed matrix | cells correct | Spearman rho (d_A*d_B vs AUC) | permutation p (rho) |
|---|---|---|---|
| current Xception | 8 / 9 | 0.919 | 0.042 (best of 24) |
| current CNN-ViT | 10 / 12 | 0.919 | 0.042 (best of 24) |
| reference Xception | 4 / 5 | 0.899 | 0.083 |

The misses are the F2F<->FS cells: F2F's d is ~0, so its sign carries no
information (the continuous score handles this; the sign call can't).
Sharpness statistics correlated with hf_ratio (lap_var, noise_std; Spearman
0.64-0.70) predict nearly as well but their FaceSwap effect is ~0.

**Reliance (reference models, 140 real test videos):** the DF-trained model's
fake probability rises as real faces get blurrier -- lap_var rho = -0.27
(p = 0.001), noise_std -0.25 (p = 0.002), boundary_grad -0.29 (p < 0.001),
all in the direction DF's training fakes moved; hf_ratio itself -0.05
(p = 0.60). The FS-trained model shows no significant reliance on any
high-frequency statistic (hf_ratio rho = +0.10, p = 0.25).

### GATE 3 call

A statistic does predict the sign pattern -- it ranks first of all 24 method
relabelings on two of three matrices and replicates across both crop sets and
both architectures -- so the paper stays on C2 (no fallback). Limits, stated
plainly: the permutation test can't go below p = 1/24, nine statistics were
tested (not significant after a Bonferroni correction), and the high-frequency
statistics are correlated, so this is a strong, specific candidate mechanism,
not proof. Reliance evidence is clear for the DF-trained model only. Phase 4
is the causal test.

### Phase 4 set-up (prepared, not yet run)

- `RandomBlurSharpen` in `src/data/dataset.py` (p = 0.9; Gaussian blur radius
  0.3-1.5 or unsharp mask radius 1-2, 50-200%), behind `src/train.py
  --hf-aug` and the driver's `--hf-aug`; label-independent, train-time only.
- `experiments/phase4/check_hf_aug.py` (`analysis/phase4/hf_aug_check.csv`,
  200 train videos/class, one crop each at 299 px): hf_ratio d vs. real goes
  DF -0.14 -> 0.01 and FS +0.42 -> +0.27. **Prediction recorded before
  training:** DF->FS should recover more than FS->NT, because the augmentation
  removes the DF cue but only weakens the FS cue.
- Launch (6 runs + 24 evals, est. ~5 h, needs keep_awake):
  `python -m experiments.phase1.run_phase1_matrix --data-root
  C:/ffpp_data/ffpp_faces_ref --splits-dir splits --runs-root
  runs/phase4_hfaug --results-dir results/phase4_hfaug --hf-aug --execute`.
  Success criterion (brief): below-chance cells rise with CI excluding the
  Phase 1 baseline mean, in-domain AUC drop <= 2 points.

## Phase 7 (partial): figures from script outputs

`analysis/make_figures.py` -> `analysis/figures/*.png|pdf` (300 dpi), every
number read from the analysis/ CSVs:
- **F1** `F1_transfer_matrices`: current 4x4 vs. reference 2x4 (seed mean +
  95% CI) Xception matrices, diverging around AUC 0.5, below-chance cells
  hatched as a non-colour cue.
- **F3** `F3_hf_effect_sizes`: hf_ratio Cohen's d per method, both crop sets.
- **F4** `F4_sign_prediction`: d_A x d_B vs. observed AUC, three matrices; the
  on-figure claim (every below-chance cell pairs FS with DF or NT) is
  asserted in code before drawing.
- **F6** `F6_architecture_agreement`: Xception vs. CNN-ViT off-diagonal AUC.
Not yet: F2 (preprocessing grid; Phase 2 not run), F5 (Phase 4 running), F7
(existing `analyze/outputs/gradcam_comparison.png` can serve). Palette: the
dataviz reference palette's validated slots 1-3; Node.js isn't installed, so
its validator script wasn't re-run here.

Phase 4 launched 2026-10-15 in visible windows (keep_awake + driver, logs
`logs/phase4_hfaug.{out,err}`), metadata confirms `hf_aug: true` at `2d38f14`.

## Literature check and an external test of the mechanism (2026-10-15)

**Correction to RESEARCH_BRIEF.md section 1.** The brief says "Published
Xception DF->FS is roughly 0.70". The published number is **0.4913**: Li et
al., *Face X-ray for More General Face Forgery Detection*, CVPR 2020, Table 1
(read from the arXiv:1912.13458v2 PDF), Xception trained on DF tested on FS.
0.70 is the *reverse* cell (FS->DF, 0.7012). Same table: NT->FS 0.4798, and
its HRNet baseline scores **0.3915** on DF->FS -- below chance on official
data. So official-data results already show FaceSwap as the worst target, at
or below chance; our mirror results (DF->FS 0.256) are a stronger form of a
published pattern, not a contradiction of it. Transcribed matrix and citation:
`analysis/external/facexray_table1_xception.csv`, `analysis/external/SOURCE.md`
(Table 1's compression level and frame/video level are not stated in the
paper).

**External test.** hf_ratio effect sizes from our mirror crops, used as
d_A x d_B, predict that published official-data Xception matrix with Spearman
rho = 0.735 over 12 off-diagonal cells, the best of all 24 method relabelings
(p = 0.042), from either crop set
(`analysis/external/sign_prediction_published.csv`). Our current matrix and
the published one correlate at rho = 0.797. This is the strongest answer so
far to the mirror-vs-official concern: the mechanism predicts an independent
lab's official-data results.

**Unverified:** a search summary attributes to DF40 (Yan et al., NeurIPS 2024
Datasets & Benchmarks, arXiv:2406.13495) below-chance results on StyleCLIP
fakes explained by real/fake resolution mismatch ("higher resolution, higher
probability to be real"). Could not open the paper to confirm (fetch size
limit, then a session limit); verify before citing.

## Phase 4: intervention (C3) -- done

**Status: done** 2026-10-19 03:39 IST. 6 runs (DF/FS x 3 seeds, Xception,
reference crops, `--hf-aug`, patience 4 as in Phase 1), 24 evals. Mirror data,
preliminary. Interrupted three times (laptop sleep/shutdown 10-15, a hang in
validation and a manual shutdown 10-18, a deliberate thermal pause 10-18/19);
every resume continued from the last finished epoch.

`analysis/phase4_delta.py` -> `analysis/phase4/delta_vs_phase1.csv`, figure
F5. Paired change vs. Phase 1 over the same 280 test videos (1000 bootstrap
resamples, seed-mean AUC under both conditions):

| Cell | Phase 1 | Phase 4 [95% CI] | change [95% CI] |
|---|---|---|---|
| DF->FS | 0.258 | 0.356 [0.301, 0.410] | +0.098 [+0.070, +0.129] |
| FS->NT | 0.428 | 0.521 [0.462, 0.582] | +0.093 [+0.059, +0.129] |
| FS->DF | 0.469 | 0.574 [0.511, 0.633] | +0.105 [+0.071, +0.136] |
| DF->F2F | 0.694 | 0.734 [0.685, 0.786] | +0.040 [+0.012, +0.070] |
| FS->F2F | 0.678 | 0.630 [0.573, 0.685] | -0.048 [-0.081, -0.016] |
| DF->NT | 0.759 | 0.745 [0.696, 0.794] | -0.014 [-0.041, +0.014] |
| DF->DF / FS->FS | 1.000 | 1.000 | ~0 |

Brief's success criterion (below-chance cells rise with CI excluding the
baseline mean; in-domain drop <= 0.02): **met for both below-chance cells**.
FS->NT is no longer below chance; **DF->FS still is** (0.356). The
pre-registered directional prediction (DF->FS recovers more than FS->NT)
**failed** -- they recovered equally. Cost: FS->F2F drops 0.048. Reading:
high-frequency energy is a causal but partial contributor.

Also fixed during this phase: `src/train.py` validation now loads in-process
and training workers persist (per-epoch worker respawn ran the 16 GB laptop
out of memory), and the driver retries failed runs.

## Phase 6: external validity (DeepFakeFace) -- done

**Status: done** 2026-10-19. No retraining. DeepFakeFace (OpenRL/DeepFakeFace,
OpenRAIL; IMDB-WIKI reals, research use only, not redistributed), 5.07 GB,
1000 images sampled per class (seed 0), reference-pipeline crops
(`experiments/phase6/prepare_dff.py`); dlib found a face in 818 real, 640
inpainting, 656 InsightFace, 901 text-to-image images (class-dependent
detection rate -- a selection-bias caveat). `experiments/phase6/eval_dff.py`
scored the Phase 1 and Phase 4 Xception checkpoints on CPU ->
`analysis/phase6/dff_scores.csv`, `dff_effect_sizes.csv`, `dff_transfer.csv`.

hf_ratio d vs. real: inpainting -0.26, InsightFace -0.52, text2img -0.40 (all
blurrier), so the sign test predicts DF-trained transfers and FS-trained
inverts on all three. Phase 1 models (3 seeds, AUC [95% CI]):

| | DF-trained | FS-trained |
|---|---|---|
| SD inpainting | 0.463 [0.436, 0.489] | 0.514 [0.487, 0.542] |
| InsightFace | 0.654 [0.630, 0.681] | 0.491 [0.464, 0.519] |
| SD text2img | 0.323 [0.302, 0.345] | 0.399 [0.376, 0.423] |

**Prediction correct on 2/4 decided cells -- no better than chance.** SD
text2img is below chance for every detector regardless of training-cue sign.
Conclusion: the high-frequency account is within-FF++ (it predicts our and the
published FF++ matrices) and does not carry to another dataset, where other
differences dominate (hypothesis, untested: IMDB-WIKI still photos are out of
distribution for detectors trained on YouTube frames). Phase 4 (hf-aug) models
behave similarly (rows with runs=phase4_hfaug in dff_transfer.csv).

## Phase 5: architecture independence (C4) -- done

**Status: done** 2026-10-19 10:28. Reference crops, DF-only and FS-only, 2
seeds, patience 2 (deviation from Phase 1's 4: validation AUC saturates at 1.0
in the first epochs, so the kept checkpoint is essentially unaffected).
ImageNet-pretrained, 299 px input like Xception: ResNet-50 (23.5M params,
batch 32) and EfficientNet-B4 (17.6M; **batch 16** -- `experiments/bench_batch.py`
measured 6.91 GB peak at batch 32 on the 6 GB card, which would page to system
RAM). Outputs `analysis/phase5/{resnet50,efficientnet_b4}_seed_mean_ci.csv`.

| Cell | Xception (3 seeds) | ResNet-50 (2) | EfficientNet-B4 (2) |
|---|---|---|---|
| DF->FS | 0.258 [0.213, 0.307] | 0.291 [0.240, 0.346] | 0.264 [0.217, 0.317] |
| FS->NT | 0.428 [0.365, 0.487] | 0.416 [0.354, 0.480] | 0.361 [0.302, 0.422] |
| FS->DF | 0.469 (at chance) | 0.450 (at chance) | 0.477 (at chance) |
| DF->DF / FS->FS | ~1.000 | ~1.000 | ~1.000 |

Both backbones reproduce Xception's below/at/above-chance call in **8/8**
cells. C4 supported: the inversion is set by the training manipulation, not
the architecture (among ImageNet-pretrained CNNs at matched input size; the
original from-scratch CNN-ViT agreed at r = 0.94).
