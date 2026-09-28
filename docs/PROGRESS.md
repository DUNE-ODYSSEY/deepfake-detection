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
