# Research Brief for Claude Code: Deepfake Detection Novelty Project

Save this at the repo root as `RESEARCH_BRIEF.md`, and add one line to `CLAUDE.md`:
"Read RESEARCH_BRIEF.md before doing any work; follow its phases, gates and ground rules."

## 0. How to work

- Read this whole file before acting. Work phase by phase.
- At the end of each phase, write `docs/PROGRESS.md` (what ran, git commit, numbers, deviations, next step) and stop at every **GATE** for my decision if the outcome is ambiguous.
- Verify file paths and function names in the repo yourself. Do not assume they match this document.
- Before any run longer than ~1 GPU-hour, estimate total hours, tell me, and prefer subsampling to blowing the budget. Check the GPU with `nvidia-smi` (assume a single consumer GPU, RTX 3060 class, unless it says otherwise).
- Total time budget for the whole plan is about 4 weeks. Cut scope rather than run over.

## 1. Where the project stands

- Repo compares XceptionNet (ImageNet-pretrained, 299 px) against a CNN-ViT hybrid (ResNet-50 stages 1-3 + 6-layer transformer trained from scratch, 224 px) on FaceForensics++ c23.
- Data came from a **Kaggle mirror, not the official release**. Preprocessing: MTCNN, 1.3x margin, crops re-saved as JPEG q95, 20 frames per video, single seed (42).
- Baseline: Xception video AUC 0.9958, hybrid 0.9836. The hybrid did **not** beat Xception.
- Cross-manipulation matrices show AUC far **below 0.5** in cells involving FaceSwap, for example Xception DF->FS 0.256, hybrid DF->FS 0.198, FS->DF about 0.32, FS->NT about 0.32, NT->FS 0.32-0.39. Published Xception DF->FS is roughly 0.70, so either our pipeline creates the inversion or standard pipelines hide it.
- Off-diagonal AUCs of the two models correlate at Pearson r = 0.94 (Spearman 0.95): transfer seems set by the training forgery, not the architecture.
- The "proxy c40" (double compression) is not official c40.

## 2. Target contribution (what makes this publishable)

Working title: *"Inverted Transfer: Why Deepfake Detectors Score Unseen Manipulations Below Chance, and How Preprocessing Toggles It."*

Falsifiable claims. Keep only the ones the data supports.

- **C1 (preprocessing sensitivity):** For the same architecture, cross-manipulation AUC on the DF/FS cells changes substantially, possibly across the 0.5 line, with preprocessing choices alone (detector, margin, storage format, alignment, frames, data source).
- **C2 (inversion mechanism):** Detectors latch onto a low-level statistic (sharpness, high-frequency energy, face-vs-border colour, boundary contrast, resolution or JPEG mismatch) whose direction relative to real faces differs between manipulations. The sign of that difference predicts which transfer cells fall below 0.5.
- **C3 (intervention):** Neutralizing that statistic in training moves the below-chance cells toward or above 0.5 for both architectures without a large in-domain drop.
- **C4 (architecture independence):** Transfer matrices are more similar across architectures than across training methods; architecture explains little variance in cross-manipulation AUC.

**Do not claim** any of the following:
- That we are the first to see below-chance cross-domain AUC. Prior work has already reported it and offered a resolution-bias explanation. Our contribution is extending it: a within-FF++ FaceSwap-centred account, sign prediction from measured statistics, interventions, and a preprocessing sweep.
- That the hybrid is better or worse "in general." The comparison is confounded (pretraining, resolution).
- Anything about c40 from the proxy.
- Any number from the Kaggle mirror as if it were official-data evidence. Label it "mirror, preliminary."

## 3. Ground rules

1. **Never fabricate, round by hand, or edit numbers.** Every number in a table or the paper must come from a file in `results/` written by a script. Save per-video scores so analyses rerun without retraining.
2. **Reproducibility:** every run gets its own directory `runs/<experiment>/<config_hash>_seed<k>/` containing `config.json`, git commit, package versions, GPU name, seed, dataset source (official / mirror / other), and metrics. Never overwrite a finished run.
3. **Fix known bugs first** (Phase 0): `--resume` resets `history = []` and overwrites `history.json`; checkpoints (`*.pt`) are committed to git (move to Git LFS or ignore); no seed argument.
4. **AUC orientation:** fake = 1. Add a unit test proving AUC(scores, labels) is computed with the correct polarity. Never flip scores post hoc in main tables. If a sign-corrected view is useful, put it in a clearly labelled supplementary table.
5. **Data and licensing:** Do not redistribute FF++. Before downloading any dataset, read its license or terms and tell me what they say. Record data provenance in every results file.
6. **Ask before** deleting files, downloading more than about 1 GB, or launching sweeps over about 1 GPU-hour.
7. **Keep the diff small.** Extend the existing code rather than rewriting it. Add scripts under `experiments/` or `analysis/`, not scattered notebooks.

## 4. Phases and gates

### Phase 0: Hygiene and audit (days 1-3)

1. Fix the resume-history bug, add `--seed`, write per-run metadata, move checkpoints out of git.
2. **Label and split audit:**
   - Assert fake = 1 everywhere.
   - Confirm train/val/test are video-disjoint using the official split files.
   - Confirm the same real test videos are used in every train->test cell.
3. **Encoding audit** with `ffprobe` on every video (or a large random sample per class, at least 50): codec, profile, resolution, fps, bitrate, pixel format, and any x264 settings string (`crf=`) in the stream. Compare real vs each method. Flag systematic mismatch, since it is a candidate cause of the inversion.
4. **Crop audit** per method: MTCNN failure rate, frames kept per video, raw face-box size distribution before resize, JPEG quantization tables.
5. Write `results/audit/*.csv` and a summary.

**GATE 0:** Report what differs between real and each method. Continue regardless, but note any confound.

### Phase 1: Reference-pipeline replication (days 3-7)

1. Implement a standard reference pipeline: dlib face detection with landmark alignment, 1.3x margin, 256x256, 32 frames per video, **PNG** storage. If a public benchmark releases preprocessed frames, check its terms and use them as an extra reference arm.
2. Train Xception on each of DF-only and FS-only, then F2F-only and NT-only if the budget allows, 3 seeds. Build the 4x4 matrix with bootstrap confidence intervals.
3. Compare against the current pipeline cell by cell.

**GATE 1 (report to me):**
- Inversion **disappears** (DF->FS returns to roughly 0.6-0.7) -> lead claim is **C1**, with the current pipeline as the case study.
- Inversion **persists** -> lead claim is **C2**, and Phase 2 becomes a robustness check.
Both branches continue.

### Phase 2: Preprocessing sensitivity grid (C1, days 6-12)

One factor at a time from the reference pipeline, training on DF-only and FS-only, evaluating on all four methods, at least 2 seeds:

- Face detector: MTCNN / dlib / RetinaFace
- Margin: 1.0 / 1.3 / 2.0
- Storage: PNG / JPEG q95 / JPEG q75
- Alignment: on / off
- Frames per video: 10 / 20 / 32
- Resize interpolation: bilinear / bicubic / area
- Data source: mirror vs official release (**only once official access exists**)

Report delta-AUC per factor and mark every cell that crosses 0.5. Cap runtime with a frame subsample and early stopping (patience 4 on validation AUC).

### Phase 3: Mechanism (C2, days 8-15)

Steps 3.1-3.3 need no training and should run first.

1. Per crop, compute:
   - Laplacian variance
   - High-frequency energy ratio (radial FFT or DCT above a cutoff)
   - Inner-face vs border mean colour difference (Lab) and luminance difference
   - Gradient magnitude along the face-mask boundary (convex hull of landmarks)
   - JPEG blockiness
   - Noise-residual standard deviation
   - Raw face-box size before resize
2. For each method vs its paired real frames, compute a signed effect size (Cohen's d, or the AUC of the statistic alone used as a fake score).
3. **Sign prediction test:** for train method A and test method B, predict the direction as sign(d_A) x sign(d_B). Score the fraction of off-diagonal cells whose below/above-0.5 status is predicted correctly, and the Spearman correlation between predicted and observed AUC. Compare against a shuffled baseline.
4. Correlate each model's logits with each statistic on the source and target methods.
5. Grad-CAM on inverted cells, as a qualitative figure only.

**GATE 3:** If no statistic predicts the sign pattern better than chance, tell me. Fallback: paper becomes C1 plus C4, plus paired-training and blending-shortcut tests (Phase 4.2-4.3).

### Phase 4: Interventions (C3, days 12-19)

Train DF-only and FS-only (both architectures) with each, then rebuild the matrix:

1. Statistic-neutralizing augmentation: random blur/sharpen, common-resolution down-up resampling of all images, random JPEG recompression applied to both classes, colour matching of the face region to its border, random low-pass filtering to equalize radial spectra.
2. **Paired sampling:** each fake is batched with its own source real video.
3. Reference comparison: a simple self-blended-image augmentation on reals only (blend a real face with a transformed copy of itself), skipped if too costly.

**Success criterion:** below-0.5 cells rise with the bootstrap CI excluding the baseline mean, and in-domain video AUC drops by no more than 2 points.

### Phase 5: Architecture independence (C4, days 15-22)

- Backbones: Xception, EfficientNet-B4, ResNet-50, a pretrained ViT-S/16 (timm), a **pretrained** hybrid at the same input size as Xception (for example a timm ResNet+ViT hybrid), and if budget allows a CLIP ViT-B/16 with LayerNorm-only tuning.
- Same protocol, 3 seeds. If the compute budget is tight, restrict to training methods DF and FS and 4 backbones.
- Analyses:
  - Mantel test / Spearman between transfer matrices across architectures, vs across seeds of the same architecture (noise ceiling).
  - Two-way or mixed-effects ANOVA on off-diagonal AUC (train method, test method, architecture, seed). Report variance explained (eta squared).

### Phase 6: External validity (days 18-25)

- **If a licensed dataset is available** (for example Celeb-DF v2 after its access form): cross-dataset evaluation.
- **Non-FF++ track (fallback, or to run in parallel):** DeepFakeFace on Hugging Face (`HACHIMISDA/DeepFakeFace`; real images from IMDB-WIKI, fakes from SD Inpainting, InsightFace and SD v1.5 text-to-image) as a 3x3 transfer matrix, with OpenFake as an external test set. Before training, verify license and audit real-vs-fake resolution, JPEG tables and file sizes, because source asymmetry is a known shortcut (higher resolution looks "real").
- Repeat a reduced Phase 3-5 on whichever external data you use.

### Phase 7: Write-up assets (days 22-28)

Produce these from scripts, not by hand:

- F1: 4x4 transfer matrices with CIs, current pipeline vs reference pipeline
- F2: preprocessing-grid heatmap of delta-AUC, with 0.5-crossings marked
- F3: signed per-method statistics
- F4: predicted vs observed sign matrix
- F5: intervention before/after
- F6: architecture-similarity plot
- F7: Grad-CAM on inverted cells

Paper outline (4-6 pages, WIFS / ICASSP style): abstract, intro, related work, setup, results, analysis, limitations, data statement.

## 5. Statistical standards

- Primary metric is **video-level AUC**; frame-level AUC is secondary.
- Bootstrap over **videos** (at least 1000 resamples), 95% CI, for every reported AUC.
- At least **3 seeds**, reported as mean and std. Gaps inside the CI are not claims.
- Report every below-0.5 cell explicitly in the main table.
- Use the same test videos and the same frame-sampling for every model.
- Give every hybrid-vs-CNN comparison matched input size, pretraining status and parameter count, or say clearly that they are unmatched.

## 6. Reporting back to me

After each phase, append to `docs/PROGRESS.md`:
1. What you ran, with the git commit
2. Key numbers (with CIs) and where the files are
3. Anything that deviated from this plan and why
4. What you propose next

If a gate outcome is ambiguous, stop and ask me instead of guessing.
