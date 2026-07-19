# Deepfake Detection: XceptionNet vs CNN-ViT Hybrid on FF++

Comparative study of a CNN baseline (XceptionNet) and a hybrid CNN-Vision
Transformer on FaceForensics++, focused on **cross-manipulation generalization**
(train on one forgery type, test on another) and **compression robustness**
(c0/c23/c40).

## Repo layout

```
src/data/preprocess.py   video -> face crops (MTCNN)
src/data/splits.py       official FF++ train/val/test splits
src/data/dataset.py      PyTorch dataset over face crops
src/models/xception.py   baseline (timm legacy_xception, pretrained)
src/models/cnn_vit.py    hybrid: ResNet-50 features -> 6-layer Transformer encoder
src/train.py             training with AMP, class balancing, early stop, resume
src/evaluate.py          evaluate any checkpoint on any manip/compression
scripts/run_all_experiments.sh  full experiment matrix
analyze/analyze.py       tables, generalization heatmaps, gap analysis
```

## 1. Setup (VS Code, local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Requires an NVIDIA GPU for realistic training times. No GPU? See
"Running on Colab" below — same code, unchanged.

## 2. Get the dataset

**Official route:** fill the FaceForensics++ access form
(https://github.com/ondyari/FaceForensics) — approval usually takes a few days —
then use their `download.py`. For this project you need:
`original_sequences/youtube` + the four `manipulated_sequences` methods, at
c23 and c40 (add c0/raw if disk allows; c23+c40 ≈ 40 GB, raw is ~500 GB).

**Faster route for a course project:** Kaggle hosts FF++ c23/c40 mirrors
(search "FaceForensics++"). Arrange the files into the official layout above.

Also download `train.json`, `val.json`, `test.json` from
https://github.com/ondyari/FaceForensics/tree/master/dataset/splits into a
`splits/` folder. **Using the official video-level splits matters** — random
frame-level splits leak identities between train and test and inflate accuracy.

## 3. Preprocess (once per compression level)

```bash
python -m src.data.preprocess --ffpp-root /data/FFpp --out-root /data/ffpp_faces \
    --compressions c23 c40 --frames-per-video 20
```

~20 frames/video × 5,000 videos/compression ≈ 100k crops per compression level.

## 4. Run experiments

Everything is driven by two scripts. Single run example:

```bash
# Train hybrid on Deepfakes only (c23)
python -m src.train --data-root /data/ffpp_faces --model cnn_vit \
    --methods Deepfakes --compression c23 --out runs/cnn_vit_DF_c23

# Test that model on Face2Face -> the cross-manipulation number
python -m src.evaluate --checkpoint runs/cnn_vit_DF_c23/best.pt \
    --data-root /data/ffpp_faces --methods Face2Face --compression c23 \
    --out results/cnn_vit_DF_to_F2F_c23.json
```

Full matrix (12 trainings + ~40 evals):

```bash
DATA_ROOT=/data/ffpp_faces bash scripts/run_all_experiments.sh
```

| Experiment | What it shows |
|---|---|
| 1. Baseline: both models trained+tested on all 4 methods, c23 | headline accuracy/F1/AUC comparison |
| 2. Cross-manipulation: 4×4 train/test matrix per model | the novelty — who survives unseen forgery types |
| 3. Compression: train c23 → test c0/c23/c40 (and train c40 →) | robustness to real-world compression |

Training is resumable (`--resume`) — safe to interrupt.

## 5. Analyze

```bash
python -m analyze.analyze --results-dir results --out-dir analysis
```

Produces `baseline_comparison.csv`, per-model 4×4 AUC matrices,
`cross_manipulation_heatmaps.png` (side-by-side, paper-ready),
`generalization_gap.csv` (diagonal vs off-diagonal AUC — your key result), and
`compression_robustness.csv`.

The claim to test: both models score high on the diagonal (same-manipulation),
but the hybrid's off-diagonal AUC drops less, i.e. a smaller generalization gap.

## Running on Colab

```python
!git clone <your-repo-url> && cd deepfake-detection && pip install -r requirements.txt
# mount Drive, point --data-root at preprocessed crops stored on Drive
from google.colab import drive; drive.mount('/content/drive')
!cd deepfake-detection && python -m src.train --data-root /content/drive/MyDrive/ffpp_faces \
    --model xception --methods Deepfakes --compression c23 \
    --out /content/drive/MyDrive/runs/xception_DF_c23 --resume
```

Keep `--out` on Drive so checkpoints survive session timeouts; `--resume`
continues where it stopped. Preprocess once (locally or in one Colab session)
and store the crops on Drive — never re-extract per session.

## Scaling down for deadlines

Full matrix ≈ several GPU-days. Course-scale options, in order of preference:
fewer frames per video (10), fewer epochs (8, early stopping usually triggers
first), or restrict Experiment 2 to 2 train methods × 4 test methods. Don't cut
the number of *videos* — video diversity is what makes the metrics trustworthy.

## Protocol notes (worth citing in the report)

- Video-level metrics use mean frame probability per video (standard FF++ protocol).
- Real/fake imbalance handled by a weighted sampler, not by dropping data.
- Both models are ImageNet-pretrained; Xception uses 299×299, the hybrid 224×224.
- Same optimizer, schedule, and data for both models — the architecture is the
  only variable.
