#!/usr/bin/env bash
# Full experiment matrix. Adjust DATA_ROOT before running.
# Total: 12 training runs (2 models x [1 all-methods + 4 single-method] + 2 c40 runs)
# and ~40 evaluations. Budget several GPU-days for the full matrix, or trim
# FRAMES/EPOCHS for a course-scale version.
set -e

DATA_ROOT=${DATA_ROOT:-/data/ffpp_faces}
SPLITS=${SPLITS:-splits}
METHODS=(Deepfakes Face2Face FaceSwap NeuralTextures)
MODELS=(xception cnn_vit)

# ---------- Experiment 1: baseline comparison (all methods, c23) ----------
for M in "${MODELS[@]}"; do
  python -m src.train --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
    --model "$M" --methods "${METHODS[@]}" --compression c23 \
    --out "runs/${M}_all_c23" --resume
  python -m src.evaluate --checkpoint "runs/${M}_all_c23/best.pt" \
    --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
    --methods "${METHODS[@]}" --compression c23 \
    --out "results/${M}_all_to_all_c23.json"
done

# ---------- Experiment 2: cross-manipulation generalization ----------
# Train on ONE method, test on EVERY method (4x4 matrix per model).
for M in "${MODELS[@]}"; do
  for TRAIN in "${METHODS[@]}"; do
    python -m src.train --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
      --model "$M" --methods "$TRAIN" --compression c23 \
      --out "runs/${M}_${TRAIN}_c23" --resume
    for TEST in "${METHODS[@]}"; do
      python -m src.evaluate --checkpoint "runs/${M}_${TRAIN}_c23/best.pt" \
        --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
        --methods "$TEST" --compression c23 \
        --out "results/${M}_${TRAIN}_to_${TEST}_c23.json"
    done
  done
done

# ---------- Experiment 3: compression robustness ----------
# Models trained on c23 (all methods), tested on c0/c23/c40.
for M in "${MODELS[@]}"; do
  for COMP in c0 c23 c40; do
    python -m src.evaluate --checkpoint "runs/${M}_all_c23/best.pt" \
      --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
      --methods "${METHODS[@]}" --compression "$COMP" \
      --out "results/${M}_all_c23_to_all_${COMP}.json"
  done
  # Also train on c40 to test the reverse direction.
  python -m src.train --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
    --model "$M" --methods "${METHODS[@]}" --compression c40 \
    --out "runs/${M}_all_c40" --resume
  for COMP in c0 c23 c40; do
    python -m src.evaluate --checkpoint "runs/${M}_all_c40/best.pt" \
      --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
      --methods "${METHODS[@]}" --compression "$COMP" \
      --out "results/${M}_all_c40_to_all_${COMP}.json"
  done
done

# ---------- Aggregate ----------
python -m analyze.analyze --results-dir results --out-dir analysis
