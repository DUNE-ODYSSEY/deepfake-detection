#!/usr/bin/env bash
# Full study minus official compression robustness (no c0/c40 access yet):
#   Exp 1: baseline comparison, both models, all 4 methods, c23
#   Exp 2: cross-manipulation 4x4 matrix, both models, c23
#   Exp 3 (proxy): c23-trained baselines evaluated on locally re-transcoded
#                  c40proxy test set (see scripts/make_c40proxy.py for caveats)
set -e

PY=${PY:-./.venv/Scripts/python.exe}
DATA_ROOT=${DATA_ROOT:-C:/ffpp_data/ffpp_faces}
SPLITS=${SPLITS:-splits}
METHODS=(Deepfakes Face2Face FaceSwap NeuralTextures)
MODELS=(xception cnn_vit)

mkdir -p runs results logs

# ---------- Experiment 1: baseline comparison (all methods, c23) ----------
for M in "${MODELS[@]}"; do
  $PY -m src.train --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
    --model "$M" --methods "${METHODS[@]}" --compression c23 \
    --out "runs/${M}_all_c23" --resume
  $PY -m src.evaluate --checkpoint "runs/${M}_all_c23/best.pt" \
    --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
    --methods "${METHODS[@]}" --compression c23 \
    --out "results/${M}_all_to_all_c23.json"
done

# ---------- Experiment 2: cross-manipulation generalization ----------
for M in "${MODELS[@]}"; do
  for TRAIN in "${METHODS[@]}"; do
    $PY -m src.train --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
      --model "$M" --methods "$TRAIN" --compression c23 \
      --out "runs/${M}_${TRAIN}_c23" --resume
    for TEST in "${METHODS[@]}"; do
      $PY -m src.evaluate --checkpoint "runs/${M}_${TRAIN}_c23/best.pt" \
        --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
        --methods "$TEST" --compression c23 \
        --out "results/${M}_${TRAIN}_to_${TEST}_c23.json"
    done
  done
done

# ---------- Experiment 3 (proxy): compression robustness ----------
for M in "${MODELS[@]}"; do
  $PY -m src.evaluate --checkpoint "runs/${M}_all_c23/best.pt" \
    --data-root "$DATA_ROOT" --splits-dir "$SPLITS" \
    --methods "${METHODS[@]}" --compression c40proxy \
    --out "results/${M}_all_c23_to_all_c40proxy.json"
done

# ---------- Aggregate ----------
$PY -m analyze.analyze --results-dir results --out-dir analysis
echo "STUDY COMPLETE"
