#!/usr/bin/env bash
set -euo pipefail

# Step 4 multistage:
#   S1 general Arabic -> S2 Qur'anic recitation -> S3 final IqraEval.
#
# Notes:
# - Use --reset-optimizer at stage boundaries to start each stage with a fresh
#   optimizer/scheduler, while still loading previous stage model weights.
# - If EveryAyah metadata is unavailable, point step4_stage2.yaml train_csv to
#   a fallback CSV before running this script.

CFG1="configs/step4_stage1.yaml"
CFG2="configs/step4_stage2.yaml"
CFG3="configs/step4_stage3.yaml"

OUT1="outputs/step4_stage1"
OUT2="outputs/step4_stage2"
OUT3="outputs/step4_stage3"

echo "[S1] Train general Arabic stage"
python src/training/train.py --config "${CFG1}" --stage S1

echo "[S2] Domain-adapt to Qur'anic recitation"
python src/training/train.py \
  --config "${CFG2}" \
  --stage S2 \
  --resume-from "${OUT1}/best.pt" \
  --reset-optimizer

echo "[S3] Final IqraEval adaptation"
python src/training/train.py \
  --config "${CFG3}" \
  --stage S3 \
  --resume-from "${OUT2}/best.pt" \
  --reset-optimizer

echo "[S3] Predict and evaluate"
python src/training/predict.py --config "${CFG3}"
python scripts/evaluate_predictions.py --pred "${OUT3}/predictions.csv" | tee "${OUT3}/eval_open_test.txt"

echo "Multistage run complete."
