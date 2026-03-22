#!/bin/bash
#SBATCH --job-name=imbalance_focal_g1
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/6__imbalance_loss_study/logs/focal_g1_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p experiments/6__imbalance_loss_study/checkpoints
mkdir -p experiments/6__imbalance_loss_study/logs/tmp

TEACHER_CKPT="checkpoints/teacher_cnn_strong_best.pt"

echo "===== EXPERIMENT 6: FOCAL LOSS (gamma=1) ====="

SEEDS=(42 123 999)

ACC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"; do
  echo "----- SEED $SEED -----"

  SEED_LOG="experiments/6__imbalance_loss_study/logs/tmp/focal_g1_seed_${SEED}.log"

  python3 -u scripts/train_student_kd_focal.py \
    --teacher_ckpt "$TEACHER_CKPT" \
    --student_size medium \
    --loss_type focal \
    --focal_gamma 1.0 \
    --alpha 0.5 \
    --temperature 4.0 \
    --seed "$SEED" \
    --student_ckpt "experiments/6__imbalance_loss_study/checkpoints/focal_g1_seed${SEED}.pt" \
    2>&1 | tee "$SEED_LOG"

  ACC_VALUES+=($(grep "Test accuracy (KD student):" "$SEED_LOG" | awk '{print $NF}'))
  MACRO_VALUES+=($(grep "Test macro-F1 (KD student):" "$SEED_LOG" | awk '{print $NF}'))
  WEIGHTED_VALUES+=($(grep "Test weighted-F1 (KD student):" "$SEED_LOG" | awk '{print $NF}'))

done

echo "===== SUMMARY: FOCAL (gamma=1) ====="

python3 - <<EOF
import numpy as np
acc = np.array([float(x) for x in "${ACC_VALUES[*]}".split()])
macro = np.array([float(x) for x in "${MACRO_VALUES[*]}".split()])
weighted = np.array([float(x) for x in "${WEIGHTED_VALUES[*]}".split()])

print(f"Accuracy     : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Macro F1     : {macro.mean():.4f} ± {macro.std():.4f}")
print(f"Weighted F1  : {weighted.mean():.4f} ± {weighted.std():.4f}")
EOF