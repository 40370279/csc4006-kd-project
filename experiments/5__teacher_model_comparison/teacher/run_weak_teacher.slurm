#!/bin/bash
#SBATCH --job-name=exp5_weak
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/users/40370279/csc4006/code/experiments/5__teacher_model_comparison/logs/weak_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/5__teacher_model_comparison"
LOG_DIR="$EXP_DIR/logs"
TMP_LOG_DIR="$LOG_DIR/tmp"
CKPT_DIR="$EXP_DIR/checkpoints"

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOG_DIR" "$TMP_LOG_DIR" "$CKPT_DIR"

echo "===== EXPERIMENT 5: WEAK TEACHER ====="

SEEDS=(42 123 999)

ACC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"
do
  echo "----- SEED $SEED -----"

  TEACHER_CKPT="$CKPT_DIR/teacher_weak_seed${SEED}.pt"
  STUDENT_CKPT="$CKPT_DIR/student_weak_seed${SEED}.pt"
  SEED_LOG="$TMP_LOG_DIR/weak_seed_${SEED}.log"

  # --- TRAIN WEAK TEACHER ---
  python3 -u scripts/train_teacher.py \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 60 \
    --patience 10 \
    --weight_decay 5e-4 \
    --seed "$SEED" \
    --teacher_ckpt "$TEACHER_CKPT" \
    2>&1 | tee "$SEED_LOG"

  # --- TRAIN STUDENT WITH KD ---
  python3 -u scripts/train_student_kd.py \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 35 \
    --patience 8 \
    --weight_decay 1e-3 \
    --alpha 0.5 \
    --temperature 4.0 \
    --seed "$SEED" \
    --student_size medium \
    --teacher_ckpt "$TEACHER_CKPT" \
    --student_ckpt "$STUDENT_CKPT" \
    2>&1 | tee -a "$SEED_LOG"

  ACC_VALUES+=($(grep -F "Test accuracy (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}'))
  MACRO_VALUES+=($(grep -F "Test macro-F1 (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}'))
  WEIGHTED_VALUES+=($(grep -F "Test weighted-F1 (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}'))
done

ACC_STR="${ACC_VALUES[*]}"
MACRO_STR="${MACRO_VALUES[*]}"
WEIGHTED_STR="${WEIGHTED_VALUES[*]}"

python3 - <<EOF
import numpy as np
acc = np.array([float(x) for x in """$ACC_STR""".split()])
macro = np.array([float(x) for x in """$MACRO_STR""".split()])
weighted = np.array([float(x) for x in """$WEIGHTED_STR""".split()])

print("Teacher: WEAK")
print(f"Accuracy     : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Macro F1     : {macro.mean():.4f} ± {macro.std():.4f}")
print(f"Weighted F1  : {weighted.mean():.4f} ± {weighted.std():.4f}")
EOF