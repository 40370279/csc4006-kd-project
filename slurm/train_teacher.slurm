#!/bin/bash
#SBATCH --job-name=teacher
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/teacher/teacher_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ===== DIRECTORIES =====
LOG_DIR="logs/teacher"
TMP_LOG_DIR="logs/teacher/tmp"
CKPT_DIR="checkpoints/teacher"

mkdir -p "$LOG_DIR" "$TMP_LOG_DIR" "$CKPT_DIR"

echo "===== TEACHER RUN ====="
echo "Seeds: 42, 123, 999"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo "Log dir: $LOG_DIR"
echo "Temp log dir: $TMP_LOG_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo

SEEDS=(42 123 999)

ACC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"
do
  echo "----- SEED $SEED -----"

  SEED_LOG="${TMP_LOG_DIR}/teacher_seed_${SEED}_${SLURM_JOB_ID:-manual}.log"

  python3 -u scripts/train_teacher.py \
    --batch_size 64 \
    --lr 8e-4 \
    --epochs 90 \
    --patience 18 \
    --warmup_epochs 5 \
    --min_lr_factor 0.02 \
    --weight_decay 1e-3 \
    --class_weight_gamma 0.20 \
    --sampler_power 0.35 \
    --label_smoothing 0.03 \
    --num_workers 2 \
    --seed "$SEED" \
    --teacher_size xlarge \
    --teacher_ckpt "${CKPT_DIR}/teacher_cnn_seed${SEED}.pt" \
    2>&1 | tee "$SEED_LOG"

  ACC_VAL=$(grep -F "Test Accuracy:" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  MACRO_VAL=$(grep -F "Test Macro-F1:" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  WEIGHTED_VAL=$(grep -F "Test Weighted-F1:" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')

  if [[ -z "${ACC_VAL:-}" || -z "${MACRO_VAL:-}" || -z "${WEIGHTED_VAL:-}" ]]; then
    echo "ERROR: Failed to parse metrics for seed $SEED"
    echo "Check log: $SEED_LOG"
    exit 1
  fi

  ACC_VALUES+=("$ACC_VAL")
  MACRO_VALUES+=("$MACRO_VAL")
  WEIGHTED_VALUES+=("$WEIGHTED_VAL")

  echo "Parsed metrics for seed $SEED:"
  echo "  Accuracy    = $ACC_VAL"
  echo "  Macro-F1    = $MACRO_VAL"
  echo "  Weighted-F1 = $WEIGHTED_VAL"
  echo
done

ACC_STR="${ACC_VALUES[*]}"
MACRO_STR="${MACRO_VALUES[*]}"
WEIGHTED_STR="${WEIGHTED_VALUES[*]}"

echo "===== TEACHER SUMMARY ====="

python3 - <<EOF
import numpy as np

acc = np.array([float(x) for x in """$ACC_STR""".split()])
macro = np.array([float(x) for x in """$MACRO_STR""".split()])
weighted = np.array([float(x) for x in """$WEIGHTED_STR""".split()])

print("Seeds        :", " ".join(map(str, [42, 123, 999])))
print(f"Accuracy     : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Macro F1     : {macro.mean():.4f} ± {macro.std():.4f}")
print(f"Weighted F1  : {weighted.mean():.4f} ± {weighted.std():.4f}")
EOF

echo
echo "Finished: $(date)"