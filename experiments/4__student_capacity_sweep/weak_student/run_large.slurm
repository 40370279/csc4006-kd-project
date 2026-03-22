#!/bin/bash
#SBATCH --job-name=cap_weak_large
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/4__student_capacity_sweep/logs/weak_large_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p experiments/4__student_capacity_sweep/checkpoints
mkdir -p experiments/4__student_capacity_sweep/logs/tmp

TEACHER_CKPT="checkpoints/teacher_cnn_best.pt"
if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
  exit 1
fi

echo "===== EXPERIMENT 4: WEAK STUDENT CAPACITY (large) ====="
echo "This job runs seeds: 42, 123, 999"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

SEEDS=(42 123 999)

ACC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"; do
  echo "----- SEED $SEED -----"

  SEED_LOG="experiments/4__student_capacity_sweep/logs/tmp/weak_large_seed_${SEED}_${SLURM_JOB_ID:-manual}.log"

  python3 -u scripts/train_student_weak_kd.py \
    --batch_size 64 \
    --lr 3e-4 \
    --epochs 50 \
    --patience 10 \
    --class_weight_gamma 0.5 \
    --alpha 0.5 \
    --temperature 4.0 \
    --student_size large \
    --num_workers 2 \
    --seed "$SEED" \
    --student_ckpt "experiments/4__student_capacity_sweep/checkpoints/weak_student_large_seed${SEED}.pt" \
    2>&1 | tee "$SEED_LOG"

  ACC_VAL=$(grep -F "Test accuracy (weak KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  MACRO_VAL=$(grep -F "Test macro-F1 (weak KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  WEIGHTED_VAL=$(grep -F "Test weighted-F1 (weak KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')

  if [[ -z "${ACC_VAL:-}" || -z "${MACRO_VAL:-}" || -z "${WEIGHTED_VAL:-}" ]]; then
    echo "ERROR: Failed to parse weak large metrics for seed $SEED"
    exit 1
  fi

  ACC_VALUES+=("$ACC_VAL")
  MACRO_VALUES+=("$MACRO_VAL")
  WEIGHTED_VALUES+=("$WEIGHTED_VAL")
done

ACC_STR="${ACC_VALUES[*]}"
MACRO_STR="${MACRO_VALUES[*]}"
WEIGHTED_STR="${WEIGHTED_VALUES[*]}"

echo "===== WEAK STUDENT LARGE SUMMARY ====="

python3 - <<EOF
import numpy as np
acc = np.array([float(x) for x in """$ACC_STR""".split()])
macro = np.array([float(x) for x in """$MACRO_STR""".split()])
weighted = np.array([float(x) for x in """$WEIGHTED_STR""".split()])
print("Student size : large")
print("Seeds        :", " ".join(map(str, [42, 123, 999])))
print(f"Accuracy     : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Macro F1     : {macro.mean():.4f} ± {macro.std():.4f}")
print(f"Weighted F1  : {weighted.mean():.4f} ± {weighted.std():.4f}")
EOF