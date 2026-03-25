#!/bin/bash
#SBATCH --job-name=exp4_large
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/users/40370279/csc4006/code/experiments/4__student_capacity_sweep/logs/large_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/4__student_capacity_sweep"
LOG_DIR="$EXP_DIR/logs"
TMP_LOG_DIR="$LOG_DIR/tmp"
CKPT_DIR="$EXP_DIR/checkpoints"

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOG_DIR" "$TMP_LOG_DIR" "$CKPT_DIR"

echo "===== EXPERIMENT 4: LARGE STUDENT ====="
echo "Teacher checkpoints source: checkpoints/"
echo "Alpha = 0.5, Temperature = 4.0"
echo "Seeds: 42, 123, 999"
echo

SEEDS=(42 123 999)

ACC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"
do
  echo "----- SEED $SEED -----"

  SEED_LOG="$TMP_LOG_DIR/large_seed_${SEED}_${SLURM_JOB_ID:-manual}.log"

  python3 -u scripts/train_student_kd.py \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 35 \
    --patience 8 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --class_weight_gamma 0.35 \
    --sampler_gamma 0.90 \
    --class1_boost 1.75 \
    --focal_gamma 1.5 \
    --label_smoothing 0.05 \
    --alpha 0.5 \
    --temperature 4.0 \
    --num_workers 2 \
    --seed "$SEED" \
    --student_size large \
    --teacher_ckpt "checkpoints/teacher_cnn_seed${SEED}.pt" \
    --student_ckpt "$CKPT_DIR/student_large_seed${SEED}.pt" \
    2>&1 | tee "$SEED_LOG"

  ACC_VALUES+=($(grep -F "Test accuracy (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}'))
  MACRO_VALUES+=($(grep -F "Test macro-F1 (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}'))
  WEIGHTED_VALUES+=($(grep -F "Test weighted-F1 (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}'))

  echo
done

ACC_STR="${ACC_VALUES[*]}"
MACRO_STR="${MACRO_VALUES[*]}"
WEIGHTED_STR="${WEIGHTED_VALUES[*]}"

echo "===== SUMMARY: LARGE STUDENT ====="

python3 - <<EOF
import numpy as np
acc = np.array([float(x) for x in """$ACC_STR""".split()])
macro = np.array([float(x) for x in """$MACRO_STR""".split()])
weighted = np.array([float(x) for x in """$WEIGHTED_STR""".split()])

print("Model        : LARGE")
print(f"Accuracy     : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Macro F1     : {macro.mean():.4f} ± {macro.std():.4f}")
print(f"Weighted F1  : {weighted.mean():.4f} ± {weighted.std():.4f}")
EOF

echo "Finished at: $(date)"