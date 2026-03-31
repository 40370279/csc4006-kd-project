#!/bin/bash
#SBATCH --job-name=exp8_classaware
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/8__class_aware_adaptive_alpha/logs/classaware_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="."
EXP_DIR="experiments/8__class_aware_adaptive_alpha"
LOG_DIR="$EXP_DIR/logs"
TMP_LOG_DIR="$LOG_DIR/tmp"
CKPT_DIR="$EXP_DIR/checkpoints"

cd "$PROJECT_DIR"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOG_DIR" "$TMP_LOG_DIR" "$CKPT_DIR"

echo "===== EXPERIMENT 8B: CLASS-AWARE ADAPTIVE ALPHA KD ====="
echo "Experiment dir: $EXP_DIR"
echo "Seeds: 42, 123, 999"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo

SEEDS=(42 123 999)

ACC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"
do
  echo "----- SEED $SEED -----"

  SEED_LOG="${TMP_LOG_DIR}/classaware_seed_${SEED}_${SLURM_JOB_ID:-manual}.log"
  TEACHER_CKPT="experiments/1__kd_vs_baseline/checkpoints/teacher_cnn_seed${SEED}.pt"
  STUDENT_CKPT="${CKPT_DIR}/student_classaware_alpha_seed${SEED}.pt"

  if [[ ! -f "$TEACHER_CKPT" ]]; then
    echo "ERROR: Teacher checkpoint not found for seed $SEED"
    echo "Expected: $TEACHER_CKPT"
    exit 1
  fi

  python3 -u scripts/train_student_kd_class_aware_alpha.py \
    --batch_size 64 \
    --lr 8e-4 \
    --epochs 70 \
    --patience 15 \
    --weight_decay 2e-3 \
    --grad_clip 1.0 \
    --class_weight_gamma 0.0 \
    --sampler_power 0.35 \
    --label_smoothing 0.0 \
    --alpha_start 0.10 \
    --alpha_end 0.10 \
    --temperature 1.0 \
    --logit_mse_weight 0.03 \
    --feature_mse_weight 0.03 \
    --projector_lr_scale 1.0 \
    --num_workers 2 \
    --seed "$SEED" \
    --student_size medium \
    --teacher_ckpt "$TEACHER_CKPT" \
    --student_ckpt "$STUDENT_CKPT" \
    --alpha_mode classwise_adaptive \
    --adaptive_alpha_min 0.05 \
    --adaptive_alpha_max 0.30 \
    2>&1 | tee "$SEED_LOG"

  ACC_VAL=$(grep -F "Test accuracy (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  MACRO_VAL=$(grep -F "Test macro-F1 (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  WEIGHTED_VAL=$(grep -F "Test weighted-F1 (KD student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')

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

echo "===== EXPERIMENT 8B: CLASS-AWARE ADAPTIVE ALPHA SUMMARY ====="

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