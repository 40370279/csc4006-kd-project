#!/bin/bash
#SBATCH --job-name=exp4_baseline_small
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/4__student_capacity_sweep/logs/studentbaseline_small_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="experiments/4__student_capacity_sweep"
LOG_DIR="$EXP_DIR/logs"
TMP_LOG_DIR="$LOG_DIR/tmp"
CKPT_DIR="$EXP_DIR/checkpoints"

cd "$PROJECT_DIR"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOG_DIR" "$TMP_LOG_DIR" "$CKPT_DIR"

echo "===== EXPERIMENT 4: STUDENT BASELINE SMALL RUN ====="
echo "Experiment dir: $EXP_DIR"
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
AUC_VALUES=()
MACRO_VALUES=()
WEIGHTED_VALUES=()

for SEED in "${SEEDS[@]}"
do
  echo "----- SEED $SEED -----"

  SEED_LOG="${TMP_LOG_DIR}/studentbaseline_small_seed_${SEED}_${SLURM_JOB_ID:-manual}.log"
  STUDENT_CKPT="${CKPT_DIR}/studentbaseline_small_seed${SEED}.pt"

  python3 -u scripts/train_student_baseline.py \
    --batch_size 64 \
    --lr 8e-4 \
    --epochs 70 \
    --patience 15 \
    --weight_decay 2e-3 \
    --grad_clip 1.0 \
    --class_weight_gamma 0.0 \
    --sampler_power 0.35 \
    --label_smoothing 0.0 \
    --num_workers 2 \
    --seed "$SEED" \
    --student_size small \
    --student_ckpt "$STUDENT_CKPT" \
    2>&1 | tee "$SEED_LOG"

  ACC_VAL=$(grep -F "Test accuracy (baseline student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  AUC_VAL=$(grep -F "Test macro-AUC (baseline student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  MACRO_VAL=$(grep -F "Test macro-F1 (baseline student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')
  WEIGHTED_VAL=$(grep -F "Test weighted-F1 (baseline student):" "$SEED_LOG" | tail -n 1 | awk '{print $NF}')

  if [[ -z "${ACC_VAL:-}" || -z "${AUC_VAL:-}" || -z "${MACRO_VAL:-}" || -z "${WEIGHTED_VAL:-}" ]]; then
    echo "ERROR: Failed to parse baseline metrics for seed $SEED"
    echo "Check log: $SEED_LOG"
    exit 1
  fi

  ACC_VALUES+=("$ACC_VAL")
  AUC_VALUES+=("$AUC_VAL")
  MACRO_VALUES+=("$MACRO_VAL")
  WEIGHTED_VALUES+=("$WEIGHTED_VAL")

  echo "Parsed metrics for seed $SEED:"
  echo "  Accuracy    = $ACC_VAL"
  echo "  Macro-AUC   = $AUC_VAL"
  echo "  Macro-F1    = $MACRO_VAL"
  echo "  Weighted-F1 = $WEIGHTED_VAL"
  echo
done

ACC_STR="${ACC_VALUES[*]}"
AUC_STR="${AUC_VALUES[*]}"
MACRO_STR="${MACRO_VALUES[*]}"
WEIGHTED_STR="${WEIGHTED_VALUES[*]}"

echo "===== EXPERIMENT 4: STUDENT BASELINE SMALL SUMMARY ====="

python3 - <<EOF
import numpy as np

acc = np.array([float(x) for x in """$ACC_STR""".split()])
auc = np.array([float(x) for x in """$AUC_STR""".split()])
macro = np.array([float(x) for x in """$MACRO_STR""".split()])
weighted = np.array([float(x) for x in """$WEIGHTED_STR""".split()])

print("Seeds        :", " ".join(map(str, [42, 123, 999])))
print(f"Accuracy     : {acc.mean():.4f} ± {acc.std():.4f}")
print(f"Macro AUC    : {auc.mean():.4f} ± {auc.std():.4f}")
print(f"Macro F1     : {macro.mean():.4f} ± {macro.std():.4f}")
print(f"Weighted F1  : {weighted.mean():.4f} ± {weighted.std():.4f}")
EOF

echo
echo "Finished: $(date)"