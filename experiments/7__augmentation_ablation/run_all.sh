#!/bin/bash
set -euo pipefail

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/7__augmentation_ablation"

cd "$PROJECT_DIR"

echo "===== SUBMITTING EXPERIMENT 7: AUGMENTATION ABLATION ====="

mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/logs/tmp"

for SCRIPT in \
  studentbaseline/run_noaug.slurm \
  studentbaseline/run_aug.slurm \
  studentkd/run_noaug.slurm \
  studentkd/run_aug.slurm
do
  SBATCH_OUTPUT=$(sbatch "$EXP_DIR/$SCRIPT")
  JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $4}')

  if [[ -z "${JOB_ID:-}" ]]; then
    echo "Failed to parse job ID for $SCRIPT"
    echo "sbatch output: $SBATCH_OUTPUT"
    exit 1
  fi

  echo "Submitted $SCRIPT -> Job $JOB_ID"
done