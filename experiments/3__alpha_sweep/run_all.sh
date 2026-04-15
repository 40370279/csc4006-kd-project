#!/bin/bash
set -euo pipefail

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/3__alpha_sweep"

cd "$PROJECT_DIR"

echo "===== SUBMITTING EXPERIMENT 3: ALPHA SWEEP ====="

mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/logs/tmp"

FILES=(
  "run_A01.slurm"
  "run_A03.slurm"
  "run_A05.slurm"
  "run_A07.slurm"
  "run_A09.slurm"
)

for FILE in "${FILES[@]}"
do
  SBATCH_OUTPUT=$(sbatch "$EXP_DIR/$FILE")
  JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $4}')

  if [[ -z "${JOB_ID:-}" ]]; then
    echo "Failed to parse job ID for $FILE"
    echo "sbatch output: $SBATCH_OUTPUT"
    exit 1
  fi

  echo "Submitted $FILE -> Job $JOB_ID"
done