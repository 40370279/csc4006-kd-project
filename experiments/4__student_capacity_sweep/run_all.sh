#!/bin/bash
set -euo pipefail

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/4__student_capacity_sweep"

cd "$PROJECT_DIR"

echo "===== SUBMITTING EXPERIMENT 4: STUDENT CAPACITY ====="

mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/logs/tmp"

for SIZE in small medium large
do
  SBATCH_OUTPUT=$(sbatch "$EXP_DIR/student/run_${SIZE}.slurm")
  JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $4}')

  if [[ -z "${JOB_ID:-}" ]]; then
    echo "Failed to parse job ID for $SIZE"
    echo "sbatch output: $SBATCH_OUTPUT"
    exit 1
  fi

  echo "Submitted $SIZE -> Job $JOB_ID"
done