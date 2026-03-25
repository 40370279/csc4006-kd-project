#!/bin/bash
set -euo pipefail

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/4__student_capacity_sweep"

cd "$PROJECT_DIR"

echo "===== SUBMITTING EXPERIMENT 4: STUDENT CAPACITY ====="

mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/logs/tmp"

for SIZE in small medium large
do
  JOB_ID=$(sbatch "$EXP_DIR/student/run_${SIZE}.slurm" | awk '{print $4}')
  echo "Submitted $SIZE -> Job $JOB_ID"
done