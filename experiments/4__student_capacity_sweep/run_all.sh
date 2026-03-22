#!/bin/bash
set -euo pipefail

echo "Submitting student capacity sweep..."

EXP_DIR="experiments/4__student_capacity_sweep"
TEACHER_CKPT="checkpoints/teacher_cnn_best.pt"

mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/logs/tmp"
mkdir -p "${EXP_DIR}/checkpoints"

if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
  exit 1
fi

sbatch "${EXP_DIR}/weak_student/run_small.slurm"
sbatch "${EXP_DIR}/weak_student/run_medium.slurm"
sbatch "${EXP_DIR}/weak_student/run_large.slurm"

sbatch "${EXP_DIR}/normal_student/run_small.slurm"
sbatch "${EXP_DIR}/normal_student/run_medium.slurm"
sbatch "${EXP_DIR}/normal_student/run_large.slurm"

echo "All student capacity jobs submitted."