#!/bin/bash
set -euo pipefail

echo "Submitting temperature sweep..."

EXP_DIR="experiments/2__temperature_sweep"
TEACHER_CKPT="checkpoints/teacher_cnn_best.pt"

mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/logs/tmp"
mkdir -p "${EXP_DIR}/checkpoints"

if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
  exit 1
fi

sbatch "${EXP_DIR}/run_T1.slurm"
sbatch "${EXP_DIR}/run_T2.slurm"
sbatch "${EXP_DIR}/run_T4.slurm"
sbatch "${EXP_DIR}/run_T8.slurm"
sbatch "${EXP_DIR}/run_T16.slurm"
sbatch "${EXP_DIR}/run_T32.slurm"

echo "All temperature sweep jobs submitted."