#!/bin/bash
set -euo pipefail

echo "Submitting alpha sweep experiments..."

EXP_DIR="experiments/3__alpha_sweep"
TEACHER_CKPT="checkpoints/teacher_cnn_best.pt"

mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/logs/tmp"
mkdir -p "${EXP_DIR}/checkpoints"

if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
  exit 1
fi

sbatch "${EXP_DIR}/run_A01.slurm"
sbatch "${EXP_DIR}/run_A03.slurm"
sbatch "${EXP_DIR}/run_A05.slurm"
sbatch "${EXP_DIR}/run_A07.slurm"
sbatch "${EXP_DIR}/run_A09.slurm"

echo "All alpha sweep jobs submitted."