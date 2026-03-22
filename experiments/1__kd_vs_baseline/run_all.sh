#!/bin/bash
set -euo pipefail

echo "Submitting KD vs Baseline experiment..."

EXP_DIR="experiments/1__kd_vs_baseline"
TEACHER_CKPT="checkpoints/teacher_cnn_best.pt"

mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/logs/tmp"
mkdir -p "${EXP_DIR}/checkpoints"

echo "Submitting baseline student job..."
BASE_JOB=$(sbatch "${EXP_DIR}/normal_student/run_baseline.slurm")
echo "$BASE_JOB"

echo "Submitting weak baseline job..."
WEAK_BASE_JOB=$(sbatch "${EXP_DIR}/weak_student/run_weak_baseline.slurm")
echo "$WEAK_BASE_JOB"

if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
  echo "Baseline jobs were submitted, but KD jobs were not."
  exit 1
fi

echo "Submitting KD student job..."
KD_JOB=$(sbatch "${EXP_DIR}/normal_student/run_kd.slurm")
echo "$KD_JOB"

echo "Submitting weak KD job..."
WEAK_KD_JOB=$(sbatch "${EXP_DIR}/weak_student/run_weak_kd.slurm")
echo "$WEAK_KD_JOB"

echo "------------------------------------"
echo "All Experiment 1 jobs submitted."
echo "------------------------------------"