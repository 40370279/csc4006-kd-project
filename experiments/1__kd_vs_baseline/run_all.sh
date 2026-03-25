#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$PROJECT_DIR/experiments/1__kd_vs_baseline"

TEACHER_SLURM="$EXP_DIR/teacher/run_teacher.slurm"
BASELINE_SLURM="$EXP_DIR/student/run_baseline.slurm"
KD_SLURM="$EXP_DIR/student/run_kd.slurm"

cd "$PROJECT_DIR"

echo "===== EXPERIMENT 1 RUN ALL ====="
echo "Project dir : $PROJECT_DIR"
echo "Experiment  : $EXP_DIR"
echo "Start time  : $(date)"
echo

if [[ ! -f "$TEACHER_SLURM" ]]; then
  echo "ERROR: Missing teacher slurm file: $TEACHER_SLURM"
  exit 1
fi

if [[ ! -f "$BASELINE_SLURM" ]]; then
  echo "ERROR: Missing baseline slurm file: $BASELINE_SLURM"
  exit 1
fi

if [[ ! -f "$KD_SLURM" ]]; then
  echo "ERROR: Missing KD slurm file: $KD_SLURM"
  exit 1
fi

echo "Submitting teacher job..."
TEACHER_SUBMIT=$(sbatch "$TEACHER_SLURM")
echo "$TEACHER_SUBMIT"
TEACHER_JOB_ID=$(echo "$TEACHER_SUBMIT" | awk '{print $NF}')

if [[ -z "${TEACHER_JOB_ID:-}" ]]; then
  echo "ERROR: Failed to parse teacher job ID"
  exit 1
fi

echo
echo "Submitting baseline job..."
BASELINE_SUBMIT=$(sbatch "$BASELINE_SLURM")
echo "$BASELINE_SUBMIT"
BASELINE_JOB_ID=$(echo "$BASELINE_SUBMIT" | awk '{print $NF}')

if [[ -z "${BASELINE_JOB_ID:-}" ]]; then
  echo "ERROR: Failed to parse baseline job ID"
  exit 1
fi

echo
echo "Submitting KD job with dependency on teacher success..."
KD_SUBMIT=$(sbatch --dependency=afterok:${TEACHER_JOB_ID} "$KD_SLURM")
echo "$KD_SUBMIT"
KD_JOB_ID=$(echo "$KD_SUBMIT" | awk '{print $NF}')

if [[ -z "${KD_JOB_ID:-}" ]]; then
  echo "ERROR: Failed to parse KD job ID"
  exit 1
fi

echo
echo "===== SUBMISSION SUMMARY ====="
echo "Teacher job ID : $TEACHER_JOB_ID"
echo "Baseline job ID: $BASELINE_JOB_ID"
echo "KD job ID      : $KD_JOB_ID"
echo "KD dependency  : afterok:$TEACHER_JOB_ID"
echo
echo "Check queue with:"
echo "  squeue -u \$USER"
echo
echo "Finished submitting at: $(date)"