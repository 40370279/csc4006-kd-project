#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/submit_all_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== JOB SUBMISSION SCRIPT ====="
echo "Time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo

echo "Submitting teacher job..."
teacher_job=$(sbatch slurm/train_teacher.slurm | awk '{print $4}')

if [[ -z "${teacher_job:-}" ]]; then
  echo "Failed to submit teacher job"
  exit 1
fi

echo "Teacher job ID:"
echo "  teacher: $teacher_job"

echo
echo "Submitting baseline student job..."
baseline_job=$(sbatch slurm/train_student_baseline.slurm | awk '{print $4}')

if [[ -z "${baseline_job:-}" ]]; then
  echo "Failed to submit baseline student job"
  exit 1
fi

echo "Baseline job ID:"
echo "  baseline: $baseline_job"

echo
echo "Submitting KD job with dependency on teacher..."
dep="afterok:${teacher_job}"
kd_job=$(sbatch --dependency=$dep slurm/train_student_kd.slurm | awk '{print $4}')

if [[ -z "${kd_job:-}" ]]; then
  echo "Failed to submit KD job"
  exit 1
fi

echo "KD job ID:"
echo "  kd: $kd_job"

echo
echo "Dependency used for KD job: $dep"

echo
echo "Submission log saved to: $LOG_FILE"
echo "Finished at: $(date)"