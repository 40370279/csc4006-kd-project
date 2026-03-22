#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

mkdir -p logs

# Create timestamped log file for this submission script
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/submit_all_${TIMESTAMP}.log"

# Redirect ALL output (stdout + stderr) to log file AND terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== JOB SUBMISSION SCRIPT ====="
echo "Time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo

echo "Submitting teacher jobs..."
teacher_regular=$(sbatch slurm/train_teacher.slurm | awk '{print $4}')
teacher_weak=$(sbatch slurm/train_teacher_weak.slurm | awk '{print $4}')
teacher_strong=$(sbatch slurm/train_teacher_strong.slurm | awk '{print $4}')

echo "Teacher job IDs:"
echo "  regular: $teacher_regular"
echo "  weak:    $teacher_weak"
echo "  strong:  $teacher_strong"

echo
echo "Submitting baseline student jobs..."
baseline_regular=$(sbatch slurm/train_student_baseline.slurm | awk '{print $4}')
baseline_weak=$(sbatch slurm/train_student_baseline_weak.slurm | awk '{print $4}')

echo "Baseline job IDs:"
echo "  regular baseline: $baseline_regular"
echo "  weak baseline:    $baseline_weak"

echo
echo "Submitting KD jobs with dependency on all teacher jobs..."
dep="afterok:${teacher_regular}:${teacher_weak}:${teacher_strong}"

kd_regular=$(sbatch --dependency=$dep slurm/train_student_kd.slurm | awk '{print $4}')
kd_weak=$(sbatch --dependency=$dep slurm/train_student_kd_weak.slurm | awk '{print $4}')

echo "KD job IDs:"
echo "  regular KD: $kd_regular"
echo "  weak KD:    $kd_weak"

echo
echo "Dependency used for KD jobs: $dep"

echo
echo "Submission log saved to: $LOG_FILE"
echo "Finished at: $(date)"