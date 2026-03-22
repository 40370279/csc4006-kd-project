#!/bin/bash
set -euo pipefail

echo "Submitting teacher comparison experiment..."

sbatch experiments/5__teacher_model_comparison/run_normal_teacher.slurm
sbatch experiments/5__teacher_model_comparison/run_weak_teacher.slurm
sbatch experiments/5__teacher_model_comparison/run_strong_teacher.slurm