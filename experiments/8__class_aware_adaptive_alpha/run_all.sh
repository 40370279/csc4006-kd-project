#!/bin/bash
set -euo pipefail

EXP_DIR="experiments/8__class_aware_adaptive_alpha"

echo "===== RUNNING EXPERIMENT 8 ====="
echo "1) Fixed alpha KD"
sbatch "$EXP_DIR/run_fixed_alpha.slurm"

echo "2) Class-aware adaptive alpha KD"
sbatch "$EXP_DIR/run_class_aware_adaptive_alpha.slurm"