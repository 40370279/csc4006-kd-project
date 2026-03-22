#!/bin/bash
set -euo pipefail

echo "Submitting Experiment 6: Imbalance Loss Study..."

sbatch experiments/6__imbalance_loss_study/run_ce.slurm
sbatch experiments/6__imbalance_loss_study/run_focal_gamma1.slurm
sbatch experiments/6__imbalance_loss_study/run_focal_gamma2.slurm
sbatch experiments/6__imbalance_loss_study/run_focal_gamma3.slurm