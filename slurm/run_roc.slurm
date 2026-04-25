#!/bin/bash
#SBATCH --job-name=roc_curves
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/roc/roc_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p logs/roc
mkdir -p results/roc_curves

echo "===== ROC CURVE RUN ====="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo

python3 -u -m src.utils.plot_roc_curves

echo
echo "Finished: $(date)"