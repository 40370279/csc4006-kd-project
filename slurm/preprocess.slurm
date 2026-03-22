#!/bin/bash
#SBATCH --job-name=preprocess_ptbxl
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/40370279/csc4006/code/logs/preprocess_ptbxl_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p logs processed

echo "===== PREPROCESS PTB-XL RUN ====="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo

python3 -u scripts/preprocess_ptbxl.py

echo
echo "Finished at: $(date)"