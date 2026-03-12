#!/bin/bash
#SBATCH --job-name=student_base_v100
#SBATCH -p k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/student_base_%j.out

module purge
module load python3/3.10.5/gcc-9.3.0

cd /users/40370279/csc4006/code
export PYTHONPATH=/users/40370279/csc4006/code:$PYTHONPATH

python3 -u scripts/train_student_baseline.py