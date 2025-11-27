#!/bin/bash
#SBATCH --job-name=student_kd_v100
#SBATCH -p k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/student_kd_%j.out

cd /users/40370279/csc4006/code
python3 -u -m src.train_student_kd
