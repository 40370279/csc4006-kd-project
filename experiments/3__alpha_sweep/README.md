# Experiment 3: Alpha Sweep

This experiment studies the effect of alpha in knowledge distillation while keeping temperature fixed at 4.

## Goal

Evaluate how different alpha values affect student model performance while keeping the rest of the KD setup unchanged.

## Alpha Values Tested

- 0.1
- 0.3
- 0.5
- 0.7
- 0.9

## Setup

All runs use:

- the same teacher checkpoints from the root checkpoints/ directory
- the same seeds: 42, 123, 999
- the same student size: small
- the same optimisation and training hyperparameters
- the same temperature: 4.0

Only the alpha value is changed.

## Structure

experiments/3__alpha_sweep/
├── checkpoints/
├── logs/
│   └── tmp/
├── README.md
├── run_A01.slurm
├── run_A03.slurm
├── run_A05.slurm
├── run_A07.slurm
├── run_A09.slurm
└── run_all.sh

## Teacher Checkpoints

These runs read teacher checkpoints from:

checkpoints/teacher_cnn_seed42.pt
checkpoints/teacher_cnn_seed123.pt
checkpoints/teacher_cnn_seed999.pt

These are shared base teacher checkpoints.

## Outputs

This experiment writes its own outputs into:

- checkpoints: experiments/3__alpha_sweep/checkpoints/
- logs: experiments/3__alpha_sweep/logs/
- temporary per-seed logs: experiments/3__alpha_sweep/logs/tmp/

Student checkpoints are saved as:

- student_A01_seed{SEED}.pt
- student_A03_seed{SEED}.pt
- student_A05_seed{SEED}.pt
- student_A07_seed{SEED}.pt
- student_A09_seed{SEED}.pt

## How to Run

Run everything:

./experiments/3__alpha_sweep/run_all.sh

Or submit manually:

sbatch experiments/3__alpha_sweep/run_A01.slurm
sbatch experiments/3__alpha_sweep/run_A03.slurm
sbatch experiments/3__alpha_sweep/run_A05.slurm
sbatch experiments/3__alpha_sweep/run_A07.slurm
sbatch experiments/3__alpha_sweep/run_A09.slurm

## Expected Insight

This experiment should show how the balance between hard-label supervision and distillation supervision affects KD performance.

Typical interpretation:

- lower alpha gives more weight to hard labels
- higher alpha gives more weight to KD targets
- a middle value often works best

This experiment identifies which alpha is best for this setup at temperature 4.