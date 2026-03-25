# Experiment 2: Temperature Sweep

This experiment studies the effect of temperature in knowledge distillation.

## Goal

Evaluate how different temperature values affect student model performance while keeping the rest of the setup fixed.

## Temperatures Tested

- 1
- 2
- 4
- 8
- 16
- 32

## Setup

All runs use:

- the same teacher checkpoints from the root `checkpoints/` directory
- the same seeds: 42, 123, 999
- the same student size: small
- the same optimisation and training hyperparameters

Only the temperature value is changed.

## Structure

experiments/2__temperature_sweep/
├── checkpoints/
├── logs/
│   └── tmp/
├── README.md
├── run_all.sh
├── run_T1.slurm
├── run_T2.slurm
├── run_T4.slurm
├── run_T8.slurm
├── run_T16.slurm
└── run_T32.slurm

## Teacher Checkpoints

These runs read teacher checkpoints from:

checkpoints/teacher_cnn_seed42.pt  
checkpoints/teacher_cnn_seed123.pt  
checkpoints/teacher_cnn_seed999.pt

These are shared base teacher checkpoints, not experiment-local teacher checkpoints.

## Outputs

This experiment writes its own outputs into:

- checkpoints: experiments/2__temperature_sweep/checkpoints/
- logs: experiments/2__temperature_sweep/logs/
- temporary per-seed logs: experiments/2__temperature_sweep/logs/tmp/

Student checkpoints are saved as:

- student_T1_seed{SEED}.pt
- student_T2_seed{SEED}.pt
- student_T4_seed{SEED}.pt
- student_T8_seed{SEED}.pt
- student_T16_seed{SEED}.pt
- student_T32_seed{SEED}.pt

## How to Run

Run everything:

./experiments/2__temperature_sweep/run_all.sh

Or submit manually:

sbatch experiments/2__temperature_sweep/run_T1.slurm
sbatch experiments/2__temperature_sweep/run_T2.slurm
sbatch experiments/2__temperature_sweep/run_T4.slurm
sbatch experiments/2__temperature_sweep/run_T8.slurm
sbatch experiments/2__temperature_sweep/run_T16.slurm
sbatch experiments/2__temperature_sweep/run_T32.slurm

## Expected Insight

This experiment should show how softer versus sharper teacher distributions affect KD performance.

Typical interpretation:

- lower temperature gives sharper targets
- higher temperature gives softer targets
- a middle temperature often works best

This experiment identifies which temperature is best for this setup.