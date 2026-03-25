# Experiment 4: Student Capacity Sweep

This experiment studies the effect of student model capacity in knowledge distillation.

## Goal

Evaluate how student model size impacts performance under a fixed KD setup.

Specifically, this experiment investigates:

- how performance scales with model capacity
- whether smaller models benefit more from KD
- whether larger models saturate performance

## Models Tested

- small
- medium
- large

These correspond to different student architectures defined in the model implementation.

## Setup

All runs use:

- teacher checkpoints from the root checkpoints/ directory
- the same seeds: 42, 123, 999
- identical training hyperparameters
- identical data processing and augmentations

KD configuration is FIXED:

- alpha = 0.5
- temperature = 4.0

Only the student model size is changed.

## Structure

experiments/4__student_capacity_sweep/
├── checkpoints/
├── logs/
│   └── tmp/
├── student/
│   ├── run_small.slurm
│   ├── run_medium.slurm
│   └── run_large.slurm
├── README.md
└── run_all.sh

## Teacher Checkpoints

All runs use shared teacher checkpoints from:

checkpoints/teacher_cnn_seed42.pt
checkpoints/teacher_cnn_seed123.pt
checkpoints/teacher_cnn_seed999.pt

These are NOT experiment-specific.

## Outputs

This experiment writes outputs into:

- checkpoints: experiments/4__student_capacity_sweep/checkpoints/
- logs: experiments/4__student_capacity_sweep/logs/
- temporary logs: experiments/4__student_capacity_sweep/logs/tmp/

Student checkpoints are saved as:

- student_small_seed{SEED}.pt
- student_medium_seed{SEED}.pt
- student_large_seed{SEED}.pt

Each run logs per-seed results and a final aggregated summary.

## How to Run

Run everything:

./experiments/4__student_capacity_sweep/run_all.sh

Or submit manually:

sbatch experiments/4__student_capacity_sweep/student/run_small.slurm
sbatch experiments/4__student_capacity_sweep/student/run_medium.slurm
sbatch experiments/4__student_capacity_sweep/student/run_large.slurm

## Expected Insight

This experiment reveals how KD interacts with model capacity.

Typical findings:

- small models benefit most from KD (largest relative improvement)
- medium models provide a strong balance of efficiency and performance
- large models achieve higher absolute performance but smaller KD gains

You may observe diminishing returns as model size increases.
