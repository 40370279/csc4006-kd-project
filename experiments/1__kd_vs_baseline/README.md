# Experiment 1: KD vs Baseline

This experiment compares:

- a standard student baseline model
- a student trained with knowledge distillation (KD)

using the same seed set and experiment-local checkpoints/logs.

## Structure

experiments/1__kd_vs_baseline/
├── checkpoints/
├── logs/
│   └── tmp/
├── student/
│   ├── run_baseline.slurm
│   └── run_kd.slurm
├── teacher/
│   └── run_teacher.slurm
├── README.md
└── run_all.sh

## Outputs

All outputs for this experiment are isolated inside this folder:

- checkpoints: experiments/1__kd_vs_baseline/checkpoints/
- logs: experiments/1__kd_vs_baseline/logs/
- temporary per-seed logs: experiments/1__kd_vs_baseline/logs/tmp/

This avoids overwriting root-level experiment outputs.

## Jobs

### 1. Teacher

Runs the teacher model across seeds:

- 42
- 123
- 999

Outputs:
- checkpoints/teacher_cnn_seed{SEED}.pt

### 2. Student Baseline

Trains the student model WITHOUT KD using the same seeds.

Outputs:
- checkpoints/student_baseline_seed{SEED}.pt

### 3. Student KD

Trains the student model WITH knowledge distillation.

Requires:
- teacher checkpoints from this experiment

Outputs:
- checkpoints/student_kd_seed{SEED}.pt

## How to Run

Run everything (recommended):

./experiments/1__kd_vs_baseline/run_all.sh

This will:

1. Train teacher
2. Then run baseline and KD (after teacher finishes)

Manual run:

sbatch experiments/1__kd_vs_baseline/teacher/run_teacher.slurm
sbatch experiments/1__kd_vs_baseline/student/run_baseline.slurm
sbatch experiments/1__kd_vs_baseline/student/run_kd.slurm

IMPORTANT:
Run teacher first before KD, or KD will fail due to missing checkpoints.

## Notes

- All runs use seeds: 42, 123, 999
- Metrics are aggregated automatically (mean ± std)
- Logs are stored per-seed for debugging
- This experiment is fully isolated from other experiments

## Expected Outcome

You should observe:

- Baseline student performance
- KD student performance
- Difference between the two
