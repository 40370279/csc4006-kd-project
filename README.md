# ECG Knowledge Distillation for Efficient PTB-XL Classification

This repository contains a reproducible research software framework for investigating **knowledge distillation (KD)** for efficient ECG classification on the **PTB-XL** dataset.

The project studies whether a compact student model can approach the performance of a higher-capacity teacher model while remaining significantly smaller and faster at inference.

## Project objectives

The software is designed to support the following goals:

- train a high-capacity **teacher** ECG classifier
- train a lightweight **baseline student** classifier
- train a lightweight **knowledge-distilled student**
- compare teacher, baseline, and KD models under controlled settings
- evaluate both **predictive performance** and **deployment-oriented efficiency**
- support reproducible ablation studies and robustness experiments

## Repository structure

```text
code/
├── src/
│   ├── data/                # Dataset wrapper and ECG augmentation
│   ├── models/              # Teacher and student model implementations
│   └── utils/               # Metrics and model statistics
├── scripts/                 # Main Python entry points
├── slurm/                   # SLURM job scripts for Kelvin2
├── experiments/             # Experiment-specific runs and logs
├── data/ptbxl/              # Raw PTB-XL dataset location
├── processed/               # Preprocessed dataset outputs
├── checkpoints/             # Trained model checkpoints
├── logs/                    # Log files
├── README.md
├── INSTALL.md
├── REPLICATION_GUIDE.md
├── LICENSE
└── requirements.txt
```

## Execution model

The project can be used in two main ways:

- **Local execution**, by running the main Python scripts directly
- **Kelvin2 / SLURM execution**, by submitting the provided batch scripts

In practice, local execution is mainly intended for setup, preprocessing, and functional verification of the core training workflows. Full grouped experiment reproduction is primarily intended for Kelvin2, because the experiment suite is organised around batch-oriented experiment scripts and may be impractically slow to run fully on local hardware.

## Main components

### Preprocessing
**Script:** `scripts/preprocess_ptbxl.py`

This script:
- loads PTB-XL metadata and waveform files
- filters to **500 Hz** recordings
- maps ECG records to **5 diagnostic superclasses**
- keeps only records with a **single unambiguous superclass**
- crops or pads each ECG to **12 × 5000**
- applies per-lead normalisation
- creates train, validation, and test splits using PTB-XL folds

**Output:**
- `processed/ptbxl_500hz_10s.npz`

### Teacher model
**Script:** `scripts/train_teacher.py`

The teacher is a higher-capacity CNN designed to learn strong ECG representations. It is used both as a standalone classifier and as the supervision source for knowledge distillation.

### Baseline student model
**Script:** `scripts/train_student_baseline.py`

The baseline student is a lightweight residual CNN trained using standard supervised learning only.

### KD student model
**Script:** `scripts/train_student_kd.py`

The KD student is trained using a combined objective including:
- hard-label cross-entropy
- soft-target distillation
- logit matching
- feature distillation

### Experiment workflows
**Location:** `experiments/`

The grouped experiment suite is organised into dedicated experiment folders covering controlled studies such as KD vs baseline comparison, temperature sweeps, alpha sweeps, student-capacity sweeps, teacher-model comparison, distillation ablations, and robustness evaluation.

The experiment-level `run_all.sh` files are intended primarily for the Kelvin2 / SLURM workflow rather than as the standard local entry point. Repeated seeds and summary statistics within these experiments are handled by the experiment scripts themselves.

## Models

### Teacher
The teacher model is a multi-scale residual 1D CNN with:
- squeeze-and-excitation attention
- statistics pooling
- higher channel capacity than the student

### Student
The student model is a lightweight residual 1D CNN with:
- standard convolutions
- adaptive average pooling
- smaller parameter count
- lower latency and reduced model size

## Dataset

This project uses the **PTB-XL** ECG dataset.

### Input format
- 12-lead ECG signals
- 500 Hz sampling rate
- 10-second recordings
- processed into tensors of shape `(12, 5000)`

### Classification task
The task is 5-class diagnostic superclass classification:
- `CD`
- `HYP`
- `MI`
- `NORM`
- `STTC`

## Experiments

The project is organised into a series of controlled experiments.

### Experiment 1 — KD vs Baseline
Compares:
- teacher
- baseline student
- KD student

### Experiment 2 — Temperature Sweep
Varies distillation temperature:
- `T = 1, 2, 4, 8, 16, 32`

### Experiment 3 — Alpha Sweep
Varies the weighting between:
- hard-label supervision
- teacher supervision

### Experiment 4 — Student Capacity Sweep
Compares:
- small
- medium
- large student models

### Experiment 5 — Teacher Model Comparison
Uses different teacher capacities to test how teacher strength affects student performance.

### Experiment 6 — Distillation Component Ablation
Compares:
- CE only
- soft targets only
- feature distillation only
- combined KD variants

### Experiment 7 — Robustness Evaluation
Evaluates model behaviour under perturbed inputs such as:
- additive noise
- amplitude scaling
- missing leads
- time masking

## Evaluation metrics

The software reports:
- Accuracy
- Macro-F1
- Weighted-F1
- Macro-AUC

**Macro-AUC** is the primary metric because it is more informative under class imbalance and better reflects class-balanced discrimination.

## Outputs

Depending on the script or experiment, outputs may include:
- training logs
- experiment logs
- model checkpoints
- printed metric summaries

Common output locations:
- `logs/`
- `checkpoints/`
- `experiments/.../logs/`
- `experiments/.../checkpoints/`

## Typical workflow

### Local execution
A practical local workflow is:
1. Install dependencies
2. Download and place PTB-XL in `data/ptbxl/`
3. Run preprocessing
4. Train the teacher
5. Train the baseline student
6. Train the KD student

This local route is mainly intended to verify that the environment, dataset paths, and main workflows operate correctly.

### Kelvin2 / SLURM execution
A practical Kelvin2 workflow is:
1. Set up the Python environment
2. Ensure PTB-XL is available in the expected location
3. Submit jobs using scripts in `slurm/`
4. Run grouped experiment workflows as required
5. Monitor logs in the relevant log directories

This is the preferred route for larger-scale experiment execution.

## Quick start

### Preprocess data
```bash
python scripts/preprocess_ptbxl.py
```

### Train teacher
```bash
python scripts/train_teacher.py
```

### Train baseline student
```bash
python scripts/train_student_baseline.py
```

### Train KD student
```bash
python scripts/train_student_kd.py
```

### Submit cluster jobs
```bash
bash slurm/submit_all.sh
```

## Reproducibility

The project supports reproducibility through:
- fixed train/validation/test fold usage
- explicit random seed control
- checkpoint saving with metadata
- experiment-specific logging
- SLURM-based batch execution

Repeated seeds and summary statistics in the grouped experiment workflows are handled by the experiment scripts themselves rather than being treated as the main standalone user workflow.

## Notes

- Some experiments depend on pretrained teacher checkpoints.
- Ensure dataset paths and checkpoint paths are correct before running jobs.
- The main Python scripts support local execution of preprocessing and the core training workflows.
- Full grouped experiment workflows are organised under `experiments/` and are primarily intended for Kelvin2 / SLURM execution.
- For installation instructions, see `INSTALL.md`.
- For reproduction steps, see `REPLICATION_GUIDE.md`.