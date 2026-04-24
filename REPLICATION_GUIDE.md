# Replication Guide

## Overview

This guide explains how to reproduce the main workflows and experiments for the ECG knowledge distillation project.

The repository supports:
- preprocessing the PTB-XL dataset
- training the teacher model
- training a baseline student
- training a knowledge-distilled student
- running controlled experiment suites

The project can be executed in two ways:
- locally, by running the main Python entry-point scripts directly
- on Kelvin2 using SLURM, by submitting the provided batch scripts

In practice, these two modes serve slightly different purposes. Local execution is mainly intended for setup, functional verification, and smaller-scale runs of the core workflows. The full grouped experiment suites are primarily intended for Kelvin2/SLURM execution, since they are organised around batch-oriented experiment scripts and may be impractically slow to reproduce fully on local hardware.

## 1. Requirements

Before reproducing results, ensure that:
- the Python environment is installed
- dependencies from `requirements.txt` are installed
- the PTB-XL dataset is available in `data/ptbxl/`
- preprocessing has either already been completed or can be run

See `INSTALL.md` first if setup is incomplete.

## 2. Data preparation

Generate the processed dataset with:

```bash
python scripts/preprocess_ptbxl.py
```

Expected output:

```text
processed/ptbxl_500hz_10s.npz
```

This step prepares the train, validation, and test arrays used by the training scripts.

## 3. Local execution workflow

For local execution, the core workflows can be run directly through the Python scripts. This is the recommended way to verify that the environment, dataset paths, dependencies, and main training code are working correctly.

### Train the teacher model

```bash
python scripts/train_teacher.py
```

Typical output:
- console training logs
- best teacher checkpoint saved under `checkpoints/`

Expected checkpoint path:

```text
checkpoints/teacher_cnn_best.pt
```

### Train the baseline student

```bash
python scripts/train_student_baseline.py
```

Expected checkpoint path:

```text
checkpoints/student_baseline_best.pt
```

### Train the KD student

```bash
python scripts/train_student_kd.py
```

Expected checkpoint path:

```text
checkpoints/student_kd_best.pt
```

Notes:
- this step requires a valid teacher checkpoint
- if the teacher checkpoint path is changed, make sure the KD script points to the correct file

Local execution is suitable for basic reproduction of the main training stages, but it is not the primary route for complete reproduction of the full experiment suite.

## 4. Kelvin2 / SLURM execution workflow

For larger-scale reproduction, use the SLURM scripts provided in `slurm/`.

Example:

```bash
bash slurm/submit_all.sh
```

This submission workflow:
- submits the teacher training job
- submits the baseline student job
- submits the KD job with a dependency on successful teacher completion
- writes a submission log under `logs/`

This is the recommended workflow for larger runs on Kelvin2.

## 5. Experiment suite execution

The project also includes a grouped experiment workflow under `experiments/`.

To launch the full experiment suite:

```bash
bash experiments/run_all_experiments.sh
```

This script runs the experiment folders in sequence, including studies such as:
- KD vs baseline
- temperature sweep
- alpha sweep
- student capacity sweep
- teacher model comparison
- distillation ablations
- robustness evaluation

The experiment-level `run_all.sh` files are intended primarily for the SLURM-based experiment workflow on Kelvin2. They are not the normal entry point for local execution. While the underlying model scripts can be run locally, the grouped experiment suites are designed mainly for batch execution in the intended cluster environment.

## 6. Expected outputs

Depending on the workflow, outputs may include:
- processed dataset files
- model checkpoints
- console logs
- experiment logs

Common locations:

```text
processed/
checkpoints/
logs/
experiments/.../logs/
experiments/.../checkpoints/
```

## 7. Reproducibility notes

This repository supports reproducibility through:
- predefined PTB-XL fold splits
- explicit seed control
- consistent training entry points
- checkpoint saving
- experiment-specific logging

Repeated seeds and summary statistics within the experiment workflows are handled by the experiment scripts themselves rather than by direct standalone use of `scripts/run_multi_seed.py`.

However:
- small variation may still occur between systems
- GPU execution can introduce limited numerical variation depending on environment and backend behaviour
- experiment-specific scripts should be checked to confirm all paths match the local setup
- complete reproduction of the full experiment suite may be impractically slow on local hardware, so Kelvin2 is the preferred environment for full-scale runs

## 8. Recommended workflow

A practical local verification sequence is:

```bash
python scripts/preprocess_ptbxl.py
python scripts/train_teacher.py
python scripts/train_student_baseline.py
python scripts/train_student_kd.py
```

This is usually sufficient to confirm that the main software workflows operate correctly.

For complete experiment-suite reproduction, the preferred workflow is to use Kelvin2 and the provided SLURM scripts.

## 9. Validation checklist

A run can be considered successful if:
- preprocessing creates `processed/ptbxl_500hz_10s.npz`
- teacher training creates a teacher checkpoint
- baseline training creates a student baseline checkpoint
- KD training creates a KD student checkpoint
- logs contain final reported metrics such as accuracy, macro-F1, weighted-F1, and macro-AUC
- experiment scripts finish without missing-path or missing-checkpoint errors

## 10. Practical advice

- start by reproducing preprocessing and one training script before running the entire suite
- confirm teacher checkpoints exist before running KD experiments
- inspect log files if metric parsing fails in experiment scripts
- use Kelvin2 for full experiment execution if local resources are limited