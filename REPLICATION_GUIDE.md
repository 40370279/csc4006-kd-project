# Replication Guide

## Overview

This guide describes how to reproduce the main workflows and experiments for the ECG knowledge distillation project.

The repository supports:
- preprocessing the PTB-XL dataset
- training the teacher model
- training a baseline student
- training a knowledge-distilled student
- running controlled experiment suites
- running multi-seed summaries

## 1. Requirements

Before reproducing results, ensure that:
- the Python environment is installed
- dependencies from `requirements.txt` are installed
- the PTB-XL dataset is available in `data/ptbxl/`
- preprocessing has either been completed already or can be run

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

## 3. Train the teacher model

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

## 4. Train the baseline student

```bash
python scripts/train_student_baseline.py
```

Expected checkpoint path:
```text
checkpoints/student_baseline_best.pt
```

## 5. Train the KD student

```bash
python scripts/train_student_kd.py
```

Expected checkpoint path:
```text
checkpoints/student_kd_best.pt
```

Note:
- this step requires a valid teacher checkpoint
- if the teacher checkpoint path is changed, make sure the KD script points to the correct file

## 6. Run the main experiment suite

To run the experiment suite organised under `experiments/`, use:

```bash
bash experiments/run_all_experiments.sh
```

This is intended to execute the grouped experiments such as:
- KD vs baseline
- temperature sweep
- alpha sweep
- student capacity sweep
- teacher model comparison
- distillation ablations
- robustness evaluation

Because experiment orchestration may vary slightly by folder, check the local scripts inside each experiment directory if any path needs adjustment.

## 7. Run multi-seed aggregation

To run the multi-seed experiment helper:

```bash
python scripts/run_multi_seed.py
```

This uses the predefined seeds:
- `42`
- `123`
- `999`

and reports metrics as:
- mean ± standard deviation

## 8. Run on Kelvin2 with SLURM

For cluster execution, use the scripts provided in `slurm/`.

Example:

```bash
bash slurm/submit_all.sh
```

This submission workflow:
- submits the teacher training job
- submits the baseline student job
- submits the KD job with a dependency on successful teacher completion
- writes a submission log under `logs/`

For experiment-specific SLURM jobs, submit the relevant `.slurm` file directly if needed.

## 9. Expected outputs

Depending on the workflow, outputs may include:
- processed dataset files
- model checkpoints
- console logs
- experiment logs
- multi-seed summaries

Common locations:
```text
processed/
checkpoints/
logs/
experiments/.../logs/
experiments/.../checkpoints/
```

## 10. Reproducibility notes

This repository supports reproducibility through:
- predefined PTB-XL fold splits
- explicit seed control
- consistent training entry points
- checkpoint saving
- experiment-specific logging
- multi-seed summaries

However:
- small variation may still occur between systems
- GPU execution can introduce limited numerical variation depending on environment and backend behaviour
- experiment-specific scripts should be checked to confirm all paths match the local setup

## 11. Recommended full workflow

A practical end-to-end sequence is:

```bash
python scripts/preprocess_ptbxl.py
python scripts/train_teacher.py
python scripts/train_student_baseline.py
python scripts/train_student_kd.py
bash experiments/run_all_experiments.sh
```

For cluster execution, the equivalent workflow can be launched through the SLURM scripts.

## 12. Validation checklist

A run can be considered successful if:
- preprocessing creates `processed/ptbxl_500hz_10s.npz`
- teacher training creates a teacher checkpoint
- baseline training creates a student baseline checkpoint
- KD training creates a KD student checkpoint
- logs contain final reported metrics such as accuracy, macro-F1, weighted-F1, and macro-AUC
- experiment scripts finish without missing-path or missing-checkpoint errors

## 13. Practical advice

- start by reproducing the preprocessing and one training script before running the entire suite
- confirm teacher checkpoints exist before running KD experiments
- inspect log files if metric parsing fails in experiment scripts
- use Kelvin2 for full experiment execution if local resources are limited