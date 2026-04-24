# Installation Guide

## Overview

This guide explains how to install and prepare the ECG knowledge distillation project for either:
- local execution of the main workflows, or
- Kelvin2 execution using the provided SLURM scripts

The software requires:
- Python
- the dependencies listed in `requirements.txt`
- the PTB-XL dataset placed in the expected directory structure

In practice, local setup is mainly intended for installation, preprocessing, and functional verification of the main training scripts. Larger experiment suites are more appropriately run on Kelvin2 because of runtime and compute requirements.

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd code
```

Replace `<your-repository-url>` with the correct repository URL.

## 2. Create a Python environment

A virtual environment is recommended.

### Option A — venv

```bash
python3 -m venv venv
source venv/bin/activate
```

### Option B — conda

```bash
conda create -n kd-ecg python=3.10
conda activate kd-ecg
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Hardware notes

Minimum practical requirements depend on the task.

### Local usage

Recommended:
- Python 3.10
- at least 8 GB RAM
- sufficient free disk space for PTB-XL, processed files, logs, and checkpoints

### Model training

Recommended:
- a CUDA-capable GPU for practical training speed
- CPU execution is possible, but may be significantly slower

Local execution is suitable for basic workflow verification and smaller-scale runs of the main scripts. Full experiment-suite reproduction may be impractically slow on local hardware.

## 5. Download the dataset

This project uses the **PTB-XL** dataset.

Download the dataset from PhysioNet and place it inside:

```text
data/ptbxl/
```

Expected structure should include the PTB-XL metadata and waveform files, for example:

```text
data/
└── ptbxl/
    ├── ptbxl_database.csv
    ├── scp_statements.csv
    ├── records500/
    └── ...
```

The exact contents may vary slightly depending on how the dataset is downloaded, but the preprocessing script must be able to locate the expected PTB-XL files under `data/ptbxl/`.

## 6. Preprocess the dataset

Run:

```bash
python scripts/preprocess_ptbxl.py
```

Expected output:

```text
processed/ptbxl_500hz_10s.npz
```

This creates the processed dataset used by the training scripts.

## 7. Verify the installation

A practical verification step is to run one of the main training scripts directly.

For example:

```bash
python scripts/train_teacher.py
```

If the installation is working correctly, you should see:
- console training output
- dataset and model information printed during execution
- a checkpoint written to `checkpoints/`

Depending on your configuration, full training may take time, so this step is mainly intended to confirm that imports, dependencies, dataset paths, and output paths are working correctly.

You may also verify the other main workflows locally with:
- `python scripts/train_student_baseline.py`
- `python scripts/train_student_kd.py`

These direct Python entry points are the normal local interface for the core workflows.

## 8. Kelvin2 / HPC usage

If running on Kelvin2, first load the required modules according to your environment.

A typical example is:

```bash
module purge
module load python3/3.10.5/gcc-9.3.0
```

Then submit jobs using the SLURM scripts provided in `slurm/`.

For example:

```bash
bash slurm/submit_all.sh
```

This submission workflow is intended for teacher, baseline student, and KD execution on Kelvin2, with dependency handling for the KD stage.

For larger-scale experiment runs, Kelvin2 is the preferred environment.

## 9. Common output locations

During execution, the project may create or update:
- `processed/`
- `checkpoints/`
- `logs/`
- `experiments/.../logs/`
- `experiments/.../checkpoints/`

## Troubleshooting

### `ModuleNotFoundError`

- confirm the environment is activated
- confirm dependencies were installed with `pip install -r requirements.txt`

### dataset file not found

- confirm PTB-XL is located under `data/ptbxl/`
- confirm required metadata files such as `ptbxl_database.csv` and `scp_statements.csv` are present

### teacher checkpoint not found during KD training

- confirm teacher training has completed successfully
- confirm the expected teacher checkpoint path matches the path used by the KD script

### CUDA or GPU errors

- confirm a compatible GPU is available
- check GPU visibility with:

```bash
nvidia-smi
```

### slow execution

- local CPU execution may be much slower than GPU or cluster execution
- for complete experiment-suite runs, Kelvin2 is recommended

## Notes

- the main Python scripts support local execution of preprocessing and the core training workflows
- cluster execution scripts are organised under `slurm/`
- grouped experiment workflows are organised under `experiments/` and are primarily intended for Kelvin2/SLURM execution
- reproduction instructions are provided in `REPLICATION_GUIDE.md`