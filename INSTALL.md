# Installation Guide

## Overview

This project implements a deep learning pipeline for ECG classification using Knowledge Distillation (KD). The system is designed to run on both local machines and the Kelvin2 HPC cluster.

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

## 2. Python Environment Setup

It is recommended to use a virtual environment.

### Option A: venv

```bash
python3 -m venv venv
source venv/bin/activate
```

### Option B: Conda

```bash
conda create -n kd-ecg python=3.9
conda activate kd-ecg
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Hardware Requirements

* GPU recommended (CUDA-enabled)
* Minimum 8GB RAM
* ~5–10GB storage for dataset and checkpoints

---

## 5. Dataset Setup

Download the **PTB-XL dataset** from PhysioNet:
https://physionet.org/content/ptb-xl/

Place the dataset in:

```
data/ptbxl/
```

Expected structure:

```
data/
└── ptbxl/
    ├── ptbxl_database.csv
    ├── records500/
    └── ...
```

---

## 6. Preprocessing

Run preprocessing to generate the processed dataset:

```bash
python scripts/preprocess_ptbxl.py
```

This will create:

```
processed/ptbxl_500hz_10s.npz
```

---

## 7. Running on Kelvin2 (HPC)

Ensure:

* SSH access is configured
* Required modules (Python, CUDA) are loaded

Example:

```bash
module load python/3.x
module load cuda
```

Then submit jobs using SLURM:

```bash
bash slurm/submit_all.sh
```

---

## 8. Verifying Installation

Run a quick training test:

```bash
python scripts/train_teacher.py
```

If successful, you should see:

* Training logs printed to console
* Checkpoints saved in `checkpoints/`

---

## Troubleshooting

* **ModuleNotFoundError** → Ensure virtual environment is activated
* **CUDA errors** → Check GPU availability (`nvidia-smi`)
* **Dataset errors** → Verify PTB-XL path and structure

---

## Notes

* All experiments are reproducible via scripts in `experiments/`
* Outputs (logs, checkpoints) are automatically generated
