# Replication Guide

## Overview

This guide describes how to reproduce all experiments and results presented in the project.

The experiments evaluate the effectiveness of Knowledge Distillation (KD) for ECG classification using the PTB-XL dataset.

---

## 1. Setup

Ensure installation is complete (see `INSTALL.md`).

Key requirements:

* Python environment configured
* PTB-XL dataset downloaded
* Preprocessing completed

---

## 2. Data Preparation

Run preprocessing:

```bash
python scripts/preprocess_ptbxl.py
```

Output:

```
processed/ptbxl_500hz_10s.npz
```

---

## 3. Train Teacher Model

```bash
python scripts/train_teacher.py
```

Output:

```
checkpoints/teacher_cnn_best.pt
```

---

## 4. Train Baseline Student

```bash
python scripts/train_student_baseline.py
```

Output:

```
checkpoints/student_baseline_best.pt
```

---

## 5. Train KD Student

```bash
python scripts/train_student_kd.py
```

Output:

```
checkpoints/student_kd_best.pt
```

---

## 6. Run All Experiments

To reproduce all experiments:

```bash
bash experiments/run_all_experiments.sh
```

This executes:

### Experiment 1: KD vs Baseline

```
experiments/1__kd_vs_baseline/
```

### Experiment 2: Temperature Sweep

```
experiments/2__temperature_sweep/
```

### Experiment 3: Alpha Sweep

```
experiments/3__alpha_sweep/
```

### Experiment 4: Student Capacity Sweep

```
experiments/4__student_capacity_sweep/
```

### Experiment 5: Teacher Model Comparison

```
experiments/5__teacher_model_comparison/
```

### Experiment 6: Distillation Type Comparison

```
experiments/6__distillation_type/
```

### Experiment 7: Robustness Evaluation

```
experiments/7__robustness_eval/
```

---

## 7. Multi-Seed Experiments (Optional)

To improve statistical reliability:

```bash
python scripts/run_multi_seed.py
```

---

## 8. Outputs

Results are saved in:

```
logs/
checkpoints/
```

Key outputs:

* Model checkpoints (.pt)
* Training logs
* Evaluation metrics (accuracy, macro-F1, weighted-F1)

---

## 9. Reproducing Figures

Figures used in the report can be regenerated from logs:

* Confusion matrices
* Performance comparisons
* Robustness plots

(If implemented)

```bash
python src/evaluation/plots.py
```

---

## 10. Running on Kelvin2 (Recommended)

For large experiments:

```bash
bash slurm/submit_all.sh
```

This will:

* Submit all jobs
* Run experiments in parallel
* Store outputs automatically

---

## 11. Expected Results

Typical outcomes:

* KD improves macro-F1 over baseline
* Higher temperatures improve soft target learning
* Larger students perform better but increase cost
* KD improves robustness to noise and perturbations

---

## 12. Reproducibility Notes

* Random seeds are controlled where possible
* Minor variation may occur due to GPU non-determinism
* Multi-seed runs are recommended for stability

---

## 13. Estimated Runtime

| Task                  | Time (GPU)                           |
| --------------------- | ------------------------------------ |
| Preprocessing         | ~10 min                              |
| Teacher training      | ~1–2 hrs                             |
| Student training      | ~30–60 min                           |
| Full experiment suite | Several hours (parallel recommended) |

---

## Summary

To fully reproduce results:

```bash
python scripts/preprocess_ptbxl.py
python scripts/train_teacher.py
bash experiments/run_all_experiments.sh
```

All outputs will be generated automatically.
