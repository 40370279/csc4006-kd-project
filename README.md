# ECG Knowledge Distillation Project

This project investigates whether **knowledge distillation (KD)** can improve the performance and robustness of lightweight models for ECG classification.

The goal is to:

* build efficient models suitable for deployment
* retain high diagnostic performance
* improve robustness under real-world conditions

---

## Core idea

A large, high-capacity **teacher model** is used to guide a smaller **student model**.

The student learns from:

* true labels (ground truth)
* teacher predictions (soft targets)
* internal representations (feature distillation)

This allows the student to:

* learn richer representations
* generalise better
* perform closer to the teacher despite lower capacity

---

## Models used

### Teacher model

* Multi-scale residual CNN
* Squeeze-and-Excitation (SE) attention
* Statistics pooling (mean + std)
* High representational capacity

---

### Student model

* Lightweight residual CNN
* Standard convolutions
* Adaptive average pooling
* Designed for:

  * low latency
  * small model size
  * efficient inference

---

## Dataset

* **PTB-XL ECG dataset**
* 12-lead ECG signals
* 5 diagnostic classes:

  * CD
  * HYP
  * MI
  * NORM
  * STTC

---

### Preprocessing

* Resampled to **500 Hz**
* Fixed length: **10 seconds (5000 samples)**
* Per-lead normalisation

---

## Knowledge distillation setup

The student is trained using a combined loss:

* Cross-Entropy (hard labels)
* KL Divergence (soft targets)
* Logit matching (MSE)
* Feature distillation (projected features)

---

## Experiments

The project is structured into a series of experiments:

---

### Experiment 1 — KD vs Baseline

* Compare:

  * Teacher model
  * Baseline student
  * KD student

Tests:

* whether KD improves student performance

---

### Experiment 2 — Temperature Sweep

* Vary distillation temperature:

  * T = 1, 2, 4, 8, 16, 32

Tests:

* how soft target smoothing affects learning

---

### Experiment 3 — Alpha Sweep

* Vary balance between:

  * hard labels
  * soft targets

Tests:

* optimal trade-off for KD

---

### Experiment 4 — Student Capacity

* Compare:

  * small, medium, large students

Tests:

* how KD scales with model capacity

---

### Experiment 5 — Teacher Strength

* Use different teacher sizes

Tests:

* whether stronger teachers improve student performance

---

### Experiment 6 — Distillation Components

* Ablation study:

  * CE only
  * soft targets only
  * feature distillation only
  * combined

Tests:

* contribution of each KD component

---

### Experiment 7 — Robustness Evaluation

* Evaluate models under corrupted inputs:

  * noise
  * amplitude changes
  * missing leads
  * time masking

Tests:

* whether KD improves real-world robustness

---

## Evaluation metrics

Primary metrics:

* Accuracy
* Macro-F1
* Weighted-F1
* **Macro-AUC (primary focus)**

Macro-AUC is used because it:

* handles class imbalance
* evaluates ranking quality
* is robust to threshold choice

---

## Seeds

All experiments are run with:

* 42
* 123
* 999

Results are reported as:

* mean ± standard deviation

---

## How to run

Each experiment uses SLURM scripts.

Example:

```
sbatch experiments/1__kd_vs_baseline/run_kd.slurm
```

---

## Dependencies

* Python 3.10
* PyTorch
* NumPy
* scikit-learn

---

## Outputs

Each experiment produces:

* logs (`logs/`, `logs/tmp/`)
* checkpoints (`checkpoints/`)
* results (JSON or printed summaries)

---

## Expected outcomes

Across experiments, KD is expected to:

* improve student performance over baseline
* reduce performance gap with teacher
* improve robustness under corrupted inputs
* maintain low latency and model size

---

## Interpretation

Knowledge distillation enables:

* transfer of richer representations
* improved generalisation
* better performance under distribution shift

This makes KD particularly suitable for:

* medical signal analysis
* real-world deployment scenarios

---

## Notes

* Training scripts are separate from evaluation scripts
* Some experiments require pre-trained checkpoints
* Ensure correct paths before running SLURM jobs
* Check logs if parsing or checkpoint errors occur

---

## Project goal

To demonstrate that:

> **small, efficient models can achieve strong and robust performance through knowledge distillation**
