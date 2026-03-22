# Experiment 4 — Student Capacity Sweep

## Objective

This experiment investigates how **student model capacity** affects performance under knowledge distillation.

Two student families are evaluated:

- **weak student family**
- **normal student family**

The goal is to compare how increasing model capacity changes:

- classification performance
- model size
- inference latency

This experiment helps show the trade-off between **accuracy and efficiency**, and whether knowledge distillation benefits smaller students more strongly.

---

## Student Families

### Weak Student Family
The weak student family uses `WeakStudentCNN`, a deliberately lower-capacity architecture designed to expose clearer knowledge distillation trends.

The following weak capacities are evaluated:

- small
- medium
- large

### Normal Student Family
The normal student family uses `StudentCNN`, a configurable student architecture.

The following normal capacities are evaluated:

- small
- medium
- large

---

## Teacher Model

All runs use the same pretrained teacher model:

checkpoints/teacher_cnn_best.pt

Teacher architecture:

TeacherCNN

The teacher model remains fixed and is only used to provide soft targets during student training.

---

## Training Configuration

All runs use the same knowledge distillation setup except for student capacity.

Common parameters:

Parameter | Value
--------- | -----
Batch size | 64
Learning rate | 3e-4
Epochs | 50
Early stopping patience | 10
Class weighting gamma | 0.5
Alpha | 0.5
Temperature | 4.0

Only the **student family** and **student capacity** change.

---

## Folder Layout

This experiment is organised into two subfolders:

- `weak_student/`
- `normal_student/`

### weak_student/
Contains the SLURM scripts, logs, and checkpoints for:

- weak small
- weak medium
- weak large

### normal_student/
Contains the SLURM scripts, logs, and checkpoints for:

- normal small
- normal medium
- normal large

---

## Running the Experiment

From the repository root run:

./experiments/4__student_capacity_sweep/run_all.sh

This submits all six student-capacity jobs:

- weak small
- weak medium
- weak large
- normal small
- normal medium
- normal large

---

## Outputs

### Logs

Logs are stored inside the corresponding family folders:

- `experiments/4__student_capacity_sweep/weak_student/logs/`
- `experiments/4__student_capacity_sweep/normal_student/logs/`

### Checkpoints

Best checkpoints are stored in:

- `experiments/4__student_capacity_sweep/weak_student/checkpoints/`
- `experiments/4__student_capacity_sweep/normal_student/checkpoints/`

---

## Metrics Recorded

For each run the following metrics are collected:

- Test Accuracy
- Test Macro-F1
- Test Weighted-F1
- Model checkpoint size
- Inference latency

Results should be recorded in:

results.csv

---

## Expected Outcome

As capacity increases, performance is expected to improve, but model size and latency should also increase.

Typical pattern:

- weakest models are fastest and smallest
- larger models perform better
- knowledge distillation may provide the largest benefit for lower-capacity students

This experiment is useful for demonstrating the trade-off between:

- compactness
- speed
- predictive performance

---

## Purpose in the Study

This experiment forms the **student capacity analysis** section of the project.

It complements the earlier experiments by showing:

- whether stronger students consistently outperform weaker ones
- whether the weak and normal student families behave differently
- how capacity affects the usefulness of knowledge distillation

Together with the KD-vs-baseline, temperature sweep, and alpha sweep experiments, this provides a strong empirical evaluation of the knowledge distillation framework.