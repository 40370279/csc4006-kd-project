# Experiment 1 — KD vs Baseline

## Objective
This experiment evaluates the effect of knowledge distillation (KD) on a lightweight student model for ECG classification using the PTB-XL dataset.

The goal is to determine whether a student model trained using knowledge distillation performs better than the same model trained with standard supervised learning.

---

## Models Compared

Model | Description
----- | -----------
Baseline Student | Student CNN trained using standard cross-entropy loss
KD Student | Student CNN trained using knowledge distillation from the teacher model

Both students use the same architecture so the only difference is the training method.

---

## Teacher Model

The KD student uses logits from a pretrained teacher network:

checkpoints/teacher_cnn_best.pt

Teacher architecture:
TeacherCNN

---

## Training Configuration

Common training parameters

Parameter | Value
--------- | -----
Batch size | 64
Learning rate | 3e-4
Epochs | 50
Early stopping patience | 10
Class weighting gamma | 0.5

KD-specific parameters

Parameter | Value
--------- | -----
Alpha | 0.5
Temperature | 4.0

---

## Running the Experiment

From the repository root run:

./experiments/1__kd_vs_baseline/run_all.sh

This script submits two SLURM jobs:

run_baseline.slurm  
run_kd.slurm

---

## Outputs

Logs are stored in:

experiments/1__kd_vs_baseline/logs/

Example files:

baseline_<jobid>.out  
kd_<jobid>.out

---

## Checkpoints

Best models are saved in:

experiments/1__kd_vs_baseline/checkpoints/

student_baseline_best.pt  
student_kd_best.pt

---

## Metrics Recorded

For each model the following metrics are collected:

• Test Accuracy  
• Test Macro-F1  
• Test Weighted-F1  
• Model checkpoint size  
• Inference latency

Results should be recorded in:

results.csv

---

## Expected Outcome

Knowledge distillation should improve student performance while keeping the model lightweight.

Example expected pattern:

Model | Test Macro-F1
----- | --------------
Baseline Student | ~0.60
KD Student | ~0.64