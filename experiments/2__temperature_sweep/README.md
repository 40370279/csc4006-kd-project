# Experiment 2 — Temperature Sweep (Weak Student)

## Objective
This experiment investigates the effect of distillation temperature (T) on the performance of a weak student model.

Temperature controls how soft the teacher probability distribution is during knowledge distillation. The goal is to identify which temperature value produces the best student performance.

---

## Student Model

This experiment uses a deliberately weaker architecture:

WeakStudentCNN

Reducing model capacity helps expose clearer trends in knowledge distillation behaviour.

---

## Teacher Model

All runs use the same pretrained teacher checkpoint:

checkpoints/teacher_cnn_best.pt

Teacher architecture:
TeacherCNN

---

## Training Configuration

Common parameters

Parameter | Value
--------- | -----
Batch size | 64
Learning rate | 3e-4
Epochs | 50
Early stopping patience | 10
Class weighting gamma | 0.5
Alpha | 0.5

The temperature parameter is varied.

---

## Temperature Values Tested

1  
2  
4  
8  
16

Each temperature value is trained in a separate SLURM job.

---

## Running the Experiment

From the repository root run:

./experiments/2__temperature_sweep/run_all.sh

This script submits the following jobs:

run_T1.slurm  
run_T2.slurm  
run_T4.slurm  
run_T8.slurm  
run_T16.slurm

---

## Outputs

Logs are stored in:

experiments/2__temperature_sweep/logs/

Example files:

T1_<jobid>.out  
T2_<jobid>.out  
T4_<jobid>.out  
T8_<jobid>.out  
T16_<jobid>.out

---

## Checkpoints

Best models for each temperature are saved in:

experiments/2__temperature_sweep/checkpoints/

weak_student_T1.pt  
weak_student_T2.pt  
weak_student_T4.pt  
weak_student_T8.pt  
weak_student_T16.pt

---

## Metrics Recorded

For each temperature the following metrics are recorded:

• Test Accuracy  
• Test Macro-F1  
• Test Weighted-F1  
• Model checkpoint size  
• Inference latency

Results should be stored in:

results.csv

---

## Expected Outcome

Knowledge distillation typically performs best at moderate temperature values.

Example trend:

Temperature | Test Macro-F1
----------- | --------------
1 | 0.60
2 | 0.63
4 | 0.66
8 | 0.65
16 | 0.62