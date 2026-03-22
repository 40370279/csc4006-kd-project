# Experiment 3 — Alpha Sweep (Weak Student)

## Objective

This experiment investigates the effect of the **alpha parameter** in knowledge distillation.

Alpha controls the balance between:

- **Hard label supervision** (cross-entropy with ground truth labels)
- **Teacher supervision** (KL divergence with teacher predictions)

The distillation loss used in training is:

Loss = alpha * CE + (1 - alpha) * KD

The goal of this experiment is to determine which balance between these two components produces the best student model performance.

---

## Student Model

This experiment uses the intentionally reduced-capacity architecture:

WeakStudentCNN

A weaker student helps expose clearer trends in knowledge distillation behaviour.

---

## Teacher Model

All experiments use the same pretrained teacher model:

checkpoints/teacher_cnn_best.pt

Teacher architecture:

TeacherCNN

The teacher model remains frozen during student training.

---

## Training Configuration

Common training parameters

Parameter | Value
--------- | -----
Batch size | 64
Learning rate | 3e-4
Epochs | 50
Early stopping patience | 10
Temperature | 4.0
Class weighting gamma | 0.5

The **alpha parameter** is varied.

---

## Alpha Values Tested

0.1  
0.3  
0.5  
0.7  
0.9  

Each value is trained in a separate SLURM job.

Interpretation:

Alpha | Training Emphasis
----- | ----------------
0.1 | Mostly teacher guidance
0.3 | Teacher-dominant supervision
0.5 | Balanced KD and labels
0.7 | Label-dominant supervision
0.9 | Mostly hard labels

---

## Running the Experiment

From the repository root run:

./experiments/3__alpha_sweep/run_all.sh

This script submits the following jobs:

run_A01.slurm  
run_A03.slurm  
run_A05.slurm  
run_A07.slurm  
run_A09.slurm  

---

## Outputs

Logs are stored in:

experiments/3__alpha_sweep/logs/

Example files:

A01_<jobid>.out  
A03_<jobid>.out  
A05_<jobid>.out  
A07_<jobid>.out  
A09_<jobid>.out  

---

## Checkpoints

Best models for each alpha value are saved in:

experiments/3__alpha_sweep/checkpoints/

weak_student_A01.pt  
weak_student_A03.pt  
weak_student_A05.pt  
weak_student_A07.pt  
weak_student_A09.pt  

---

## Metrics Recorded

For each alpha value the following metrics are recorded:

• Test Accuracy  
• Test Macro-F1  
• Test Weighted-F1  
• Model checkpoint size  
• Inference latency  

Results should be stored in:

results.csv

---

## Expected Outcome

Knowledge distillation typically performs best with a balanced alpha value.

Example expected pattern:

Alpha | Test Macro-F1
----- | --------------
0.1 | 0.63
0.3 | 0.65
0.5 | 0.66
0.7 | 0.64
0.9 | 0.61

Values too close to 1.0 reduce the influence of teacher knowledge, while values too close to 0 rely too heavily on teacher predictions.

---

## Purpose in the Study

This experiment helps determine the optimal balance between:

- Direct supervision from labelled ECG data
- Soft supervision from the teacher model

It forms an important part of the **knowledge distillation ablation study**.