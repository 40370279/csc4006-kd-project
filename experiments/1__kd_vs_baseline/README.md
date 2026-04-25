# Experiment 1: Teacher vs KD Student vs Baseline Student

## Overview

This experiment establishes the core comparison for the project by evaluating three models on the PTB-XL 5-class ECG classification task:

1. Teacher model – a high-capacity TeacherCNN  
2. Baseline student – a lightweight StudentCNN trained with standard cross-entropy only  
3. KD student – the same lightweight StudentCNN trained using knowledge distillation from the teacher  

The goal is to determine whether knowledge distillation improves a small student model while preserving efficiency.

---

## Research question

Does knowledge distillation improve a small ECG student model compared with standard supervised training, while remaining much more efficient than the teacher?

---

## Models compared

Teacher:
- Architecture: TeacherCNN  
- Size: large  
- Role: provides soft targets and feature guidance  

Baseline student:
- Architecture: StudentCNN  
- Size: small  
- Training: standard supervised learning  

KD student:
- Architecture: StudentCNN  
- Size: small  
- Training: hard labels + teacher soft targets + auxiliary losses  

---

## Files

Main scripts:
- teacher/run_teacher.slurm  
- student/run_baseline.slurm  
- student/run_kd.slurm  
- run_all.sh  

Outputs:
- checkpoints/  
- logs/  
- logs/tmp/  

---

## Dataset

- PTB-XL (processed)
- Shape: (N, C, T)
- C = 12 leads
- T = 5000 samples

Classes:
- CD, HYP, MI, NORM, STTC  

---

## Seeds

Runs are repeated with:
42, 123, 999  

Results are reported as mean ± standard deviation.

---

## Training configuration

Teacher:
- batch size: 64  
- lr: 8e-4  
- epochs: 90  
- patience: 18  
- weight decay: 1e-3  
- label smoothing: 0.03  

Baseline student:
- batch size: 64  
- lr: 8e-4  
- epochs: 70  
- patience: 15  
- weight decay: 2e-3  
- student size: small  

KD student:
- alpha: 0.10  
- temperature: 1.0  
- logit MSE: 0.03  
- feature MSE: 0.03  

---

## Metrics

Reported metrics:
- Accuracy  
- Macro-AUC  
- Macro-F1  
- Weighted-F1  

Also recorded:
- best epoch  
- latency  
- checkpoint size  

---

## How to run

Run all:
sbatch experiments/1__kd_vs_baseline/run_all.sh  

Run teacher:
sbatch experiments/1__kd_vs_baseline/teacher/run_teacher.slurm  

Run baseline:
sbatch experiments/1__kd_vs_baseline/student/run_baseline.slurm  

Run KD:
sbatch experiments/1__kd_vs_baseline/student/run_kd.slurm  

---

## Checkpoints

Teacher:
- teacher_cnn_seed42.pt  
- teacher_cnn_seed123.pt  
- teacher_cnn_seed999.pt  

Baseline:
- student_baseline_seed42.pt  
- student_baseline_seed123.pt  
- student_baseline_seed999.pt  

KD:
- student_kd_seed42.pt  
- student_kd_seed123.pt  
- student_kd_seed999.pt  

---

## Expected outcome

- Teacher = best performance  
- Baseline = weaker but efficient  
- KD = improved baseline without increasing size  

Example results:
- Accuracy: 0.766 → 0.776  
- Macro-AUC: 0.852 → 0.891  
- Macro-F1: 0.568 → 0.598  

---

## Interpretation

- KD improves weak student performance  
- No increase in model size  
- Maintains efficiency advantages  
- Justifies further KD experiments  

---

## Notes

- Train teacher before KD  
- Metrics parsed automatically  
- Check logs/tmp if parsing fails  

---

## Related experiments

- Experiment 2 – temperature sweep  
- Experiment 3 – alpha sweep  
- Experiment 4 – capacity sweep  
- Experiment 5 – teacher comparison  
- Experiment 6 – ablation study  