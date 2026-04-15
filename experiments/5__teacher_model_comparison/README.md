# Experiment 5: Teacher Model Comparison

## Overview

This experiment investigates how the **capacity (size) of the teacher model** affects the performance of knowledge distillation.

Different teacher models are trained with varying sizes, and their effectiveness in transferring knowledge to a fixed student model is evaluated.

---

## Research question

How does the size and strength of the teacher model influence the performance of the distilled student?

---

## Key idea

In knowledge distillation, the teacher model provides guidance to the student through soft predictions.

- Smaller teachers:
  - Less accurate  
  - Provide weaker supervision  

- Larger teachers:
  - More accurate  
  - Provide richer and more informative soft targets  

This experiment tests whether a stronger teacher leads to better student performance.

---

## Experimental setup

### Teacher models

Teacher sizes evaluated:
- medium  
- large  
- xlarge  

Each teacher is trained independently.

---

### Student model

- Architecture: StudentCNN  
- Size: small  
- Training: knowledge distillation  
- Fixed parameters:
  - Temperature: T = 1  
  - Alpha: 0.10 → 0.10  

---

### Fixed components

- Dataset: PTB-XL  
- Training settings: identical across all runs  
- Student architecture: unchanged  

---

## Seeds

Each teacher configuration is evaluated with:
- 42  
- 123  
- 999  

Results are reported as mean ± standard deviation.

---

## Files

Teacher training:
- `teacher/run_medium_teacher.slurm`
- `teacher/run_large_teacher.slurm`
- `teacher/run_xlarge_teacher.slurm`

Main script:
- `run_all.sh`

Outputs:
- `checkpoints/`
- `logs/`
- `logs/tmp/`

---

## Metrics

For each teacher:

- Accuracy  
- Macro-AUC  
- Macro-F1  
- Weighted-F1  

Additionally:
- teacher performance itself  
- resulting student performance after KD  

Each script:
- runs all seeds  
- parses results automatically  
- prints mean ± std summary  

---

## How to run

Run all teacher comparison experiments:

sbatch experiments/5__teacher_model_comparison/run_all.sh  

Run individual teachers:

sbatch experiments/5__teacher_model_comparison/teacher/run_medium_teacher.slurm  
sbatch experiments/5__teacher_model_comparison/teacher/run_large_teacher.slurm  
sbatch experiments/5__teacher_model_comparison/teacher/run_xlarge_teacher.slurm  

---

## Checkpoints

Teacher checkpoints:
- teacher_medium_seed42.pt  
- teacher_large_seed42.pt  
- teacher_xlarge_seed42.pt  

Student checkpoints (KD using each teacher):
- student_kd_medium_teacher_seed42.pt  
- student_kd_large_teacher_seed42.pt  
- student_kd_xlarge_teacher_seed42.pt  

(and similarly for seeds 123 and 999)

---

## Expected outcome

- Larger teachers should achieve higher accuracy  
- Stronger teachers are expected to produce better KD students  
- However, gains may diminish as teacher size increases  

---

## Interpretation

This experiment evaluates the impact of **teacher quality on knowledge distillation**.

Key insights:
- Better teachers provide more useful soft targets  
- Student performance depends on teacher strength  
- There may be diminishing returns beyond a certain teacher size  

This helps determine whether increasing teacher capacity is worthwhile.

---

## Notes

- Student architecture is fixed  
- Only teacher size is varied  
- Teacher must be trained before KD student runs  
- If parsing fails, check `logs/tmp/`  

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)  
- Uses best KD settings from Experiments 2 and 3  
- Complements Experiment 4 (student capacity)  
- Explores the role of teacher strength in KD  