# Experiment 2: Temperature Sweep for Knowledge Distillation

## Overview

This experiment investigates the effect of the **temperature parameter (T)** in knowledge distillation on the performance of a small student model.

The teacher and student architectures remain fixed, and the only parameter that is varied is the **temperature used in the soft target distribution**.

---

## Research question

How does the distillation temperature influence the effectiveness of knowledge transfer from the teacher to the student?

---

## Key idea

In knowledge distillation, the temperature controls how “soft” the teacher’s output probabilities are.

- Low temperature (T = 1):
  - Produces sharp probability distributions  
  - Similar to standard training  

- High temperature (T > 1):
  - Produces softer distributions  
  - Reveals relationships between classes (“dark knowledge”)  

This experiment tests whether softer targets improve student learning.

---

## Experimental setup

### Fixed components
- Teacher: TeacherCNN (pretrained from Experiment 1)
- Student: StudentCNN (small)
- Alpha: 0.10 → 0.10 (fixed)
- Training settings: identical across all runs

### Variable
- Temperature (T)

Values tested:
- T = 1  
- T = 2  
- T = 4  
- T = 8  
- T = 16  
- T = 32  

---

## Seeds

Each temperature is evaluated with:
- 42  
- 123  
- 999  

Final results are reported as mean ± standard deviation.

---

## Files

- `run_T1.slurm`
- `run_T2.slurm`
- `run_T4.slurm`
- `run_T8.slurm`
- `run_T16.slurm`
- `run_T32.slurm`
- `run_all.sh`

Outputs:
- `checkpoints/`
- `logs/`
- `logs/tmp/`

---

## Metrics

For each temperature:

- Accuracy  
- Macro-AUC  
- Macro-F1  
- Weighted-F1  

Each script:
- runs all seeds  
- parses results automatically  
- prints mean ± std summary  

---

## How to run

Run all temperature experiments:

sbatch experiments/2__temperature_sweep/run_all.sh  

Run a single temperature:

sbatch experiments/2__temperature_sweep/run_T1.slurm  
sbatch experiments/2__temperature_sweep/run_T2.slurm  
sbatch experiments/2__temperature_sweep/run_T4.slurm  
sbatch experiments/2__temperature_sweep/run_T8.slurm  
sbatch experiments/2__temperature_sweep/run_T16.slurm  
sbatch experiments/2__temperature_sweep/run_T32.slurm  

---

## Checkpoints

Each run saves student checkpoints:

- student_T1_seed42.pt  
- student_T2_seed42.pt  
- student_T4_seed42.pt  
- ...  

(and similarly for seeds 123 and 999)

---

## Expected outcome

- T = 1 behaves like standard KD (limited soft information)  
- Moderate temperatures (T = 2–8) are expected to perform best  
- Very high temperatures (T = 16–32) may degrade performance due to overly smooth targets  

---

## Interpretation

This experiment helps identify the **optimal temperature for distillation**.

Key insights:
- Temperature controls the quality of knowledge transfer  
- Moderate softening improves generalisation  
- Too much smoothing reduces useful signal  

The best-performing temperature is used in later experiments.

---

## Notes

- Only temperature is changed — all other hyperparameters remain constant  
- Teacher checkpoints must exist before running this experiment  
- If metric parsing fails, check `logs/tmp/`  

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)  
- Helps tune KD hyperparameters  
- Informs later experiments using the best temperature  