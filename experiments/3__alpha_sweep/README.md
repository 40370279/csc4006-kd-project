# Experiment 3: Alpha Sweep for Knowledge Distillation

## Overview

This experiment investigates the effect of the **alpha parameter (α)** in knowledge distillation.

The teacher and student architectures remain fixed, and the temperature is fixed at **T = 1**.  
The only parameter varied is **alpha**, which controls the balance between hard labels and teacher guidance.

---

## Research question

How does the weighting between hard-label supervision and teacher knowledge affect student performance?

---

## Key idea

In knowledge distillation, alpha controls the trade-off between:

- **Hard loss (ground truth labels)**
- **Soft loss (teacher predictions)**

Interpretation:

- Low alpha (e.g. 0.1):
  - Strong reliance on teacher knowledge  
  - Less emphasis on true labels  

- High alpha (e.g. 0.9):
  - Mostly standard supervised learning  
  - Minimal influence from teacher  

This experiment tests how much teacher guidance is actually beneficial.

---

## Experimental setup

### Fixed components
- Teacher: TeacherCNN (from Experiment 1)
- Student: StudentCNN (small)
- Temperature: T = 1
- Training settings: identical across all runs

### Variable
- Alpha (α)

Values tested:
- α = 0.1  
- α = 0.3  
- α = 0.5  
- α = 0.7  
- α = 0.9  

---

## Seeds

Each alpha value is evaluated with:
- 42  
- 123  
- 999  

Final results are reported as mean ± standard deviation.

---

## Files

- `run_A01.slurm`
- `run_A03.slurm`
- `run_A05.slurm`
- `run_A07.slurm`
- `run_A09.slurm`
- `run_all.sh`

Outputs:
- `checkpoints/`
- `logs/`
- `logs/tmp/`

---

## Metrics

For each alpha value:

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

Run all alpha experiments:

sbatch experiments/3__alpha_sweep/run_all.sh  

Run a single alpha:

sbatch experiments/3__alpha_sweep/run_A01.slurm  
sbatch experiments/3__alpha_sweep/run_A03.slurm  
sbatch experiments/3__alpha_sweep/run_A05.slurm  
sbatch experiments/3__alpha_sweep/run_A07.slurm  
sbatch experiments/3__alpha_sweep/run_A09.slurm  

---

## Checkpoints

Each run saves student checkpoints:

- student_A01_seed42.pt  
- student_A03_seed42.pt  
- student_A05_seed42.pt  
- ...  

(and similarly for seeds 123 and 999)

---

## Expected outcome

- Low alpha (0.1–0.3):
  - Strong teacher influence  
  - May improve generalisation  

- Medium alpha (0.5):
  - Balanced learning  
  - Often performs best  

- High alpha (0.7–0.9):
  - Behaves like baseline training  
  - Reduced KD benefit  

---

## Interpretation

This experiment determines the **optimal balance between teacher knowledge and ground truth labels**.

Key insights:
- Too little teacher influence limits KD benefits  
- Too much teacher influence may reduce alignment with true labels  
- A balanced alpha often yields the best performance  

The best alpha value is used in later experiments.

---

## Notes

- Only alpha is changed — all other hyperparameters remain constant  
- Temperature is fixed at T = 1  
- Teacher checkpoints must exist before running  
- If metric parsing fails, check `logs/tmp/`  

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)  
- Complements Experiment 2 (temperature sweep)  
- Helps tune KD hyperparameters for later experiments  