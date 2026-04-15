# Experiment 4: Student Capacity Sweep

## Overview

This experiment investigates how the **capacity (size) of the student model** affects performance in both standard training and knowledge distillation.

Two types of models are evaluated:
- Baseline students (no distillation)
- KD students (with distillation)

The goal is to understand how model size influences:
- performance
- efficiency
- the effectiveness of knowledge distillation

---

## Research question

How does student model capacity affect classification performance, and does knowledge distillation provide consistent benefits across different model sizes?

---

## Key idea

Model capacity determines how much information a model can learn.

- Small models:
  - Fast and efficient  
  - Limited representational power  

- Large models:
  - More expressive  
  - Better performance but higher cost  

This experiment tests whether knowledge distillation helps:
- small models more than large ones  
- or all model sizes equally  

---

## Experimental setup

### Models evaluated

Student sizes:
- small  
- medium  
- large  

For each size, two models are trained:

1. **Baseline student**
   - Standard supervised training  

2. **KD student**
   - Uses teacher guidance  

---

### Fixed components

- Teacher: TeacherCNN (from Experiment 1)  
- Temperature: T = 1  
- Alpha: 0.10 → 0.10  
- Training settings: identical across all runs  

---

## Seeds

Each configuration is evaluated with:
- 42  
- 123  
- 999  

Results are reported as mean ± standard deviation.

---

## Files

Baseline runs:
- `studentbaseline/run_small.slurm`
- `studentbaseline/run_medium.slurm`
- `studentbaseline/run_large.slurm`

KD runs:
- `studentkd/run_small.slurm`
- `studentkd/run_medium.slurm`
- `studentkd/run_large.slurm`

Main script:
- `run_all.sh`

Outputs:
- `checkpoints/`
- `logs/`
- `logs/tmp/`

---

## Metrics

For each model:

- Accuracy  
- Macro-AUC  
- Macro-F1  
- Weighted-F1  

Additional measurements:
- model parameter count  
- checkpoint size  
- inference latency  

Each script:
- runs all seeds  
- parses results automatically  
- prints mean ± std summary  

---

## How to run

Run all capacity experiments:

sbatch experiments/4__student_capacity_sweep/run_all.sh  

Run individual jobs:

Baseline:
sbatch experiments/4__student_capacity_sweep/studentbaseline/run_small.slurm  
sbatch experiments/4__student_capacity_sweep/studentbaseline/run_medium.slurm  
sbatch experiments/4__student_capacity_sweep/studentbaseline/run_large.slurm  

KD:
sbatch experiments/4__student_capacity_sweep/studentkd/run_small.slurm  
sbatch experiments/4__student_capacity_sweep/studentkd/run_medium.slurm  
sbatch experiments/4__student_capacity_sweep/studentkd/run_large.slurm  

---

## Checkpoints

Baseline:
- student_baseline_small_seed42.pt  
- student_baseline_medium_seed42.pt  
- student_baseline_large_seed42.pt  

KD:
- student_kd_small_seed42.pt  
- student_kd_medium_seed42.pt  
- student_kd_large_seed42.pt  

(and similarly for seeds 123 and 999)

---

## Expected outcome

- Larger models should achieve higher performance  
- Smaller models should be more efficient  
- Knowledge distillation is expected to:
  - significantly improve small models  
  - moderately improve medium models  
  - have smaller gains for large models  

---

## Interpretation

This experiment evaluates the trade-off between **model size, performance, and efficiency**.

Key insights:
- Model capacity directly impacts classification performance  
- Knowledge distillation is most beneficial for weaker (smaller) models  
- Larger students may already approximate teacher behaviour  

This helps identify the **optimal student size for deployment**.

---

## Notes

- All hyperparameters are fixed except model size  
- Teacher checkpoints must exist before KD runs  
- If metric parsing fails, check `logs/tmp/`  

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)  
- Uses best settings from Experiments 2 and 3  
- Explores scaling behaviour of student models  
- Helps identify best performance-efficiency trade-off  