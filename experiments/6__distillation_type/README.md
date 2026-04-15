# Experiment 6: Distillation Component Ablation

## Overview

This experiment investigates the contribution of different **knowledge distillation components** to student performance.

Instead of using the full KD loss, individual components are isolated to determine their impact on learning.

---

## Research question

Which components of the knowledge distillation framework contribute most to improving student performance?

---

## Key idea

The full KD loss consists of multiple components:

1. **Hard loss (cross-entropy)** – ground truth supervision  
2. **Soft targets (KL divergence)** – teacher probability distribution  
3. **Logit alignment (MSE)** – similarity between student and teacher outputs  
4. **Feature alignment (MSE)** – similarity between internal representations  

This experiment removes or isolates components to understand their individual importance.

---

## Experimental setup

### Configurations tested

1. **Baseline (CE only)**
   - Standard supervised learning  
   - No teacher information  

2. **Soft targets only**
   - Uses KL divergence with teacher outputs  
   - No feature alignment  

3. **Feature alignment only**
   - Matches internal representations  
   - No soft target loss  

4. **Soft targets + feature alignment**
   - Combines both KD components  
   - Excludes logit MSE  

---

### Fixed components

- Student: StudentCNN (small)  
- Teacher: TeacherCNN (from previous experiments)  
- Temperature: T = 1  
- Alpha: 0.10 → 0.10  
- Training settings: identical across runs  

---

## Seeds

Each configuration is evaluated with:
- 42  
- 123  
- 999  

Results are reported as mean ± standard deviation.

---

## Files

- `baseline_ce_only.slurm`
- `soft_targets_only.slurm`
- `feature_only.slurm`
- `soft_targets_plus_features.slurm`
- `run_all.sh`

Outputs:
- `checkpoints/`
- `logs/`
- `logs/tmp/`

---

## Metrics

For each configuration:

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

Run all ablation experiments:

sbatch experiments/6__distillation_type/run_all.sh  

Run individual configurations:

sbatch experiments/6__distillation_type/baseline_ce_only.slurm  
sbatch experiments/6__distillation_type/soft_targets_only.slurm  
sbatch experiments/6__distillation_type/feature_only.slurm  
sbatch experiments/6__distillation_type/soft_targets_plus_features.slurm  

---

## Checkpoints

Example outputs:

- student_baseline_seed42.pt  
- student_soft_targets_seed42.pt  
- student_feature_only_seed42.pt  
- student_soft_targets_features_seed42.pt  

(and similarly for seeds 123 and 999)

---

## Expected outcome

- Baseline should perform worst  
- Soft targets should provide significant improvement  
- Feature alignment alone may give limited gains  
- Combined components should perform best  

---

## Interpretation

This experiment isolates the effect of each distillation component.

Key insights:
- Soft targets are typically the most important factor  
- Feature alignment provides additional improvements  
- Combining components yields the strongest performance  
- Some components may contribute little in isolation  

This helps justify the design of the full KD loss function.

---

## Notes

- Only loss components are changed  
- Model architecture remains fixed  
- Teacher must be trained beforehand  
- If parsing fails, check `logs/tmp/`  

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)  
- Explains why KD works  
- Complements Experiments 2 and 3 (hyperparameters)  
- Provides insight into model behaviour and design choices  