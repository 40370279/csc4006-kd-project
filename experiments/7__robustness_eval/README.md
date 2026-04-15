# Experiment 7: Robustness Evaluation

## Overview

This experiment evaluates the **robustness of the student models** by comparing:

- Baseline student (no distillation)
- KD student (trained with knowledge distillation)

Both models are subjected to various **input perturbations** to assess how performance degrades under realistic noise and corruption scenarios.

---

## Research question

Does knowledge distillation improve the robustness of student models to noise, signal distortion, and missing data?

---

## Key idea

Real-world ECG signals are often noisy or incomplete. A good model should:

- maintain performance under perturbations  
- degrade gracefully under increasing corruption  

This experiment tests whether KD improves:
- generalisation  
- stability under distribution shifts  

---

## Experimental setup

### Models compared

- Baseline student (standard training)
- KD student (trained using teacher guidance)

Both models:
- use the same architecture (StudentCNN small)  
- are trained with identical hyperparameters (except KD loss)  

---

### Corruptions tested

The following perturbations are applied:

1. **Clean (no corruption)**  
2. **Gaussian noise**
   - severity = 0.05  
   - severity = 0.10  

3. **Amplitude scaling**
   - severity = 0.10  
   - severity = 0.20  

4. **Lead dropout**
   - 1 lead removed  
   - 2 leads removed  

5. **Time masking**
   - severity = 0.05  
   - severity = 0.10  

---

## Metric reported

Primary metric:
- **Macro-F1 advantage (KD – baseline)**

This directly measures:
> how much better KD performs compared to baseline under each condition

Also implicitly evaluates:
- robustness under noise  
- stability across perturbations  

---

## Seeds

Each configuration is evaluated with:
- 42  
- 123  
- 999  

Results are reported as mean ± standard deviation.

---

## Files

- `run_eval_small.slurm`
- `evaluate_robustness.py`
- `run_all.sh`

Outputs:
- `results/robustness_small_seed*.json`
- `logs/`
- `logs/tmp/`

---

## How to run

Run robustness evaluation:

sbatch experiments/7__robustness_eval/run_eval_small.slurm  

Run all (if configured):

sbatch experiments/7__robustness_eval/run_all.sh  

---

## Input dependencies

This experiment requires pre-trained models from Experiment 1:

- `student_baseline_seed*.pt`
- `student_kd_seed*.pt`

Location:
experiments/1__kd_vs_baseline/checkpoints/

---

## Expected outcome

- KD should outperform baseline on clean data  
- KD is expected to show **greater robustness under perturbations**  
- Performance gap should increase under:
  - noise  
  - missing leads  
  - signal masking  

---

## Interpretation

This experiment evaluates **robustness as a key advantage of knowledge distillation**.

Key insights:
- KD transfers smoother, more generalisable representations  
- KD reduces overfitting to clean training data  
- KD improves resilience to corrupted inputs  

If KD consistently shows positive advantage:
→ strong evidence that distillation improves real-world reliability  

---

## Notes

- No training occurs in this experiment  
- Only evaluation is performed  
- Ensure checkpoints exist before running  
- If parsing fails, check `logs/tmp/` and JSON outputs  

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)  
- Uses best settings from Experiments 2–3  
- Complements Experiment 6 (ablation)  
- Provides real-world validation of KD benefits  