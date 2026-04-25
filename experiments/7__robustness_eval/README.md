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

- maintain class-discriminative performance under perturbations
- degrade gracefully under increasing corruption

This experiment tests whether KD improves:

- generalisation
- stability under distribution shifts
- class-balanced discrimination under corrupted inputs

---

## Experimental setup

### Models compared

- Baseline student (standard training)
- KD student (trained using teacher guidance)

Both models:

- use the same architecture (`StudentCNN` small)
- are trained with identical hyperparameters except for the KD objective

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

- **Macro-AUC advantage (KD – baseline)**

This directly measures:

> how much better the KD student performs than the baseline student in class-balanced discriminative performance under each condition

Macro-AUC is used as the primary robustness metric because it is the main evaluation metric for the wider project and is more informative than accuracy under the imbalanced PTB-XL five-class setting.

Additional metrics may also be recorded in the JSON outputs, including:

- accuracy
- macro-F1
- weighted-F1
- macro-F1 advantage

However, macro-AUC advantage is the main metric used to interpret robustness in the research article.

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

```bash
sbatch experiments/7__robustness_eval/run_eval_small.slurm
```

Run all, if configured:

```bash
bash experiments/7__robustness_eval/run_all.sh
```

---

## Input dependencies

This experiment requires pre-trained models from Experiment 1:

- `student_baseline_seed*.pt`
- `student_kd_seed*.pt`

Location:

```text
experiments/1__kd_vs_baseline/checkpoints/
```

---

## Expected outcome

- KD should outperform baseline on clean data
- KD is expected to show greater robustness under perturbations
- KD should maintain a positive macro-AUC advantage under:
  - noise
  - missing leads
  - signal masking

---

## Interpretation

This experiment evaluates **robustness as a key advantage of knowledge distillation**.

Key insights:

- KD may transfer smoother, more generalisable decision boundaries
- KD may reduce overfitting to clean training data
- KD may improve resilience to corrupted inputs
- KD should be judged primarily by whether it preserves macro-AUC advantage under corruption

If KD consistently shows a positive macro-AUC advantage, this provides evidence that distillation improves class-balanced robustness under degraded ECG inputs.

---

## Notes

- No training occurs in this experiment
- Only evaluation is performed
- Ensure checkpoints exist before running
- If parsing fails, check `logs/tmp/` and JSON outputs
- Macro-F1 may be useful as a secondary metric, but macro-AUC is the primary metric for consistency with the research article

---

## Relation to other experiments

- Builds on Experiment 1 (KD vs baseline)
- Uses best settings from Experiments 2–3
- Complements Experiment 6 (ablation)
- Provides robustness evidence for the KD performance-efficiency trade-off