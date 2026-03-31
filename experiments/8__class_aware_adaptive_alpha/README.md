# Experiment 8: Class-Aware Adaptive Alpha

This experiment compares two knowledge distillation settings:

1. Fixed alpha KD
2. Class-aware adaptive alpha KD

The motivation is that previous experiments showed:
- strong teacher influence (low alpha) improves macro F1
- KD particularly helps minority classes
- PTB-XL is class imbalanced

To address this, the adaptive method applies lower alpha to minority-class samples,
increasing reliance on teacher supervision where it is most needed.

## Modes

- `fixed`
  - standard KD using a single batch-wide alpha
- `classwise_adaptive`
  - per-sample alpha based on class frequency

## Output

Logs:
- `experiments/8__class_aware_adaptive_alpha/logs/`

Checkpoints:
- `experiments/8__class_aware_adaptive_alpha/checkpoints/`