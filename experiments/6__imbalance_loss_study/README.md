# Experiment 6: Imbalance Loss Study

## Goal
Investigate the impact of loss functions on class imbalance in ECG classification.

## Motivation
Previous experiments showed poor performance on minority classes (especially class 1).
This experiment evaluates whether focal loss improves macro-F1.

## Setup
- Teacher: Strong Teacher
- Student: Medium
- KD: alpha=0.5, T=4
- Seeds: 42, 123, 999

## Variants
1. Cross Entropy (baseline)
2. Focal Loss (gamma=1)
3. Focal Loss (gamma=2)
4. Focal Loss (gamma=3)

## Metrics
- Accuracy
- Macro F1 (PRIMARY)
- Weighted F1

## Expected Outcome
Focal loss should improve macro-F1 by improving minority class recall.