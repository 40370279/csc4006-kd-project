# Experiment 7: Augmentation Ablation

This experiment tests the effect of data augmentation on medium student models for both baseline training and KD training.

## Conditions

- studentbaseline / no augmentation
- studentbaseline / augmentation
- studentkd / no augmentation
- studentkd / augmentation

## Fixed settings

- student size: medium
- seeds: 42, 123, 999

## Goal

To determine:
- whether augmentation improves baseline student performance
- whether augmentation improves KD student performance
- whether KD remains beneficial even when augmentation is disabled
- whether KD and augmentation provide complementary gains